"""
Test SAM 3's ability to segment coronary vessels using TEXT PROMPTS ONLY.

This is the critical first experiment:
- Can SAM 3 understand coronary anatomy from text descriptions?
- Can it segment vessels without point/box prompts?
- Can it distinguish LAD from LCX in multi-vessel views?

If this works well (IoU > 0.5), then SAM 3 could be the segmentation backbone
for fully automated QCA with minimal fine-tuning.

If not, we'll need to train a seed point detector to prompt SAM 3.
"""

import numpy as np
import cv2
import pydicom
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

# Import SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Import our utilities
from json_to_masks import (
    load_json_measurements,
    json_contours_to_mask,
    extract_vessel_info,
    visualize_mask_overlay,
    get_diverse_test_cases
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TEXT PROMPTS TO TEST
# Starting broad, then getting more specific
TEXT_PROMPTS = {
    'generic_vessel': [
        "blood vessel",
        "artery",
        "vessel with contrast",
        "coronary artery",
    ],
    'specific_vessels': [
        "LAD coronary artery",
        "left anterior descending artery",
        "LCX coronary artery",
        "left circumflex artery",
        "RCA coronary artery",
        "right coronary artery",
    ],
    'anatomical': [
        "coronary vessel lumen",
        "contrast-filled coronary artery",
        "epicardial coronary artery",
    ],
    'pathology': [
        "stenosis",
        "narrowed artery",
        "coronary lesion",
    ]
}


def load_cine_frame(cine_path: str, frame_idx: int) -> np.ndarray:
    """
    Load a single frame from numpy cine file.

    Args:
        cine_path: Path to .npy cine file
        frame_idx: Frame index to load

    Returns:
        Image array (grayscale, normalized to 0-255)
    """
    # Load cine (shape: num_frames x height x width)
    cine = np.load(cine_path)

    # Get specified frame
    if len(cine.shape) == 3:
        image = cine[frame_idx]
        logger.info(f"Loaded frame {frame_idx}/{cine.shape[0]} from cine")
    else:
        # Single frame
        image = cine
        logger.info("Single frame cine")

    # Normalize to 0-255
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255).astype(np.uint8)

    return image


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Dice coefficient between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    if total == 0:
        return 0.0

    return 2 * intersection / total


def test_sam3_single_image(
    sam_processor: Sam3Processor,
    inference_state,
    image: np.ndarray,
    text_prompts: List[str],
    ground_truth_mask: np.ndarray,
    vessel_name: str
) -> Dict:
    """
    Test multiple text prompts on a single image.

    Args:
        sam_processor: SAM 3 processor
        inference_state: SAM 3 inference state for the image
        image: Angiogram image (H, W) grayscale
        text_prompts: List of text prompts to test
        ground_truth_mask: Expert-labeled mask from JSON
        vessel_name: Name of vessel (for logging)

    Returns:
        Dict with results for each prompt:
        - prompt: text prompt
        - mask: predicted mask
        - iou: IoU vs ground truth
        - dice: Dice vs ground truth
        - num_pixels: number of predicted vessel pixels
    """
    results = []

    for prompt in tqdm(text_prompts, desc=f"Testing prompts for {vessel_name}"):
        try:
            # SAM 3 text-based segmentation
            output = sam_processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

            masks = output.get("masks", [])
            boxes = output.get("boxes", [])
            scores = output.get("scores", [])

            # Convert tensors to numpy if needed
            import torch
            if masks is not None and len(masks) > 0:
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                num_masks = len(masks) if hasattr(masks, '__len__') else 0
            else:
                masks = []
                num_masks = 0

            if scores is not None and len(scores) > 0:
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
            else:
                scores = []

            # Combine all masks for this prompt into a single binary mask
            if num_masks > 0:
                # Convert masks to binary and combine
                combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                for mask in masks:
                    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                    combined_mask = np.logical_or(combined_mask, mask_np > 0.5).astype(np.uint8)

                pred_mask = combined_mask
                avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
            else:
                pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                avg_score = 0.0

            # Calculate metrics
            iou = calculate_iou(pred_mask, ground_truth_mask)
            dice = calculate_dice(pred_mask, ground_truth_mask)

            results.append({
                'prompt': prompt,
                'mask': pred_mask,
                'iou': float(iou),
                'dice': float(dice),
                'num_pixels': int(pred_mask.sum()),
                'num_objects': int(num_masks),
                'score': float(avg_score)
            })

            logger.info(f"  '{prompt}': IoU={iou:.3f}, Dice={dice:.3f}, Pixels={pred_mask.sum()}, Objects={num_masks}")

        except Exception as e:
            logger.error(f"Error with prompt '{prompt}': {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'prompt': prompt,
                'mask': np.zeros(image.shape[:2], dtype=np.uint8),
                'iou': 0.0,
                'dice': 0.0,
                'num_pixels': 0,
                'num_objects': 0,
                'score': 0.0,
                'error': str(e)
            })

    return results


def visualize_results(
    image: np.ndarray,
    ground_truth: np.ndarray,
    results: List[Dict],
    vessel_name: str,
    save_path: str
):
    """
    Create visualization comparing different text prompts.

    Args:
        image: Original angiogram
        ground_truth: Expert-labeled mask
        results: List of prediction results from test_sam3_single_image
        vessel_name: Vessel name for title
        save_path: Path to save visualization
    """
    # Select top 4 prompts by IoU
    sorted_results = sorted(results, key=lambda x: x['iou'], reverse=True)
    top_results = sorted_results[:4]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{vessel_name} - SAM 3 Text Prompt Results', fontsize=16)

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Ground truth
    gt_overlay = visualize_mask_overlay(image, ground_truth, color=(0, 255, 0), alpha=0.4)
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title(f'Ground Truth\n({ground_truth.sum()} pixels)')
    axes[0, 1].axis('off')

    # Best prompt
    if len(top_results) > 0:
        best = top_results[0]
        best_overlay = visualize_mask_overlay(image, best['mask'], color=(255, 0, 0), alpha=0.4)
        axes[0, 2].imshow(best_overlay)
        axes[0, 2].set_title(
            f"Best: '{best['prompt']}'\n"
            f"IoU={best['iou']:.3f}, Dice={best['dice']:.3f}"
        )
        axes[0, 2].axis('off')

    # Top 3 other prompts
    for i, result in enumerate(top_results[1:4]):
        row = 1
        col = i
        overlay = visualize_mask_overlay(image, result['mask'], color=(0, 0, 255), alpha=0.4)
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(
            f"'{result['prompt']}'\n"
            f"IoU={result['iou']:.3f}, Dice={result['dice']:.3f}"
        )
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization saved to {save_path}")


def main():
    """
    Main testing pipeline:
    1. Load 5 diverse test cases
    2. Initialize SAM 3
    3. Test text prompts on each case
    4. Generate visualizations and metrics
    5. Summarize results
    """
    # Paths
    csv_path = r"E:\AngioMLDL_data\corrected_dataset.csv"
    output_dir = Path(r"C:\Users\odressler\AngioMLDL_SAM\sam3_test_results")
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("SAM 3 TEXT PROMPT TEST FOR CORONARY ARTERY SEGMENTATION")
    logger.info("=" * 80)

    # Step 1: Select diverse test cases
    logger.info("\n[1/5] Selecting diverse test cases...")
    test_cases = get_diverse_test_cases(csv_path, n_cases=5)

    logger.info(f"Selected {len(test_cases)} test cases:")
    for i, case in enumerate(test_cases):
        logger.info(f"  {i+1}. {case['patient_id']} - {case['vessel_pattern']} {case['phase']} ({case['main_vessel']})")

    # Step 2: Initialize SAM 3
    logger.info("\n[2/5] Initializing SAM 3 model...")
    try:
        # Build SAM 3 model
        logger.info("Building SAM 3 image model (this may download checkpoints)...")
        sam_model = build_sam3_image_model(device='cuda')

        # Create processor
        sam_processor = Sam3Processor(sam_model)

        logger.info("SAM 3 initialized successfully on CUDA")
    except Exception as e:
        logger.error(f"Failed to initialize SAM 3: {e}")
        logger.error("Make sure you have:")
        logger.error("1. Requested access to SAM 3 checkpoints on Hugging Face")
        logger.error("2. Logged in with: huggingface-cli login")
        raise

    # Step 3: Test each case
    logger.info("\n[3/5] Testing text prompts on each case...")

    all_results = []

    for i, case in enumerate(test_cases):
        case_id = f"{case['patient_id']}_{case['vessel_pattern']}_{case['phase']}"
        logger.info(f"\n--- Test Case {i+1}/{len(test_cases)}: {case_id} ---")

        # Load cine frame
        cine_path = case['cine_path']
        frame_idx = case['frame_index']
        image = load_cine_frame(cine_path, frame_idx)
        logger.info(f"Image shape: {image.shape}")

        # Load JSON and create ground truth mask
        json_path = case['contours_path']
        json_data = load_json_measurements(json_path)
        vessel_info = extract_vessel_info(json_data)
        gt_mask = json_contours_to_mask(json_data, image_shape=image.shape[:2])

        logger.info(f"Vessel: {vessel_info['vessel']}, CASS: {vessel_info['cass_segment']}")
        logger.info(f"View angles: {vessel_info['view_angles']}")
        logger.info(f"Ground truth: {gt_mask.sum()} vessel pixels")

        # Convert grayscale to RGB for SAM 3
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        # Set image in SAM 3 processor
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image_rgb)
        inference_state = sam_processor.set_image(pil_image)

        # Collect all prompts to test
        all_prompts = []
        for category_prompts in TEXT_PROMPTS.values():
            all_prompts.extend(category_prompts)

        # Test SAM 3 with text prompts
        results = test_sam3_single_image(
            sam_processor=sam_processor,
            inference_state=inference_state,
            image=image,
            text_prompts=all_prompts,
            ground_truth_mask=gt_mask,
            vessel_name=case_id
        )

        # Save results
        case_result = {
            'case_id': case_id,
            'patient_id': case['patient_id'],
            'vessel_pattern': case['vessel_pattern'],
            'phase': case['phase'],
            'vessel': vessel_info['vessel'],
            'cass_segment': vessel_info['cass_segment'],
            'view_angles': vessel_info['view_angles'],
            'prompt_results': results
        }
        all_results.append(case_result)

        # Visualize
        vis_path = output_dir / f"case_{i+1}_{case['patient_id']}_{vessel_info['vessel']}.png"
        visualize_results(image, gt_mask, results, case_id, str(vis_path))

    # Step 4: Aggregate metrics
    logger.info("\n[4/5] Aggregating results across all cases...")

    # Calculate per-prompt average performance
    prompt_stats = {}
    for case_result in all_results:
        for prompt_result in case_result['prompt_results']:
            prompt = prompt_result['prompt']
            if prompt not in prompt_stats:
                prompt_stats[prompt] = {'ious': [], 'dices': []}

            if 'error' not in prompt_result:
                prompt_stats[prompt]['ious'].append(prompt_result['iou'])
                prompt_stats[prompt]['dices'].append(prompt_result['dice'])

    # Summarize
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: AVERAGE PERFORMANCE BY TEXT PROMPT")
    logger.info("=" * 80)

    prompt_summary = []
    for prompt, stats in prompt_stats.items():
        if len(stats['ious']) > 0:
            avg_iou = np.mean(stats['ious'])
            avg_dice = np.mean(stats['dices'])
            std_iou = np.std(stats['ious'])

            prompt_summary.append({
                'prompt': prompt,
                'avg_iou': avg_iou,
                'avg_dice': avg_dice,
                'std_iou': std_iou,
                'n_cases': len(stats['ious'])
            })

    # Sort by average IoU
    prompt_summary.sort(key=lambda x: x['avg_iou'], reverse=True)

    logger.info("\nTop 10 Prompts:")
    for i, summary in enumerate(prompt_summary[:10]):
        logger.info(
            f"{i+1:2d}. '{summary['prompt']:<40s}' - "
            f"IoU: {summary['avg_iou']:.3f} ± {summary['std_iou']:.3f}, "
            f"Dice: {summary['avg_dice']:.3f}"
        )

    # Step 5: Save detailed results
    logger.info("\n[5/5] Saving results...")

    # Save summary as JSON
    results_json = output_dir / 'sam3_text_prompt_results.json'
    with open(results_json, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for case_result in all_results:
            case_copy = case_result.copy()
            case_copy['prompt_results'] = [
                {k: v.tolist() if isinstance(v, np.ndarray) else v
                 for k, v in pr.items() if k != 'mask'}
                for pr in case_result['prompt_results']
            ]
            serializable_results.append(case_copy)

        json.dump({
            'test_cases': serializable_results,
            'prompt_summary': prompt_summary
        }, f, indent=2)

    logger.info(f"Results saved to {results_json}")

    # Print final assessment
    logger.info("\n" + "=" * 80)
    logger.info("ASSESSMENT")
    logger.info("=" * 80)

    best_avg_iou = prompt_summary[0]['avg_iou'] if prompt_summary else 0.0

    if best_avg_iou > 0.7:
        logger.info("✅ EXCELLENT: SAM 3 text prompts work well for coronary arteries!")
        logger.info("   → Proceed with view-angle based prompting (Task 4)")
        logger.info("   → Can build fully automated pipeline with minimal fine-tuning")
    elif best_avg_iou > 0.5:
        logger.info("✓ GOOD: SAM 3 shows promise but needs improvement")
        logger.info("   → Consider fine-tuning SAM 3 on coronary artery data")
        logger.info("   → Or combine with seed point detector")
    elif best_avg_iou > 0.3:
        logger.info("⚠ MODERATE: SAM 3 text prompts have limited success")
        logger.info("   → Recommend training seed point detector → SAM 3 point prompts")
        logger.info("   → Or try SAM-VMNet (medical-specialized)")
    else:
        logger.info("❌ POOR: SAM 3 text prompts don't work for coronary arteries")
        logger.info("   → Fallback: Train custom segmentation model")
        logger.info("   → Or use seed point detector + SAM 3 point-based prompting")

    logger.info(f"\nBest prompt: '{prompt_summary[0]['prompt']}'")
    logger.info(f"Best average IoU: {best_avg_iou:.3f}")
    logger.info(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
