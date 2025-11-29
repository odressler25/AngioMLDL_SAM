"""
Test SAM 3 with exemplar-based prompting using YOUR expert data.

Strategy:
1. Select 5 diverse exemplar cases (RCA, LAD, LCX)
2. For each exemplar, extract bounding box from vessel mask
3. Use these boxes as geometric prompts for SAM 3
4. Test if SAM 3 can learn "coronary vessel" concept from examples
5. Evaluate on held-out test cases

This is Phase 1 of the exemplar strategy from SAM3_EXEMPLAR_STRATEGY.md
"""

import numpy as np
import torch
from pathlib import Path
import json
from PIL import Image
import cv2
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

# Import SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_tight_bbox(mask: np.ndarray, padding: int = 10) -> List[int]:
    """
    Get tight bounding box around vessel mask.

    Args:
        mask: Binary mask (H, W)
        padding: Pixels to pad around bbox

    Returns:
        [x1, y1, x2, y2] in pixel coordinates
    """
    # Find vessel pixels
    vessel_pixels = np.argwhere(mask > 0)

    if len(vessel_pixels) == 0:
        return None

    # Get bounding box (note: argwhere returns [row, col] = [y, x])
    y_coords = vessel_pixels[:, 0]
    x_coords = vessel_pixels[:, 1]

    y1 = max(0, y_coords.min() - padding)
    y2 = min(mask.shape[0], y_coords.max() + padding)
    x1 = max(0, x_coords.min() - padding)
    x2 = min(mask.shape[1], x_coords.max() + padding)

    # Return in [x1, y1, x2, y2] format
    return [int(x1), int(y1), int(x2), int(y2)]


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert normalized float image to uint8."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    return image


def load_exemplar_case(case_id: str, base_path: Path, frame_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load exemplar image, mask, and metadata.

    Args:
        case_id: e.g., "101-0025_MID_RCA_PRE"
        base_path: Path to corrected_vessel_dataset
        frame_idx: Which frame from cine to use

    Returns:
        (image, mask, metadata)
    """
    # Load cine
    cine_path = base_path / "cines" / f"{case_id}_cine.npy"
    cine = np.load(cine_path)
    image = cine[frame_idx]  # Shape: (H, W)

    # Load mask
    mask_path = base_path / "vessel_masks" / f"{case_id}_mask.npy"
    mask = np.load(mask_path)

    # Load metadata
    json_path = base_path / "contours" / f"{case_id}_contours.json"
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    return image, mask, metadata


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculate Intersection over Union."""
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask.astype(np.uint8),
                              (gt_mask.shape[1], gt_mask.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def test_with_exemplar_bboxes(
    sam_processor,
    test_image: np.ndarray,
    exemplar_bboxes: List[List[int]],
    ground_truth_mask: np.ndarray,
    case_id: str
) -> Dict:
    """
    Test SAM 3 with geometric prompts from exemplar bounding boxes.

    Args:
        sam_processor: SAM 3 processor
        test_image: Test angiogram (H, W)
        exemplar_bboxes: List of [x1, y1, x2, y2] from exemplars
        ground_truth_mask: GT mask for evaluation
        case_id: Test case identifier

    Returns:
        Dictionary with results
    """
    # Convert to uint8 and PIL
    test_image_uint8 = normalize_to_uint8(test_image)
    pil_image = Image.fromarray(test_image_uint8).convert("RGB")

    # Set image in SAM 3
    state = sam_processor.set_image(pil_image)

    # Add geometric prompts from exemplars
    for i, bbox in enumerate(exemplar_bboxes):
        state = sam_processor.add_geometric_prompt(
            bbox=bbox,
            label=f"vessel_exemplar_{i}",
            state=state
        )

    # Get predictions
    try:
        output = sam_processor._forward_grounding(state)

        # Extract masks
        masks = output.get("masks", [])
        scores = output.get("scores", [])

        # Convert tensors to numpy
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        # Combine all predicted masks
        if masks is not None and len(masks) > 0:
            combined_mask = np.zeros_like(ground_truth_mask)
            for mask in masks:
                if len(mask.shape) == 3:
                    mask = mask[0]  # Remove batch dim if present

                # Resize to match GT
                if mask.shape != ground_truth_mask.shape:
                    mask = cv2.resize(mask.astype(np.uint8),
                                    (ground_truth_mask.shape[1], ground_truth_mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

                combined_mask = np.logical_or(combined_mask, mask > 0.5)

            # Calculate metrics
            iou = calculate_iou(combined_mask, ground_truth_mask)
            dice = 2 * iou / (1 + iou) if iou > 0 else 0.0

            avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0

            return {
                'case_id': case_id,
                'iou': float(iou),
                'dice': float(dice),
                'num_masks': len(masks),
                'avg_score': avg_score,
                'success': True
            }
        else:
            return {
                'case_id': case_id,
                'iou': 0.0,
                'dice': 0.0,
                'num_masks': 0,
                'avg_score': 0.0,
                'success': False,
                'error': 'No masks predicted'
            }

    except Exception as e:
        logger.error(f"Error processing {case_id}: {e}")
        return {
            'case_id': case_id,
            'iou': 0.0,
            'dice': 0.0,
            'num_masks': 0,
            'avg_score': 0.0,
            'success': False,
            'error': str(e)
        }


def main():
    """
    Main exemplar-based testing pipeline.
    """
    # Paths
    base_path = Path(r"E:\AngioMLDL_data\corrected_vessel_dataset")

    # Define exemplar cases (diverse vessels and views) - VERIFIED TO EXIST
    exemplar_cases = [
        "101-0025_MID_RCA_PRE",   # RCA
        "101-0086_MID_LAD_PRE",   # LAD
        "101-0052_DIST_LCX_PRE",  # LCX
    ]

    # Define test cases (different from exemplars) - VERIFIED TO EXIST
    test_cases = [
        "101-0058_MID_RCA_PRE",   # RCA (different patient)
        "101-0095_PROX_LAD_PRE",  # LAD (different location)
        "101-0052_OM2_SB_PRE",    # OM branch
    ]

    # Initialize SAM 3
    logger.info("Loading SAM 3 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam_model = build_sam3_image_model(device=device)
    sam_processor = Sam3Processor(sam_model)

    # Load exemplars and extract bounding boxes
    logger.info(f"Loading {len(exemplar_cases)} exemplar cases...")
    exemplar_bboxes = []

    for case_id in exemplar_cases:
        try:
            image, mask, metadata = load_exemplar_case(case_id, base_path, frame_idx=0)
            bbox = get_tight_bbox(mask, padding=20)

            if bbox is not None:
                exemplar_bboxes.append(bbox)
                vessel = metadata.get('segment_name', 'Unknown')
                logger.info(f"  {case_id}: {vessel}, bbox={bbox}")
            else:
                logger.warning(f"  {case_id}: No vessel pixels found in mask")

        except Exception as e:
            logger.error(f"  {case_id}: Failed to load - {e}")

    logger.info(f"Loaded {len(exemplar_bboxes)} exemplar bounding boxes")

    if len(exemplar_bboxes) == 0:
        logger.error("No exemplar bboxes loaded. Exiting.")
        return

    # Test on held-out cases
    logger.info(f"\nTesting on {len(test_cases)} held-out cases...")
    results = []

    for case_id in tqdm(test_cases, desc="Testing"):
        try:
            # Load test case
            test_image, gt_mask, metadata = load_exemplar_case(case_id, base_path, frame_idx=0)

            # Test with exemplar bboxes
            result = test_with_exemplar_bboxes(
                sam_processor,
                test_image,
                exemplar_bboxes,
                gt_mask,
                case_id
            )

            # Add metadata
            result['vessel'] = metadata.get('segment_name', 'Unknown')
            result['view_angles'] = metadata.get('view_angles', {})

            results.append(result)

            logger.info(f"  {case_id}: IoU={result['iou']:.3f}, Dice={result['dice']:.3f}, "
                       f"Masks={result['num_masks']}, Score={result['avg_score']:.3f}")

        except Exception as e:
            logger.error(f"  {case_id}: Failed - {e}")
            results.append({
                'case_id': case_id,
                'iou': 0.0,
                'dice': 0.0,
                'success': False,
                'error': str(e)
            })

    # Summary statistics
    successful = [r for r in results if r['success']]
    if successful:
        avg_iou = np.mean([r['iou'] for r in successful])
        avg_dice = np.mean([r['dice'] for r in successful])
        logger.info(f"\n=== SUMMARY ({len(successful)}/{len(results)} successful) ===")
        logger.info(f"Average IoU: {avg_iou:.3f}")
        logger.info(f"Average Dice: {avg_dice:.3f}")
    else:
        logger.warning("\n=== NO SUCCESSFUL PREDICTIONS ===")

    # Save results
    output_dir = Path("sam3_test_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "sam3_exemplar_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'exemplar_cases': exemplar_cases,
            'test_cases': test_cases,
            'results': results,
            'summary': {
                'avg_iou': float(np.mean([r['iou'] for r in successful])) if successful else 0.0,
                'avg_dice': float(np.mean([r['dice'] for r in successful])) if successful else 0.0,
                'success_rate': len(successful) / len(results) if results else 0.0
            }
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
