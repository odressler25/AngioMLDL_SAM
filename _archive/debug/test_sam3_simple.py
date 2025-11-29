"""
Simple test of SAM 3 text prompts on angiograms with FIXED API.

Tests if SAM 3 can segment coronary vessels from text prompts alone.
Uses the cases we KNOW exist.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
import json
import logging
from typing import Dict

# Import SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Import our utilities
from json_to_masks import json_contours_to_mask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TEXT PROMPTS TO TEST
TEXT_PROMPTS = [
    # Generic vessel terms
    "blood vessel",
    "artery",
    "vessel",
    "coronary artery",

    # Medical/anatomical terms
    "coronary vessel",
    "cardiac vessel",
    "heart vessel",
    "contrast-filled vessel",

    # X-ray specific
    "bright tube",
    "white vessel",
    "contrast agent",
    "radiopaque structure",
]


def test_single_case(sam_processor, case_id: str, base_path: Path):
    """
    Test one angiogram with text prompts.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {case_id}")
    logger.info(f"{'='*60}")

    # Load cine (take first frame)
    cine_path = base_path / "cines" / f"{case_id}_cine.npy"
    cine = np.load(cine_path)
    image = cine[0]  # First frame

    # Convert to uint8 RGB
    image_uint8 = (image * 255).astype(np.uint8)
    image_rgb = np.stack([image_uint8] * 3, axis=-1)
    pil_image = Image.fromarray(image_rgb)

    logger.info(f"Image shape: {image.shape}")

    # Load ground truth mask
    mask_path = base_path / "vessel_masks" / f"{case_id}_mask.npy"
    gt_mask = np.load(mask_path)
    gt_pixels = gt_mask.sum()
    logger.info(f"Ground truth: {gt_pixels:,} vessel pixels")

    # Load metadata
    json_path = base_path / "contours" / f"{case_id}_contours.json"
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    vessel = metadata.get('segment_name', 'Unknown')
    logger.info(f"Vessel: {vessel}")

    # Set image in SAM 3
    state = sam_processor.set_image(pil_image)

    # Test each prompt
    results = {}
    best_iou = 0.0
    best_prompt = None

    for prompt in TEXT_PROMPTS:
        try:
            # Use FIXED API call
            output = sam_processor.set_text_prompt(
                state=state,
                prompt=prompt
            )

            # Extract masks
            masks = output.get("masks", [])
            scores = output.get("scores", [])

            # Convert tensors
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            num_masks = len(masks) if hasattr(masks, '__len__') else 0

            # Calculate IoU if masks found
            iou = 0.0
            if num_masks > 0:
                # Take best mask
                best_mask = masks[0]
                if len(best_mask.shape) == 3:
                    best_mask = best_mask[0]

                # Resize to match GT
                if best_mask.shape != gt_mask.shape:
                    import cv2
                    best_mask = cv2.resize(
                        best_mask.astype(np.float32),
                        (gt_mask.shape[1], gt_mask.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )

                # Calculate IoU
                intersection = np.logical_and(best_mask > 0.5, gt_mask > 0).sum()
                union = np.logical_or(best_mask > 0.5, gt_mask > 0).sum()
                if union > 0:
                    iou = float(intersection / union)

                if iou > best_iou:
                    best_iou = iou
                    best_prompt = prompt

            results[prompt] = {
                'num_masks': num_masks,
                'iou': iou,
                'score': float(scores[0]) if len(scores) > 0 else 0.0
            }

            # Log result
            if num_masks > 0:
                logger.info(f"  '{prompt}': {num_masks} masks, IoU={iou:.3f}")
            else:
                logger.info(f"  '{prompt}': No masks found")

        except Exception as e:
            logger.error(f"  '{prompt}': Error - {e}")
            results[prompt] = {'error': str(e)}

    # Summary for this case
    logger.info(f"\nBest result: '{best_prompt}' with IoU={best_iou:.3f}")

    return results, best_iou


def main():
    """
    Test SAM 3 on known angiogram cases.
    """
    print("Starting SAM 3 test...")  # Direct print to ensure output
    logger.info("=== SAM 3 Text Prompt Test (Fixed API) ===\n")

    # Paths
    base_path = Path(r"E:\AngioMLDL_data\corrected_vessel_dataset")
    print(f"Data path: {base_path}")  # Debug print

    # Test cases we KNOW exist
    test_cases = [
        "101-0025_MID_RCA_PRE",   # RCA
        "101-0086_MID_LAD_PRE",   # LAD
        "101-0052_DIST_LCX_PRE",  # LCX
    ]

    # Initialize SAM 3
    print("About to load SAM 3...")  # Debug print
    logger.info("Loading SAM 3 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    print(f"Device: {device}")  # Debug print

    try:
        sam_model = build_sam3_image_model(device=device)
        sam_processor = Sam3Processor(sam_model)
        logger.info("SAM 3 loaded successfully\n")
        print("SAM 3 loaded!")  # Debug print
    except Exception as e:
        print(f"ERROR loading SAM 3: {e}")
        raise

    # Test each case
    all_results = []

    print(f"Testing {len(test_cases)} cases...")  # Debug print

    for i, case_id in enumerate(test_cases):
        print(f"\nProcessing case {i+1}/{len(test_cases)}: {case_id}")  # Debug print
        try:
            results, best_iou = test_single_case(sam_processor, case_id, base_path)
            all_results.append({
                'case_id': case_id,
                'best_iou': best_iou,
                'results': results
            })
            print(f"Case {case_id} completed with best IoU: {best_iou:.3f}")
        except Exception as e:
            print(f"ERROR testing {case_id}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)

    avg_iou = np.mean([r['best_iou'] for r in all_results])

    for result in all_results:
        logger.info(f"\n{result['case_id']}: Best IoU = {result['best_iou']:.3f}")

    logger.info(f"\nAverage best IoU: {avg_iou:.3f}")

    if avg_iou > 0.1:
        logger.info("\n✓ SAM 3 can partially segment vessels with text prompts!")
    else:
        logger.info("\n✗ SAM 3 cannot segment vessels from text alone.")
        logger.info("  Need to try geometric prompts (bounding boxes) next.")


if __name__ == '__main__':
    main()