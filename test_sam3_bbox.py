"""
Test SAM 3 with bounding box prompts on angiograms.

Since text prompts failed (IoU ~0.01), let's test if SAM 3 can segment
vessels when given the exact bounding box around them.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
import json
import logging
import cv2

# Import SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_tight_bbox(mask: np.ndarray, padding: int = 10) -> list:
    """
    Get bounding box around vessel mask.

    Args:
        mask: Binary mask
        padding: Pixels to pad

    Returns:
        [x1, y1, x2, y2] or None if no vessel pixels
    """
    vessel_pixels = np.argwhere(mask > 0)

    if len(vessel_pixels) == 0:
        return None

    # argwhere returns [y, x] format
    y_coords = vessel_pixels[:, 0]
    x_coords = vessel_pixels[:, 1]

    y1 = max(0, y_coords.min() - padding)
    y2 = min(mask.shape[0], y_coords.max() + padding)
    x1 = max(0, x_coords.min() - padding)
    x2 = min(mask.shape[1], x_coords.max() + padding)

    return [int(x1), int(y1), int(x2), int(y2)]


def test_bbox_prompt(sam_processor, case_id: str, base_path: Path):
    """
    Test SAM 3 with bounding box from ground truth.
    """
    print(f"\nTesting {case_id} with bounding box prompt...")

    # Load cine (first frame)
    cine_path = base_path / "cines" / f"{case_id}_cine.npy"
    cine = np.load(cine_path)
    image = cine[0]

    # Convert to RGB
    image_uint8 = (image * 255).astype(np.uint8)
    image_rgb = np.stack([image_uint8] * 3, axis=-1)
    pil_image = Image.fromarray(image_rgb)

    # Load ground truth mask
    mask_path = base_path / "vessel_masks" / f"{case_id}_mask.npy"
    gt_mask = np.load(mask_path)

    # Get bounding box from ground truth
    bbox = get_tight_bbox(gt_mask, padding=20)

    if bbox is None:
        print(f"  No vessel pixels found in ground truth!")
        return 0.0

    print(f"  Ground truth bbox: {bbox}")
    print(f"  Ground truth pixels: {gt_mask.sum():,}")

    # Set image in SAM 3
    state = sam_processor.set_image(pil_image)

    # Try different prompt methods
    results = {}

    # Method 1: Direct bbox prompt (if available)
    try:
        print("  Trying direct bbox prompt...")
        output = sam_processor.set_bbox_prompt(
            state=state,
            bbox=bbox
        )
        masks = output.get("masks", [])

        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        if len(masks) > 0:
            pred_mask = masks[0]
            if len(pred_mask.shape) == 3:
                pred_mask = pred_mask[0]

            # Calculate IoU
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.float32),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0).sum()
            union = np.logical_or(pred_mask > 0.5, gt_mask > 0).sum()
            iou = float(intersection / union) if union > 0 else 0.0

            results['direct_bbox'] = iou
            print(f"  Direct bbox IoU: {iou:.3f}")

    except AttributeError:
        print("  set_bbox_prompt not available")

    # Method 2: Geometric prompt (from exemplar strategy)
    try:
        print("  Trying geometric prompt...")
        state = sam_processor.set_image(pil_image)  # Reset state

        # Add geometric prompt
        state = sam_processor.add_geometric_prompt(
            bbox=bbox,
            label="vessel",
            state=state
        )

        # Get predictions
        output = sam_processor._forward_grounding(state)
        masks = output.get("masks", [])

        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        if len(masks) > 0:
            pred_mask = masks[0]
            if len(pred_mask.shape) == 3:
                pred_mask = pred_mask[0]

            # Calculate IoU
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.float32),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0).sum()
            union = np.logical_or(pred_mask > 0.5, gt_mask > 0).sum()
            iou = float(intersection / union) if union > 0 else 0.0

            results['geometric_prompt'] = iou
            print(f"  Geometric prompt IoU: {iou:.3f}")

    except Exception as e:
        print(f"  Geometric prompt error: {e}")

    # Method 3: Point prompt at center of bbox
    try:
        print("  Trying point prompt...")
        state = sam_processor.set_image(pil_image)  # Reset state

        # Calculate center point
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        output = sam_processor.set_point_prompt(
            state=state,
            point=[center_x, center_y],
            label=1  # Positive point
        )
        masks = output.get("masks", [])

        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        if len(masks) > 0:
            pred_mask = masks[0]
            if len(pred_mask.shape) == 3:
                pred_mask = pred_mask[0]

            # Calculate IoU
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.float32),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0).sum()
            union = np.logical_or(pred_mask > 0.5, gt_mask > 0).sum()
            iou = float(intersection / union) if union > 0 else 0.0

            results['point_prompt'] = iou
            print(f"  Point prompt IoU: {iou:.3f}")

    except Exception as e:
        print(f"  Point prompt error: {e}")

    # Return best result
    best_iou = max(results.values()) if results else 0.0
    best_method = max(results, key=results.get) if results else "none"

    print(f"  Best result: {best_method} with IoU={best_iou:.3f}")

    return best_iou


def main():
    """
    Test SAM 3 with bounding box prompts.
    """
    print("\n=== SAM 3 Bounding Box Prompt Test ===\n")

    # Paths
    base_path = Path(r"E:\AngioMLDL_data\corrected_vessel_dataset")

    # Test cases
    test_cases = [
        "101-0025_MID_RCA_PRE",   # RCA
        "101-0086_MID_LAD_PRE",   # LAD
        "101-0052_DIST_LCX_PRE",  # LCX
    ]

    # Initialize SAM 3
    print("Loading SAM 3...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam_model = build_sam3_image_model(device=device)
    sam_processor = Sam3Processor(sam_model)
    print("SAM 3 loaded!\n")

    # Test each case
    results = []

    for case_id in test_cases:
        try:
            best_iou = test_bbox_prompt(sam_processor, case_id, base_path)
            results.append(best_iou)
        except Exception as e:
            print(f"Error testing {case_id}: {e}")
            results.append(0.0)

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    for case_id, iou in zip(test_cases, results):
        print(f"{case_id}: IoU = {iou:.3f}")

    avg_iou = np.mean(results)
    print(f"\nAverage IoU: {avg_iou:.3f}")

    if avg_iou > 0.3:
        print("\n✓ SAM 3 can segment vessels with bounding box prompts!")
        print("  This means SAM 3 understands the visual pattern.")
        print("  Fine-tuning should work well.")
    elif avg_iou > 0.1:
        print("\n⚠ SAM 3 partially segments vessels with bbox prompts.")
        print("  May need fine-tuning to improve.")
    else:
        print("\n✗ SAM 3 cannot segment vessels even with bbox prompts.")
        print("  Medical domain too different from training data.")
        print("  Need extensive fine-tuning or different approach.")


if __name__ == '__main__':
    main()