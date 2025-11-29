"""
Test SAM 3 with CORRECT bounding box syntax from official examples.

Key learnings from sam3_image_predictor_example.ipynb:
1. Box parameter is 'box' not 'bbox'
2. Box must be normalized to [0,1] range
3. Box format is (cx, cy, w, h) - center-based, not corner-based
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
import json
import cv2

# Import SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh

print("Starting SAM 3 bbox test with CORRECT API...")


def get_vessel_bbox_xywh(mask: np.ndarray, padding: int = 10):
    """
    Get bounding box in (x, y, w, h) format.

    Returns:
        [x, y, width, height] or None
    """
    vessel_pixels = np.argwhere(mask > 0)

    if len(vessel_pixels) == 0:
        return None

    # argwhere returns [y, x]
    y_coords = vessel_pixels[:, 0]
    x_coords = vessel_pixels[:, 1]

    y_min = max(0, y_coords.min() - padding)
    y_max = min(mask.shape[0], y_coords.max() + padding)
    x_min = max(0, x_coords.min() - padding)
    x_max = min(mask.shape[1], x_coords.max() + padding)

    # Return in (x, y, width, height) format
    x = x_min
    y = y_min
    width = x_max - x_min
    height = y_max - y_min

    return [float(x), float(y), float(width), float(height)]


def normalize_bbox(bbox_cxcywh, img_width, img_height):
    """
    Normalize bbox to [0, 1] range.

    Args:
        bbox_cxcywh: [cx, cy, w, h] in pixel coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Normalized [cx, cy, w, h] in [0, 1] range
    """
    return [
        bbox_cxcywh[0] / img_width,   # cx
        bbox_cxcywh[1] / img_height,  # cy
        bbox_cxcywh[2] / img_width,   # w
        bbox_cxcywh[3] / img_height   # h
    ]


def test_bbox_on_case(processor, case_id: str, base_path: Path):
    """
    Test SAM 3 with correct bbox prompt.
    """
    print(f"\nTesting {case_id}...")

    # Load cine (first frame)
    cine_path = base_path / "cines" / f"{case_id}_cine.npy"
    cine = np.load(cine_path)
    image = cine[0]

    # Convert to RGB PIL
    image_uint8 = (image * 255).astype(np.uint8)
    image_rgb = np.stack([image_uint8] * 3, axis=-1)
    pil_image = Image.fromarray(image_rgb)
    width, height = pil_image.size

    print(f"  Image size: {width}x{height}")

    # Load ground truth
    mask_path = base_path / "vessel_masks" / f"{case_id}_mask.npy"
    gt_mask = np.load(mask_path)
    print(f"  GT pixels: {gt_mask.sum():,}")

    # Get bbox in (x, y, w, h) format
    bbox_xywh = get_vessel_bbox_xywh(gt_mask, padding=20)

    if bbox_xywh is None:
        print(f"  No vessel pixels!")
        return 0.0

    print(f"  Box (x,y,w,h): {bbox_xywh}")

    # Convert to center format (cx, cy, w, h)
    bbox_xywh_tensor = torch.tensor(bbox_xywh).view(-1, 4)
    bbox_cxcywh = box_xywh_to_cxcywh(bbox_xywh_tensor)
    bbox_cxcywh = bbox_cxcywh.flatten().tolist()

    print(f"  Box (cx,cy,w,h): {bbox_cxcywh}")

    # Normalize to [0, 1]
    norm_box = normalize_bbox(bbox_cxcywh, width, height)
    print(f"  Normalized box: {norm_box}")

    # Set image
    inference_state = processor.set_image(pil_image)

    # Reset prompts
    processor.reset_all_prompts(inference_state)

    # Add geometric prompt with correct syntax
    inference_state = processor.add_geometric_prompt(
        state=inference_state,
        box=norm_box,  # Normalized (cx, cy, w, h)
        label=True     # Positive box
    )

    # Get masks
    masks = inference_state.get("masks", [])

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    if len(masks) == 0:
        print(f"  No masks predicted!")
        return 0.0

    # Take first mask
    pred_mask = masks[0]
    if len(pred_mask.shape) == 3:
        pred_mask = pred_mask[0]

    print(f"  Predicted pixels: {(pred_mask > 0.5).sum():,}")

    # Resize if needed
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(
            pred_mask.astype(np.float32),
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # Calculate IoU
    intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0).sum()
    union = np.logical_or(pred_mask > 0.5, gt_mask > 0).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    print(f"  IoU: {iou:.3f}")

    return iou


def main():
    """
    Test with correct API.
    """
    # Setup
    base_path = Path(r"E:\AngioMLDL_data\corrected_vessel_dataset")

    test_cases = [
        "101-0025_MID_RCA_PRE",   # RCA
        "101-0086_MID_LAD_PRE",   # LAD
        "101-0052_DIST_LCX_PRE",  # LCX
    ]

    # Load SAM 3
    print("\nLoading SAM 3...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use bfloat16 like in example
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    print("SAM 3 loaded!")

    # Test each case
    results = []

    for case_id in test_cases:
        try:
            iou = test_bbox_on_case(processor, case_id, base_path)
            results.append(iou)
        except Exception as e:
            print(f"Error on {case_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append(0.0)

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    for case_id, iou in zip(test_cases, results):
        print(f"{case_id}: IoU = {iou:.3f}")

    avg_iou = np.mean(results)
    print(f"\nAverage IoU: {avg_iou:.3f}")

    if avg_iou > 0.5:
        print("\n✓ SAM 3 successfully segments vessels with bbox prompts!")
        print("  The model understands vessel visual patterns.")
    elif avg_iou > 0.2:
        print("\n⚠ SAM 3 partially segments vessels with bbox.")
        print("  Fine-tuning should improve performance.")
    else:
        print("\n✗ SAM 3 struggles even with bbox prompts.")
        print("  May need significant fine-tuning on medical data.")


if __name__ == '__main__':
    main()