"""
Visualize what SAM 3 is seeing and predicting with bounding box prompts.

Shows:
1. Original angiogram
2. Ground truth mask
3. Bounding box we're giving SAM 3
4. What SAM 3 predicts
5. Overlap comparison
"""

import numpy as np
import torch
from PIL import Image, ImageDraw
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh


def get_vessel_bbox_xywh(mask: np.ndarray, padding: int = 20):
    """Get bounding box in (x, y, w, h) format."""
    vessel_pixels = np.argwhere(mask > 0)
    if len(vessel_pixels) == 0:
        return None

    y_coords = vessel_pixels[:, 0]
    x_coords = vessel_pixels[:, 1]

    y_min = max(0, y_coords.min() - padding)
    y_max = min(mask.shape[0], y_coords.max() + padding)
    x_min = max(0, x_coords.min() - padding)
    x_max = min(mask.shape[1], x_coords.max() + padding)

    x = x_min
    y = y_min
    width = x_max - x_min
    height = y_max - y_min

    return [float(x), float(y), float(width), float(height)]


def normalize_bbox(bbox_cxcywh, img_width, img_height):
    """Normalize bbox to [0, 1] range."""
    return [
        bbox_cxcywh[0] / img_width,
        bbox_cxcywh[1] / img_height,
        bbox_cxcywh[2] / img_width,
        bbox_cxcywh[3] / img_height
    ]


def visualize_case(processor, case_id: str, base_path: Path, save_dir: Path):
    """
    Process one case and create comprehensive visualization.
    """
    print(f"\nProcessing {case_id}...")

    # Load cine (first frame)
    cine_path = base_path / "cines" / f"{case_id}_cine.npy"
    cine = np.load(cine_path)
    image = cine[0]

    # Convert to RGB
    image_uint8 = (image * 255).astype(np.uint8)
    image_rgb = np.stack([image_uint8] * 3, axis=-1)
    pil_image = Image.fromarray(image_rgb)
    width, height = pil_image.size

    # Load ground truth mask
    mask_path = base_path / "vessel_masks" / f"{case_id}_mask.npy"
    gt_mask = np.load(mask_path)

    # Load metadata
    json_path = base_path / "contours" / f"{case_id}_contours.json"
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    vessel_name = metadata.get('segment_name', 'Unknown')
    view_angles = metadata.get('view_angles', {})

    # Get bbox in (x, y, w, h) format
    bbox_xywh = get_vessel_bbox_xywh(gt_mask, padding=20)
    if bbox_xywh is None:
        print(f"  No vessel pixels found!")
        return

    # Convert to center format
    bbox_xywh_tensor = torch.tensor(bbox_xywh).view(-1, 4)
    bbox_cxcywh = box_xywh_to_cxcywh(bbox_xywh_tensor)
    bbox_cxcywh = bbox_cxcywh.flatten().tolist()

    # Normalize
    norm_box = normalize_bbox(bbox_cxcywh, width, height)

    # Get SAM 3 prediction
    inference_state = processor.set_image(pil_image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.add_geometric_prompt(
        state=inference_state,
        box=norm_box,
        label=True
    )

    # Extract prediction
    masks = inference_state.get("masks", [])
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    pred_mask = None
    if len(masks) > 0:
        pred_mask = masks[0]
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]

        # Resize to match GT
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask.astype(np.float32),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

    # Calculate metrics
    if pred_mask is not None:
        intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0).sum()
        union = np.logical_or(pred_mask > 0.5, gt_mask > 0).sum()
        iou = float(intersection / union) if union > 0 else 0.0
    else:
        iou = 0.0
        pred_mask = np.zeros_like(gt_mask)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{case_id}\nVessel: {vessel_name}, View: LAO={view_angles.get("lao", "?")}° RAO={view_angles.get("rao", "?")}\nIoU: {iou:.3f}', fontsize=14)

    # 1. Original angiogram
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Angiogram')
    axes[0, 0].axis('off')

    # 2. Ground truth mask
    axes[0, 1].imshow(image, cmap='gray', alpha=0.7)
    axes[0, 1].imshow(gt_mask, cmap='Reds', alpha=0.5)
    axes[0, 1].set_title(f'Ground Truth\n({gt_mask.sum():,} pixels)')
    axes[0, 1].axis('off')

    # 3. Bounding box on image
    axes[0, 2].imshow(image, cmap='gray')
    rect = patches.Rectangle((bbox_xywh[0], bbox_xywh[1]),
                             bbox_xywh[2], bbox_xywh[3],
                             linewidth=3, edgecolor='lime', facecolor='none')
    axes[0, 2].add_patch(rect)
    axes[0, 2].set_title(f'Bounding Box Prompt\nBox: [{int(bbox_xywh[0])}, {int(bbox_xywh[1])}, {int(bbox_xywh[2])}, {int(bbox_xywh[3])}]')
    axes[0, 2].axis('off')

    # 4. SAM 3 prediction
    axes[1, 0].imshow(image, cmap='gray', alpha=0.7)
    axes[1, 0].imshow(pred_mask > 0.5, cmap='Blues', alpha=0.5)
    axes[1, 0].set_title(f'SAM 3 Prediction\n({(pred_mask > 0.5).sum():,} pixels)')
    axes[1, 0].axis('off')

    # 5. Overlay comparison
    overlay = np.zeros((image.shape[0], image.shape[1], 3))
    overlay[gt_mask > 0] = [1, 0, 0]  # Red for GT
    overlay[pred_mask > 0.5] = overlay[pred_mask > 0.5] * 0.5 + np.array([0, 0, 1]) * 0.5  # Blue for prediction

    axes[1, 1].imshow(image, cmap='gray', alpha=0.5)
    axes[1, 1].imshow(overlay, alpha=0.5)
    axes[1, 1].set_title('Overlay\n(Red=GT, Blue=Pred, Purple=Overlap)')
    axes[1, 1].axis('off')

    # 6. Difference map
    diff = np.zeros((image.shape[0], image.shape[1], 3))
    # False positives (predicted but not in GT) - yellow
    fp_mask = np.logical_and(pred_mask > 0.5, gt_mask == 0)
    diff[fp_mask] = [1, 1, 0]
    # False negatives (in GT but not predicted) - red
    fn_mask = np.logical_and(pred_mask <= 0.5, gt_mask > 0)
    diff[fn_mask] = [1, 0, 0]
    # True positives (correctly predicted) - green
    tp_mask = np.logical_and(pred_mask > 0.5, gt_mask > 0)
    diff[tp_mask] = [0, 1, 0]

    axes[1, 2].imshow(image, cmap='gray', alpha=0.3)
    axes[1, 2].imshow(diff, alpha=0.7)
    axes[1, 2].set_title('Error Analysis\n(Green=Correct, Red=Missed, Yellow=Extra)')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = save_dir / f"{case_id}_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization to {output_path}")
    print(f"  IoU: {iou:.3f}")

    return iou


def main():
    """
    Create visualizations for all test cases.
    """
    # Setup
    base_path = Path(r"E:\AngioMLDL_data\corrected_vessel_dataset")
    save_dir = Path("sam3_visualizations")
    save_dir.mkdir(exist_ok=True)

    test_cases = [
        "101-0025_MID_RCA_PRE",   # RCA
        "101-0086_MID_LAD_PRE",   # LAD
        "101-0052_DIST_LCX_PRE",  # LCX
    ]

    # Load SAM 3
    print("Loading SAM 3...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    print("SAM 3 loaded!")

    # Process each case
    results = []

    for case_id in test_cases:
        try:
            iou = visualize_case(processor, case_id, base_path, save_dir)
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

    print(f"\nVisualizations saved in: {save_dir}")
    print("\nEach visualization shows:")
    print("  - Top row: Original image, Ground truth mask, Bounding box prompt")
    print("  - Bottom row: SAM 3 prediction, Overlay comparison, Error analysis")


if __name__ == '__main__':
    main()