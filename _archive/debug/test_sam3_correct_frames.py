"""
Test SAM 3 on the ACTUAL CONTRAST-FILLED FRAMES where vessels are visible.

Uses frame_num from JSON which is the frame that was analyzed by experts.
"""

import numpy as np
import torch
from PIL import Image
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

    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def normalize_bbox(bbox_cxcywh, img_width, img_height):
    """Normalize bbox to [0, 1] range."""
    return [
        bbox_cxcywh[0] / img_width,
        bbox_cxcywh[1] / img_height,
        bbox_cxcywh[2] / img_width,
        bbox_cxcywh[3] / img_height
    ]


def test_case_with_visualization(processor, case_id: str, base_path: Path, save_dir: Path):
    """
    Test SAM 3 on the CORRECT frame with contrast.
    """
    print(f"\nProcessing {case_id}...")

    # Load metadata FIRST to get correct frame number
    json_path = base_path / "contours" / f"{case_id}_contours.json"
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    frame_num = metadata.get('frame_num', 0)
    vessel_name = metadata.get('segment_name', 'Unknown')
    view_angles = metadata.get('view_angles', {})

    print(f"  Using frame {frame_num} (contrast-filled)")
    print(f"  Vessel: {vessel_name}")

    # Load cine and get CORRECT frame
    cine_path = base_path / "cines" / f"{case_id}_cine.npy"
    cine = np.load(cine_path)

    # Check if frame_num is valid
    if frame_num >= len(cine):
        print(f"  WARNING: frame_num {frame_num} >= cine length {len(cine)}, using last frame")
        frame_num = len(cine) - 1

    image = cine[frame_num]  # USE CORRECT FRAME!

    # Convert to RGB
    image_uint8 = (image * 255).astype(np.uint8)
    image_rgb = np.stack([image_uint8] * 3, axis=-1)
    pil_image = Image.fromarray(image_rgb)
    width, height = pil_image.size

    # Load ground truth mask
    mask_path = base_path / "vessel_masks" / f"{case_id}_mask.npy"
    gt_mask = np.load(mask_path)

    # Get bbox
    bbox_xywh = get_vessel_bbox_xywh(gt_mask, padding=20)
    if bbox_xywh is None:
        print(f"  No vessel pixels found!")
        return 0.0

    # Convert to center format
    bbox_xywh_tensor = torch.tensor(bbox_xywh).view(-1, 4)
    bbox_cxcywh = box_xywh_to_cxcywh(bbox_xywh_tensor)
    bbox_cxcywh = bbox_cxcywh.flatten().tolist()

    # Normalize
    norm_box = normalize_bbox(bbox_cxcywh, width, height)

    # Test 1: Bounding box prompt
    inference_state = processor.set_image(pil_image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.add_geometric_prompt(
        state=inference_state,
        box=norm_box,
        label=True
    )

    # Get prediction
    masks_bbox = inference_state.get("masks", [])
    if isinstance(masks_bbox, torch.Tensor):
        masks_bbox = masks_bbox.cpu().numpy()

    pred_mask_bbox = None
    if len(masks_bbox) > 0:
        pred_mask_bbox = masks_bbox[0]
        if len(pred_mask_bbox.shape) == 3:
            pred_mask_bbox = pred_mask_bbox[0]

        if pred_mask_bbox.shape != gt_mask.shape:
            pred_mask_bbox = cv2.resize(
                pred_mask_bbox.astype(np.float32),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

    # Test 2: Text prompt (now that vessel is visible)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(
        state=inference_state,
        prompt="bright white vessel with contrast"
    )

    masks_text = inference_state.get("masks", [])
    if isinstance(masks_text, torch.Tensor):
        masks_text = masks_text.cpu().numpy()

    pred_mask_text = None
    if len(masks_text) > 0:
        pred_mask_text = masks_text[0]
        if len(pred_mask_text.shape) == 3:
            pred_mask_text = pred_mask_text[0]

        if pred_mask_text.shape != gt_mask.shape:
            pred_mask_text = cv2.resize(
                pred_mask_text.astype(np.float32),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

    # Calculate IoU for both
    iou_bbox = 0.0
    if pred_mask_bbox is not None:
        intersection = np.logical_and(pred_mask_bbox > 0.5, gt_mask > 0).sum()
        union = np.logical_or(pred_mask_bbox > 0.5, gt_mask > 0).sum()
        iou_bbox = float(intersection / union) if union > 0 else 0.0

    iou_text = 0.0
    if pred_mask_text is not None:
        intersection = np.logical_and(pred_mask_text > 0.5, gt_mask > 0).sum()
        union = np.logical_or(pred_mask_text > 0.5, gt_mask > 0).sum()
        iou_text = float(intersection / union) if union > 0 else 0.0

    print(f"  Bbox IoU: {iou_bbox:.3f}")
    print(f"  Text IoU: {iou_text:.3f}")

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{case_id} - Frame {frame_num}\n{vessel_name}\nBbox IoU: {iou_bbox:.3f}, Text IoU: {iou_text:.3f}', fontsize=14)

    # Row 1: Input and ground truth
    axes[0, 0].imshow(cine[0], cmap='gray')
    axes[0, 0].set_title(f'Frame 0 (NO CONTRAST)\nWhat we were using before!')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].set_title(f'Frame {frame_num} (WITH CONTRAST)\nCorrect frame!')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(image, cmap='gray', alpha=0.7)
    axes[0, 2].imshow(gt_mask, cmap='Reds', alpha=0.5)
    axes[0, 2].set_title(f'Ground Truth\n({gt_mask.sum():,} pixels)')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(image, cmap='gray')
    rect = patches.Rectangle((bbox_xywh[0], bbox_xywh[1]),
                             bbox_xywh[2], bbox_xywh[3],
                             linewidth=3, edgecolor='lime', facecolor='none')
    axes[0, 3].add_patch(rect)
    axes[0, 3].set_title('Bounding Box Prompt')
    axes[0, 3].axis('off')

    # Row 2: Predictions
    if pred_mask_bbox is not None:
        axes[1, 0].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 0].imshow(pred_mask_bbox > 0.5, cmap='Blues', alpha=0.5)
        axes[1, 0].set_title(f'Bbox Prediction\n({(pred_mask_bbox > 0.5).sum():,} pixels)')
    else:
        axes[1, 0].imshow(image, cmap='gray')
        axes[1, 0].set_title('Bbox: No prediction')
    axes[1, 0].axis('off')

    if pred_mask_text is not None:
        axes[1, 1].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(pred_mask_text > 0.5, cmap='Greens', alpha=0.5)
        axes[1, 1].set_title(f'Text Prediction\n({(pred_mask_text > 0.5).sum():,} pixels)')
    else:
        axes[1, 1].imshow(image, cmap='gray')
        axes[1, 1].set_title('Text: No prediction')
    axes[1, 1].axis('off')

    # Overlay comparison
    overlay = np.zeros((image.shape[0], image.shape[1], 3))
    overlay[gt_mask > 0, 0] = 1  # Red for GT
    if pred_mask_bbox is not None:
        overlay[pred_mask_bbox > 0.5, 2] = 0.5  # Blue for bbox pred

    axes[1, 2].imshow(image, cmap='gray', alpha=0.5)
    axes[1, 2].imshow(overlay, alpha=0.5)
    axes[1, 2].set_title('GT (Red) vs Bbox (Blue)')
    axes[1, 2].axis('off')

    # Error map for best prediction
    best_pred = pred_mask_bbox if iou_bbox >= iou_text else pred_mask_text
    if best_pred is not None:
        diff = np.zeros((image.shape[0], image.shape[1], 3))
        fp_mask = np.logical_and(best_pred > 0.5, gt_mask == 0)
        diff[fp_mask] = [1, 1, 0]  # Yellow - false positive
        fn_mask = np.logical_and(best_pred <= 0.5, gt_mask > 0)
        diff[fn_mask] = [1, 0, 0]  # Red - false negative
        tp_mask = np.logical_and(best_pred > 0.5, gt_mask > 0)
        diff[tp_mask] = [0, 1, 0]  # Green - true positive

        axes[1, 3].imshow(image, cmap='gray', alpha=0.3)
        axes[1, 3].imshow(diff, alpha=0.7)
        axes[1, 3].set_title('Error Analysis (Best)\nGreen=Correct, Red=Missed, Yellow=Extra')
    else:
        axes[1, 3].imshow(image, cmap='gray')
        axes[1, 3].set_title('No predictions')
    axes[1, 3].axis('off')

    plt.tight_layout()

    # Save
    output_path = save_dir / f"{case_id}_correct_frame.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")

    return max(iou_bbox, iou_text)


def main():
    """
    Test with CORRECT frames.
    """
    base_path = Path(r"E:\AngioMLDL_data\corrected_vessel_dataset")
    save_dir = Path("sam3_correct_frames")
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

    # Test each case
    results = []

    for case_id in test_cases:
        try:
            iou = test_case_with_visualization(processor, case_id, base_path, save_dir)
            results.append(iou)
        except Exception as e:
            print(f"Error on {case_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append(0.0)

    # Summary
    print("\n" + "="*50)
    print("SUMMARY (Using CORRECT contrast-filled frames)")
    print("="*50)

    for case_id, iou in zip(test_cases, results):
        print(f"{case_id}: Best IoU = {iou:.3f}")

    avg_iou = np.mean(results)
    print(f"\nAverage IoU: {avg_iou:.3f}")

    if avg_iou > 0.5:
        print("\n✓ SAM 3 works well with visible vessels!")
    elif avg_iou > 0.2:
        print("\n⚠ SAM 3 partially segments visible vessels")
    else:
        print("\n✗ SAM 3 struggles even with visible vessels")

    print(f"\nVisualizations saved in: {save_dir}/")


if __name__ == '__main__':
    main()