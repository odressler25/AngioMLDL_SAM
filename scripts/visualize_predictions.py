"""
Visualize SAM3 model predictions during training.
Run this script to see how your model is currently performing on validation images.

IMPORTANT: Run this AFTER stopping training, or it will compete for GPU memory.

Usage:
    python scripts/visualize_predictions.py
    python scripts/visualize_predictions.py --num-images 10
    python scripts/visualize_predictions.py --checkpoint path/to/checkpoint.pt
    python scripts/visualize_predictions.py --gt-only  # No model inference
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Add sam3 to path
sys.path.insert(0, "C:/Users/odressler/sam3")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def decode_rle(rle_dict, height, width):
    """Decode RLE mask to binary mask."""
    if rle_dict is None:
        return None

    counts = rle_dict.get('counts', [])
    if isinstance(counts, str):
        import pycocotools.mask as mask_util
        mask = mask_util.decode(rle_dict)
        return mask

    mask = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            mask[pos:pos + count] = 1
        pos += count
    return mask.reshape((height, width), order='F')


def load_ground_truth(ann_file, image_id):
    """Load ground truth annotations for an image."""
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    return [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load SAM3 model and apply training checkpoint weights."""
    print(f"Building SAM3 model...")

    model = build_sam3_image_model(
        bpe_path="C:/Users/odressler/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        device='cpu',
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Loaded from epoch: {epoch}")

    return model


def visualize_with_gt_only(image_path, ann_file, output_dir, image_info):
    """Visualize just the ground truth for comparison."""
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    gt_annotations = load_ground_truth(ann_file, image_info['id'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(image_np)
    axes[0].set_title(f"Original: {image_info['file_name']}", fontsize=10)
    axes[0].axis('off')

    gt_overlay = image_np.copy()
    gt_mask_combined = np.zeros((height, width), dtype=np.uint8)

    for ann in gt_annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(gt_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if 'segmentation' in ann and ann['segmentation']:
            seg = ann['segmentation']
            if isinstance(seg, dict) and 'counts' in seg:
                mask = decode_rle(seg, height, width)
                if mask is not None:
                    gt_mask_combined = np.maximum(gt_mask_combined, mask)

    gt_colored = gt_overlay.copy().astype(np.float32)
    mask_overlay = np.zeros_like(gt_colored)
    mask_overlay[gt_mask_combined > 0] = [0, 255, 0]
    gt_colored = gt_colored * 0.6 + mask_overlay * 0.4
    gt_colored = np.clip(gt_colored, 0, 255).astype(np.uint8)

    axes[1].imshow(gt_colored)
    axes[1].set_title(f"Ground Truth: {len(gt_annotations)} vessel annotations", fontsize=10)
    axes[1].axis('off')

    plt.tight_layout()
    output_path = output_dir / f"gt_{image_info['id']:04d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(output_path)


def visualize_with_predictions(image_path, ann_file, output_dir, image_info, processor, text_prompt="coronary artery"):
    """Visualize ground truth alongside model predictions."""
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    gt_annotations = load_ground_truth(ann_file, image_info['id'])

    # Run inference
    state = processor.set_image(image)
    state = processor.set_text_prompt(text_prompt, state)

    pred_masks = state.get('masks', None)
    pred_boxes = state.get('boxes', None)
    pred_scores = state.get('scores', None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original
    axes[0].imshow(image_np)
    axes[0].set_title(f"Original: {image_info['file_name']}", fontsize=10)
    axes[0].axis('off')

    # Ground truth
    gt_overlay = image_np.copy()
    gt_mask_combined = np.zeros((height, width), dtype=np.uint8)

    for ann in gt_annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(gt_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if 'segmentation' in ann and ann['segmentation']:
            seg = ann['segmentation']
            if isinstance(seg, dict) and 'counts' in seg:
                mask = decode_rle(seg, height, width)
                if mask is not None:
                    gt_mask_combined = np.maximum(gt_mask_combined, mask)

    gt_colored = gt_overlay.copy().astype(np.float32)
    mask_overlay = np.zeros_like(gt_colored)
    mask_overlay[gt_mask_combined > 0] = [0, 255, 0]
    gt_colored = gt_colored * 0.6 + mask_overlay * 0.4
    gt_colored = np.clip(gt_colored, 0, 255).astype(np.uint8)

    axes[1].imshow(gt_colored)
    axes[1].set_title(f"Ground Truth: {len(gt_annotations)} annotations", fontsize=10)
    axes[1].axis('off')

    # Predictions
    pred_overlay = image_np.copy()
    pred_mask_combined = np.zeros((height, width), dtype=np.uint8)
    num_predictions = 0

    if pred_masks is not None and len(pred_masks) > 0:
        pred_masks_np = pred_masks.squeeze(1).cpu().numpy()
        num_predictions = len(pred_masks_np)

        for i, mask in enumerate(pred_masks_np):
            binary_mask = (mask > 0.5).astype(np.uint8)
            pred_mask_combined = np.maximum(pred_mask_combined, binary_mask)

            if pred_boxes is not None and i < len(pred_boxes):
                box = pred_boxes[i].cpu().numpy()
                x0, y0, x1, y1 = [int(v) for v in box]
                score = pred_scores[i].item() if pred_scores is not None else 0
                cv2.rectangle(pred_overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.putText(pred_overlay, f"{score:.2f}", (x0, y0-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    pred_colored = pred_overlay.copy().astype(np.float32)
    mask_overlay = np.zeros_like(pred_colored)
    mask_overlay[pred_mask_combined > 0] = [255, 0, 0]
    pred_colored = pred_colored * 0.6 + mask_overlay * 0.4
    pred_colored = np.clip(pred_colored, 0, 255).astype(np.uint8)

    axes[2].imshow(pred_colored)
    axes[2].set_title(f"Prediction: {num_predictions} detections", fontsize=10)
    axes[2].axis('off')

    plt.tight_layout()
    output_path = output_dir / f"pred_{image_info['id']:04d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Calculate IoU
    if gt_mask_combined.sum() > 0 and pred_mask_combined.sum() > 0:
        intersection = (gt_mask_combined & pred_mask_combined).sum()
        union = (gt_mask_combined | pred_mask_combined).sum()
        iou = intersection / union if union > 0 else 0
    else:
        iou = 0

    return str(output_path), num_predictions, iou


def main():
    parser = argparse.ArgumentParser(description='Visualize SAM3 predictions')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: latest)')
    parser.add_argument('--num-images', type=int, default=5,
                        help='Number of images to visualize')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualizations')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to use')
    parser.add_argument('--gt-only', action='store_true',
                        help='Only show ground truth (no model inference)')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Confidence threshold for predictions')
    parser.add_argument('--prompt', type=str, default='coronary artery',
                        help='Text prompt for inference')
    args = parser.parse_args()

    data_root = Path("E:/AngioMLDL_data/coco_format_v2")
    exp_dir = Path("C:/Users/odressler/AngioMLDL_SAM/experiments/phase1_vessel_seg")

    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_file = data_root / args.split / "annotations.json"
    img_dir = data_root / args.split / "images"

    if not ann_file.exists():
        print(f"Error: Annotations not found at {ann_file}")
        return

    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    np.random.seed(42)
    selected_indices = np.random.choice(len(images), min(args.num_images, len(images)), replace=False)
    selected_images = [images[i] for i in selected_indices]

    print(f"=" * 60)
    print(f"SAM3 Visualization")
    print(f"=" * 60)
    print(f"Data: {img_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(selected_images)} from {args.split} set")
    print()

    if args.gt_only:
        print("Mode: Ground truth only")
        print()
        for i, img_info in enumerate(selected_images):
            image_path = img_dir / img_info['file_name']
            print(f"[{i+1}/{len(selected_images)}] {img_info['file_name']}")
            if not image_path.exists():
                print(f"  Warning: Image not found")
                continue
            output_path = visualize_with_gt_only(image_path, ann_file, output_dir, img_info)
            gt_annotations = load_ground_truth(ann_file, img_info['id'])
            print(f"  {len(gt_annotations)} vessels -> {output_path}")
    else:
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            ckpt_dir = exp_dir / "checkpoints"
            checkpoint_path = ckpt_dir / "checkpoint.pt"
            if not checkpoint_path.exists():
                ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
                if ckpts:
                    checkpoint_path = ckpts[-1]

        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            print("Tip: Use --gt-only to visualize ground truth without model")
            return

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Text prompt: '{args.prompt}'")
        print(f"Confidence: {args.confidence}")
        print()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            print(f"GPU free memory: {free_mem / 1e9:.1f} GB")

        try:
            model = load_model_from_checkpoint(checkpoint_path, device=device)
            processor = Sam3Processor(model, resolution=1008, device=device,
                                       confidence_threshold=args.confidence)
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return

        print()
        print("Running inference...")
        print()

        total_iou = 0
        valid_count = 0

        for i, img_info in enumerate(selected_images):
            image_path = img_dir / img_info['file_name']
            print(f"[{i+1}/{len(selected_images)}] {img_info['file_name']}")

            if not image_path.exists():
                print(f"  Warning: Image not found")
                continue

            try:
                output_path, num_preds, iou = visualize_with_predictions(
                    image_path, ann_file, output_dir, img_info, processor, args.prompt
                )
                gt_annotations = load_ground_truth(ann_file, img_info['id'])
                print(f"  GT: {len(gt_annotations)}, Pred: {num_preds}, IoU: {iou:.3f}")
                total_iou += iou
                valid_count += 1
            except Exception as e:
                print(f"  Error: {e}")

        if valid_count > 0:
            print(f"\nAverage IoU: {total_iou / valid_count:.3f}")

    print()
    print(f"Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
