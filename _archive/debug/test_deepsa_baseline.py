"""
Test DeepSA pretrained model on our 3 angiography test cases

Compare performance with SAM 3 baseline (0.372 IoU)
"""

import sys
sys.path.append("DeepSA/")

import torch
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage import morphology

# DeepSA imports
from DeepSA.models import UNet
from DeepSA.datasets import tophat
import torchvision.transforms as T

# Size expected by DeepSA
SIZE = 512

# Preprocessing transforms from demo.py
tfmc1 = T.Compose([
    T.Resize(SIZE),
    T.Lambda(lambda img: tophat(img, 50)),
    T.ToTensor(),
    T.Normalize((0.5), (0.5))
])

tfmc2 = T.Compose([
    T.Resize(SIZE),
    T.ToTensor(),
    T.Normalize((0.5), (0.5))
])


def load_deepsa_model(checkpoint_path='DeepSA/ckpt/fscad_36249.ckpt', device='cuda'):
    """
    Load pretrained DeepSA model

    Available checkpoints:
    - fscad_36249.ckpt: Dice 0.828 on FS-CAD dataset
    - xcad_4afe3.ckpt: Trained on XCAD dataset
    - pt_bc62a.ckpt: Pretrained model
    """
    print(f"Loading DeepSA model from {checkpoint_path}...")

    # Create UNet (1 channel in, 1 channel out, 32 base filters)
    model = UNet(1, 1, 32, bilinear=True)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Remove 'module.' prefix from state dict keys
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['netE'].items()}
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model


def predict_deepsa(model, image_pil, use_tophat=True, device='cuda'):
    """
    Run DeepSA prediction on a PIL image

    Args:
        model: DeepSA UNet model
        image_pil: PIL Image (grayscale)
        use_tophat: Whether to use tophat preprocessing
        device: 'cuda' or 'cpu'

    Returns:
        seg_mask: Binary segmentation mask (numpy array)
        sub_img: Subtraction image (vessel-enhanced)
    """
    # Convert to grayscale if needed
    if image_pil.mode != 'L':
        image_pil = image_pil.convert('L')

    # Preprocess
    if use_tophat:
        x = tfmc1(image_pil)
    else:
        x = tfmc2(image_pil)

    # Add batch dimension
    x = x.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred_y = model(x)

    # Process subtraction image (vessel-enhanced)
    sub_img = pred_y.squeeze().cpu().numpy()
    sub_img = ((sub_img + 1) / 2 * 255).astype('uint8')

    # Process segmentation mask
    seg_mask = torch.sign(pred_y)
    seg_mask = ((seg_mask.cpu() + 1) / 2).numpy().astype(bool)

    # Remove small objects (noise)
    seg_mask = morphology.remove_small_objects(seg_mask[0, 0], min_size=500)

    return seg_mask.astype(np.uint8), sub_img


def load_ground_truth_mask(json_path, original_frame_shape, target_shape=(512, 512)):
    """
    Load ground truth mask from JSON contours

    Args:
        json_path: Path to contours JSON file
        original_frame_shape: Original cine frame dimensions (H, W)
        target_shape: Target shape for output mask

    Returns:
        mask: Binary mask (0/1) at target_shape
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get image dimensions from first contour
    # Assume contours are in original image coordinates
    contours_data = data.get('centerline', [])

    if not contours_data:
        print(f"WARNING: No centerline contours in {json_path}")
        return np.zeros(target_shape, dtype=np.uint8)

    # Use actual cine frame dimensions (not estimated from contour coordinates!)
    orig_h, orig_w = original_frame_shape

    # Create mask at original cine size
    mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)

    # Draw polygon from left_edge + reversed right_edge
    if 'left_edge' in data and 'right_edge' in data:
        left_edge = np.array(data['left_edge'], dtype=np.int32)
        right_edge = np.array(data['right_edge'], dtype=np.int32)

        # Create closed polygon
        polygon = np.vstack([left_edge, right_edge[::-1]])

        cv2.fillPoly(mask_orig, [polygon], 1)

    # Resize to target shape with proper aspect ratio preserved
    mask_resized = cv2.resize(mask_orig, target_shape, interpolation=cv2.INTER_NEAREST)

    return mask_resized


def compute_iou(pred, gt):
    """Compute Intersection over Union"""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_dice(pred, gt):
    """Compute Dice coefficient"""
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 0.0
    return 2 * intersection / total


def visualize_results(frame, pred_mask, gt_mask, case_name, output_dir):
    """
    Create visualization comparing prediction with ground truth
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original, Prediction, Ground Truth
    axes[0, 0].imshow(frame, cmap='gray')
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(frame, cmap='gray')
    axes[0, 1].imshow(pred_mask, cmap='Greens', alpha=0.5)
    axes[0, 1].set_title('DeepSA Prediction')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(frame, cmap='gray')
    axes[0, 2].imshow(gt_mask, cmap='Reds', alpha=0.5)
    axes[0, 2].set_title('Ground Truth (Medis)')
    axes[0, 2].axis('off')

    # Row 2: Overlay comparison, Prediction mask only, GT mask only
    axes[1, 0].imshow(frame, cmap='gray')
    axes[1, 0].imshow(pred_mask, cmap='Greens', alpha=0.3)
    axes[1, 0].imshow(gt_mask, cmap='Reds', alpha=0.3)
    axes[1, 0].set_title('Green=Prediction, Red=Ground Truth')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_mask, cmap='gray')
    axes[1, 1].set_title('Prediction Mask')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(gt_mask, cmap='gray')
    axes[1, 2].set_title('Ground Truth Mask')
    axes[1, 2].axis('off')

    plt.suptitle(f'{case_name}\nDeepSA Baseline Test', fontsize=16)
    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / f'{case_name}_deepsa.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {output_path}")


def test_deepsa_on_cases():
    """
    Test DeepSA on our 3 angiography test cases
    """

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cpu':
        print("WARNING: Running on CPU will be slower. Consider using GPU.")

    # Load DeepSA model
    model = load_deepsa_model(
        checkpoint_path='DeepSA/ckpt/fscad_36249.ckpt',
        device=device
    )

    # Test cases (load from .npy cine files)
    test_cases = [
        {
            'name': '101-0025_MID_RCA_PRE',
            'vessel': 'RCA Mid',
            'cine_path': 'E:/AngioMLDL_data/corrected_vessel_dataset/cines/101-0025_MID_RCA_PRE_cine.npy',
            'json_path': 'E:/AngioMLDL_data/corrected_vessel_dataset/contours/101-0025_MID_RCA_PRE_contours.json',
            'frame_num': 37
        },
        {
            'name': '101-0086_MID_LAD_PRE',
            'vessel': 'LAD Mid',
            'cine_path': 'E:/AngioMLDL_data/corrected_vessel_dataset/cines/101-0086_MID_LAD_PRE_cine.npy',
            'json_path': 'E:/AngioMLDL_data/corrected_vessel_dataset/contours/101-0086_MID_LAD_PRE_contours.json',
            'frame_num': 30
        },
        {
            'name': '101-0052_DIST_LCX_PRE',
            'vessel': 'LCX Dist',
            'cine_path': 'E:/AngioMLDL_data/corrected_vessel_dataset/cines/101-0052_DIST_LCX_PRE_cine.npy',
            'json_path': 'E:/AngioMLDL_data/corrected_vessel_dataset/contours/101-0052_DIST_LCX_PRE_contours.json',
            'frame_num': 40
        }
    ]

    results = []
    output_dir = Path('deepsa_results')
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("TESTING DEEPSA ON CORONARY ANGIOGRAPHY")
    print("="*70)

    for case in test_cases:
        print(f"\n{'='*70}")
        print(f"Case: {case['name']}")
        print(f"Vessel: {case['vessel']}")
        print(f"Frame: {case['frame_num']}")
        print(f"{'='*70}")

        # Load cine and extract frame
        cine_path = Path(case['cine_path'])

        if not cine_path.exists():
            print(f"ERROR: Cine not found at {cine_path}")
            continue

        # Load cine (numpy array)
        cine = np.load(str(cine_path))
        frame_idx = case['frame_num']

        if frame_idx >= len(cine):
            print(f"ERROR: Frame {frame_idx} out of range (cine has {len(cine)} frames)")
            continue

        # Extract frame
        frame = cine[frame_idx]

        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (frame / frame.max() * 255).astype(np.uint8)

        # Convert to PIL Image
        frame_pil = Image.fromarray(frame)

        # Get DeepSA prediction
        print("Running DeepSA prediction...")
        pred_mask, sub_img = predict_deepsa(model, frame_pil, use_tophat=True, device=device)

        # Load ground truth
        print("Loading ground truth mask...")
        gt_mask = load_ground_truth_mask(
            case['json_path'],
            original_frame_shape=frame.shape,  # Use actual cine frame dimensions
            target_shape=(SIZE, SIZE)
        )

        # Compute metrics
        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)

        print(f"\nResults:")
        print(f"  IoU:  {iou:.3f}")
        print(f"  Dice: {dice:.3f}")

        # Visualize
        visualize_results(
            cv2.resize(frame, (SIZE, SIZE)),
            pred_mask,
            gt_mask,
            case['name'],
            output_dir
        )

        results.append({
            'case': case['name'],
            'vessel': case['vessel'],
            'iou': iou,
            'dice': dice
        })

    # Summary
    print("\n" + "="*70)
    print("DEEPSA BASELINE SUMMARY")
    print("="*70)

    for r in results:
        print(f"{r['vessel']:12s} - IoU: {r['iou']:.3f}, Dice: {r['dice']:.3f}")

    avg_iou = np.mean([r['iou'] for r in results])
    avg_dice = np.mean([r['dice'] for r in results])

    print(f"\n{'Average':12s} - IoU: {avg_iou:.3f}, Dice: {avg_dice:.3f}")

    # Comparison with SAM 3
    sam3_baseline = 0.372
    improvement = (avg_iou - sam3_baseline) / sam3_baseline * 100

    print("\n" + "="*70)
    print("COMPARISON WITH SAM 3 BASELINE")
    print("="*70)
    print(f"SAM 3 Average IoU:   {sam3_baseline:.3f}")
    print(f"DeepSA Average IoU:  {avg_iou:.3f}")
    print(f"Improvement:         {improvement:+.1f}%")

    if avg_iou > sam3_baseline:
        print(f"\nDeepSA is BETTER than SAM 3 baseline by {abs(improvement):.1f}%")
    else:
        print(f"\nDeepSA is WORSE than SAM 3 baseline by {abs(improvement):.1f}%")

    print("\n" + "="*70)
    print(f"Results saved to: {output_dir.absolute()}")
    print("="*70)

    return results


if __name__ == '__main__':
    results = test_deepsa_on_cases()
