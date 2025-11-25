"""
Diagnose what the model has learned:
1. Check alignment between raw images and GT masks
2. Test view angle conditioning (does it respond to different angles?)
3. Visualize feature activations
4. Compare with DeepSA predictions if available
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from train_stage1_with_ddp import SAM3FineTuneModel, AngiographyDataset


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    model = SAM3FineTuneModel(image_size=1008, freeze_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith('module.') else k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


def check_alignment(dataset, num_samples=3):
    """Check if GT masks align with input images"""
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    print("\n" + "="*70)
    print("ALIGNMENT CHECK: Do GT masks align with visible vessels?")
    print("="*70)

    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))

        # Get original data paths
        cine_path = dataset.cine_paths[idx]
        mask_path = dataset.mask_paths[idx]
        frame_idx = dataset.frame_indices[idx]

        # Load ORIGINAL (before preprocessing)
        cine_orig = np.load(cine_path)
        if frame_idx >= len(cine_orig):
            frame_idx = len(cine_orig) - 1
        frame_orig = cine_orig[frame_idx]

        # Convert to float
        if frame_orig.ndim == 2:
            frame_orig = np.stack([frame_orig, frame_orig, frame_orig], axis=-1)
        frame_orig = frame_orig.astype(np.float32)
        if frame_orig.max() > 1:
            frame_orig = frame_orig / 255.0

        mask_orig = np.load(mask_path).astype(np.float32)
        if mask_orig.max() > 1:
            mask_orig = mask_orig / 255.0

        print(f"\nSample {idx}:")
        print(f"  Original image shape: {frame_orig.shape}")
        print(f"  Original mask shape: {mask_orig.shape}")

        # Load PREPROCESSED (from dataset)
        image, mask_preprocessed, primary_angle, secondary_angle = dataset[idx]
        image_np = image.permute(1, 2, 0).numpy()
        mask_preprocessed_np = mask_preprocessed.numpy()

        print(f"  Preprocessed image shape: {image_np.shape}")
        print(f"  Preprocessed mask shape: {mask_preprocessed_np.shape}")

        # Check alignment
        vessel_pixels_orig = (mask_orig > 0.5).sum()
        vessel_pixels_prep = (mask_preprocessed_np > 0.5).sum()
        print(f"  Vessel pixels - Original: {vessel_pixels_orig}, Preprocessed: {vessel_pixels_prep}")

        # Plot
        axes[i, 0].imshow(frame_orig)
        axes[i, 0].set_title(f'Original Image {frame_orig.shape}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(frame_orig)
        axes[i, 1].contour(mask_orig, colors='red', linewidths=2, levels=[0.5])
        axes[i, 1].set_title('Original + GT Overlay')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(image_np)
        axes[i, 2].set_title(f'Preprocessed {image_np.shape[:2]}')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(image_np)
        axes[i, 3].contour(mask_preprocessed_np, colors='red', linewidths=2, levels=[0.5])
        axes[i, 3].set_title('Preprocessed + GT Overlay')
        axes[i, 3].axis('off')

    plt.tight_layout()
    return fig


def test_view_angle_sensitivity(model, dataset, device='cpu'):
    """Test if model responds to different view angles"""

    print("\n" + "="*70)
    print("VIEW ANGLE SENSITIVITY: Does the model use view information?")
    print("="*70)

    # Get one sample
    idx = 50
    image, mask_gt, primary_orig, secondary_orig = dataset[idx]

    print(f"\nOriginal view angles: ({primary_orig.item():.1f}°, {secondary_orig.item():.1f}°)")

    # Test different view angles
    test_angles = [
        (0, 0, "Straight on"),
        (30, 0, "RAO 30°"),
        (-30, 0, "LAO 30°"),
        (0, 30, "Cranial 30°"),
        (0, -30, "Caudal 30°"),
        (primary_orig.item(), secondary_orig.item(), "Original"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    with torch.no_grad():
        for i, (prim, sec, label) in enumerate(test_angles):
            image_batch = image.unsqueeze(0).to(device)
            primary_batch = torch.tensor([prim], dtype=torch.float32).to(device)
            secondary_batch = torch.tensor([sec], dtype=torch.float32).to(device)

            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                mask_pred_logits = model(image_batch, primary_batch, secondary_batch)

            mask_pred = torch.sigmoid(mask_pred_logits).squeeze().float().cpu().numpy()

            # Calculate stats
            mean_prob = mask_pred.mean()
            max_prob = mask_pred.max()
            vessel_pixels = (mask_pred > 0.5).sum()

            print(f"{label:20s}: Mean={mean_prob:.3f}, Max={max_prob:.3f}, Pixels>{0.5:.1f}={vessel_pixels}")

            axes[i].imshow(image.permute(1, 2, 0).numpy())
            axes[i].imshow(mask_pred, cmap='Reds', alpha=0.6)
            axes[i].set_title(f'{label}\nPixels: {vessel_pixels}')
            axes[i].axis('off')

    plt.tight_layout()
    return fig


def check_deepsa_availability(dataset):
    """Check if DeepSA predictions are available"""
    print("\n" + "="*70)
    print("DEEPSA PSEUDO-LABELS: Are they available?")
    print("="*70)

    # Check if there's a DeepSA predictions directory
    data_dir = Path(dataset.cine_paths[0]).parent.parent

    potential_deepsa_dirs = [
        data_dir / "deepsa_predictions",
        data_dir / "deepsa_masks",
        data_dir / "pseudo_labels",
        Path("E:/AngioMLDL_data/deepsa_predictions"),
    ]

    for deepsa_dir in potential_deepsa_dirs:
        print(f"Checking: {deepsa_dir}")
        if deepsa_dir.exists():
            num_files = len(list(deepsa_dir.glob("*.npy")))
            print(f"  ✓ Found! Contains {num_files} .npy files")
            return deepsa_dir
        else:
            print(f"  ✗ Not found")

    print("\nDeepSA pseudo-labels NOT found.")
    print("Recommendation: Generate them using DeepSA for better training labels!")
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/stage1_ddp_full_ft_best.pth')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    val_dataset = AngiographyDataset(
        csv_path=r'E:\AngioMLDL_data\corrected_dataset_training.csv',
        split='val',
        image_size=1008
    )

    # 1. Check alignment
    print("\n📍 TEST 1: Checking GT mask alignment...")
    fig1 = check_alignment(val_dataset, num_samples=3)
    plt.savefig('diagnostic_1_alignment.png', dpi=150, bbox_inches='tight')
    print("Saved to: diagnostic_1_alignment.png")

    # 2. Check DeepSA availability
    deepsa_dir = check_deepsa_availability(val_dataset)

    # 3. Test view angle sensitivity
    if Path(args.checkpoint).exists():
        print(f"\n📍 TEST 2: Loading model from {args.checkpoint}...")
        model = load_checkpoint(args.checkpoint, device=device)

        print("\n🔍 Testing view angle sensitivity...")
        fig2 = test_view_angle_sensitivity(model, val_dataset, device=device)
        plt.savefig('diagnostic_2_view_angles.png', dpi=150, bbox_inches='tight')
        print("Saved to: diagnostic_2_view_angles.png")

    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE!")
    print("="*70)
    print("\nKey findings will be in:")
    print("  1. diagnostic_1_alignment.png - Shows if GT masks align with images")
    print("  2. diagnostic_2_view_angles.png - Shows if model uses view information")

    plt.show()


if __name__ == '__main__':
    main()
