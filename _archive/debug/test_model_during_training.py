"""
Test SAM 3 model during training
- Loads checkpoint without interfering with training
- Runs on CPU or a separate GPU if available
- Visualizes predictions on a few validation samples
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import model - try new DeepSA model first, fallback to old
try:
    from train_stage1_deepsa import SAM3DeepSAModel, DeepSADataset
    USE_DEEPSA = True
except ImportError:
    from train_stage1_with_ddp import SAM3FineTuneModel, AngiographyDataset
    USE_DEEPSA = False


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Build model based on which training script we're using
    if USE_DEEPSA:
        model = SAM3DeepSAModel(
            image_size=1008,
            freeze_backbone=False
        )
    else:
        model = SAM3FineTuneModel(
            image_size=1008,
            freeze_backbone=False
        )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Remove 'module.' prefix from DDP wrapped model
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Val Dice: {checkpoint['val_dice']:.4f}")

    return model


def predict_and_visualize(model, dataset, num_samples=3, device='cpu'):
    """Predict on samples and visualize"""

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            idx = np.random.randint(0, len(dataset))
            image, mask_gt, primary_angle, secondary_angle = dataset[idx]

            # Add batch dimension
            image_batch = image.unsqueeze(0).to(device)
            primary_batch = primary_angle.unsqueeze(0).to(device)
            secondary_batch = secondary_angle.unsqueeze(0).to(device)

            # Predict
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                mask_pred_logits = model(image_batch, primary_batch, secondary_batch)

            # Convert to float32 before numpy (bfloat16 not supported by numpy)
            mask_pred = torch.sigmoid(mask_pred_logits).squeeze().float().cpu().numpy()

            # Convert to numpy for visualization
            image_np = image.permute(1, 2, 0).numpy()
            mask_gt_np = mask_gt.numpy()

            # Calculate Dice
            mask_pred_binary = (mask_pred > 0.5).astype(np.float32)
            intersection = (mask_pred_binary * mask_gt_np).sum()
            dice = (2. * intersection + 1.0) / (mask_pred_binary.sum() + mask_gt_np.sum() + 1.0)

            # Plot
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f'Input Image (Sample {idx})')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_gt_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(mask_pred, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Predicted Mask (prob)')
            axes[i, 2].axis('off')

            # Overlay
            axes[i, 3].imshow(image_np)
            axes[i, 3].imshow(mask_pred_binary, cmap='Reds', alpha=0.5)
            axes[i, 3].contour(mask_gt_np, colors='green', linewidths=2, levels=[0.5])
            axes[i, 3].set_title(f'Overlay (Dice: {dice:.3f})\nView: ({primary_angle.item():.1f}°, {secondary_angle.item():.1f}°)')
            axes[i, 3].axis('off')

    plt.tight_layout()
    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test model during training')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/phase1_deepsa_best.pth',
                       help='Path to checkpoint')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu or cuda:0, cuda:1)')
    parser.add_argument('--save', type=str, default='test_predictions.png',
                       help='Path to save visualization')

    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Training may not have saved a checkpoint yet.")
        return

    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model = load_checkpoint(args.checkpoint, device=device)

    # Load validation dataset
    print("\nLoading validation dataset...")
    if USE_DEEPSA:
        val_dataset = DeepSADataset(
            csv_path=r'E:\AngioMLDL_data\corrected_dataset_training.csv',
            split='val',
            image_size=1008
        )
    else:
        val_dataset = AngiographyDataset(
            csv_path=r'E:\AngioMLDL_data\corrected_dataset_training.csv',
            split='val',
            image_size=1008
        )

    # Predict and visualize
    print(f"\nGenerating predictions on {args.num_samples} samples...")
    fig = predict_and_visualize(model, val_dataset, num_samples=args.num_samples, device=device)

    # Save
    plt.savefig(args.save, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {args.save}")

    # Show
    plt.show()


if __name__ == '__main__':
    main()
