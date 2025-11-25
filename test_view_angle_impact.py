"""
Test if the model actually uses view angle information.

Experiment:
1. Take same image, run with correct view angles -> get prediction
2. Run same image with WRONG view angles -> get prediction
3. If model uses view angles, predictions should differ
4. If predictions are identical, view angles are being ignored

Also tests:
- Systematic angle sweeps to see how predictions change
- Compare Dice with correct vs random angles
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, r'C:\Users\odressler\sam3')

from train_stage1_deepsa import SAM3DeepSAModel, DeepSADataset


def load_model(checkpoint_path, device='cpu'):
    """Load trained model"""
    model = SAM3DeepSAModel(image_size=1008, freeze_backbone=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Remove 'module.' prefix from DDP
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, Val Dice: {checkpoint['val_dice']:.4f})")
    return model


def dice_score(pred, target):
    """Calculate Dice score"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1.0) / (pred.sum() + target.sum() + 1.0)


def test_angle_sensitivity(model, dataset, device, num_samples=10):
    """
    Test 1: Compare predictions with correct vs wrong angles
    """
    print("\n" + "="*70)
    print("TEST 1: Correct vs Wrong View Angles")
    print("="*70)

    correct_dices = []
    wrong_dices = []
    prediction_diffs = []

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            image, mask_gt, primary_correct, secondary_correct = dataset[idx]

            image = image.unsqueeze(0).to(device)
            mask_gt = mask_gt.to(device)
            primary_correct = primary_correct.unsqueeze(0).to(device)
            secondary_correct = secondary_correct.unsqueeze(0).to(device)

            # Prediction with CORRECT angles
            pred_correct = torch.sigmoid(model(image, primary_correct, secondary_correct)).squeeze()

            # Prediction with WRONG angles (opposite sign, different values)
            primary_wrong = torch.tensor([-primary_correct.item() + 20], device=device)
            secondary_wrong = torch.tensor([-secondary_correct.item() - 15], device=device)
            pred_wrong = torch.sigmoid(model(image, primary_wrong, secondary_wrong)).squeeze()

            # Prediction with RANDOM angles
            primary_random = torch.tensor([np.random.uniform(-40, 40)], device=device, dtype=torch.float32)
            secondary_random = torch.tensor([np.random.uniform(-40, 40)], device=device, dtype=torch.float32)
            pred_random = torch.sigmoid(model(image, primary_random, secondary_random)).squeeze()

            # Calculate metrics
            dice_correct = dice_score(pred_correct, mask_gt).item()
            dice_wrong = dice_score(pred_wrong, mask_gt).item()

            # How different are the predictions?
            pred_diff = torch.abs(pred_correct - pred_wrong).mean().item()

            correct_dices.append(dice_correct)
            wrong_dices.append(dice_wrong)
            prediction_diffs.append(pred_diff)

            print(f"Sample {idx}: Correct={dice_correct:.3f}, Wrong={dice_wrong:.3f}, Pred diff={pred_diff:.4f}")

    print(f"\nSummary:")
    print(f"  Avg Dice with CORRECT angles: {np.mean(correct_dices):.4f}")
    print(f"  Avg Dice with WRONG angles:   {np.mean(wrong_dices):.4f}")
    print(f"  Avg prediction difference:    {np.mean(prediction_diffs):.4f}")

    if np.mean(prediction_diffs) < 0.01:
        print("\n  WARNING: Predictions barely change with different angles!")
        print("  The model may not be using view angle information effectively.")
    elif np.mean(correct_dices) > np.mean(wrong_dices):
        print(f"\n  GOOD: Correct angles give better Dice (+{np.mean(correct_dices) - np.mean(wrong_dices):.4f})")
        print("  The model IS using view angle information!")
    else:
        print("\n  INCONCLUSIVE: Wrong angles give similar or better Dice")
        print("  View angles may not be helping.")

    return correct_dices, wrong_dices, prediction_diffs


def test_angle_sweep(model, dataset, device, sample_idx=0):
    """
    Test 2: Sweep through angles and visualize how predictions change
    """
    print("\n" + "="*70)
    print("TEST 2: Angle Sweep Visualization")
    print("="*70)

    image, mask_gt, primary_orig, secondary_orig = dataset[sample_idx]
    image = image.unsqueeze(0).to(device)
    mask_gt = mask_gt.numpy()

    # Sweep primary angle from -40 to +40
    angles = np.linspace(-40, 40, 9)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Sample {sample_idx}: Primary angle sweep (Secondary fixed at {secondary_orig.item():.1f})',
                 fontsize=14, fontweight='bold')

    # First plot: original image
    axes[0, 0].imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')

    # Ground truth
    axes[1, 0].imshow(mask_gt, cmap='gray')
    axes[1, 0].set_title(f'GT (orig angle: {primary_orig.item():.1f})')
    axes[1, 0].axis('off')

    predictions = []
    with torch.no_grad():
        for i, angle in enumerate(angles[:4]):
            primary = torch.tensor([angle], device=device, dtype=torch.float32)
            secondary = secondary_orig.unsqueeze(0).to(device)

            pred = torch.sigmoid(model(image, primary, secondary)).squeeze().cpu().numpy()
            predictions.append(pred)

            dice = dice_score(torch.tensor(pred), torch.tensor(mask_gt)).item()

            ax_idx = i + 1
            axes[0, ax_idx].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[0, ax_idx].set_title(f'Primary={angle:.0f} (Dice={dice:.3f})')
            axes[0, ax_idx].axis('off')

        for i, angle in enumerate(angles[4:8]):
            primary = torch.tensor([angle], device=device, dtype=torch.float32)
            secondary = secondary_orig.unsqueeze(0).to(device)

            pred = torch.sigmoid(model(image, primary, secondary)).squeeze().cpu().numpy()
            predictions.append(pred)

            dice = dice_score(torch.tensor(pred), torch.tensor(mask_gt)).item()

            axes[1, i + 1].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[1, i + 1].set_title(f'Primary={angle:.0f} (Dice={dice:.3f})')
            axes[1, i + 1].axis('off')

    plt.tight_layout()
    plt.savefig('view_angle_sweep.png', dpi=150, bbox_inches='tight')
    print(f"Saved angle sweep visualization to view_angle_sweep.png")

    # Calculate variance across predictions
    pred_stack = np.stack(predictions)
    pixel_variance = pred_stack.var(axis=0).mean()
    print(f"\nMean pixel variance across angle sweep: {pixel_variance:.6f}")

    if pixel_variance < 0.001:
        print("WARNING: Very low variance - predictions barely change with angle!")
    else:
        print("Predictions DO change with viewing angle.")

    return predictions


def test_embedding_distances(model, device):
    """
    Test 3: Check if view encoder produces different embeddings for different angles
    """
    print("\n" + "="*70)
    print("TEST 3: View Encoder Embedding Analysis")
    print("="*70)

    view_encoder = model.view_encoder

    # Test angles
    test_angles = [
        (0, 0),      # Neutral
        (30, 0),     # RAO
        (-30, 0),    # LAO
        (0, 30),     # CRAN
        (0, -30),    # CAUD
        (30, 30),    # RAO-CRAN
        (-30, -30),  # LAO-CAUD
    ]

    embeddings = []
    with torch.no_grad():
        for primary, secondary in test_angles:
            p = torch.tensor([primary], device=device, dtype=torch.float32)
            s = torch.tensor([secondary], device=device, dtype=torch.float32)
            emb = view_encoder(p, s)
            embeddings.append(emb.squeeze().cpu().numpy())

    embeddings = np.stack(embeddings)

    # Compute pairwise cosine similarities
    print("\nCosine similarity matrix:")
    print("Angles:  ", [f"({p},{s})" for p, s in test_angles])

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    similarities = normalized @ normalized.T

    print("\n        ", end="")
    for i in range(len(test_angles)):
        print(f"  {i}   ", end="")
    print()

    for i, (p, s) in enumerate(test_angles):
        print(f"({p:3},{s:3})", end=" ")
        for j in range(len(test_angles)):
            print(f"{similarities[i,j]:.3f} ", end="")
        print()

    # Check if embeddings are all too similar
    off_diagonal = similarities[np.triu_indices(len(test_angles), k=1)]
    avg_similarity = off_diagonal.mean()

    print(f"\nAverage off-diagonal similarity: {avg_similarity:.4f}")

    if avg_similarity > 0.95:
        print("WARNING: Embeddings are too similar! View encoder may not be discriminative.")
    elif avg_similarity > 0.8:
        print("Embeddings have moderate variation. Could be improved.")
    else:
        print("GOOD: Embeddings are sufficiently different for different angles.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test view angle impact')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/phase1_deepsa_best.pth')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-samples', type=int, default=10)

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load dataset
    dataset = DeepSADataset(
        csv_path=r'E:\AngioMLDL_data\corrected_dataset_training.csv',
        split='val',
        image_size=1008
    )

    # Run tests
    test_embedding_distances(model, device)
    test_angle_sensitivity(model, dataset, device, num_samples=args.num_samples)
    test_angle_sweep(model, dataset, device, sample_idx=0)

    plt.show()


if __name__ == '__main__':
    main()
