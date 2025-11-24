"""
Stage 1: View-Conditioned Vessel Segmentation

Training SAM 3 with LoRA to segment coronary vessels using:
1. DeepSA pseudo-labels (full vessel tree)
2. View angle conditioning (LAO/RAO, Cranial/Caudal)

Key innovation: Model learns view-dependent vessel appearance, improving
spatial understanding for later vessel identification and CASS classification.
"""

import sys
sys.path.append("DeepSA/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image
import json

# Local imports
from view_angle_encoder import ViewAngleEncoder, ViewConditionedFeatureFusion
from sam3_lora_wrapper import SAM3WithLoRA


class ViewConditionedVesselDataset(Dataset):
    """
    Dataset for Stage 1 with view angle conditioning

    Returns:
    - Angiography frame
    - View angles (LAO/RAO, Cranial/Caudal)
    - Vessel mask (DeepSA pseudo-label)
    """

    def __init__(self, csv_path, pseudo_label_dir, image_size=1024, split='train'):
        """
        Args:
            csv_path: Path to corrected_dataset_training.csv
            pseudo_label_dir: Directory with DeepSA pseudo-labels
            image_size: Input size for SAM 3 (default 1024)
            split: 'train', 'val', or 'test'
        """
        self.df = pd.read_csv(csv_path)

        # Filter by split
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.pseudo_label_dir = Path(pseudo_label_dir)
        self.image_size = image_size

        # Track unique cines (avoid processing same cine multiple times)
        self.unique_cines = self.df['cine_path'].unique()

        print(f"Loaded {len(self.df)} cases for {split} split")
        print(f"Unique cines: {len(self.unique_cines)}")

    def __len__(self):
        return len(self.df)

    def load_view_angles(self, contours_path):
        """
        Load view angles from Medis JSON

        Returns:
            primary_angle: LAO/RAO (+ = LAO, - = RAO)
            secondary_angle: Cranial/Caudal (+ = Cranial, - = Caudal)
        """
        with open(contours_path, 'r') as f:
            data = json.load(f)

        angles = data.get('view_angles', {})
        primary = float(angles.get('primary_angle', 0.0))
        secondary = float(angles.get('secondary_angle', 0.0))

        return primary, secondary

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load cine
        cine_path = row['cine_path']
        cine = np.load(cine_path)

        # Get correct frame
        frame_idx = int(row['frame_index'])
        if frame_idx >= len(cine):
            frame_idx = len(cine) - 1

        frame = cine[frame_idx]

        # Normalize to [0, 255]
        if frame.dtype != np.uint8:
            frame = (frame / frame.max() * 255).astype(np.uint8)

        # Load view angles
        contours_path = row['contours_path']
        primary_angle, secondary_angle = self.load_view_angles(contours_path)

        # Load DeepSA pseudo-label
        # Extract unique ID from cine filename
        cine_filename = Path(cine_path).stem.replace('_cine', '')
        pseudo_label_path = self.pseudo_label_dir / f"{cine_filename}_full_vessel_mask.npy"

        if not pseudo_label_path.exists():
            # Fallback: generate placeholder (should not happen if preprocessing ran)
            print(f"WARNING: Pseudo-label not found: {pseudo_label_path}")
            vessel_mask = np.zeros((512, 512), dtype=np.uint8)
        else:
            vessel_mask = np.load(pseudo_label_path)

        # Resize frame and mask
        frame_resized = cv2.resize(frame, (self.image_size, self.image_size))
        mask_resized = cv2.resize(vessel_mask, (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        # SAM 3 expects 3-channel RGB input (we'll replicate grayscale)
        frame_rgb = np.stack([frame_resized] * 3, axis=-1)  # (H, W, 3)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        mask_tensor = torch.from_numpy(mask_resized).float() / 255.0  # (H, W)

        return {
            'image': frame_tensor,  # (3, H, W)
            'mask': mask_tensor,    # (H, W)
            'primary_angle': torch.tensor(primary_angle, dtype=torch.float32),
            'secondary_angle': torch.tensor(secondary_angle, dtype=torch.float32),
            'case_id': row['case_id'],
            'cine_path': cine_path
        }


class ViewConditionedSAM3(nn.Module):
    """
    SAM 3 with LoRA + View Angle Conditioning

    Architecture:
    1. View angle encoder → view embedding
    2. SAM 3 image encoder (with LoRA) → image features
    3. Feature fusion (FiLM modulation)
    4. SAM 3 decoder → vessel mask prediction
    """

    def __init__(self, sam3_lora_wrapper, view_encoder, feature_fusion):
        super().__init__()

        self.sam3 = sam3_lora_wrapper
        self.view_encoder = view_encoder
        self.feature_fusion = feature_fusion

    def forward(self, images, primary_angles, secondary_angles):
        """
        Args:
            images: (B, 3, H, W) RGB images
            primary_angles: (B,) LAO/RAO angles
            secondary_angles: (B,) Cranial/Caudal angles

        Returns:
            masks: (B, H, W) predicted vessel masks
        """
        # Encode view angles
        view_embedding = self.view_encoder(primary_angles, secondary_angles)  # (B, 256)

        # TODO: Integrate with actual SAM 3 forward pass
        # For now, placeholder implementation

        # In actual implementation:
        # 1. SAM 3 image encoder extracts features
        # 2. Fuse view embedding with features using FiLM or attention
        # 3. SAM 3 decoder generates mask

        # Placeholder: random prediction
        B = images.shape[0]
        masks = torch.rand(B, images.shape[2], images.shape[3])

        return masks


def dice_loss(pred, target, smooth=1.0):
    """Dice loss for binary segmentation"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def combined_loss(pred, target):
    """Combined BCE + Dice loss"""
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice


def train_epoch(model, dataloader, optimizer, device='cuda'):
    """Train one epoch"""
    model.train()

    total_loss = 0
    total_dice = 0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image'].to(device)
        masks_gt = batch['mask'].to(device)
        primary_angles = batch['primary_angle'].to(device)
        secondary_angles = batch['secondary_angle'].to(device)

        # Forward pass
        pred_masks = model(images, primary_angles, secondary_angles)

        # Loss
        loss = combined_loss(pred_masks, masks_gt)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        with torch.no_grad():
            dice = 1 - dice_loss(pred_masks, masks_gt)
            total_dice += dice.item()

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)

    return avg_loss, avg_dice


def validate(model, dataloader, device='cuda'):
    """Validate model"""
    model.eval()

    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            masks_gt = batch['mask'].to(device)
            primary_angles = batch['primary_angle'].to(device)
            secondary_angles = batch['secondary_angle'].to(device)

            # Forward
            pred_masks = model(images, primary_angles, secondary_angles)

            # Metrics
            dice = 1 - dice_loss(pred_masks, masks_gt)
            total_dice += dice.item()

            # IoU
            pred_binary = (pred_masks > 0.5).float()
            intersection = (pred_binary * masks_gt).sum()
            union = (pred_binary + masks_gt).clamp(0, 1).sum()
            iou = intersection / (union + 1e-6)
            total_iou += iou.item()

    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_dice, avg_iou


def main():
    """Stage 1 training with view conditioning"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Paths
    csv_path = "E:/AngioMLDL_data/corrected_dataset_training.csv"
    pseudo_label_dir = "E:/AngioMLDL_data/deepsa_pseudo_labels"

    # Hyperparameters
    batch_size = 8  # Can use larger batch with LoRA + dual GPUs
    learning_rate = 1e-4
    epochs = 20
    image_size = 1024

    print("="*70)
    print("STAGE 1: VIEW-CONDITIONED VESSEL SEGMENTATION")
    print("="*70)
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {image_size}x{image_size}")
    print("="*70 + "\n")

    # Create datasets
    print("Loading datasets...")
    train_dataset = ViewConditionedVesselDataset(
        csv_path=csv_path,
        pseudo_label_dir=pseudo_label_dir,
        image_size=image_size,
        split='train'
    )

    val_dataset = ViewConditionedVesselDataset(
        csv_path=csv_path,
        pseudo_label_dir=pseudo_label_dir,
        image_size=image_size,
        split='val'
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")

    # Initialize models
    print("Initializing models...")

    # SAM 3 with LoRA
    sam3_lora = SAM3WithLoRA(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        use_multi_gpu=True
    ).to(device)

    # View encoder
    view_encoder = ViewAngleEncoder(
        embedding_dim=256,
        output_mode='embedding'
    ).to(device)

    # Feature fusion
    feature_fusion = ViewConditionedFeatureFusion(
        feature_dim=256,
        fusion_mode='film'
    ).to(device)

    # Combined model
    model = ViewConditionedSAM3(sam3_lora, view_encoder, feature_fusion)

    print("[OK] Models initialized\n")

    # Optimizer (only LoRA parameters + view encoder)
    trainable_params = [
        {'params': sam3_lora.parameters()},  # LoRA params only
        {'params': view_encoder.parameters()},
        {'params': feature_fusion.parameters()}
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print("="*70)
    print("TRAINING")
    print("="*70)

    best_dice = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-"*70)

        # Train
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, device)
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")

        # Validate
        val_dice, val_iou = validate(model, val_loader, device)
        print(f"Val   - Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        # Learning rate step
        scheduler.step()
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_dice': best_dice
            }
            torch.save(checkpoint, f'checkpoints/stage1_best_dice{val_dice:.4f}.pth')
            print(f"[OK] New best Dice: {best_dice:.4f}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print("="*70)


if __name__ == '__main__':
    # Create checkpoints directory
    Path('checkpoints').mkdir(exist_ok=True)

    main()
