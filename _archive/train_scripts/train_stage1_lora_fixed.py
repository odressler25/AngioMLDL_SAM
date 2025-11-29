"""
Stage 1: Training with SAM 3 + LoRA (FIXED VERSION with 1008x1008)
View-conditioned vessel segmentation for coronary angiography

This version adds LoRA back now that we know the fix for RoPE errors.
Key fix: Use 1008x1008 resolution (SAM 3's expected size, not 1024x1024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
import time

# Add sam3 to path
sam3_path = r"C:\Users\odressler\sam3"
if os.path.exists(sam3_path):
    sys.path.insert(0, sam3_path)

from sam3 import build_sam3_image_model
from sam3_lora_wrapper import SAM3WithLoRA
from view_angle_encoder import ViewAngleEncoder, ViewConditionedFeatureFusion


class SAM3LoRAModel(nn.Module):
    """
    SAM 3 with LoRA for domain adaptation + view conditioning
    FIXED: Uses 1008x1008 resolution + .contiguous() for DataParallel compatibility
    """
    def __init__(self, lora_r=16, lora_alpha=32, image_size=1008, use_multi_gpu=True):
        super().__init__()

        print("=" * 70)
        print("Building SAM 3 with LoRA (FIXED - 1008x1008 + DataParallel)")
        print("=" * 70)

        # Load SAM 3 with LoRA
        print("\nLoading SAM 3 with LoRA...")
        self.sam3_lora = SAM3WithLoRA(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            use_multi_gpu=use_multi_gpu
        )

        # SAM3WithLoRA already prints parameter summary
        # Just get the counts for our own summary later
        sam3_trainable = sum(p.numel() for p in self.sam3_lora.parameters() if p.requires_grad)
        sam3_total = sum(p.numel() for p in self.sam3_lora.parameters())

        # View angle encoder (TRAINABLE)
        self.view_encoder = ViewAngleEncoder(embedding_dim=256)

        # Feature fusion (TRAINABLE)
        self.feature_fusion = ViewConditionedFeatureFusion(
            feature_dim=256,
            fusion_mode='film'
        )

        self.image_size = image_size

        # Segmentation head (TRAINABLE)
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # Count all trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  Total trainable (LoRA + view + head): {trainable/1e6:.1f}M")
        print("=" * 70)

    def forward(self, images, primary_angles, secondary_angles):
        """
        Forward pass with LoRA-adapted SAM 3 backbone

        Args:
            images: (B, 3, H, W) in [0, 1]
            primary_angles: (B,) LAO/RAO angles
            secondary_angles: (B,) Cranial/Caudal angles
        Returns:
            masks: (B, H, W) predicted vessel masks
        """
        # Encode view angles (TRAINABLE)
        view_embedding = self.view_encoder(primary_angles, secondary_angles)

        # Normalize images for SAM 3
        images_normalized = (images - 0.5) / 0.5

        # Extract SAM 3 features with LoRA adaptation
        # The wrapper now handles preprocessing to 1024x1024 and returns embeddings
        with torch.amp.autocast('cuda'):
            features = self.sam3_lora(images_normalized)

        # Ensure (B, C, H, W) format
        if len(features.shape) == 3:
            B_feat, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.permute(0, 2, 1).reshape(B_feat, C, H, W)

        # Project to 256 channels if needed
        if features.shape[1] != 256:
            if not hasattr(self, 'feature_proj'):
                self.feature_proj = nn.Conv2d(features.shape[1], 256, kernel_size=1).to(features.device)
            features = self.feature_proj(features)

        # Fuse with view embedding (TRAINABLE)
        fused_features = self.feature_fusion(features, view_embedding)

        # Decode to segmentation mask (TRAINABLE)
        mask_logits = self.seg_head(fused_features)

        # Upsample to original size
        masks = F.interpolate(mask_logits, size=(self.image_size, self.image_size), mode='bilinear')
        masks = masks.squeeze(1)  # (B, H, W)

        return masks


def dice_loss(pred, target, smooth=1.0):
    """Dice loss"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def dice_score(pred, target, smooth=1.0):
    """Dice coefficient for evaluation"""
    pred = (pred > 0.5).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


class AngiographyDataset(Dataset):
    """Dataset for coronary angiography"""
    def __init__(self, csv_path, split='train', image_size=1008):
        import pandas as pd
        import json

        self.split = split
        self.image_size = image_size

        # Load CSV
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)

        self.cine_paths = df['cine_path'].tolist()
        self.mask_paths = df['vessel_mask_actual_path'].tolist()
        self.contours_paths = df['contours_path'].tolist()
        self.frame_indices = df['frame_index'].tolist()

        print(f"Loaded {len(self)} {split} samples")

    def __len__(self):
        return len(self.cine_paths)

    def __getitem__(self, idx):
        import json

        # Load cine (video) and extract frame
        cine = np.load(self.cine_paths[idx])  # Shape: (T, H, W) or (T, H, W, C)
        frame_idx = self.frame_indices[idx]

        # Extract single frame
        if frame_idx >= len(cine):
            frame_idx = len(cine) - 1
        frame = cine[frame_idx]

        # Handle grayscale: (H, W) -> (H, W, 3)
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)

        # Normalize to [0, 1]
        frame = frame.astype(np.float32)
        if frame.max() > 1:
            frame = frame / 255.0

        # Convert to tensor (H, W, C) -> (C, H, W)
        image = torch.from_numpy(frame).permute(2, 0, 1)

        # Load vessel mask
        mask = np.load(self.mask_paths[idx])  # Shape: (H, W)
        mask = mask.astype(np.float32)
        if mask.max() > 1:
            mask = mask / 255.0
        mask = torch.from_numpy(mask)

        # Resize to target size (needed for batching - images have different sizes)
        # The model will preprocess to 1008x1008 anyway, but we need consistent sizes for batching
        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Load view angles from contours JSON
        with open(self.contours_paths[idx], 'r') as f:
            contours = json.load(f)

        view_angles = contours.get('view_angles', {})
        primary_angle = float(view_angles.get('primary', 0.0))  # LAO/RAO
        secondary_angle = float(view_angles.get('secondary', 0.0))  # Cranial/Caudal

        primary_angle = torch.tensor(primary_angle, dtype=torch.float32)
        secondary_angle = torch.tensor(secondary_angle, dtype=torch.float32)

        return image, mask, primary_angle, secondary_angle


def train_epoch(model, dataloader, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, masks, primary_angles, secondary_angles) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        primary_angles = primary_angles.to(device)
        secondary_angles = secondary_angles.to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        with torch.amp.autocast('cuda'):
            pred_masks = model(images, primary_angles, secondary_angles)
            loss = dice_loss(torch.sigmoid(pred_masks), masks)

        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        with torch.no_grad():
            dice = dice_score(torch.sigmoid(pred_masks), masks)

        total_loss += loss.item()
        total_dice += dice

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)

    return avg_loss, avg_dice


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_dice = 0

    for images, masks, primary_angles, secondary_angles in tqdm(dataloader, desc='Validation'):
        images = images.to(device)
        masks = masks.to(device)
        primary_angles = primary_angles.to(device)
        secondary_angles = secondary_angles.to(device)

        with torch.amp.autocast('cuda'):
            pred_masks = model(images, primary_angles, secondary_angles)
            loss = dice_loss(torch.sigmoid(pred_masks), masks)

        dice = dice_score(torch.sigmoid(pred_masks), masks)

        total_loss += loss.item()
        total_dice += dice

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)

    return avg_loss, avg_dice


def main():
    # Hyperparameters
    batch_size = 16  # Target ~20GB per GPU with DataParallel (now fixed!)
    learning_rate = 1e-4
    epochs = 30
    image_size = 1008  # SAM 3 expects 1008x1008 (wrapper also uses 1008)
    lora_r = 16
    lora_alpha = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # Build model with LoRA
    use_multi_gpu = torch.cuda.device_count() > 1
    model = SAM3LoRAModel(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        image_size=image_size,
        use_multi_gpu=use_multi_gpu
    ).to(device)

    # Use DataParallel if multiple GPUs (now fixed with .contiguous() and 1008 resolution)
    if use_multi_gpu:
        print(f"\nUsing {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Optimizer (trainable parameters: LoRA + view encoder + seg head)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda')

    # Datasets
    csv_path = r'E:\AngioMLDL_data\corrected_dataset_training.csv'
    train_dataset = AngiographyDataset(csv_path, split='train', image_size=image_size)
    val_dataset = AngiographyDataset(csv_path, split='val', image_size=image_size)

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

    # Training loop
    best_dice = 0.0

    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, device, scaler)

        # Validate
        val_loss, val_dice = validate(model, val_loader, device)

        # Scheduler step
        scheduler.step()

        # Print metrics
        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint_path = 'checkpoints/stage1_lora_fixed_best.pth'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, checkpoint_path)
            print(f"  -> Saved best model (Dice: {val_dice:.4f})")

        # GPU memory stats
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Max GPU memory: {mem_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
