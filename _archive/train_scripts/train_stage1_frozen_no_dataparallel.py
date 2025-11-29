"""
Stage 1: Frozen SAM 3 WITHOUT DataParallel (RoPE Fix)

The issue: SAM 3's RoPE doesn't work with DataParallel
Solution: Extract features on single GPU, then use both GPUs for trainable parts

This reduces SAM 3 to single-GPU but still uses both GPUs for training heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm

# Add sam3 to path
sam3_path = r"C:\Users\odressler\sam3"
if os.path.exists(sam3_path):
    sys.path.insert(0, sam3_path)

from sam3 import build_sam3_image_model
from view_angle_encoder import ViewAngleEncoder, ViewConditionedFeatureFusion


class FrozenSAM3Model(nn.Module):
    """
    SAM 3 with completely frozen backbone (no LoRA, no training)
    Only trainable components: view encoder + segmentation head

    NOTE: SAM 3 backbone runs on single GPU due to RoPE + DataParallel incompatibility
    """
    def __init__(self, image_size=1008):
        super().__init__()

        print("=" * 70)
        print("Building Frozen SAM 3 Model (Single GPU for SAM 3)")
        print("=" * 70)

        # Load SAM 3 and freeze it completely
        print("\nLoading SAM 3...")
        self.sam3 = build_sam3_image_model()
        self.sam3.eval()  # Always in eval mode
        self.sam3.cuda()  # Put on GPU 0

        # Freeze ALL SAM 3 parameters
        for param in self.sam3.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel() for p in self.sam3.parameters())
        print(f"SAM 3 loaded: {frozen_params/1e6:.1f}M params (all frozen, GPU 0 only)")

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

        # Feature projection (will be initialized if needed)
        self.feature_proj = None

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        print(f"\nParameter Summary:")
        print(f"  Total params: {total/1e6:.1f}M")
        print(f"  Trainable: {trainable/1e6:.1f}M")
        print(f"  Frozen (SAM 3): {frozen_params/1e6:.1f}M")
        print(f"  Trainable %: {trainable/total*100:.2f}%")
        print("=" * 70)

    def preprocess_for_backbone(self, image, target_size=1008):
        """
        Resize and pad image to exactly 1008x1008 for SAM 3 backbone
        """
        b, c, h, w = image.shape

        # Resize longest side to target_size
        scale = target_size / max(h, w)
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Pad to square
        curr_h, curr_w = image.shape[-2:]
        pad_h = target_size - curr_h
        pad_w = target_size - curr_w
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return image

    def extract_sam3_features(self, images):
        """
        Extract features from SAM 3 backbone on GPU 0 only
        This avoids DataParallel issues with RoPE
        """
        # Force images to GPU 0
        images = images.cuda(0)

        # Preprocess
        images_padded = self.preprocess_for_backbone(images, target_size=1008)
        images_normalized = (images_padded - 0.5) / 0.5

        # Extract features (no gradients)
        with torch.no_grad():
            self.sam3.eval()
            backbone_out = self.sam3.backbone.forward_image(images_normalized)

        # Get features
        if isinstance(backbone_out, dict):
            features = backbone_out.get('vision_features', backbone_out.get('image_embeddings'))
        else:
            features = backbone_out

        # Ensure (B, C, H, W) format
        if len(features.shape) == 3:
            B_feat, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.permute(0, 2, 1).reshape(B_feat, C, H, W)

        return features

    def forward(self, images, primary_angles, secondary_angles):
        """
        Forward pass - SAM 3 on GPU 0, trainable parts on all GPUs
        """
        # View encoder can run on any device
        view_embedding = self.view_encoder(primary_angles, secondary_angles)

        # SAM 3 features extracted on GPU 0 only (outside DataParallel)
        # This method will be called by DataParallel, but we force GPU 0
        features = self.extract_sam3_features(images)

        # Move features to same device as view_embedding for fusion
        features = features.to(view_embedding.device)

        # Project to 256 channels if needed
        if features.shape[1] != 256:
            if self.feature_proj is None:
                self.feature_proj = nn.Conv2d(features.shape[1], 256, kernel_size=1).to(features.device)
                # Register as a module so optimizer sees it
                self.add_module('feature_proj', self.feature_proj)
            features = self.feature_proj(features)

        # Fuse with view embedding (TRAINABLE)
        fused_features = self.feature_fusion(features, view_embedding)

        # Decode to segmentation mask (TRAINABLE)
        mask_logits = self.seg_head(fused_features)

        # Upsample to original size
        masks = F.interpolate(mask_logits, size=(self.image_size, self.image_size), mode='bilinear')
        masks = masks.squeeze(1)

        return masks


def dice_loss(pred, target, smooth=1.0):
    """Dice loss"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def combined_loss(pred_logits, target, bce_weight=0.5):
    """
    Combined BCE + Dice loss for better training with class imbalance

    Args:
        pred_logits: Raw logits (before sigmoid)
        target: Ground truth masks [0, 1]
        bce_weight: Weight for BCE loss (1 - bce_weight for Dice)
    """
    # BCE loss with logits (more numerically stable)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='mean')

    # Dice loss on sigmoid predictions
    pred_probs = torch.sigmoid(pred_logits)
    dice = dice_loss(pred_probs, target)

    # Combined
    return bce_weight * bce + (1 - bce_weight) * dice


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

        # Load cine and extract frame
        cine = np.load(self.cine_paths[idx])
        frame_idx = self.frame_indices[idx]
        if frame_idx >= len(cine):
            frame_idx = len(cine) - 1
        frame = cine[frame_idx]

        # Handle grayscale
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)

        # Normalize to [0, 1]
        frame = frame.astype(np.float32)
        if frame.max() > 1:
            frame = frame / 255.0

        # Convert to tensor
        image = torch.from_numpy(frame).permute(2, 0, 1)

        # Load mask
        mask = np.load(self.mask_paths[idx]).astype(np.float32)
        if mask.max() > 1:
            mask = mask / 255.0
        mask = torch.from_numpy(mask)

        # Resize for consistent batching
        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size),
                                 mode='bilinear', align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(self.image_size, self.image_size),
                                mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Load view angles
        with open(self.contours_paths[idx], 'r') as f:
            contours = json.load(f)
        view_angles = contours.get('view_angles', {})
        # FIX: Keys are 'primary_angle' and 'secondary_angle', not 'primary' and 'secondary'
        primary_angle = torch.tensor(float(view_angles.get('primary_angle', 0.0)), dtype=torch.float32)
        secondary_angle = torch.tensor(float(view_angles.get('secondary_angle', 0.0)), dtype=torch.float32)

        return image, mask, primary_angle, secondary_angle


def train_epoch(model, dataloader, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, masks, primary_angles, secondary_angles in pbar:
        images = images.to(device)
        masks = masks.to(device)
        primary_angles = primary_angles.to(device)
        secondary_angles = secondary_angles.to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        with torch.amp.autocast('cuda'):
            pred_logits = model(images, primary_angles, secondary_angles)
            # Use combined BCE + Dice loss
            loss = combined_loss(pred_logits, masks, bce_weight=0.5)

        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        with torch.no_grad():
            dice = dice_score(torch.sigmoid(pred_logits), masks)

        total_loss += loss.item()
        total_dice += dice

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})

    return total_loss / len(dataloader), total_dice / len(dataloader)


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
            pred_logits = model(images, primary_angles, secondary_angles)
            loss = combined_loss(pred_logits, masks, bce_weight=0.5)

        dice = dice_score(torch.sigmoid(pred_logits), masks)
        total_loss += loss.item()
        total_dice += dice

    return total_loss / len(dataloader), total_dice / len(dataloader)


def main():
    # Hyperparameters
    batch_size = 16  # Can still use larger batch since features are small
    learning_rate = 1e-4
    epochs = 30
    image_size = 1008

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # Build model
    model = FrozenSAM3Model(image_size=image_size).to(device)

    # NOTE: We don't use DataParallel because SAM 3's RoPE doesn't work with it
    # The trainable parts are small enough to run on one GPU
    print("\nNOTE: Not using DataParallel (SAM 3 RoPE incompatibility)")
    print("SAM 3 backbone on GPU 0 only, trainable heads on GPU 0\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Scaler
    scaler = torch.amp.GradScaler('cuda')

    # Datasets
    csv_path = r'E:\AngioMLDL_data\corrected_dataset_training.csv'
    train_dataset = AngiographyDataset(csv_path, split='train', image_size=image_size)
    val_dataset = AngiographyDataset(csv_path, split='val', image_size=image_size)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    # Training loop
    best_dice = 0.0
    print("Starting training...")
    print("=" * 70)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_dice = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_dice = validate(model, val_loader, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_dice > best_dice:
            best_dice = val_dice
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, 'checkpoints/stage1_frozen_no_dp_best.pth')
            print(f"  -> Saved best model (Dice: {val_dice:.4f})")

        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Max GPU memory: {mem_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
