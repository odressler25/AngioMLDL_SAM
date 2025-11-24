"""
Stage 1: View-Conditioned Vessel Segmentation - OVERNIGHT TRAINING

Simplified for overnight run with comprehensive logging.
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
from datetime import datetime

# Local imports
from view_angle_encoder import ViewAngleEncoder, ViewConditionedFeatureFusion
from sam3_lora_wrapper import SAM3WithLoRA


class ViewConditionedVesselDataset(Dataset):
    """Dataset for Stage 1 with view angle conditioning"""

    def __init__(self, csv_path, pseudo_label_dir, image_size=1024):
        self.df = pd.read_csv(csv_path)
        self.pseudo_label_dir = Path(pseudo_label_dir)
        self.image_size = image_size

        print(f"[DATA] Loaded {len(self.df)} cases")

    def load_view_angles(self, contours_path):
        """Load view angles from Medis JSON"""
        with open(contours_path, 'r') as f:
            data = json.load(f)
        angles = data.get('view_angles', {})
        return float(angles.get('primary_angle', 0.0)), float(angles.get('secondary_angle', 0.0))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load frame
        cine = np.load(row['cine_path'])
        frame_idx = int(row['frame_index'])
        if frame_idx >= len(cine):
            frame_idx = len(cine) - 1
        frame = cine[frame_idx]

        # Normalize
        if frame.dtype != np.uint8:
            frame = (frame / frame.max() * 255).astype(np.uint8)

        # Load view angles
        primary_angle, secondary_angle = self.load_view_angles(row['contours_path'])

        # Load pseudo-label
        cine_filename = Path(row['cine_path']).stem.replace('_cine', '')
        pseudo_label_path = self.pseudo_label_dir / f"{cine_filename}_full_vessel_mask.npy"

        if not pseudo_label_path.exists():
            print(f"[WARN] Missing pseudo-label: {cine_filename}")
            vessel_mask = np.zeros((512, 512), dtype=np.uint8)
        else:
            vessel_mask = np.load(pseudo_label_path)

        # Resize
        frame_resized = cv2.resize(frame, (self.image_size, self.image_size))
        mask_resized = cv2.resize(vessel_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # To tensor (SAM 3 expects RGB)
        frame_rgb = np.stack([frame_resized] * 3, axis=-1)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized).float() / 255.0

        return {
            'image': frame_tensor,
            'mask': mask_tensor,
            'primary_angle': torch.tensor(primary_angle, dtype=torch.float32),
            'secondary_angle': torch.tensor(secondary_angle, dtype=torch.float32),
            'case_id': row['case_id']
        }


class SimpleViewConditionedSAM3(nn.Module):
    """
    Simplified SAM 3 + LoRA + View Conditioning for overnight training

    Uses a straightforward segmentation head on top of SAM 3 features
    """

    def __init__(self, sam3_lora, view_encoder, feature_fusion, image_size=1024):
        super().__init__()
        self.sam3_lora = sam3_lora
        self.view_encoder = view_encoder
        self.feature_fusion = feature_fusion
        self.image_size = image_size

        # Simple segmentation head
        # SAM 3 features are typically 256-dim, we'll add a decoder
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def preprocess_for_backbone(self, image, target_size=1024):
        """
        Resize and pad image to exactly target_size x target_size.
        This fixes the RoPE assertion by ensuring sequence length matches expectations.

        Args:
            image: (B, C, H, W) tensor in [0, 1] range
            target_size: int, SAM 3 expects 1024x1024
        Returns:
            preprocessed: (B, C, target_size, target_size)
        """
        b, c, h, w = image.shape

        # 1. Resize longest side to target_size (maintain aspect ratio)
        scale = target_size / max(h, w)
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # 2. Pad to form perfect square (target_size x target_size)
        # SAM models expect padding on right and bottom
        curr_h, curr_w = image.shape[-2:]
        pad_h = target_size - curr_h
        pad_w = target_size - curr_w

        # F.pad format: (left, right, top, bottom)
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return image

    def forward(self, images, primary_angles, secondary_angles):
        """
        Args:
            images: (B, 3, H, W)
            primary_angles: (B,)
            secondary_angles: (B,)
        Returns:
            masks: (B, H, W) predicted vessel masks
        """
        B = images.shape[0]

        # Encode view angles
        view_embedding = self.view_encoder(primary_angles, secondary_angles)

        # Extract SAM 3 features using backbone.forward_image()
        # This is how Sam3Processor gets features for inference

        # Access the actual model (handle DataParallel wrapper if present)
        sam3_model = self.sam3_lora.model
        if hasattr(sam3_model, 'module'):
            sam3_model = sam3_model.module

        # Get backbone
        backbone = sam3_model.backbone

        # Extract features using SAM 3's backbone forward_image
        with torch.amp.autocast('cuda'):
            # CRITICAL FIX: Preprocess to exactly 1024x1024 to fix RoPE assertion
            # RoPE expects fixed sequence length from fixed image dimensions
            images_padded = self.preprocess_for_backbone(images, target_size=1024)

            # SAM 3 backbone.forward_image expects:
            # - Shape: (B, 3, 1024, 1024) - FIXED SIZE for RoPE
            # - Normalized with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

            # Our images are already [0, 1], normalize them
            images_normalized = (images_padded - 0.5) / 0.5

            # Get backbone features
            backbone_out = backbone.forward_image(images_normalized)

            # backbone_out is a dict with various keys
            # The main visual features are in 'vision_features'
            if 'vision_features' in backbone_out:
                features = backbone_out['vision_features']
            elif 'image_embeddings' in backbone_out:
                features = backbone_out['image_embeddings']
            else:
                # Fallback: try to find the first tensor-valued feature
                for k, v in backbone_out.items():
                    if isinstance(v, torch.Tensor) and len(v.shape) >= 3:
                        features = v
                        break

            # Ensure features are (B, C, H, W) format
            if len(features.shape) == 3:
                # (B, H*W, C) -> (B, C, H, W)
                B_feat, N, C = features.shape
                H = W = int(N ** 0.5)
                features = features.permute(0, 2, 1).reshape(B_feat, C, H, W)

            # Ensure 256 channels (match view embedding dimension)
            if features.shape[1] != 256:
                # Project to 256 channels if needed
                if not hasattr(self, 'feature_proj'):
                    self.feature_proj = nn.Conv2d(features.shape[1], 256, kernel_size=1).to(features.device)
                features = self.feature_proj(features)

        # Fuse with view embedding (view_embedding is B,256 -> broadcast to B,256,H,W)
        fused_features = self.feature_fusion(features, view_embedding)

        # Decode to segmentation mask
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


def combined_loss(pred_logits, target):
    """Combined BCE + Dice loss"""
    pred = torch.sigmoid(pred_logits)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice


def log_message(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def train_epoch(model, dataloader, optimizer, device='cuda', epoch=0):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Train")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks_gt = batch['mask'].to(device)
        primary_angles = batch['primary_angle'].to(device)
        secondary_angles = batch['secondary_angle'].to(device)

        # Forward
        pred_masks = model(images, primary_angles, secondary_angles)
        loss = combined_loss(pred_masks, masks_gt)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        with torch.no_grad():
            pred_sigmoid = torch.sigmoid(pred_masks)
            dice = 1 - dice_loss(pred_sigmoid, masks_gt)
            total_dice += dice.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    return avg_loss, avg_dice


def validate(model, dataloader, device='cuda'):
    """Validate"""
    model.eval()
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            masks_gt = batch['mask'].to(device)
            primary_angles = batch['primary_angle'].to(device)
            secondary_angles = batch['secondary_angle'].to(device)

            pred_masks = model(images, primary_angles, secondary_angles)
            pred = torch.sigmoid(pred_masks)

            # Dice
            dice = 1 - dice_loss(pred, masks_gt)
            total_dice += dice.item()

            # IoU
            pred_binary = (pred > 0.5).float()
            intersection = (pred_binary * masks_gt).sum()
            union = (pred_binary + masks_gt).clamp(0, 1).sum()
            iou = intersection / (union + 1e-6)
            total_iou += iou.item()

            pbar.set_postfix({'dice': f'{dice.item():.4f}', 'iou': f'{iou.item():.4f}'})

    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    return avg_dice, avg_iou


def main():
    """Overnight training"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_message(f"Device: {device}")
    log_message(f"GPU Count: {torch.cuda.device_count()}")

    # Paths
    csv_path = "E:/AngioMLDL_data/corrected_dataset_training.csv"
    pseudo_label_dir = "E:/AngioMLDL_data/deepsa_pseudo_labels"

    # Hyperparameters - MAXIMIZE GPU USAGE (2x RTX 3090 = 48GB total)
    batch_size = 16  # Target ~20GB per GPU with DataParallel
    learning_rate = 1e-4
    epochs = 30  # Overnight run
    image_size = 1024  # Full SAM 3 resolution

    log_message("="*70)
    log_message("STAGE 1: VIEW-CONDITIONED VESSEL SEGMENTATION - OVERNIGHT")
    log_message("="*70)
    log_message(f"Batch size: {batch_size}")
    log_message(f"Learning rate: {learning_rate}")
    log_message(f"Epochs: {epochs}")
    log_message(f"Image size: {image_size}x{image_size}")
    log_message("="*70)

    # Dataset
    log_message("\nCreating dataset...")
    dataset = ViewConditionedVesselDataset(csv_path, pseudo_label_dir, image_size)

    # 80/20 train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    log_message(f"Train samples: {len(train_dataset)}")
    log_message(f"Val samples: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    log_message(f"Train batches: {len(train_loader)}")
    log_message(f"Val batches: {len(val_loader)}")

    # Models
    log_message("\nInitializing models...")
    sam3_lora = SAM3WithLoRA(lora_r=16, lora_alpha=32, use_multi_gpu=True).to(device)
    view_encoder = ViewAngleEncoder(embedding_dim=256).to(device)
    feature_fusion = ViewConditionedFeatureFusion(feature_dim=256, fusion_mode='film').to(device)

    model = SimpleViewConditionedSAM3(sam3_lora, view_encoder, feature_fusion, image_size)

    # Apply DataParallel BEFORE moving to device (critical for multi-GPU)
    if torch.cuda.device_count() > 1:
        log_message(f"Wrapping model in DataParallel for {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    log_message("[OK] Models initialized")

    # Optimizer
    trainable_params = list(model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    log_message("\n" + "="*70)
    log_message("TRAINING START")
    log_message("="*70)

    best_dice = 0
    Path('checkpoints').mkdir(exist_ok=True)

    for epoch in range(epochs):
        log_message(f"\n{'='*70}")
        log_message(f"Epoch {epoch+1}/{epochs}")
        log_message(f"{'='*70}")

        # Train
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, device, epoch)
        log_message(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")

        # Validate
        val_dice, val_iou = validate(model, val_loader, device)
        log_message(f"Val   - Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        # LR step
        scheduler.step()
        log_message(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_iou': val_iou
            }
            save_path = Path('checkpoints') / f'stage1_best_epoch{epoch+1}_dice{val_dice:.4f}.pth'
            torch.save(checkpoint, save_path)
            log_message(f"[CHECKPOINT] Saved: {save_path.name}")
            log_message(f"[BEST] New best Dice: {best_dice:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            save_path = Path('checkpoints') / f'stage1_epoch{epoch+1}.pth'
            torch.save(checkpoint, save_path)
            log_message(f"[CHECKPOINT] Periodic save: {save_path.name}")

    log_message("\n" + "="*70)
    log_message("TRAINING COMPLETE")
    log_message(f"Best Validation Dice: {best_dice:.4f}")
    log_message("="*70)


if __name__ == '__main__':
    Path('checkpoints').mkdir(exist_ok=True)

    try:
        main()
    except Exception as e:
        log_message(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
