"""
Train Obstruction Detection Head on Phase 1/2 Model

Builds on the trained vessel segmentation model, adds obstruction head.
Uses backbone features from Phase 1/2, trains only the new obstruction decoder.
"""

import os
import sys
sys.path.insert(0, r"C:\Users\odressler\sam3")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from PIL import Image
from pathlib import Path
import json
import yaml
from tqdm import tqdm
from datetime import datetime


class ObstructionDataset(Dataset):
    """Dataset for obstruction detection training."""

    def __init__(self, coco_json, images_dir, image_size=512):
        self.images_dir = Path(images_dir)
        self.image_size = image_size

        with open(coco_json) as f:
            self.coco = json.load(f)

        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.image_ids = list(self.images.keys())
        print(f"Loaded {len(self.image_ids)} images with {len(self.coco['annotations'])} annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        image_path = self.images_dir / img_info['file_name']
        image = Image.open(image_path).convert('RGB')

        orig_size = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Create mask from annotations
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        anns = self.annotations.get(img_id, [])

        import cv2
        for ann in anns:
            for seg in ann['segmentation']:
                polygon = np.array(seg, dtype=np.float32).reshape(-1, 2)
                polygon[:, 0] = polygon[:, 0] * self.image_size / orig_size[0]
                polygon[:, 1] = polygon[:, 1] * self.image_size / orig_size[1]
                polygon = polygon.astype(np.int32)
                cv2.fillPoly(mask, [polygon], 1.0)

        # Normalize image (ImageNet stats)
        image_np = np.array(image) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_np - mean) / std

        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor, img_info.get('stenosis_pct', 50.0)


class ObstructionHead(nn.Module):
    """Lightweight obstruction detection head."""

    def __init__(self, in_channels=256, hidden_dim=128):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv_out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.conv_out(x)


class ObstructionModel(nn.Module):
    """
    Model for obstruction detection using Phase 1/2 backbone features.

    Architecture:
    - Backbone: From Phase 1/2 checkpoint (frozen)
    - Obstruction head: New trainable decoder
    """

    def __init__(self, backbone_channels=256, image_size=512):
        super().__init__()
        self.image_size = image_size

        # Simple feature extractor (will be replaced by Phase 1/2 backbone)
        # Using a lightweight CNN for now - can be swapped for SAM3 backbone
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # /2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /16
            nn.Conv2d(256, backbone_channels, 3, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
        )

        self.obstruction_head = ObstructionHead(backbone_channels, hidden_dim=128)

    def forward(self, images):
        # Extract features
        features = self.encoder(images)  # (B, 256, H/16, W/16)

        # Obstruction prediction
        pred_logits = self.obstruction_head(features)

        # Upsample to input size
        pred_logits = F.interpolate(
            pred_logits,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )

        return pred_logits


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation."""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance."""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def combined_loss(pred, target, bce_weight=0.3, dice_weight=0.5, focal_weight=0.2):
    """Combined BCE + Dice + Focal loss."""
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    return bce_weight * bce + dice_weight * dice + focal_weight * focal


def dice_score(pred, target, threshold=0.5):
    """Calculate Dice score."""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def iou_score(pred, target, threshold=0.5):
    """Calculate IoU score."""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, masks, stenosis_pcts in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            pred_logits = model(images)
            loss = combined_loss(pred_logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            dice = dice_score(pred_logits, masks)
            iou = iou_score(pred_logits, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}',
            'iou': f'{iou.item():.4f}'
        })

    n = len(dataloader)
    return total_loss / n, total_dice / n, total_iou / n


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks, stenosis_pcts in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)

            with autocast('cuda'):
                pred_logits = model(images)
                loss = combined_loss(pred_logits, masks)

            dice = dice_score(pred_logits, masks)
            iou = iou_score(pred_logits, masks)

            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()

    n = len(dataloader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    print("=" * 70)
    print("Obstruction Detection Training (Multi-head approach)")
    print("=" * 70)

    # Config
    config_path = Path(r"E:\AngioMLDL_data\unified_obstruction_coco\train_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Output directories
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['output']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    print("\nLoading datasets...")
    train_dataset = ObstructionDataset(
        config['dataset']['train_json'],
        config['dataset']['train_images'],
        image_size=config['model']['image_size']
    )
    val_dataset = ObstructionDataset(
        config['dataset']['val_json'],
        config['dataset']['val_images'],
        image_size=config['model']['image_size']
    )

    # DataLoaders - 2x RTX 3090 (24GB each), target ~20GB per GPU
    # Model ~50MB, input 512x512 ~3MB, features/grads ~50MB per sample
    # Target: 40GB / ~60MB = ~64-128 batch size
    batch_size = 64  # Maximizes 2x RTX 3090 utilization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    print("\nBuilding model...")
    model = ObstructionModel(
        backbone_channels=256,
        image_size=config['model']['image_size']
    )

    # Multi-GPU support - ALWAYS use both GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"WARNING: Only {num_gpus} GPU detected, expected 2x RTX 3090")
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler with warmup
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training']['warmup_epochs']

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return config['training']['min_lr'] / config['training']['learning_rate'] + \
                   (1 - config['training']['min_lr'] / config['training']['learning_rate']) * \
                   (1 + np.cos(np.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')

    # Training loop
    best_dice = 0
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"Batch size: {batch_size}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )
        val_loss, val_dice, val_iou = validate(model, val_loader, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = checkpoint_dir / "obstruction_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'config': config
            }, save_path)
            print(f"  *** New best model saved (Dice: {val_dice:.4f}) ***")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_path = checkpoint_dir / f"obstruction_epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'config': config
            }, save_path)

    # Save final model
    save_path = checkpoint_dir / "obstruction_final.pt"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
        'config': config
    }, save_path)

    print("\n" + "=" * 70)
    print(f"Training complete!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
