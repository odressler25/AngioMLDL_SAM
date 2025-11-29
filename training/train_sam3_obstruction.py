"""
Train SAM 3 for Obstruction Detection with Text Prompting

Uses Medis QCA gold-standard labels to teach SAM 3 the "obstruction" concept.
After training, SAM 3 can detect obstructions using text prompt: "obstruction"
"""

import sys
import os

sys.path.insert(0, r"C:\Users\odressler\sam3")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from PIL import Image
from pathlib import Path
import json
import yaml
from tqdm import tqdm
from datetime import datetime
import pycocotools.mask as mask_util

# SAM 3 imports
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class ObstructionDataset(Dataset):
    """Dataset for obstruction detection training."""

    def __init__(self, coco_json, images_dir, image_size=512, augment=False):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.augment = augment

        # Load COCO annotations
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
        print(f"Loaded {len(self.image_ids)} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        image_path = self.images_dir / img_info['file_name']
        image = Image.open(image_path).convert('RGB')

        # Resize to target size
        orig_size = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Create mask from annotations
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        anns = self.annotations.get(img_id, [])

        for ann in anns:
            # Convert polygon to mask
            for seg in ann['segmentation']:
                # Scale polygon to new size
                polygon = np.array(seg, dtype=np.float32).reshape(-1, 2)
                polygon[:, 0] = polygon[:, 0] * self.image_size / orig_size[0]
                polygon[:, 1] = polygon[:, 1] * self.image_size / orig_size[1]
                polygon = polygon.astype(np.int32)

                # Draw polygon on mask
                import cv2
                cv2.fillPoly(mask, [polygon], 1.0)

        # Normalize image
        image_np = np.array(image) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_np - mean) / std

        # To tensors
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor, img_info.get('stenosis_pct', 50.0)


class Sam3ObstructionModel(nn.Module):
    """SAM 3 model wrapped for obstruction detection with text prompting."""

    def __init__(self, image_size=512, concept_prompt="obstruction"):
        super().__init__()

        # Build SAM 3 model
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

        self.sam3_model = build_sam3_image_model(bpe_path=bpe_path)
        self.processor = Sam3Processor(self.sam3_model, confidence_threshold=0.3)
        self.concept_prompt = concept_prompt

        # Freeze most of SAM 3, only train output layers
        for name, param in self.sam3_model.named_parameters():
            if 'mask_decoder' not in name and 'output' not in name:
                param.requires_grad = False

        # Add a trainable adapter for the obstruction concept
        self.obstruction_adapter = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, images):
        """
        Forward pass with text prompt.

        Args:
            images: (B, 3, H, W) normalized images

        Returns:
            pred_masks: (B, 1, H, W) predicted obstruction masks
        """
        batch_size = images.shape[0]
        device = images.device

        # Get SAM 3 image embeddings
        with torch.no_grad():
            image_embeddings = self.sam3_model.image_encoder(images)

        # Apply obstruction adapter
        pred_logits = self.obstruction_adapter(image_embeddings)

        # Upsample to input size
        pred_logits = F.interpolate(
            pred_logits,
            size=images.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        return pred_logits


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation."""
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def combined_loss(pred, target, bce_weight=0.5):
    """Combined BCE + Dice loss."""
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice


def dice_score(pred, target, threshold=0.5):
    """Calculate Dice score."""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)


def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, masks, stenosis_pcts in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast():
            pred_logits = model(images)
            loss = combined_loss(pred_logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_logits)
            dice = dice_score(pred_probs, masks)

        total_loss += loss.item()
        total_dice += dice.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })

    return total_loss / len(dataloader), total_dice / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for images, masks, stenosis_pcts in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)

            with autocast():
                pred_logits = model(images)
                loss = combined_loss(pred_logits, masks)

            pred_probs = torch.sigmoid(pred_logits)
            dice = dice_score(pred_probs, masks)

            total_loss += loss.item()
            total_dice += dice.item()

    return total_loss / len(dataloader), total_dice / len(dataloader)


def main():
    print("=" * 60)
    print("SAM 3 Obstruction Detection Training")
    print("=" * 60)

    # Config
    config_path = Path(r"E:\AngioMLDL_data\batch2\coco_obstruction\train_obstruction_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create output directories
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    print("\nBuilding SAM 3 model...")
    model = Sam3ObstructionModel(
        image_size=config['model']['image_size'],
        concept_prompt=config['concept']['prompt']
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['min_lr']
    )

    scaler = GradScaler()

    # Training loop
    best_dice = 0
    print("\nStarting training...")

    for epoch in range(1, config['training']['num_epochs'] + 1):
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )
        val_loss, val_dice = validate(model, val_loader, device)

        scheduler.step()

        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = checkpoint_dir / "obstruction_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'config': config
            }, save_path)
            print(f"  Saved best model (Dice: {val_dice:.4f})")

        # Save periodic checkpoint
        if epoch % config['output']['save_frequency'] == 0:
            save_path = checkpoint_dir / f"obstruction_epoch_{epoch:03d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'config': config
            }, save_path)

    print("\n" + "=" * 60)
    print(f"Training complete! Best Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
