"""
Stage 1: SAM 3 Fine-tuning with DistributedDataParallel (DDP)

The solution to RoPE + multi-GPU: Use DDP instead of DataParallel
- DataParallel: Single process, broadcasts inputs, doesn't work with RoPE
- DDP: Multiple processes, each with own model copy, works with RoPE

Based on:
- SAM 3 training script (sam3/train/train.py)
- MedSAM2 paper (trained on 12 H100s with DDP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import os
import sys
from tqdm import tqdm

# Add sam3 to path
sam3_path = r"C:\Users\odressler\sam3"
if os.path.exists(sam3_path):
    sys.path.insert(0, sam3_path)

from sam3 import build_sam3_image_model
from view_angle_encoder_v2 import ViewAngleEncoder, ViewConditionedFeatureFusion


def setup_ddp(rank, world_size, port=12355):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # Initialize process group
    # Use 'gloo' backend on Windows (NCCL not supported on Windows)
    backend = 'gloo' if os.name == 'nt' else 'nccl'

    dist.init_process_group(
        backend=backend,
        init_method=f'tcp://localhost:{port}',
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"Using DDP backend: {backend}")


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


class SAM3FineTuneModel(nn.Module):
    """
    SAM 3 with fine-tuning support

    Options:
    1. Frozen backbone + trainable heads (fast, 0.6M params)
    2. Full fine-tuning (slow, 840M params, like MedSAM2)
    3. LoRA fine-tuning (TODO: 3-5M params, best of both)
    """
    def __init__(self, image_size=1008, freeze_backbone=True):
        super().__init__()

        self.image_size = image_size
        self.freeze_backbone = freeze_backbone

        # Load SAM 3
        self.sam3 = build_sam3_image_model()

        if freeze_backbone:
            # Freeze backbone (like your current approach)
            self.sam3.eval()
            for param in self.sam3.parameters():
                param.requires_grad = False
            print(f"SAM 3 backbone: FROZEN")
        else:
            # Full fine-tuning (like MedSAM2)
            self.sam3.train()
            for param in self.sam3.parameters():
                param.requires_grad = True
            print(f"SAM 3 backbone: TRAINABLE (full fine-tuning)")

        # View angle encoder (TRAINABLE)
        self.view_encoder = ViewAngleEncoder(embedding_dim=256)

        # Feature fusion (TRAINABLE)
        self.feature_fusion = ViewConditionedFeatureFusion(
            feature_dim=256,
            fusion_mode='film'
        )

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

        # Feature projection
        self.feature_proj = None

        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen

        print(f"\nParameter Summary:")
        print(f"  Total: {total/1e6:.1f}M")
        print(f"  Trainable: {trainable/1e6:.1f}M ({trainable/total*100:.2f}%)")
        print(f"  Frozen: {frozen/1e6:.1f}M")

    def preprocess_for_backbone(self, image, target_size=1008):
        """Resize and pad to 1008x1008"""
        b, c, h, w = image.shape

        scale = target_size / max(h, w)
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

        curr_h, curr_w = image.shape[-2:]
        pad_h = target_size - curr_h
        pad_w = target_size - curr_w
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return image

    def extract_sam3_features(self, images):
        """Extract features from SAM 3 backbone"""
        # Preprocess
        images_padded = self.preprocess_for_backbone(images, target_size=1008)
        images_normalized = (images_padded - 0.5) / 0.5

        # Extract features
        if self.freeze_backbone:
            with torch.no_grad():
                self.sam3.eval()
                backbone_out = self.sam3.backbone.forward_image(images_normalized)
        else:
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
        """Forward pass"""
        # View encoding
        view_embedding = self.view_encoder(primary_angles, secondary_angles)

        # SAM 3 features
        features = self.extract_sam3_features(images)

        # Project to 256 channels if needed
        if features.shape[1] != 256:
            if self.feature_proj is None:
                self.feature_proj = nn.Conv2d(features.shape[1], 256, kernel_size=1).to(features.device)
                # Register as a module
                self.add_module('feature_proj', self.feature_proj)
            features = self.feature_proj(features)

        # Fuse with view embedding
        fused_features = self.feature_fusion(features, view_embedding)

        # Decode to segmentation mask
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
    """Combined BCE + Dice loss"""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='mean')
    pred_probs = torch.sigmoid(pred_logits)
    dice = dice_loss(pred_probs, target)
    return bce_weight * bce + (1 - bce_weight) * dice


def dice_score(pred, target, smooth=1.0):
    """Dice coefficient"""
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

        # Resize
        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size),
                                 mode='bilinear', align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(self.image_size, self.image_size),
                                mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Load view angles (FIXED keys)
        with open(self.contours_paths[idx], 'r') as f:
            contours = json.load(f)
        view_angles = contours.get('view_angles', {})
        primary_angle = torch.tensor(float(view_angles.get('primary_angle', 0.0)), dtype=torch.float32)
        secondary_angle = torch.tensor(float(view_angles.get('secondary_angle', 0.0)), dtype=torch.float32)

        return image, mask, primary_angle, secondary_angle


def train_epoch(model, dataloader, optimizer, device, scaler, rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0

    if rank == 0:
        pbar = tqdm(dataloader, desc='Training')
    else:
        pbar = dataloader

    for images, masks, primary_angles, secondary_angles in pbar:
        images = images.to(device)
        masks = masks.to(device)
        primary_angles = primary_angles.to(device)
        secondary_angles = secondary_angles.to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        with torch.amp.autocast('cuda'):
            pred_logits = model(images, primary_angles, secondary_angles)
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

        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})

    return total_loss / len(dataloader), total_dice / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device, rank):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_dice = 0

    if rank == 0:
        pbar = tqdm(dataloader, desc='Validation')
    else:
        pbar = dataloader

    for images, masks, primary_angles, secondary_angles in pbar:
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


def train_worker(rank, world_size, args):
    """Worker function for DDP training"""
    # Setup DDP
    setup_ddp(rank, world_size, port=args['port'])
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Training with DDP: {world_size} GPUs")
        print(f"{'='*70}\n")

    # Build model
    model = SAM3FineTuneModel(
        image_size=args['image_size'],
        freeze_backbone=args['freeze_backbone']
    ).to(device)

    # Wrap with DDP
    # Note: gloo backend on Windows doesn't support device_ids parameter
    # Also disable broadcast_buffers to avoid "Invalid scalar type" errors with gloo
    if os.name == 'nt':
        model = DDP(
            model,
            find_unused_parameters=True,
            broadcast_buffers=False  # Fixes gloo "Invalid scalar type" error
        )
    else:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Optimizer (following MedSAM2's approach)
    if args['freeze_backbone']:
        # Only train heads
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args['learning_rate'],
            weight_decay=0.01
        )
    else:
        # Differential learning rates like MedSAM2
        # Lower LR for backbone, higher for heads
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'sam3.backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args['learning_rate'] * 0.1},  # 10x lower for backbone
            {'params': head_params, 'lr': args['learning_rate']}
        ], weight_decay=0.01)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])

    # Scaler
    scaler = torch.amp.GradScaler('cuda')

    # Datasets
    train_dataset = AngiographyDataset(args['csv_path'], split='train', image_size=args['image_size'])
    val_dataset = AngiographyDataset(args['csv_path'], split='val', image_size=args['image_size'])

    # Samplers for DDP
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Training loop
    best_dice = 0.0
    if rank == 0:
        print("Starting training...")
        print("=" * 70)

    for epoch in range(args['epochs']):
        # Set epoch for sampler (ensures different shuffling each epoch)
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args['epochs']}")

        train_loss, train_dice = train_epoch(model, train_loader, optimizer, device, scaler, rank)
        val_loss, val_dice = validate(model, val_loader, device, rank)
        scheduler.step()

        if rank == 0:
            print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_dice > best_dice:
                best_dice = val_dice
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # Save unwrapped model
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                }, f'checkpoints/stage1_ddp_{args["name"]}_best.pth')
                print(f"  -> Saved best model (Dice: {val_dice:.4f})")

            mem_used = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"  Max GPU {rank} memory: {mem_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        print("\n" + "=" * 70)
        print("Training complete!")
        print(f"Best validation Dice: {best_dice:.4f}")

    cleanup_ddp()


def main():
    # Hyperparameters
    args = {
        'csv_path': r'E:\AngioMLDL_data\corrected_dataset_training.csv',
        'image_size': 1008,
        'batch_size': 4,  # Per GPU (effective batch = 4 * 2 = 8)
                          # Reduced for full fine-tuning to fit in VRAM
        'learning_rate': 1e-4,
        'epochs': 30,
        'freeze_backbone': False,  # FALSE = Full fine-tuning like MedSAM2 (840M params)
                                    # TRUE = Frozen backbone (0.6M params)
        'port': 12355,
        'name': 'full_ft'  # 'full_ft' for full fine-tuning, 'frozen' for frozen
    }

    world_size = torch.cuda.device_count()

    if world_size < 2:
        print("Warning: Only 1 GPU detected. DDP works but is unnecessary.")
        print("Consider using train_stage1_frozen_no_dataparallel.py instead.")

    print(f"\nDevice: cuda")
    print(f"GPUs available: {world_size}")
    for i in range(world_size):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    # Launch DDP training
    mp.spawn(
        train_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
