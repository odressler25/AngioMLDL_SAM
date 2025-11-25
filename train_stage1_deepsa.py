"""
Stage 1: SAM 3 Fine-tuning for Full Vessel Segmentation

Training on DeepSA pseudo-labels with categorical view angles.
Uses DDP for multi-GPU training.

Phase 1 of the 4-phase plan:
1. Train on DeepSA labels for full vessel segmentation + categorical view angles
2. Use Medis segments to create bboxes for CASS segment classification
3. Vessel obstruction detection
4. Train on Medis GT values (after fixing alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add sam3 to path
sam3_path = r"C:\Users\odressler\sam3"
if os.path.exists(sam3_path):
    sys.path.insert(0, sam3_path)

from sam3 import build_sam3_image_model


def setup_ddp(rank, world_size, port=12355):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    backend = 'gloo' if os.name == 'nt' else 'nccl'
    dist.init_process_group(
        backend=backend,
        init_method=f'tcp://localhost:{port}',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"Using DDP backend: {backend}")


def cleanup_ddp():
    dist.destroy_process_group()


class CategoricalViewEncoder(nn.Module):
    """
    Categorical view angle encoding.

    Bins angles into discrete categories and uses learned embeddings.
    Primary angle: RAO/LAO (-40 to +40 degrees)
    Secondary angle: CRAN/CAUD (-40 to +40 degrees)
    """
    def __init__(self, embedding_dim=256, num_bins=9):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins  # 9 bins = 10-degree intervals from -40 to +40

        # Bin edges: [-40, -30, -20, -10, 0, 10, 20, 30, 40]
        # Creates bins: <-35, -35 to -25, -25 to -15, ..., >35
        self.bin_edges = torch.linspace(-40, 40, num_bins + 1)

        # Learned embeddings for each bin
        self.primary_embedding = nn.Embedding(num_bins, embedding_dim // 2)
        self.secondary_embedding = nn.Embedding(num_bins, embedding_dim // 2)

        # MLP to combine
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def angle_to_bin(self, angle):
        """Convert angle to bin index"""
        # Clamp to valid range
        angle = torch.clamp(angle, -40, 40)
        # Find bin index
        bin_idx = torch.bucketize(angle, self.bin_edges[1:-1].to(angle.device))
        return bin_idx

    def forward(self, primary_angle, secondary_angle):
        """
        Args:
            primary_angle: (B,) tensor of primary angles in degrees
            secondary_angle: (B,) tensor of secondary angles in degrees
        Returns:
            (B, embedding_dim) view embedding
        """
        # Convert to bin indices
        primary_bin = self.angle_to_bin(primary_angle)
        secondary_bin = self.angle_to_bin(secondary_angle)

        # Get embeddings
        primary_emb = self.primary_embedding(primary_bin)  # (B, embedding_dim // 2)
        secondary_emb = self.secondary_embedding(secondary_bin)  # (B, embedding_dim // 2)

        # Concatenate and process
        combined = torch.cat([primary_emb, secondary_emb], dim=-1)  # (B, embedding_dim)
        output = self.mlp(combined)

        return output


class ViewConditionedFeatureFusion(nn.Module):
    """FiLM-based feature fusion for view conditioning"""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.gamma_net = nn.Linear(feature_dim, feature_dim)
        self.beta_net = nn.Linear(feature_dim, feature_dim)

    def forward(self, features, view_embedding):
        """
        Args:
            features: (B, C, H, W) spatial features
            view_embedding: (B, C) view embedding
        Returns:
            (B, C, H, W) modulated features
        """
        gamma = self.gamma_net(view_embedding).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_net(view_embedding).unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


class SAM3DeepSAModel(nn.Module):
    """
    SAM 3 for full vessel segmentation with categorical view angles.
    Trained on DeepSA pseudo-labels.
    """
    def __init__(self, image_size=1008, freeze_backbone=False):
        super().__init__()
        self.image_size = image_size
        self.freeze_backbone = freeze_backbone

        # Load SAM 3
        self.sam3 = build_sam3_image_model()

        if freeze_backbone:
            self.sam3.eval()
            for param in self.sam3.parameters():
                param.requires_grad = False
            print("SAM 3 backbone: FROZEN")
        else:
            self.sam3.train()
            for param in self.sam3.parameters():
                param.requires_grad = True
            print("SAM 3 backbone: TRAINABLE (full fine-tuning)")

        # Categorical view encoder
        self.view_encoder = CategoricalViewEncoder(embedding_dim=256, num_bins=9)

        # Feature fusion
        self.feature_fusion = ViewConditionedFeatureFusion(feature_dim=256)

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # Feature projection (lazy init)
        self.feature_proj = None

        # Parameter count
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        print(f"\nParameter Summary:")
        print(f"  Total: {total/1e6:.1f}M")
        print(f"  Trainable: {trainable/1e6:.1f}M ({trainable/total*100:.2f}%)")
        print(f"  Frozen: {frozen/1e6:.1f}M")

    def preprocess_for_backbone(self, image, target_size=1008):
        """Resize and pad to target size"""
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
        images_padded = self.preprocess_for_backbone(images, target_size=1008)
        images_normalized = (images_padded - 0.5) / 0.5

        if self.freeze_backbone:
            with torch.no_grad():
                self.sam3.eval()
                backbone_out = self.sam3.backbone.forward_image(images_normalized)
        else:
            backbone_out = self.sam3.backbone.forward_image(images_normalized)

        if isinstance(backbone_out, dict):
            features = backbone_out.get('vision_features', backbone_out.get('image_embeddings'))
        else:
            features = backbone_out

        if len(features.shape) == 3:
            B_feat, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.permute(0, 2, 1).reshape(B_feat, C, H, W)

        return features

    def forward(self, images, primary_angles, secondary_angles):
        """Forward pass"""
        # View encoding (categorical)
        view_embedding = self.view_encoder(primary_angles, secondary_angles)

        # SAM 3 features
        features = self.extract_sam3_features(images)

        # Project to 256 channels if needed
        if features.shape[1] != 256:
            if self.feature_proj is None:
                self.feature_proj = nn.Conv2d(features.shape[1], 256, kernel_size=1).to(features.device)
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


class DeepSADataset(Dataset):
    """
    Dataset for Phase 1: Full vessel segmentation using DeepSA pseudo-labels.

    Uses CSV as single source of truth:
    - cine_path: path to cine video
    - frame_index: which frame to use
    - deepsa_pseudo_label_path: DeepSA full vessel mask
    - contours_path: JSON with view angles
    """
    def __init__(self, csv_path, split='train', image_size=1008):
        self.split = split
        self.image_size = image_size

        # Load CSV
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)

        # Store paths from CSV
        self.cine_paths = df['cine_path'].tolist()
        self.deepsa_paths = df['deepsa_pseudo_label_path'].tolist()
        self.contours_paths = df['contours_path'].tolist()
        self.frame_indices = df['frame_index'].tolist()

        # Verify DeepSA paths exist
        missing = [p for p in self.deepsa_paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} DeepSA labels. First: {missing[0]}")

        print(f"Loaded {len(self)} {split} samples with DeepSA labels")

    def __len__(self):
        return len(self.cine_paths)

    def __getitem__(self, idx):
        # Load cine and extract correct frame
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

        # Convert to tensor (C, H, W)
        image = torch.from_numpy(frame).permute(2, 0, 1)

        # Load DeepSA mask
        mask = np.load(self.deepsa_paths[idx]).astype(np.float32)
        # DeepSA labels are binary (0/1), no need to normalize by 255
        mask = torch.from_numpy(mask)

        # Resize image and mask to target size
        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # DeepSA masks are 512x512, need to resize to match image
        if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='nearest'  # Use nearest for binary masks to preserve sharp edges
            ).squeeze(0).squeeze(0)

        # Load view angles from contours JSON
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

    pbar = tqdm(dataloader, desc='Training') if rank == 0 else dataloader

    for images, masks, primary_angles, secondary_angles in pbar:
        images = images.to(device)
        masks = masks.to(device)
        primary_angles = primary_angles.to(device)
        secondary_angles = secondary_angles.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            pred_logits = model(images, primary_angles, secondary_angles)
            loss = combined_loss(pred_logits, masks, bce_weight=0.5)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

    pbar = tqdm(dataloader, desc='Validation') if rank == 0 else dataloader

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
    setup_ddp(rank, world_size, port=args['port'])
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print(f"\n{'='*70}")
        print("PHASE 1: Full Vessel Segmentation with DeepSA Labels")
        print(f"Training with DDP: {world_size} GPUs")
        print(f"{'='*70}\n")

    # Build model
    model = SAM3DeepSAModel(
        image_size=args['image_size'],
        freeze_backbone=args['freeze_backbone']
    ).to(device)

    # Wrap with DDP
    if os.name == 'nt':
        model = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
    else:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Optimizer
    if args['freeze_backbone']:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args['learning_rate'],
            weight_decay=0.01
        )
    else:
        # Differential learning rates (MedSAM2 approach)
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'sam3.backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args['learning_rate'] * 0.1},
            {'params': head_params, 'lr': args['learning_rate']}
        ], weight_decay=0.01)

        if rank == 0:
            print(f"Backbone params: {sum(p.numel() for p in backbone_params)/1e6:.1f}M (LR: {args['learning_rate'] * 0.1})")
            print(f"Head params: {sum(p.numel() for p in head_params)/1e6:.1f}M (LR: {args['learning_rate']})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    scaler = torch.amp.GradScaler('cuda')

    # Datasets
    train_dataset = DeepSADataset(args['csv_path'], split='train', image_size=args['image_size'])
    val_dataset = DeepSADataset(args['csv_path'], split='val', image_size=args['image_size'])

    # Samplers for DDP
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

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
        print(f"\nStarting training...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Batch size per GPU: {args['batch_size']}, Effective batch: {args['batch_size'] * world_size}")
        print("=" * 70)

    for epoch in range(args['epochs']):
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
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                }, f'checkpoints/phase1_deepsa_best.pth')
                print(f"  -> Saved best model (Dice: {val_dice:.4f})")

            # Also save latest
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, f'checkpoints/phase1_deepsa_latest.pth')

            mem_used = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"  Max GPU {rank} memory: {mem_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        print("\n" + "=" * 70)
        print("Phase 1 training complete!")
        print(f"Best validation Dice: {best_dice:.4f}")
        print(f"Checkpoint: checkpoints/phase1_deepsa_best.pth")

    cleanup_ddp()


def main():
    args = {
        'csv_path': r'E:\AngioMLDL_data\corrected_dataset_training.csv',
        'image_size': 1008,
        'batch_size': 4,  # Per GPU (effective batch = 4 * 2 = 8)
        'learning_rate': 1e-4,
        'epochs': 50,
        'freeze_backbone': False,  # Full fine-tuning like MedSAM2
        'port': 12355,
    }

    world_size = torch.cuda.device_count()

    print(f"\n{'='*70}")
    print("SAM 3 Phase 1: Full Vessel Segmentation")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Training data: DeepSA pseudo-labels")
    print(f"  View encoding: Categorical (9 bins per angle)")
    print(f"  Backbone: {'Frozen' if args['freeze_backbone'] else 'Full fine-tuning'}")
    print(f"  Image size: {args['image_size']}")
    print(f"  Batch size: {args['batch_size']} per GPU")
    print(f"  Learning rate: {args['learning_rate']}")
    print(f"  Epochs: {args['epochs']}")
    print(f"\nGPUs available: {world_size}")
    for i in range(world_size):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    mp.spawn(
        train_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
