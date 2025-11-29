"""
Phase 2: CASS Segment Classification

Goal: Teach the model to identify WHICH coronary artery segment is in the image.

This is where view angles become critical:
- Different viewing angles show different parts of the coronary tree
- RAO views emphasize LAD, LAO views emphasize LCX/RCA
- The model must learn the relationship between view angle and visible segments

Architecture:
- Load Phase 1 checkpoint (vessel segmentation backbone)
- Add CASS segment classification head
- Use Medis contours to create bounding boxes for the analyzed segment
- Predict: segment ID (1-28) and segment location (bbox)

CASS Segments in our data:
- 1: Proximal RCA
- 2: Mid RCA
- 3: Distal RCA
- 4: PDA
- 13: LAD (prox/mid)
- 14: Diagonal 1
- 15: Diagonal 2
- 18: Proximal LCX
- 19: Distal LCX
- 20: OM1
- 21: OM2
- 28: Ramus
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

sys.path.insert(0, r'C:\Users\odressler\sam3')

from sam3 import build_sam3_image_model
from train_stage1_deepsa import CategoricalViewEncoder, ViewConditionedFeatureFusion


def setup_ddp(rank, world_size, port=12356):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    backend = 'gloo' if os.name == 'nt' else 'nccl'
    dist.init_process_group(backend=backend, init_method=f'tcp://localhost:{port}',
                           world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"Using DDP backend: {backend}")


def cleanup_ddp():
    dist.destroy_process_group()


# CASS segment ID to index mapping (for classification)
# Standard CASS convention:
# RCA: 1=prox, 2=mid, 3=distal, 4=PDA
# LAD: 12=prox, 13=mid, 14=distal, 15=D1, 16=D2
# LCX: 18=prox, 19=distal, 20=OM1, 21=OM2
# 28=Ramus
CASS_SEGMENTS = [1, 2, 3, 4, 12, 13, 14, 15, 16, 18, 19, 20, 21, 28]
CASS_TO_IDX = {seg: idx for idx, seg in enumerate(CASS_SEGMENTS)}
IDX_TO_CASS = {idx: seg for seg, idx in CASS_TO_IDX.items()}
NUM_CLASSES = len(CASS_SEGMENTS)

# Segment names for interpretability
CASS_NAMES = {
    1: 'Proximal RCA',
    2: 'Mid RCA',
    3: 'Distal RCA',
    4: 'PDA',
    12: 'Proximal LAD',
    13: 'Mid LAD',
    14: 'Distal LAD',
    15: 'Diagonal 1',
    16: 'Diagonal 2',
    18: 'Proximal LCX',
    19: 'Distal LCX',
    20: 'OM1',
    21: 'OM2',
    28: 'Ramus'
}


class SAM3CASSModel(nn.Module):
    """
    SAM 3 for CASS segment classification.

    Takes:
    - Image
    - View angles (critical for determining which segments are visible)

    Outputs:
    - Vessel segmentation mask (from Phase 1)
    - CASS segment classification (which segment is being analyzed)
    - Bounding box for the segment (where in the image)
    """
    def __init__(self, image_size=1008, num_classes=NUM_CLASSES, phase1_checkpoint=None):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        # Load SAM 3 backbone
        self.sam3 = build_sam3_image_model()

        # View encoder (same as Phase 1)
        self.view_encoder = CategoricalViewEncoder(embedding_dim=256, num_bins=9)

        # Feature fusion
        self.feature_fusion = ViewConditionedFeatureFusion(feature_dim=256)

        # Segmentation head (from Phase 1)
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

        # NEW: CASS classification head
        # Uses global pooled features + view embedding
        self.cass_classifier = nn.Sequential(
            nn.Linear(256 + 256, 512),  # features + view embedding
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # NEW: Bounding box regression head
        # Predicts normalized [x_center, y_center, width, height]
        self.bbox_head = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()  # Normalized to [0, 1]
        )

        # Load Phase 1 checkpoint if provided
        if phase1_checkpoint and Path(phase1_checkpoint).exists():
            self._load_phase1(phase1_checkpoint)

        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\nParameter Summary:")
        print(f"  Total: {total/1e6:.1f}M")
        print(f"  Trainable: {trainable/1e6:.1f}M")

    def _load_phase1(self, checkpoint_path):
        """Load Phase 1 weights for segmentation"""
        print(f"Loading Phase 1 checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Load matching weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items()
                          if k in model_dict and model_dict[k].shape == v.shape}

        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} layers from Phase 1")
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        print(f"  Phase 1 Val Dice: {checkpoint.get('val_dice', 'N/A')}")

    def preprocess_for_backbone(self, image, target_size=1008):
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
        images_padded = self.preprocess_for_backbone(images, target_size=1008)
        images_normalized = (images_padded - 0.5) / 0.5
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
        """
        Returns:
            seg_masks: (B, H, W) vessel segmentation logits
            cass_logits: (B, num_classes) CASS segment classification logits
            bbox_pred: (B, 4) normalized bounding box [x_center, y_center, w, h]
        """
        # View encoding
        view_embedding = self.view_encoder(primary_angles, secondary_angles)

        # SAM 3 features
        features = self.extract_sam3_features(images)

        # Project to 256 channels if needed
        if features.shape[1] != 256:
            if self.feature_proj is None:
                self.feature_proj = nn.Conv2d(features.shape[1], 256, kernel_size=1).to(features.device)
                self.add_module('feature_proj', self.feature_proj)
            features = self.feature_proj(features)

        # Fuse with view embedding for segmentation
        fused_features = self.feature_fusion(features, view_embedding)

        # Segmentation output
        seg_logits = self.seg_head(fused_features)
        seg_masks = F.interpolate(seg_logits, size=(self.image_size, self.image_size), mode='bilinear')
        seg_masks = seg_masks.squeeze(1)

        # Global pooled features for classification
        global_features = F.adaptive_avg_pool2d(fused_features, 1).squeeze(-1).squeeze(-1)  # (B, 256)

        # Concatenate with view embedding (view is critical for CASS classification!)
        combined = torch.cat([global_features, view_embedding], dim=-1)  # (B, 512)

        # CASS classification
        cass_logits = self.cass_classifier(combined)

        # Bounding box regression
        bbox_pred = self.bbox_head(combined)

        return seg_masks, cass_logits, bbox_pred


class CASSDataset(Dataset):
    """
    Dataset for Phase 2: CASS segment classification.

    Provides:
    - Image
    - DeepSA vessel mask (for segmentation loss)
    - View angles
    - CASS segment ID (classification target)
    - Bounding box from Medis contours (bbox target)
    """
    def __init__(self, csv_path, split='train', image_size=1008):
        self.split = split
        self.image_size = image_size

        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)

        # Filter to only samples with valid CASS segments
        df = df[df['cass_segment'].isin(CASS_SEGMENTS)].reset_index(drop=True)

        self.cine_paths = df['cine_path'].tolist()
        self.deepsa_paths = df['deepsa_pseudo_label_path'].tolist()
        self.contours_paths = df['contours_path'].tolist()
        self.frame_indices = df['frame_index'].tolist()
        self.cass_segments = df['cass_segment'].tolist()

        print(f"Loaded {len(self)} {split} samples for CASS classification")
        print(f"  Classes: {sorted(set(self.cass_segments))}")

    def __len__(self):
        return len(self.cine_paths)

    def _compute_bbox_from_contours(self, contours_data, original_size):
        """Compute normalized bounding box from Medis contours

        Contours and masks are in the same coordinate space as the image.
        """
        # Get all contour points
        all_points = []
        for key in ['centerline', 'left_edge', 'right_edge']:
            if key in contours_data and contours_data[key]:
                all_points.extend(contours_data[key])

        if not all_points:
            # Return center bbox if no contours
            return torch.tensor([0.5, 0.5, 0.3, 0.3], dtype=torch.float32)

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Add small padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(original_size, x_max + padding)
        y_max = min(original_size, y_max + padding)

        # Convert to normalized [x_center, y_center, width, height]
        x_center = (x_min + x_max) / 2 / original_size
        y_center = (y_min + y_max) / 2 / original_size
        width = (x_max - x_min) / original_size
        height = (y_max - y_min) / original_size

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

    def __getitem__(self, idx):
        # Load image
        cine = np.load(self.cine_paths[idx])
        frame_idx = self.frame_indices[idx]
        if frame_idx >= len(cine):
            frame_idx = len(cine) - 1
        frame = cine[frame_idx]

        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)

        frame = frame.astype(np.float32)
        original_size = frame.shape[0]
        if frame.max() > 1:
            frame = frame / 255.0

        image = torch.from_numpy(frame).permute(2, 0, 1)

        # Load DeepSA mask
        mask = np.load(self.deepsa_paths[idx]).astype(np.float32)
        mask = torch.from_numpy(mask)

        # Resize
        if image.shape[1] != self.image_size:
            image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size),
                                 mode='bilinear', align_corners=False).squeeze(0)
        if mask.shape[0] != self.image_size:
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(self.image_size, self.image_size),
                                mode='nearest').squeeze(0).squeeze(0)

        # Load contours for view angles and bbox
        with open(self.contours_paths[idx], 'r') as f:
            contours = json.load(f)

        view_angles = contours.get('view_angles', {})
        primary_angle = torch.tensor(float(view_angles.get('primary_angle', 0.0)), dtype=torch.float32)
        secondary_angle = torch.tensor(float(view_angles.get('secondary_angle', 0.0)), dtype=torch.float32)

        # CASS segment label
        cass_segment = self.cass_segments[idx]
        cass_idx = torch.tensor(CASS_TO_IDX[cass_segment], dtype=torch.long)

        # Bounding box from contours
        bbox = self._compute_bbox_from_contours(contours, original_size)

        return image, mask, primary_angle, secondary_angle, cass_idx, bbox


def combined_loss_phase2(seg_logits, seg_target, cass_logits, cass_target,
                         bbox_pred, bbox_target, seg_weight=0.3, cls_weight=0.5, bbox_weight=0.2):
    """
    Combined loss for Phase 2:
    - Segmentation loss (BCE + Dice) - maintain vessel segmentation ability
    - Classification loss (CrossEntropy) - learn CASS segments
    - Bbox loss (SmoothL1) - learn segment location
    """
    # Segmentation loss
    bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target, reduction='mean')
    pred_probs = torch.sigmoid(seg_logits)
    pred_flat = pred_probs.contiguous().view(-1)
    target_flat = seg_target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = 1 - (2. * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)
    seg_loss = 0.5 * bce + 0.5 * dice

    # Classification loss
    cls_loss = F.cross_entropy(cass_logits, cass_target)

    # Bbox loss
    bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_target)

    total_loss = seg_weight * seg_loss + cls_weight * cls_loss + bbox_weight * bbox_loss

    return total_loss, seg_loss.item(), cls_loss.item(), bbox_loss.item()


def train_epoch(model, dataloader, optimizer, device, scaler, rank):
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training') if rank == 0 else dataloader

    for images, masks, primary, secondary, cass_idx, bbox in pbar:
        images = images.to(device)
        masks = masks.to(device)
        primary = primary.to(device)
        secondary = secondary.to(device)
        cass_idx = cass_idx.to(device)
        bbox = bbox.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            seg_logits, cass_logits, bbox_pred = model(images, primary, secondary)
            loss, seg_l, cls_l, bbox_l = combined_loss_phase2(
                seg_logits, masks, cass_logits, cass_idx, bbox_pred, bbox
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accuracy
        pred_class = cass_logits.argmax(dim=1)
        correct += (pred_class == cass_idx).sum().item()
        total += cass_idx.size(0)

        total_loss += loss.item()
        total_seg_loss += seg_l
        total_cls_loss += cls_l
        total_bbox_loss += bbox_l

        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'cls': f'{cls_l:.3f}',
                'acc': f'{100*correct/total:.1f}%'
            })

    n = len(dataloader)
    return total_loss/n, total_seg_loss/n, total_cls_loss/n, total_bbox_loss/n, correct/total


@torch.no_grad()
def validate(model, dataloader, device, rank):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Validation') if rank == 0 else dataloader

    for images, masks, primary, secondary, cass_idx, bbox in pbar:
        images = images.to(device)
        masks = masks.to(device)
        primary = primary.to(device)
        secondary = secondary.to(device)
        cass_idx = cass_idx.to(device)
        bbox = bbox.to(device)

        with torch.amp.autocast('cuda'):
            seg_logits, cass_logits, bbox_pred = model(images, primary, secondary)
            loss, _, _, _ = combined_loss_phase2(
                seg_logits, masks, cass_logits, cass_idx, bbox_pred, bbox
            )

        pred_class = cass_logits.argmax(dim=1)
        correct += (pred_class == cass_idx).sum().item()
        total += cass_idx.size(0)
        total_loss += loss.item()

    return total_loss / len(dataloader), correct / total


def train_worker(rank, world_size, args):
    setup_ddp(rank, world_size, port=args['port'])
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print(f"\n{'='*70}")
        print("PHASE 2: CASS Segment Classification")
        print(f"Training with DDP: {world_size} GPUs")
        print(f"{'='*70}\n")
        print(f"CASS Segments: {CASS_SEGMENTS}")
        print(f"Number of classes: {NUM_CLASSES}")

    # Build model with Phase 1 checkpoint
    model = SAM3CASSModel(
        image_size=args['image_size'],
        num_classes=NUM_CLASSES,
        phase1_checkpoint=args['phase1_checkpoint']
    ).to(device)

    # Wrap with DDP
    if os.name == 'nt':
        model = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
    else:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Optimizer - train all parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    scaler = torch.amp.GradScaler('cuda')

    # Datasets
    train_dataset = CASSDataset(args['csv_path'], split='train', image_size=args['image_size'])
    val_dataset = CASSDataset(args['csv_path'], split='val', image_size=args['image_size'])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], sampler=train_sampler,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], sampler=val_sampler,
                           num_workers=4, pin_memory=True)

    best_acc = 0.0
    if rank == 0:
        print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")
        print("="*70)

    for epoch in range(args['epochs']):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args['epochs']}")

        train_loss, seg_l, cls_l, bbox_l, train_acc = train_epoch(
            model, train_loader, optimizer, device, scaler, rank
        )
        val_loss, val_acc = validate(model, val_loader, device, rank)
        scheduler.step()

        if rank == 0:
            print(f"  Train - Loss: {train_loss:.4f}, Seg: {seg_l:.4f}, Cls: {cls_l:.4f}, Bbox: {bbox_l:.4f}")
            print(f"  Train Acc: {100*train_acc:.1f}%, Val Acc: {100*val_acc:.1f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'val_acc': val_acc,
                }, 'checkpoints/phase2_cass_best.pth')
                print(f"  -> Saved best model (Acc: {100*val_acc:.1f}%)")

    if rank == 0:
        print(f"\nPhase 2 complete! Best Val Acc: {100*best_acc:.1f}%")

    cleanup_ddp()


def main():
    args = {
        'csv_path': r'E:\AngioMLDL_data\corrected_dataset_training.csv',
        'phase1_checkpoint': r'checkpoints/phase1_deepsa_best.pth',
        'image_size': 1008,
        'batch_size': 4,
        'learning_rate': 5e-5,  # Lower LR since we're fine-tuning
        'epochs': 30,
        'port': 12356,  # Different port than Phase 1
    }

    world_size = torch.cuda.device_count()

    print(f"\n{'='*70}")
    print("SAM 3 Phase 2: CASS Segment Classification")
    print(f"{'='*70}")
    print(f"\nThis phase teaches the model:")
    print(f"  1. Which CASS segment is being analyzed")
    print(f"  2. Where the segment is located (bounding box)")
    print(f"  3. How view angles relate to visible segments")
    print(f"\nGPUs: {world_size}")

    if not Path(args['phase1_checkpoint']).exists():
        print(f"\nWARNING: Phase 1 checkpoint not found at {args['phase1_checkpoint']}")
        print("Training from scratch (not recommended)")

    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
