"""
Stage 1: Frozen SAM 3 with CATEGORICAL view angles

Key difference: Discretizes continuous angles into standard angiography views
This makes it easier for the model to learn view-vessel relationships
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


def discretize_angle(angle, bins):
    """
    Discretize continuous angle into nearest bin

    Args:
        angle: Continuous angle in degrees
        bins: List of bin centers (e.g., [-45, -30, -15, 0, 15, 30, 45])

    Returns:
        Index of nearest bin
    """
    bins = np.array(bins)
    idx = np.argmin(np.abs(bins - angle))
    return idx


# Standard angiography view angles
PRIMARY_BINS = [-45, -30, -15, 0, 15, 30, 45, 60, 90]  # LAO/RAO
SECONDARY_BINS = [-45, -30, -15, 0, 15, 30, 45]  # Cranial/Caudal

NUM_PRIMARY = len(PRIMARY_BINS)
NUM_SECONDARY = len(SECONDARY_BINS)


class CategoricalViewEncoder(nn.Module):
    """
    Encodes view angles as categorical embeddings

    Instead of continuous sinusoidal encoding, uses learned embeddings
    for discrete view angle bins
    """

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Learnable embeddings for each view angle bin
        self.primary_embeddings = nn.Embedding(NUM_PRIMARY, embedding_dim // 2)
        self.secondary_embeddings = nn.Embedding(NUM_SECONDARY, embedding_dim // 2)

        # MLP to process combined embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, primary_indices, secondary_indices):
        """
        Args:
            primary_indices: (B,) tensor of primary angle bin indices
            secondary_indices: (B,) tensor of secondary angle bin indices

        Returns:
            view_embedding: (B, embedding_dim)
        """
        # Look up embeddings
        primary_emb = self.primary_embeddings(primary_indices)  # (B, embedding_dim // 2)
        secondary_emb = self.secondary_embeddings(secondary_indices)  # (B, embedding_dim // 2)

        # Concatenate
        combined = torch.cat([primary_emb, secondary_emb], dim=1)  # (B, embedding_dim)

        # Process with MLP
        view_embedding = self.mlp(combined)

        return view_embedding


class ViewConditionedFeatureFusion(nn.Module):
    """Feature-wise Linear Modulation (FiLM)"""

    def __init__(self, feature_dim=256):
        super().__init__()
        self.film_gamma = nn.Linear(feature_dim, feature_dim)
        self.film_beta = nn.Linear(feature_dim, feature_dim)

    def forward(self, image_features, view_embedding):
        """
        Args:
            image_features: (B, C, H, W)
            view_embedding: (B, C)
        """
        gamma = self.film_gamma(view_embedding)[:, :, None, None]
        beta = self.film_beta(view_embedding)[:, :, None, None]
        return gamma * image_features + beta


class FrozenSAM3Model(nn.Module):
    """
    SAM 3 with categorical view encoding
    """
    def __init__(self, image_size=1008):
        super().__init__()

        print("=" * 70)
        print("Building Frozen SAM 3 with CATEGORICAL View Encoding")
        print("=" * 70)

        # Load SAM 3
        print("\nLoading SAM 3...")
        self.sam3 = build_sam3_image_model()
        self.sam3.eval()
        self.sam3.cuda()

        for param in self.sam3.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel() for p in self.sam3.parameters())
        print(f"SAM 3 loaded: {frozen_params/1e6:.1f}M params (frozen)")

        # Categorical view encoder
        self.view_encoder = CategoricalViewEncoder(embedding_dim=256)

        print(f"\nView discretization:")
        print(f"  Primary bins: {PRIMARY_BINS}")
        print(f"  Secondary bins: {SECONDARY_BINS}")
        print(f"  Total view combinations: {NUM_PRIMARY} × {NUM_SECONDARY} = {NUM_PRIMARY * NUM_SECONDARY}")

        # Feature fusion
        self.feature_fusion = ViewConditionedFeatureFusion(feature_dim=256)

        self.image_size = image_size

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

        self.feature_proj = None

        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        print(f"\nParameter Summary:")
        print(f"  Total params: {total/1e6:.1f}M")
        print(f"  Trainable: {trainable/1e6:.1f}M")
        print(f"  Frozen (SAM 3): {frozen_params/1e6:.1f}M")
        print(f"  Trainable %: {trainable/total*100:.2f}%")
        print("=" * 70)

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
        """Extract SAM 3 features on GPU 0"""
        images = images.cuda(0)
        images_padded = self.preprocess_for_backbone(images, target_size=1008)
        images_normalized = (images_padded - 0.5) / 0.5

        with torch.no_grad():
            self.sam3.eval()
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

    def forward(self, images, primary_indices, secondary_indices):
        """
        Args:
            images: (B, 3, H, W)
            primary_indices: (B,) categorical indices
            secondary_indices: (B,) categorical indices
        """
        # Encode categorical view angles
        view_embedding = self.view_encoder(primary_indices, secondary_indices)

        # SAM 3 features
        features = self.extract_sam3_features(images)
        features = features.to(view_embedding.device)

        # Project to 256 if needed
        if features.shape[1] != 256:
            if self.feature_proj is None:
                self.feature_proj = nn.Conv2d(features.shape[1], 256, kernel_size=1).to(features.device)
                self.add_module('feature_proj', self.feature_proj)
            features = self.feature_proj(features)

        # Fuse with view
        fused_features = self.feature_fusion(features, view_embedding)

        # Decode
        mask_logits = self.seg_head(fused_features)
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
    """Dataset with categorical view angles"""
    def __init__(self, csv_path, split='train', image_size=1008):
        import pandas as pd
        import json

        self.split = split
        self.image_size = image_size

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

        # Load image
        cine = np.load(self.cine_paths[idx])
        frame_idx = self.frame_indices[idx]
        if frame_idx >= len(cine):
            frame_idx = len(cine) - 1
        frame = cine[frame_idx]

        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)

        frame = frame.astype(np.float32)
        if frame.max() > 1:
            frame = frame / 255.0

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

        # Load and DISCRETIZE view angles
        with open(self.contours_paths[idx], 'r') as f:
            contours = json.load(f)
        view_angles = contours.get('view_angles', {})
        primary_angle = float(view_angles.get('primary_angle', 0.0))
        secondary_angle = float(view_angles.get('secondary_angle', 0.0))

        # Convert to categorical indices
        primary_idx = discretize_angle(primary_angle, PRIMARY_BINS)
        secondary_idx = discretize_angle(secondary_angle, SECONDARY_BINS)

        primary_idx = torch.tensor(primary_idx, dtype=torch.long)
        secondary_idx = torch.tensor(secondary_idx, dtype=torch.long)

        return image, mask, primary_idx, secondary_idx


def train_epoch(model, dataloader, optimizer, device, scaler):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, masks, primary_idx, secondary_idx in pbar:
        images = images.to(device)
        masks = masks.to(device)
        primary_idx = primary_idx.to(device)
        secondary_idx = secondary_idx.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            pred_logits = model(images, primary_idx, secondary_idx)
            loss = combined_loss(pred_logits, masks, bce_weight=0.5)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            dice = dice_score(torch.sigmoid(pred_logits), masks)

        total_loss += loss.item()
        total_dice += dice

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})

    return total_loss / len(dataloader), total_dice / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate"""
    model.eval()
    total_loss = 0
    total_dice = 0

    for images, masks, primary_idx, secondary_idx in tqdm(dataloader, desc='Validation'):
        images = images.to(device)
        masks = masks.to(device)
        primary_idx = primary_idx.to(device)
        secondary_idx = secondary_idx.to(device)

        with torch.amp.autocast('cuda'):
            pred_logits = model(images, primary_idx, secondary_idx)
            loss = combined_loss(pred_logits, masks, bce_weight=0.5)

        dice = dice_score(torch.sigmoid(pred_logits), masks)
        total_loss += loss.item()
        total_dice += dice

    return total_loss / len(dataloader), total_dice / len(dataloader)


def main():
    batch_size = 16
    learning_rate = 1e-4
    epochs = 30
    image_size = 1008

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    model = FrozenSAM3Model(image_size=image_size).to(device)

    print("\nNOTE: Using categorical view encoding (discrete bins)")
    print("This should be easier to learn than continuous angles\n")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    csv_path = r'E:\AngioMLDL_data\corrected_dataset_training.csv'
    train_dataset = AngiographyDataset(csv_path, split='train', image_size=image_size)
    val_dataset = AngiographyDataset(csv_path, split='val', image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

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
            }, 'checkpoints/stage1_frozen_categorical_best.pth')
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
