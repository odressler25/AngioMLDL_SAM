"""
Debug script to check what's happening in training
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\odressler\sam3")

from sam3 import build_sam3_image_model
from view_angle_encoder import ViewAngleEncoder, ViewConditionedFeatureFusion
import pandas as pd
import json

# Load one sample
csv_path = r'E:\AngioMLDL_data\corrected_dataset_training.csv'
df = pd.read_csv(csv_path)
df = df[df['split'] == 'train'].reset_index(drop=True)

# Get first sample
idx = 0
cine = np.load(df['cine_path'].iloc[idx])
mask = np.load(df['vessel_mask_actual_path'].iloc[idx])
frame_idx = df['frame_index'].iloc[idx]

frame = cine[frame_idx]
if frame.ndim == 2:
    frame = np.stack([frame, frame, frame], axis=-1)

frame = frame.astype(np.float32)
if frame.max() > 1:
    frame = frame / 255.0

mask = mask.astype(np.float32)
if mask.max() > 1:
    mask = mask / 255.0

print("=" * 70)
print("Debugging Training Pipeline")
print("=" * 70)

print(f"\n1. Input image:")
print(f"   Shape: {frame.shape}")
print(f"   Range: [{frame.min():.3f}, {frame.max():.3f}]")
print(f"   Mean: {frame.mean():.3f}")

print(f"\n2. Ground truth mask:")
print(f"   Shape: {mask.shape}")
print(f"   Range: [{mask.min():.3f}, {mask.max():.3f}]")
print(f"   Vessel pixels: {(mask > 0.5).sum()} / {mask.size} ({(mask > 0.5).sum() / mask.size * 100:.1f}%)")

# Convert to tensor and add batch dim
image_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).cuda()
mask_tensor = torch.from_numpy(mask).unsqueeze(0).cuda()

print(f"\n3. Testing SAM 3 feature extraction:")
sam3 = build_sam3_image_model().cuda()
sam3.eval()

# Resize to 1008
if image_tensor.shape[2] != 1008 or image_tensor.shape[3] != 1008:
    image_resized = F.interpolate(image_tensor, size=(1008, 1008), mode='bilinear', align_corners=False)
    mask_resized = F.interpolate(mask_tensor.unsqueeze(0), size=(1008, 1008), mode='bilinear', align_corners=False).squeeze(0)
else:
    image_resized = image_tensor
    mask_resized = mask_tensor

print(f"   Resized image: {image_resized.shape}")

# Normalize for SAM 3
image_norm = (image_resized - 0.5) / 0.5

with torch.no_grad():
    features = sam3.backbone.forward_image(image_norm)

if isinstance(features, dict):
    features = features.get('vision_features', features.get('image_embeddings'))

print(f"   SAM 3 features shape: {features.shape}")
print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
print(f"   Feature mean: {features.mean():.3f}")
print(f"   Feature std: {features.std():.3f}")

# Convert to (B, C, H, W) if needed
if len(features.shape) == 3:
    B, N, C = features.shape
    H = W = int(N ** 0.5)
    features = features.permute(0, 2, 1).reshape(B, C, H, W)
    print(f"   Reshaped to: {features.shape}")

# Project to 256 if needed
if features.shape[1] != 256:
    proj = torch.nn.Conv2d(features.shape[1], 256, kernel_size=1).cuda()
    features = proj(features)
    print(f"   Projected to: {features.shape}")

print(f"\n4. Testing view encoder:")
with open(df['contours_path'].iloc[idx], 'r') as f:
    contours = json.load(f)
view_angles = contours.get('view_angles', {})
primary = torch.tensor([float(view_angles.get('primary', 0.0))]).cuda()
secondary = torch.tensor([float(view_angles.get('secondary', 0.0))]).cuda()

print(f"   View angles: primary={primary.item():.1f}, secondary={secondary.item():.1f}")

view_encoder = ViewAngleEncoder(embedding_dim=256).cuda()
view_emb = view_encoder(primary, secondary)
print(f"   View embedding: {view_emb.shape}")
print(f"   View embedding range: [{view_emb.min():.3f}, {view_emb.max():.3f}]")

print(f"\n5. Testing feature fusion:")
fusion = ViewConditionedFeatureFusion(feature_dim=256, fusion_mode='film').cuda()
fused = fusion(features, view_emb)
print(f"   Fused features: {fused.shape}")
print(f"   Fused range: [{fused.min():.3f}, {fused.max():.3f}]")

print(f"\n6. Testing segmentation head:")
seg_head = torch.nn.Sequential(
    torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(128),
    torch.nn.ReLU(),
    torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 1, kernel_size=1)
).cuda()

mask_logits = seg_head(fused)
print(f"   Mask logits: {mask_logits.shape}")
print(f"   Logits range: [{mask_logits.min():.3f}, {mask_logits.max():.3f}]")

# Upsample to original size
masks = F.interpolate(mask_logits, size=(1008, 1008), mode='bilinear').squeeze(1)
pred_mask = torch.sigmoid(masks)

print(f"   Predicted mask: {pred_mask.shape}")
print(f"   Pred range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
print(f"   Pred mean: {pred_mask.mean():.3f}")
print(f"   Pred > 0.5: {(pred_mask > 0.5).sum()} / {pred_mask.numel()} ({(pred_mask > 0.5).sum() / pred_mask.numel() * 100:.1f}%)")

print(f"\n7. Computing Dice:")
pred_binary = (pred_mask > 0.5).float()
target = mask_resized

intersection = (pred_binary * target).sum()
dice = (2. * intersection + 1.0) / (pred_binary.sum() + target.sum() + 1.0)
print(f"   Dice score: {dice.item():.4f}")

if dice.item() < 0.1:
    print("\n   ⚠️  PROBLEM: Dice is very low!")
    print(f"   Ground truth vessel %: {(target > 0.5).sum() / target.numel() * 100:.2f}%")
    print(f"   Predicted vessel %: {(pred_binary > 0.5).sum() / pred_binary.numel() * 100:.2f}%")
    print(f"   Intersection: {intersection.item()}")

    if pred_mask.mean() < 0.1:
        print(f"   → Model predicting mostly background (mean={pred_mask.mean():.4f})")
    elif pred_mask.mean() > 0.9:
        print(f"   → Model predicting mostly foreground (mean={pred_mask.mean():.4f})")

print("\n" + "=" * 70)
