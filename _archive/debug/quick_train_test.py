"""
Quick 1-batch training test to verify everything works before overnight run
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

# Local imports
from view_angle_encoder import ViewAngleEncoder, ViewConditionedFeatureFusion
from sam3_lora_wrapper import SAM3WithLoRA

print("="*70)
print("QUICK TRAINING TEST (1 BATCH)")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
print(f"GPU Count: {torch.cuda.device_count()}")

# Hyperparameters
batch_size = 8  # Smaller for quick test
image_size = 1024

print(f"\nTest config:")
print(f"  Batch size: {batch_size}")
print(f"  Image size: {image_size}x{image_size}")

# Initialize models
print("\n[1/3] Building models...")
sam3_lora = SAM3WithLoRA(lora_r=16, lora_alpha=32, use_multi_gpu=True).to(device)
view_encoder = ViewAngleEncoder(embedding_dim=256).to(device)
feature_fusion = ViewConditionedFeatureFusion(feature_dim=256, fusion_mode='film').to(device)

print("[OK] Models built")

# Create dummy data
print("\n[2/3] Creating dummy batch...")
dummy_images = torch.randn(batch_size, 3, image_size, image_size).to(device)
dummy_masks = torch.rand(batch_size, image_size, image_size).to(device)
dummy_primary = torch.randn(batch_size).to(device) * 30
dummy_secondary = torch.randn(batch_size).to(device) * 30

print(f"  Images: {dummy_images.shape}")
print(f"  Masks: {dummy_masks.shape}")
print(f"  Angles: {dummy_primary.shape}, {dummy_secondary.shape}")

# Forward pass test
print("\n[3/3] Testing forward pass...")

try:
    # Access backbone (same as training script)
    sam3_model = sam3_lora.model
    backbone = sam3_model.backbone

    # Normalize images
    images_normalized = (dummy_images - 0.5) / 0.5

    # Get features
    with torch.amp.autocast('cuda'):
        print("  Calling backbone.forward_image()...")
        backbone_out = backbone.forward_image(images_normalized)

        # Extract features
        if 'vision_features' in backbone_out:
            features = backbone_out['vision_features']
        elif 'image_embeddings' in backbone_out:
            features = backbone_out['image_embeddings']
        else:
            for k, v in backbone_out.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 3:
                    features = v
                    break

        print(f"  Features extracted: {features.shape}")

        # Convert to (B, C, H, W)
        if len(features.shape) == 3:
            B_feat, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.permute(0, 2, 1).reshape(B_feat, C, H, W)
            print(f"  Reshaped to: {features.shape}")

        # Project to 256 channels
        if features.shape[1] != 256:
            feature_proj = nn.Conv2d(features.shape[1], 256, kernel_size=1).to(device)
            features = feature_proj(features)
            print(f"  Projected to: {features.shape}")

        # View encoding
        print("  Encoding view angles...")
        view_embedding = view_encoder(dummy_primary, dummy_secondary)
        print(f"  View embedding: {view_embedding.shape}")

        # Fusion
        print("  Fusing features...")
        fused_features = feature_fusion(features, view_embedding)
        print(f"  Fused: {fused_features.shape}")

        # Segmentation head
        print("  Creating segmentation masks...")
        seg_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        ).to(device)

        mask_logits = seg_head(fused_features)
        masks = F.interpolate(mask_logits, size=(image_size, image_size), mode='bilinear')
        masks = masks.squeeze(1)
        print(f"  Output masks: {masks.shape}")

    print("\n[OK] Forward pass successful!")

except Exception as e:
    print(f"\n[FAIL] Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Memory check
print("\n" + "="*70)
print("GPU MEMORY USAGE")
print("="*70)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

print("\n" + "="*70)
print("SUCCESS - Ready for overnight training!")
print("="*70)
print("\nYou can now safely run:")
print("  python train_stage1_overnight.py")
print("="*70)
