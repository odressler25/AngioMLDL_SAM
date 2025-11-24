"""
CRITICAL TEST: Verify SAM 3 backbone.forward_image() works correctly
before running overnight training.

This test simulates EXACTLY what the training script will do.
"""
import torch
import torch.nn as nn
from sam3.model_builder import build_sam3_image_model
from lora_layers import add_lora_to_model
from view_angle_encoder import ViewAngleEncoder, ViewConditionedFeatureFusion

print("="*70)
print("CRITICAL VERIFICATION TEST")
print("="*70)

# Step 1: Build SAM 3 with LoRA (exactly like training script)
print("\n[1/5] Building SAM 3 with LoRA...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

sam3_base = build_sam3_image_model(device=device)
print("[OK] SAM 3 base loaded")

# Add LoRA
target_modules = ["attn.qkv", "attn.proj"]
sam3_with_lora, num_lora_params = add_lora_to_model(
    model=sam3_base,
    target_modules=target_modules,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
)
print(f"✓ LoRA added ({num_lora_params:,} trainable params)")

# Step 2: Test backbone.forward_image()
print("\n[2/5] Testing backbone.forward_image()...")

# Create dummy batch (exactly like training)
batch_size = 4
image_size = 1024
dummy_images = torch.randn(batch_size, 3, image_size, image_size).to(device)
print(f"Input shape: {dummy_images.shape}")

# Normalize (exactly like training script will do)
images_normalized = (dummy_images - 0.5) / 0.5
print(f"Normalized range: [{images_normalized.min():.2f}, {images_normalized.max():.2f}]")

try:
    # Get backbone
    backbone = sam3_with_lora.backbone
    print(f"✓ Backbone accessed: {type(backbone).__name__}")

    # Forward pass
    print("\nCalling backbone.forward_image()...")
    with torch.amp.autocast('cuda'):
        backbone_out = backbone.forward_image(images_normalized)

    print(f"✓ backbone.forward_image() succeeded!")
    print(f"  Output type: {type(backbone_out)}")

    if isinstance(backbone_out, dict):
        print(f"  Output keys: {list(backbone_out.keys())}")
        for k, v in backbone_out.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.shape}, dtype={v.dtype}")
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
                print(f"    {k}: list of {len(v)} tensors, first shape={v[0].shape}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*70)
    print("ERROR: Cannot proceed with training!")
    print("="*70)
    exit(1)

# Step 3: Extract features (exactly like training script)
print("\n[3/5] Extracting features...")

try:
    if 'vision_features' in backbone_out:
        features = backbone_out['vision_features']
        print(f"✓ Found 'vision_features': {features.shape}")
    elif 'image_embeddings' in backbone_out:
        features = backbone_out['image_embeddings']
        print(f"✓ Found 'image_embeddings': {features.shape}")
    else:
        print("⚠ Looking for alternative feature key...")
        for k, v in backbone_out.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 3:
                features = v
                print(f"✓ Using '{k}': {features.shape}")
                break

    # Convert to (B, C, H, W) if needed
    if len(features.shape) == 3:
        print(f"  Converting from (B, H*W, C) to (B, C, H, W)...")
        B_feat, N, C = features.shape
        H = W = int(N ** 0.5)
        features = features.permute(0, 2, 1).reshape(B_feat, C, H, W)
        print(f"  Result: {features.shape}")

    print(f"✓ Feature extraction successful!")
    print(f"  Final feature shape: {features.shape}")
    print(f"  Channels: {features.shape[1]}")

except Exception as e:
    print(f"✗ Feature extraction FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Test view encoding + fusion
print("\n[4/5] Testing view encoding and fusion...")

try:
    # Create view encoder
    view_encoder = ViewAngleEncoder(embedding_dim=256).to(device)
    feature_fusion = ViewConditionedFeatureFusion(feature_dim=256, fusion_mode='film').to(device)

    # Dummy view angles
    primary_angles = torch.randn(batch_size).to(device) * 30  # ±30 degrees
    secondary_angles = torch.randn(batch_size).to(device) * 30

    # Encode
    view_embedding = view_encoder(primary_angles, secondary_angles)
    print(f"✓ View embedding: {view_embedding.shape}")

    # Project features to 256 channels if needed
    if features.shape[1] != 256:
        print(f"  Projecting {features.shape[1]} → 256 channels...")
        feature_proj = nn.Conv2d(features.shape[1], 256, kernel_size=1).to(device)
        features = feature_proj(features)
        print(f"  Projected: {features.shape}")

    # Fuse
    fused_features = feature_fusion(features, view_embedding)
    print(f"✓ Fused features: {fused_features.shape}")

except Exception as e:
    print(f"✗ View encoding/fusion FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Test segmentation head
print("\n[5/5] Testing segmentation head...")

try:
    seg_head = nn.Sequential(
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 1, kernel_size=1)
    ).to(device)

    # Decode
    mask_logits = seg_head(fused_features)
    print(f"✓ Mask logits: {mask_logits.shape}")

    # Upsample
    masks = torch.nn.functional.interpolate(
        mask_logits,
        size=(image_size, image_size),
        mode='bilinear'
    )
    masks = masks.squeeze(1)
    print(f"✓ Final masks: {masks.shape}")

except Exception as e:
    print(f"✗ Segmentation FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 6: Calculate memory usage
print("\n" + "="*70)
print("MEMORY USAGE")
print("="*70)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Success!
print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nThe training script should work correctly.")
print("Expected behavior:")
print(f"  - Batch size 16 will split to 8 per GPU")
print(f"  - Each GPU should use ~15-20GB VRAM")
print(f"  - Forward pass produces masks of shape (B, {image_size}, {image_size})")
print("\nYou can now run: python train_stage1_overnight.py")
print("="*70)
