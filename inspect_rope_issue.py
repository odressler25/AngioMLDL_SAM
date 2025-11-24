"""
Debug the RoPE assertion error to understand what SAM 3 expects
"""

import torch
import sys
sys.path.insert(0, r"C:\Users\odressler\sam3")

from sam3 import build_sam3_image_model
import torch.nn.functional as F

device = 'cuda'

print("Loading SAM 3...")
sam3 = build_sam3_image_model().to(device)
sam3.eval()

# Test image: exactly 1024x1024
images = torch.rand(1, 3, 1024, 1024).to(device)
images_normalized = (images - 0.5) / 0.5

print(f"\nInput shape: {images.shape}")
print(f"Normalized shape: {images_normalized.shape}")

# Try to trace where it fails
print("\nAttempting forward pass...")

try:
    with torch.no_grad():
        backbone_out = sam3.backbone.forward_image(images_normalized)
    print("SUCCESS!")
except AssertionError as e:
    print(f"FAILED with assertion error")
    print("\nLet's inspect the RoPE frequencies...")

    # Access the vision backbone
    vision_backbone = sam3.backbone.vision_backbone
    trunk = vision_backbone.trunk

    print(f"\nTrunk type: {type(trunk)}")
    print(f"Number of blocks: {len(trunk.blocks)}")

    # Check first attention block
    first_block = trunk.blocks[0]
    first_attn = first_block.attn

    print(f"\nFirst attention block:")
    print(f"  Type: {type(first_attn)}")

    if hasattr(first_attn, 'freqs_cis'):
        freqs_cis = first_attn.freqs_cis
        print(f"  RoPE freqs_cis shape: {freqs_cis.shape}")
    else:
        print(f"  No freqs_cis found!")

    # Check what dimensions the model expects
    print(f"\nTrying to understand expected dimensions...")

    # Get patch embedding info
    if hasattr(trunk, 'patch_embed'):
        patch_embed = trunk.patch_embed
        print(f"\nPatch embedding:")
        print(f"  Type: {type(patch_embed)}")
        if hasattr(patch_embed, 'patch_size'):
            print(f"  Patch size: {patch_embed.patch_size}")
        if hasattr(patch_embed, 'img_size'):
            print(f"  Expected img size: {patch_embed.img_size}")

    # Calculate expected sequence length
    if hasattr(trunk, 'patch_size'):
        patch_size = trunk.patch_size[0] if isinstance(trunk.patch_size, tuple) else trunk.patch_size
        img_size = 1024
        num_patches = (img_size // patch_size) ** 2
        print(f"\nCalculated:")
        print(f"  Patch size: {patch_size}")
        print(f"  Number of patches: {num_patches}")

        if hasattr(first_attn, 'freqs_cis'):
            print(f"  RoPE expects: {freqs_cis.shape[0]} positions")
            print(f"  We provide: {num_patches} tokens")

            if freqs_cis.shape[0] != num_patches:
                print(f"\n  MISMATCH! RoPE was initialized for different image size")
                print(f"  Expected image size: {int((freqs_cis.shape[0] ** 0.5) * patch_size)}")
