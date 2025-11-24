"""
Test that the RoPE fix (padding to 1024x1024) resolves the assertion error.
"""

import torch
import sys
import os

# Add sam3 to path
sam3_path = r"C:\Users\odressler\sam3"
if os.path.exists(sam3_path):
    sys.path.insert(0, sam3_path)

from sam3_lora_wrapper import SAM3WithLoRA
import torch.nn.functional as F

def preprocess_for_backbone(image, target_size=1024):
    """
    Resize and pad image to exactly target_size x target_size.
    This fixes the RoPE assertion by ensuring sequence length matches expectations.
    """
    b, c, h, w = image.shape

    # 1. Resize longest side to target_size (maintain aspect ratio)
    scale = target_size / max(h, w)
    if scale != 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # 2. Pad to form perfect square (target_size x target_size)
    curr_h, curr_w = image.shape[-2:]
    pad_h = target_size - curr_h
    pad_w = target_size - curr_w

    # F.pad format: (left, right, top, bottom)
    image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)

    return image


def test_rope_fix():
    print("=" * 60)
    print("Testing RoPE Fix with Image Preprocessing")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Step 1: Build SAM 3 with LoRA
    print("\n[1/4] Building SAM 3 with LoRA...")
    try:
        sam3_lora = SAM3WithLoRA(
            lora_r=16,
            lora_alpha=32,
            use_multi_gpu=False  # Single GPU test
        ).to(device)
        print("[OK] SAM 3 with LoRA built successfully")

        # Get backbone
        sam3_model = sam3_lora.model
        if hasattr(sam3_model, 'module'):
            sam3_model = sam3_model.module
        backbone = sam3_model.backbone

    except Exception as e:
        print(f"[FAIL] Failed to build SAM 3: {e}")
        return False

    # Step 2: Test with various image sizes (non-square, different resolutions)
    print("\n[2/4] Testing with various image sizes...")

    test_cases = [
        (1, 3, 800, 600),   # Non-square, smaller
        (2, 3, 512, 768),   # Non-square batch
        (1, 3, 1024, 1024), # Exact size
        (4, 3, 640, 480),   # Batch of 4
    ]

    for i, shape in enumerate(test_cases):
        b, c, h, w = shape
        print(f"\n  Test {i+1}: Input shape {shape}")

        try:
            # Create dummy image
            images = torch.rand(b, c, h, w).to(device)

            # Apply preprocessing (THIS IS THE FIX)
            images_padded = preprocess_for_backbone(images, target_size=1024)
            print(f"    -> After padding: {images_padded.shape}")

            # Normalize
            images_normalized = (images_padded - 0.5) / 0.5

            # Test forward pass
            with torch.no_grad():
                backbone_out = backbone.forward_image(images_normalized)

            print(f"    [OK] Forward pass succeeded!")

            # Check output
            if 'vision_features' in backbone_out:
                features = backbone_out['vision_features']
                print(f"    [OK] Got vision_features: {features.shape}")
            elif 'image_embeddings' in backbone_out:
                features = backbone_out['image_embeddings']
                print(f"    [OK] Got image_embeddings: {features.shape}")
            else:
                print(f"    [OK] Got output dict with keys: {backbone_out.keys()}")

        except AssertionError as e:
            import traceback
            print(f"    [FAIL] RoPE assertion failed!")
            print(f"    Error: {e}")
            print(f"    Full traceback:")
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"    [FAIL] Other error: {e}")
            return False

    # Step 3: Test memory usage
    print("\n[3/4] Testing memory usage with realistic batch...")
    try:
        batch_size = 8
        images = torch.rand(batch_size, 3, 800, 600).to(device)
        images_padded = preprocess_for_backbone(images, target_size=1024)
        images_normalized = (images_padded - 0.5) / 0.5

        with torch.no_grad():
            backbone_out = backbone.forward_image(images_normalized)

        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [OK] Batch size {batch_size}: {mem_used:.2f} GB used")

    except Exception as e:
        print(f"  [FAIL] Memory test failed: {e}")
        return False

    # Step 4: Verify gradients work with LoRA
    print("\n[4/4] Testing backward pass (LoRA gradients)...")
    try:
        sam3_lora.train()  # Enable training mode
        images = torch.rand(2, 3, 512, 512).to(device)
        images_padded = preprocess_for_backbone(images, target_size=1024)
        images_normalized = (images_padded - 0.5) / 0.5

        # Forward with gradients
        backbone_out = backbone.forward_image(images_normalized)

        if 'vision_features' in backbone_out:
            features = backbone_out['vision_features']
        elif 'image_embeddings' in backbone_out:
            features = backbone_out['image_embeddings']
        else:
            features = list(backbone_out.values())[0]

        # Create dummy loss
        loss = features.mean()
        loss.backward()

        print(f"  [OK] Backward pass succeeded!")

        # Check LoRA gradients
        lora_params_with_grad = sum(1 for p in sam3_lora.lora_params if p.grad is not None)
        print(f"  [OK] LoRA parameters with gradients: {lora_params_with_grad}/{len(sam3_lora.lora_params)}")

    except Exception as e:
        print(f"  [FAIL] Backward pass failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("[OK] ALL TESTS PASSED!")
    print("The RoPE fix works - ready for overnight training!")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = test_rope_fix()
    sys.exit(0 if success else 1)
