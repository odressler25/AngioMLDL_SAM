"""
Quick test of frozen SAM 3 approach (no LoRA) to verify RoPE fix
"""

import torch
import sys
import os

sam3_path = r"C:\Users\odressler\sam3"
if os.path.exists(sam3_path):
    sys.path.insert(0, sam3_path)

from sam3 import build_sam3_image_model
import torch.nn.functional as F


def preprocess_for_backbone(image, target_size=1008):
    """Resize and pad to 1008x1008 (SAM 3's expected resolution)"""
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


def test_frozen_sam3():
    print("="*60)
    print("Testing Frozen SAM 3 (No LoRA)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load SAM 3 and freeze it
    print("\n[1/4] Loading and freezing SAM 3...")
    try:
        sam3 = build_sam3_image_model().to(device)
        sam3.eval()

        # Freeze ALL parameters
        for param in sam3.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel() for p in sam3.parameters())
        trainable_params = sum(p.numel() for p in sam3.parameters() if p.requires_grad)

        print(f"  [OK] SAM 3 loaded")
        print(f"       Total params: {frozen_params/1e6:.1f}M")
        print(f"       Trainable: {trainable_params/1e6:.1f}M")
        print(f"       Frozen: {frozen_params/1e6:.1f}M")

    except Exception as e:
        print(f"  [FAIL] Failed to load SAM 3: {e}")
        return False

    # Test with various image sizes
    print("\n[2/4] Testing forward pass with various image sizes...")

    test_cases = [
        (1, 3, 800, 600),
        (2, 3, 512, 768),
        (1, 3, 1024, 1024),
        (4, 3, 640, 480),
    ]

    for i, shape in enumerate(test_cases):
        b, c, h, w = shape
        print(f"\n  Test {i+1}: Input shape {shape}")

        try:
            # Create dummy image
            images = torch.rand(b, c, h, w).to(device)

            # Preprocess
            images_padded = preprocess_for_backbone(images, target_size=1008)
            print(f"    -> After padding: {images_padded.shape}")

            # Normalize
            images_normalized = (images_padded - 0.5) / 0.5

            # Forward pass (NO GRADIENTS)
            with torch.no_grad():
                sam3.eval()
                backbone_out = sam3.backbone.forward_image(images_normalized)

            print(f"    [OK] Forward pass succeeded!")

            # Check output
            if 'vision_features' in backbone_out:
                features = backbone_out['vision_features']
                print(f"    [OK] Got vision_features: {features.shape}")
            elif 'image_embeddings' in backbone_out:
                features = backbone_out['image_embeddings']
                print(f"    [OK] Got image_embeddings: {features.shape}")
            else:
                print(f"    [OK] Got output dict with keys: {list(backbone_out.keys())}")

        except AssertionError as e:
            print(f"    [FAIL] RoPE assertion still failing!")
            print(f"           This means freezing SAM 3 didn't fix it")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"    [FAIL] Other error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Test memory usage
    print("\n[3/4] Testing memory usage with realistic batch...")
    try:
        batch_size = 8
        images = torch.rand(batch_size, 3, 800, 600).to(device)
        images_padded = preprocess_for_backbone(images, target_size=1008)
        images_normalized = (images_padded - 0.5) / 0.5

        with torch.no_grad():
            sam3.eval()
            backbone_out = sam3.backbone.forward_image(images_normalized)

        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [OK] Batch size {batch_size}: {mem_used:.2f} GB used")

    except Exception as e:
        print(f"  [FAIL] Memory test failed: {e}")
        return False

    # Clean up memory before gradient test
    torch.cuda.empty_cache()

    # Verify SAM 3 is frozen (check requires_grad)
    print("\n[4/4] Verifying SAM 3 is truly frozen...")
    sam3_params_trainable = sum(1 for p in sam3.parameters() if p.requires_grad)
    sam3_params_total = sum(1 for p in sam3.parameters())

    if sam3_params_trainable == 0:
        print(f"  [OK] SAM 3 is frozen - all {sam3_params_total} params have requires_grad=False")
    else:
        print(f"  [WARN] SAM 3 has {sam3_params_trainable}/{sam3_params_total} trainable params!")
        print(f"         Expected 0")
        return False

    print("\n" + "="*60)
    print("[OK] ALL TESTS PASSED!")
    print("Frozen SAM 3 approach works - ready for overnight training!")
    print("="*60)

    return True


if __name__ == '__main__':
    import sys
    success = test_frozen_sam3()
    sys.exit(0 if success else 1)
