"""Test if SAM 3 expects 1008x1008 images, not 1024x1024"""

import torch
import sys
sys.path.insert(0, r"C:\Users\odressler\sam3")

from sam3 import build_sam3_image_model

device = 'cuda'

print("Loading SAM 3...")
sam3 = build_sam3_image_model().to(device)
sam3.eval()

# Test different resolutions
test_resolutions = [384, 576, 1008, 1024, 512]

for res in test_resolutions:
    print(f"\nTesting {res}x{res}...")
    images = torch.rand(1, 3, res, res).to(device)
    images_normalized = (images - 0.5) / 0.5

    try:
        with torch.no_grad():
            backbone_out = sam3.backbone.forward_image(images_normalized)
        print(f"  [OK] {res}x{res} WORKS!")

        # Check output
        if 'vision_features' in backbone_out:
            features = backbone_out['vision_features']
            print(f"       Output shape: {features.shape}")
        break  # Found it!

    except AssertionError:
        print(f"  [FAIL] {res}x{res} causes RoPE assertion")
    except Exception as e:
        print(f"  [FAIL] {res}x{res} causes error: {type(e).__name__}")
