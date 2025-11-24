"""Quick test that LoRA + 1008x1008 resolution works"""

import torch
import sys
sys.path.insert(0, r"C:\Users\odressler\sam3")

from sam3_lora_wrapper import SAM3WithLoRA
import torch.nn.functional as F

device = 'cuda'

print("="*60)
print("Testing SAM 3 + LoRA with 1008x1008 Resolution")
print("="*60)

# Build SAM 3 with LoRA
print("\nLoading SAM 3 with LoRA...")
sam3_lora = SAM3WithLoRA(lora_r=16, lora_alpha=32, use_multi_gpu=False).to(device)

# Get backbone
sam3_model = sam3_lora.model
backbone = sam3_model.backbone

print("\nTesting forward pass with 1008x1008...")

test_cases = [
    (1, 3, 1008, 1008),  # Exact size
    (2, 3, 512, 768),    # Will be padded
    (4, 3, 800, 600),    # Will be padded
]

def preprocess(image, target_size=1008):
    """Resize and pad to 1008x1008"""
    b, c, h, w = image.shape
    scale = target_size / max(h, w)
    if scale != 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    curr_h, curr_w = image.shape[-2:]
    pad_h, pad_w = target_size - curr_h, target_size - curr_w
    image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return image

for i, shape in enumerate(test_cases):
    print(f"\nTest {i+1}: Input {shape}")
    try:
        images = torch.rand(*shape).to(device)
        images_padded = preprocess(images, 1008)
        images_normalized = (images_padded - 0.5) / 0.5

        # Forward with LoRA
        with torch.amp.autocast('cuda'):
            backbone_out = backbone.forward_image(images_normalized)

        features = backbone_out.get('vision_features', list(backbone_out.values())[0])
        print(f"  [OK] Forward succeeded! Features: {features.shape}")

    except AssertionError as e:
        print(f"  [FAIL] RoPE assertion error!")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*60)
print("[OK] ALL TESTS PASSED!")
print("LoRA + 1008x1008 works - ready for training!")
print("="*60)
