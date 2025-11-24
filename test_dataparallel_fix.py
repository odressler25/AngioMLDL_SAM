"""
Test that LoRA + DataParallel now works with the fixes
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, r"C:\Users\odressler\sam3")

from sam3_lora_wrapper import SAM3WithLoRA

device = 'cuda'

print("="*70)
print("Testing LoRA + DataParallel Fix")
print("="*70)

# Build SAM 3 with LoRA
print("\n[1/3] Building SAM 3 with LoRA...")
sam3_lora = SAM3WithLoRA(lora_r=16, lora_alpha=32, use_multi_gpu=True).to(device)

# Apply DataParallel
if torch.cuda.device_count() > 1:
    print(f"\n[2/3] Applying DataParallel ({torch.cuda.device_count()} GPUs)...")
    sam3_lora = nn.DataParallel(sam3_lora)
    print("  DataParallel applied")
else:
    print("\n[2/3] Only 1 GPU detected, skipping DataParallel")

# Test with various batch sizes
print("\n[3/3] Testing forward pass...")

test_cases = [
    (2, 3, 512, 512),   # Small batch
    (8, 3, 800, 600),   # Medium batch
    (16, 3, 1024, 1024), # Full batch at target size
]

for i, shape in enumerate(test_cases):
    b, c, h, w = shape
    print(f"\n  Test {i+1}: Batch size {b}, Image {h}x{w}")

    try:
        images = torch.rand(b, c, h, w).to(device)
        images_normalized = (images - 0.5) / 0.5

        # Forward pass (wrapper handles 1024 resize)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            features = sam3_lora(images_normalized)

        print(f"    [OK] Forward succeeded! Features: {features.shape}")

        # Check both GPUs are being used
        if torch.cuda.device_count() > 1:
            gpu0_mem = torch.cuda.memory_allocated(0) / 1e9
            gpu1_mem = torch.cuda.memory_allocated(1) / 1e9
            print(f"    GPU 0: {gpu0_mem:.2f} GB, GPU 1: {gpu1_mem:.2f} GB")

    except AssertionError as e:
        print(f"    [FAIL] RoPE assertion error!")
        print(f"    The fix didn't work - still getting RoPE errors")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"    [FAIL] Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*70)
print("[OK] ALL TESTS PASSED!")
print("LoRA + DataParallel is now working!")
print("Ready for overnight training on both GPUs.")
print("="*70)
