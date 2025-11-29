"""
Minimal test to isolate the LoRA + DataParallel + RoPE issue
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, r"C:\Users\odressler\sam3")

from sam3_lora_wrapper import SAM3WithLoRA

print("=" * 70)
print("Minimal LoRA + DataParallel Test")
print("=" * 70)

# Build wrapper
print("\n[1/4] Building SAM 3 with LoRA...")
wrapper = SAM3WithLoRA(lora_r=16, lora_alpha=32, use_multi_gpu=False).cuda()
print("  Model built")

# Test WITHOUT DataParallel first
print("\n[2/4] Testing forward WITHOUT DataParallel...")
test_img = torch.randn(2, 3, 1008, 1008).cuda()
test_img_norm = (test_img - 0.5) / 0.5

try:
    with torch.no_grad():
        out = wrapper(test_img_norm)
    print(f"  ✓ SUCCESS! Output shape: {out.shape}")
except AssertionError as e:
    print(f"  ✗ FAILED with RoPE assertion (even without DataParallel!)")
    print(f"  This means the issue is NOT DataParallel-specific")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now test WITH DataParallel
if torch.cuda.device_count() > 1:
    print(f"\n[3/4] Wrapping with DataParallel ({torch.cuda.device_count()} GPUs)...")
    wrapper_dp = nn.DataParallel(wrapper)
    print("  DataParallel applied")

    print("\n[4/4] Testing forward WITH DataParallel...")
    test_img2 = torch.randn(4, 3, 1008, 1008).cuda()  # Larger batch
    test_img2_norm = (test_img2 - 0.5) / 0.5

    try:
        with torch.no_grad():
            out = wrapper_dp(test_img2_norm)
        print(f"  ✓ SUCCESS! Output shape: {out.shape}")
        print(f"\n✓ Both tests passed! LoRA + DataParallel works!")
    except AssertionError as e:
        print(f"  ✗ FAILED with RoPE assertion")
        print(f"  The issue IS DataParallel-specific")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("\n[3/4] Only 1 GPU detected, skipping DataParallel test")

print("\n" + "=" * 70)
