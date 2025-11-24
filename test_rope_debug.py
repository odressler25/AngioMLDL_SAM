"""
Debug script to understand RoPE error
"""
import torch
import sys
sys.path.insert(0, r"C:\Users\odressler\sam3")

from sam3 import build_sam3_image_model

print("=" * 70)
print("Testing SAM 3 RoPE with different resolutions")
print("=" * 70)

# Test resolutions
test_resolutions = [1008, 1024, 1016, 512]

for res in test_resolutions:
    print(f"\nTesting {res}x{res}...")

    try:
        # Build model
        model = build_sam3_image_model('cuda')
        model.eval()

        # Create test image
        img = torch.randn(1, 3, res, res).cuda()

        # Forward pass
        with torch.no_grad():
            out = model.backbone.forward_image(img)

        # Get shape
        if isinstance(out, dict):
            shape = out.get('vision_features', out.get('image_embeddings')).shape
        else:
            shape = out.shape

        print(f"  ✓ SUCCESS! Output shape: {shape}")

        # Print RoPE info if we can access it
        try:
            for name, module in model.named_modules():
                if hasattr(module, 'freqs_cis'):
                    print(f"  RoPE cache shape: {module.freqs_cis.shape}")
                    break
        except:
            pass

    except AssertionError as e:
        print(f"  ✗ FAILED with RoPE assertion")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

    # Clean up
    del model
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
