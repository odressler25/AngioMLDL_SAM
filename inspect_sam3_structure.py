"""
Quick script to inspect SAM 3 model structure
"""
import torch
from sam3.model_builder import build_sam3_image_model

print("Building SAM 3 model...")
model = build_sam3_image_model(device='cpu')

print("\n" + "="*70)
print("SAM 3 Model Structure")
print("="*70)

print(f"\nModel type: {type(model)}")
print(f"Model class: {model.__class__.__name__}")

print("\nTop-level attributes:")
for attr in dir(model):
    if not attr.startswith('_'):
        try:
            val = getattr(model, attr)
            if isinstance(val, torch.nn.Module):
                print(f"  - {attr}: {type(val).__name__}")
        except:
            pass

print("\nModel structure (first level):")
for name, module in model.named_children():
    print(f"  - {name}: {type(module).__name__}")

print("\nModel structure (second level):")
for name, module in model.named_modules():
    if name and '.' not in name[1:]:  # Only first-level children
        print(f"  - {name}: {type(module).__name__}")

print("\nSearching for 'encoder' in attribute names:")
for attr in dir(model):
    if 'encoder' in attr.lower():
        print(f"  - {attr}")

print("\nSearching for 'backbone' in attribute names:")
for attr in dir(model):
    if 'backbone' in attr.lower():
        print(f"  - {attr}")
        try:
            backbone = getattr(model, attr)
            print(f"    Type: {type(backbone)}")
            if isinstance(backbone, torch.nn.Module):
                print(f"    Children:")
                for name, child in backbone.named_children():
                    print(f"      - {name}: {type(child).__name__}")
        except:
            pass

print("\n" + "="*70)
print("Checking if we can do a forward pass")
print("="*70)

# Try a dummy forward pass
try:
    dummy_image = torch.randn(1, 3, 1024, 1024)
    print(f"\nInput shape: {dummy_image.shape}")

    # Try direct forward
    output = model(dummy_image)
    print(f"\nDirect forward() works!")
    print(f"Output type: {type(output)}")
    if isinstance(output, dict):
        print(f"Output keys: {output.keys()}")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
    elif isinstance(output, torch.Tensor):
        print(f"Output shape: {output.shape}")

except Exception as e:
    print(f"\nDirect forward() failed: {e}")

print("\n" + "="*70)
