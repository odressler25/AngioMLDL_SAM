"""
Inspect SAM 3 backbone structure
"""
import torch
from sam3.model_builder import build_sam3_image_model

print("Building SAM 3 model...")
model = build_sam3_image_model(device='cpu')

print("\n" + "="*70)
print("SAM 3 Backbone Structure")
print("="*70)

backbone = model.backbone
print(f"\nBackbone type: {type(backbone)}")
print(f"Backbone class: {backbone.__class__.__name__}")

print("\nBackbone children:")
for name, module in backbone.named_children():
    print(f"  - {name}: {type(module).__name__}")

print("\nVision Backbone:")
vision_backbone = backbone.vision_backbone
print(f"  Type: {type(vision_backbone)}")
print(f"  Class: {vision_backbone.__class__.__name__}")

print("\n  Vision Backbone children:")
for name, module in vision_backbone.named_children():
    print(f"    - {name}: {type(module).__name__}")

print("\nSearching for 'trunk' or 'encoder' in vision_backbone:")
for attr in dir(vision_backbone):
    if 'trunk' in attr.lower() or 'encoder' in attr.lower():
        print(f"  - {attr}")
        try:
            val = getattr(vision_backbone, attr)
            if isinstance(val, torch.nn.Module):
                print(f"    Type: {type(val).__name__}")
        except:
            pass

print("\n" + "="*70)
print("Testing Forward Pass Through Backbone")
print("="*70)

try:
    # Try with just an image tensor
    dummy_image = torch.randn(1, 3, 1024, 1024)
    print(f"\nInput shape: {dummy_image.shape}")

    # Try backbone forward
    print("\nTrying: backbone(dummy_image)")
    output = backbone(dummy_image)
    print(f"Success! Output type: {type(output)}")

    if isinstance(output, dict):
        print(f"Output keys: {output.keys()}")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
    elif isinstance(output, torch.Tensor):
        print(f"Output shape: {output.shape}")

except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\nTrying: backbone.vision_backbone(dummy_image)")
    output = vision_backbone(dummy_image)
    print(f"Success! Output type: {type(output)}")

    if isinstance(output, dict):
        print(f"Output keys: {output.keys()}")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            elif isinstance(v, list):
                print(f"  {k}: list of {len(v)} items")
                if v and isinstance(v[0], torch.Tensor):
                    print(f"    First item shape: {v[0].shape}")
    elif isinstance(output, torch.Tensor):
        print(f"Output shape: {output.shape}")
    elif isinstance(output, (list, tuple)):
        print(f"Output is {type(output).__name__} with {len(output)} items")
        for i, item in enumerate(output):
            if isinstance(item, torch.Tensor):
                print(f"  Item {i}: {item.shape}")

except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
