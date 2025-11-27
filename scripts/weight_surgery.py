"""
Weight Surgery: Convert Custom Phase 1 checkpoint to Native SAM3 format.

The Custom checkpoint has keys like: sam3.backbone.vision_backbone...
The Native SAM3 expects keys like: backbone.vision_backbone...

This script:
1. Loads the custom checkpoint
2. Strips the 'sam3.' prefix from backbone keys
3. Discards custom heads (seg_head, view_encoder, feature_fusion)
4. Saves in a format the Native trainer can load

Usage:
    python scripts/weight_surgery.py
    python scripts/weight_surgery.py --input checkpoints/phase1_deepsa_best.pth --output checkpoints/phase1_native_format.pth
"""

import argparse
import torch
from pathlib import Path


def convert_custom_to_native(input_path: str, output_path: str):
    """Convert custom checkpoint to native SAM3 format."""

    print("=" * 60)
    print("Weight Surgery: Custom -> Native SAM3 Format")
    print("=" * 60)

    # Load custom checkpoint
    print(f"\nLoading: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        val_dice = checkpoint.get('val_dice', 'unknown')
        print(f"  Epoch: {epoch}")
        print(f"  Val Dice: {val_dice}")
    else:
        state_dict = checkpoint
        epoch = 'unknown'
        val_dice = 'unknown'

    print(f"  Total keys: {len(state_dict)}")

    # Categorize keys
    sam3_backbone_keys = []
    custom_head_keys = []
    other_keys = []

    for key in state_dict.keys():
        if key.startswith('sam3.'):
            sam3_backbone_keys.append(key)
        elif key.startswith(('seg_head.', 'view_encoder.', 'feature_fusion.', 'feature_proj.')):
            custom_head_keys.append(key)
        else:
            other_keys.append(key)

    print(f"\n  SAM3 backbone keys: {len(sam3_backbone_keys)}")
    print(f"  Custom head keys: {len(custom_head_keys)} (will be discarded)")
    print(f"  Other keys: {len(other_keys)}")

    # Convert keys
    print("\nPerforming surgery...")
    native_state_dict = {}

    for key in sam3_backbone_keys:
        # Strip 'sam3.' prefix
        new_key = key[5:]  # Remove 'sam3.'
        native_state_dict[new_key] = state_dict[key]

    print(f"  Converted {len(native_state_dict)} keys")

    # Show sample conversions
    print("\nSample key conversions:")
    samples = list(zip(sam3_backbone_keys[:5], [k[5:] for k in sam3_backbone_keys[:5]]))
    for old, new in samples:
        print(f"  {old[:50]}...")
        print(f"    -> {new[:50]}...")

    # Discarded keys
    print(f"\nDiscarded custom heads:")
    for key in custom_head_keys[:5]:
        print(f"  - {key}")
    if len(custom_head_keys) > 5:
        print(f"  ... and {len(custom_head_keys) - 5} more")

    # Save in native format
    # The native trainer expects either:
    # 1. A state_dict directly
    # 2. A checkpoint with 'model' key
    native_checkpoint = {
        'model': native_state_dict,
        'source': 'weight_surgery',
        'original_checkpoint': str(input_path),
        'original_epoch': epoch,
        'original_val_dice': val_dice,
    }

    print(f"\nSaving: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(native_checkpoint, output_path)

    # Verify file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("Surgery complete!")
    print("=" * 60)
    print(f"\nTo use in training config:")
    print(f"  checkpoint:")
    print(f"    resume_from: {output_path}")

    return native_state_dict


def verify_compatibility(native_state_dict: dict):
    """Verify the converted checkpoint is compatible with native SAM3."""
    import sys
    sys.path.insert(0, 'C:/Users/odressler/sam3')

    print("\nVerifying compatibility with native SAM3...")

    try:
        from sam3.model_builder import build_sam3_image_model
        model = build_sam3_image_model(device='cpu', eval_mode=True)
        model_keys = set(model.state_dict().keys())
        converted_keys = set(native_state_dict.keys())

        # Check overlap
        matching = model_keys & converted_keys
        missing_in_converted = model_keys - converted_keys
        extra_in_converted = converted_keys - model_keys

        print(f"  Native model keys: {len(model_keys)}")
        print(f"  Converted keys: {len(converted_keys)}")
        print(f"  Matching: {len(matching)}")
        print(f"  Missing in converted: {len(missing_in_converted)}")
        print(f"  Extra in converted: {len(extra_in_converted)}")

        if missing_in_converted:
            print("\n  Missing keys (will use pretrained):")
            for k in list(missing_in_converted)[:5]:
                print(f"    - {k}")
            if len(missing_in_converted) > 5:
                print(f"    ... and {len(missing_in_converted) - 5} more")

        # Try loading
        print("\n  Attempting to load weights...")
        missing, unexpected = model.load_state_dict(native_state_dict, strict=False)
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        print("  SUCCESS - Checkpoint is compatible!")

    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert custom checkpoint to native SAM3 format')
    parser.add_argument('--input', type=str,
                        default='checkpoints/phase1_deepsa_best.pth',
                        help='Path to custom checkpoint')
    parser.add_argument('--output', type=str,
                        default='checkpoints/phase1_native_format.pth',
                        help='Output path for native format checkpoint')
    parser.add_argument('--verify', action='store_true',
                        help='Verify compatibility with native SAM3')
    args = parser.parse_args()

    native_state_dict = convert_custom_to_native(args.input, args.output)

    if args.verify:
        verify_compatibility(native_state_dict)


if __name__ == '__main__':
    main()
