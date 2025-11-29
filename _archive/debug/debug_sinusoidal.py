"""
Debug the sinusoidal encoding
"""
import torch
import numpy as np

embedding_dim = 256

# Recreate the sinusoidal encoder
angle_frequencies = torch.randn(embedding_dim // 4) * 0.01

print("=" * 70)
print("Debugging Sinusoidal Encoding")
print("=" * 70)

print(f"\nangle_frequencies shape: {angle_frequencies.shape}")
print(f"angle_frequencies: {angle_frequencies[:10]}")  # First 10 values

# Test angles
angles = torch.tensor([[32.92, 3.28], [-30.0, 25.0]], dtype=torch.float32)
print(f"\nInput angles: {angles}")

# Convert to radians
angles_rad = angles * (np.pi / 180.0)
print(f"Angles in radians: {angles_rad}")

# Expand
print(f"\nangle_frequencies shape: {angle_frequencies.shape}")  # (64,)
print(f"angles_rad shape: {angles_rad.shape}")  # (2, 2)

angles_expanded = angles_rad.unsqueeze(-1) * angle_frequencies.unsqueeze(0).unsqueeze(0)
print(f"angles_expanded shape: {angles_expanded.shape}")  # Should be (2, 2, 64)

# Sin/cos
sin_encoded = torch.sin(angles_expanded)
cos_encoded = torch.cos(angles_expanded)
print(f"sin_encoded shape: {sin_encoded.shape}")
print(f"cos_encoded shape: {cos_encoded.shape}")

# Concatenate
encoded = torch.cat([sin_encoded, cos_encoded], dim=-1)
print(f"encoded shape after cat: {encoded.shape}")  # (2, 2, 128)

# Flatten
encoded_flat = encoded.reshape(encoded.shape[0], -1)
print(f"encoded_flat shape: {encoded_flat.shape}")  # (2, 256)

print(f"\nencoded_flat for first angle: {encoded_flat[0, :10]}")
print(f"encoded_flat for second angle: {encoded_flat[1, :10]}")

# Check if they're the same
diff = (encoded_flat[0] - encoded_flat[1]).abs().sum()
print(f"\nAbsolute difference between encodings: {diff.item():.6f}")

if diff < 0.01:
    print("⚠️  WARNING: Encodings are nearly IDENTICAL!")
    print("This means the sinusoidal encoding is NOT working!")

    print("\n🔍 Investigating why...")
    print(f"angle_frequencies range: [{angle_frequencies.min():.6f}, {angle_frequencies.max():.6f}]")
    print(f"angle_frequencies mean: {angle_frequencies.mean():.6f}")
    print(f"angle_frequencies std: {angle_frequencies.std():.6f}")

    if angle_frequencies.std() < 0.001:
        print("\n❌ PROBLEM: Frequencies have very low variance!")
        print("   Initialized with randn() * 0.01, which is TOO SMALL")
        print("   All frequencies are ~0, so sin(angle * 0) ≈ 0 for all angles")
else:
    print("✅ Encodings are different, sinusoidal encoding works")
