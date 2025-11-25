"""
Test if raw sinusoidal encoding (without MLP) preserves differences
"""
import torch
import numpy as np

embedding_dim = 256

# Just the sinusoidal part, no MLP
angle_frequencies = torch.randn(embedding_dim // 4) * 0.1

def sinusoidal_encode(angles):
    angles_rad = angles * (np.pi / 180.0)
    angles_expanded = angles_rad.unsqueeze(-1) * angle_frequencies.unsqueeze(0).unsqueeze(0)
    sin_encoded = torch.sin(angles_expanded)
    cos_encoded = torch.cos(angles_expanded)
    encoded = torch.cat([sin_encoded, cos_encoded], dim=-1)
    encoded = encoded.reshape(encoded.shape[0], -1)
    return encoded

# Test angles
primary_angles = torch.tensor([32.92, -30.0, 45.0, 0.0])
secondary_angles = torch.tensor([3.28, 25.0, -15.0, 0.0])
angles = torch.stack([primary_angles, secondary_angles], dim=1)

# Encode
encoded = sinusoidal_encode(angles)

print("RAW Sinusoidal Encoding (no MLP):")
print(f"Shape: {encoded.shape}")

cos_sim_01 = torch.nn.functional.cosine_similarity(encoded[0:1], encoded[1:2])
cos_sim_02 = torch.nn.functional.cosine_similarity(encoded[0:1], encoded[2:3])
cos_sim_03 = torch.nn.functional.cosine_similarity(encoded[0:1], encoded[3:4])

print(f"\nCosine similarity:")
print(f"  32.92 vs -30.0: {cos_sim_01.item():.6f}")
print(f"  32.92 vs  45.0: {cos_sim_02.item():.6f}")
print(f"  32.92 vs   0.0: {cos_sim_03.item():.6f}")

if cos_sim_01.item() < 0.95:
    print("\nGOOD: Raw sinusoidal encoding preserves differences!")
else:
    print("\nBAD: Even raw sinusoidal encoding is too similar!")

    # Check frequency magnitude
    print(f"\nFrequency stats:")
    print(f"  Mean: {angle_frequencies.mean():.6f}")
    print(f"  Std: {angle_frequencies.std():.6f}")
    print(f"  Range: [{angle_frequencies.min():.6f}, {angle_frequencies.max():.6f}]")

    # Check if frequencies are too similar
    print(f"\nAngle differences:")
    print(f"  32.92 - (-30.0) = {32.92 - (-30.0):.2f}°")
    print(f"  32.92 - 45.0 = {32.92 - 45.0:.2f}°")

    # Max frequency effect
    max_angle_rad = 90 * np.pi / 180
    max_freq = angle_frequencies.abs().max()
    print(f"\nMax phase shift: {max_angle_rad * max_freq:.6f} radians")
    print(f"This is: {(max_angle_rad * max_freq) / np.pi:.6f} * pi")

    if (max_angle_rad * max_freq) < 0.5:
        print("\nPROBLEM: Frequencies too small!")
        print("Even 90° angle difference causes < 0.5 radian phase shift")
        print("Need to increase frequency initialization much more!")
