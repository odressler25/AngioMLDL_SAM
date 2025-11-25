"""
Debug why MLP makes all outputs identical
"""
import torch
import torch.nn as nn
import numpy as np

embedding_dim = 256

# Recreate the full encoder
class ViewAngleEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.angle_frequencies = nn.Parameter(
            torch.randn(embedding_dim // 4) * 0.01,
            requires_grad=True
        )

        self.angle_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def sinusoidal_encode(self, angles):
        angles_rad = angles * (np.pi / 180.0)
        angles_expanded = angles_rad.unsqueeze(-1) * self.angle_frequencies.unsqueeze(0).unsqueeze(0)
        sin_encoded = torch.sin(angles_expanded)
        cos_encoded = torch.cos(angles_expanded)
        encoded = torch.cat([sin_encoded, cos_encoded], dim=-1)
        encoded = encoded.reshape(encoded.shape[0], -1)
        return encoded

    def forward(self, primary_angles, secondary_angles):
        angles = torch.stack([primary_angles, secondary_angles], dim=1)
        encoded = self.sinusoidal_encode(angles)
        view_embedding = self.angle_mlp(encoded)
        return view_embedding, encoded  # Return both for debugging

print("=" * 70)
print("Debugging MLP Collapse")
print("=" * 70)

encoder = ViewAngleEncoder(embedding_dim=256)

# Test angles
primary_angles = torch.tensor([32.92, -30.0, 45.0], dtype=torch.float32)
secondary_angles = torch.tensor([3.28, 25.0, -15.0], dtype=torch.float32)

# Forward pass
view_embeddings, encoded_features = encoder(primary_angles, secondary_angles)

print("\n1. Raw sinusoidal encoding (before MLP):")
print(f"   Shape: {encoded_features.shape}")
print(f"   Sample 0: {encoded_features[0, :5]}")
print(f"   Sample 1: {encoded_features[1, :5]}")
print(f"   Difference (0 vs 1): {(encoded_features[0] - encoded_features[1]).abs().sum():.4f}")

print("\n2. After MLP:")
print(f"   Shape: {view_embeddings.shape}")
print(f"   Sample 0: {view_embeddings[0, :5]}")
print(f"   Sample 1: {view_embeddings[1, :5]}")
print(f"   Difference (0 vs 1): {(view_embeddings[0] - view_embeddings[1]).abs().sum():.4f}")

# Compute cosine similarity
cos_sim = nn.functional.cosine_similarity(view_embeddings[0:1], view_embeddings[1:2])
print(f"\n3. Cosine similarity between samples 0 and 1: {cos_sim.item():.6f}")

if cos_sim.item() > 0.99:
    print("\n   Problem: MLP output is nearly IDENTICAL for different inputs!")
    print("\n   Possible reasons:")
    print("   1. Input magnitude too small (frequencies = 0.01 scale)")
    print("   2. LayerNorm collapses small differences")
    print("   3. Random initialization makes MLP ignore small inputs")

    print(f"\n   Input magnitude:")
    print(f"   - Encoded features norm: {encoded_features.norm(dim=1)}")
    print(f"   - Encoded features mean: {encoded_features.mean():.6f}")
    print(f"   - Encoded features std: {encoded_features.std():.6f}")

    print(f"\n   First Linear layer weights:")
    first_linear = encoder.angle_mlp[0]  # First Linear layer
    print(f"   - Weight shape: {first_linear.weight.shape}")
    print(f"   - Weight mean: {first_linear.weight.mean():.6f}")
    print(f"   - Weight std: {first_linear.weight.std():.6f}")

    # Compute first layer output
    first_out = first_linear(encoded_features)
    print(f"\n   After first Linear:")
    print(f"   - Output range: [{first_out.min():.4f}, {first_out.max():.4f}]")
    print(f"   - Output diff (0 vs 1): {(first_out[0] - first_out[1]).abs().sum():.4f}")

    # After LayerNorm
    layernorm = encoder.angle_mlp[1]
    normed = layernorm(first_out)
    print(f"\n   After LayerNorm:")
    print(f"   - Output range: [{normed.min():.4f}, {normed.max():.4f}]")
    print(f"   - Output mean: {normed.mean():.6f}")
    print(f"   - Output std: {normed.std():.6f}")
    print(f"   - Output diff (0 vs 1): {(normed[0] - normed[1]).abs().sum():.4f}")

    cos_sim_normed = nn.functional.cosine_similarity(normed[0:1], normed[1:2])
    print(f"   - Cosine similarity: {cos_sim_normed.item():.6f}")

    if cos_sim_normed.item() > 0.99:
        print("\n   ROOT CAUSE: LayerNorm is collapsing the differences!")
        print("   When inputs are very small (0.01 scale), LayerNorm normalizes")
        print("   to mean=0, std=1, which removes most of the angle information.")
