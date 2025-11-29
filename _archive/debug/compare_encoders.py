"""
Compare continuous vs categorical view encoders on same data
"""
import torch
import sys
sys.path.insert(0, r"C:\Users\odressler\sam3")

from view_angle_encoder import ViewAngleEncoder, ViewConditionedFeatureFusion
import numpy as np

# Import categorical encoder from the training script
PRIMARY_BINS = [-45, -30, -15, 0, 15, 30, 45, 60, 90]
SECONDARY_BINS = [-45, -30, -15, 0, 15, 30, 45]

def discretize_angle(angle, bins):
    bins = np.array(bins)
    idx = np.argmin(np.abs(bins - angle))
    return idx

class CategoricalViewEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.primary_embeddings = torch.nn.Embedding(len(PRIMARY_BINS), embedding_dim // 2)
        self.secondary_embeddings = torch.nn.Embedding(len(SECONDARY_BINS), embedding_dim // 2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.LayerNorm(embedding_dim)
        )

    def forward(self, primary_indices, secondary_indices):
        primary_emb = self.primary_embeddings(primary_indices)
        secondary_emb = self.secondary_embeddings(secondary_indices)
        combined = torch.cat([primary_emb, secondary_emb], dim=1)
        return self.mlp(combined)

print("=" * 70)
print("Comparing Continuous vs Categorical View Encoders")
print("=" * 70)

# Test angles (from real data)
test_angles = [
    (32.92, 3.28),   # Sample from dataset
    (-30.0, 25.0),   # RAO 30, Cranial 25
    (45.0, -15.0),   # LAO 45, Caudal 15
    (0.0, 0.0)       # Straight on
]

print("\nTest angles:")
for i, (p, s) in enumerate(test_angles):
    print(f"  {i+1}. Primary: {p:6.2f}°, Secondary: {s:6.2f}°")

# Create encoders
continuous_encoder = ViewAngleEncoder(embedding_dim=256)
categorical_encoder = CategoricalViewEncoder(embedding_dim=256)

# Test continuous
print("\n" + "=" * 70)
print("CONTINUOUS ENCODER")
print("=" * 70)

primary_continuous = torch.tensor([p for p, s in test_angles], dtype=torch.float32)
secondary_continuous = torch.tensor([s for p, s in test_angles], dtype=torch.float32)

continuous_emb = continuous_encoder(primary_continuous, secondary_continuous)
print(f"Output shape: {continuous_emb.shape}")
print(f"Output range: [{continuous_emb.min():.3f}, {continuous_emb.max():.3f}]")
print(f"Output mean: {continuous_emb.mean():.3f}")
print(f"Output std: {continuous_emb.std():.3f}")

# Test categorical
print("\n" + "=" * 70)
print("CATEGORICAL ENCODER")
print("=" * 70)

primary_indices = torch.tensor([discretize_angle(p, PRIMARY_BINS) for p, s in test_angles], dtype=torch.long)
secondary_indices = torch.tensor([discretize_angle(s, SECONDARY_BINS) for p, s in test_angles], dtype=torch.long)

print("Discretized indices:")
for i, (p_idx, s_idx) in enumerate(zip(primary_indices, secondary_indices)):
    p_bin = PRIMARY_BINS[p_idx]
    s_bin = SECONDARY_BINS[s_idx]
    print(f"  {i+1}. Primary: {p_bin}° (idx {p_idx}), Secondary: {s_bin}° (idx {s_idx})")

categorical_emb = categorical_encoder(primary_indices, secondary_indices)
print(f"\nOutput shape: {categorical_emb.shape}")
print(f"Output range: [{categorical_emb.min():.3f}, {categorical_emb.max():.3f}]")
print(f"Output mean: {categorical_emb.mean():.3f}")
print(f"Output std: {categorical_emb.std():.3f}")

# Compare embedding norms
print("\n" + "=" * 70)
print("EMBEDDING COMPARISON")
print("=" * 70)

continuous_norms = torch.norm(continuous_emb, dim=1)
categorical_norms = torch.norm(categorical_emb, dim=1)

print("\nEmbedding norms:")
print(f"  Continuous: {continuous_norms}")
print(f"  Categorical: {categorical_norms}")

# Check similarity between close angles
print("\nSimilarity between angles 1 and 2 (32.92° vs -30°):")
cos_sim_continuous = torch.nn.functional.cosine_similarity(
    continuous_emb[0:1], continuous_emb[1:2]
)
cos_sim_categorical = torch.nn.functional.cosine_similarity(
    categorical_emb[0:1], categorical_emb[1:2]
)
print(f"  Continuous: {cos_sim_continuous.item():.4f}")
print(f"  Categorical: {cos_sim_categorical.item():.4f}")

print("\nSimilarity between angles 1 and 3 (32.92° vs 45°):")
cos_sim_continuous_13 = torch.nn.functional.cosine_similarity(
    continuous_emb[0:1], continuous_emb[2:3]
)
cos_sim_categorical_13 = torch.nn.functional.cosine_similarity(
    categorical_emb[0:1], categorical_emb[2:3]
)
print(f"  Continuous: {cos_sim_continuous_13.item():.4f}")
print(f"  Categorical: {cos_sim_categorical_13.item():.4f}")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print("\nKey observations:")
print("1. Continuous encoder uses sinusoidal positional encoding")
print("   - Inherently captures angle similarity")
print("   - 32.92° and 30° should have high similarity")
print()
print("2. Categorical encoder uses random initialized embeddings")
print("   - Must learn similarity from data")
print("   - With 63 categories and 748 samples, this is SPARSE")
print()
print("3. The categorical embeddings start random, so:")
print("   - Similar angles (30° vs 32.92°) may have very different embeddings")
print("   - Model must learn from scratch that nearby bins are similar")
print("   - This requires many examples per bin")
