"""
View Angle Encoder - VERSION 2 (Actually Fixed)

ROOT CAUSE:
- Frequency initialization was torch.randn() * 0.01, causing phase shifts of only 0.13π for 90° angles
- This made sin/cos encoding nearly identical for all angles
- MLP + LayerNorm then collapsed these similar inputs to identical outputs

FIX:
- Use learnable frequencies spanning [1, 64] like standard positional encoding
- Remove LayerNorm that collapses small differences
- This allows 90° → phase shift of ~100 radians → very different sin/cos patterns
"""

import torch
import torch.nn as nn
import numpy as np


class ViewAngleEncoder(nn.Module):
    """
    Encodes XA positioner angles into learnable embeddings

    Input: [LAO/RAO angle, Cranial/Caudal angle]
    Output: Spatial feature map or embedding vector
    """

    def __init__(self, embedding_dim=256, output_mode='embedding'):
        """
        Args:
            embedding_dim: Dimension of output embedding
            output_mode: 'embedding' (vector) or 'spatial' (feature map)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_mode = output_mode

        # FIXED: Use exponential frequency spacing like Transformer positional encoding
        # Frequencies from 1 to 64 (like 10000^(2i/d_model) in Transformers)
        freq_bands = embedding_dim // 4
        frequencies = torch.exp(
            torch.linspace(0, np.log(64), freq_bands)
        )
        self.angle_frequencies = nn.Parameter(frequencies, requires_grad=True)

        # MLP to process encoded angles - NO LayerNorm!
        self.angle_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Optional: project to spatial feature map
        if output_mode == 'spatial':
            self.to_spatial = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 4),
                nn.GELU(),
                nn.Linear(embedding_dim * 4, embedding_dim)
            )

    def sinusoidal_encode(self, angles):
        """
        Encode angles using sinusoidal functions

        Args:
            angles: (B, 2) tensor [primary_angle, secondary_angle] in degrees

        Returns:
            encoded: (B, embedding_dim) tensor
        """
        # Convert degrees to radians
        angles_rad = angles * (np.pi / 180.0)

        # Expand with different frequencies
        # angles_rad: (B, 2), frequencies: (freq_bands,)
        # Result: (B, 2, freq_bands)
        angles_expanded = angles_rad.unsqueeze(-1) * self.angle_frequencies.unsqueeze(0).unsqueeze(0)

        # Apply sin and cos
        sin_encoded = torch.sin(angles_expanded)  # (B, 2, freq_bands)
        cos_encoded = torch.cos(angles_expanded)  # (B, 2, freq_bands)

        # Concatenate sin and cos: (B, 2, 2*freq_bands)
        encoded = torch.cat([sin_encoded, cos_encoded], dim=-1)

        # Flatten to (B, 2 * 2*freq_bands) = (B, embedding_dim)
        encoded = encoded.reshape(encoded.shape[0], -1)

        return encoded

    def forward(self, primary_angles, secondary_angles):
        """
        Encode view angles

        Args:
            primary_angles: (B,) LAO/RAO angles in degrees (+ = LAO, - = RAO)
            secondary_angles: (B,) Cranial/Caudal angles (+ = Cranial, - = Caudal)

        Returns:
            view_embedding: (B, embedding_dim) or (B, embedding_dim, H, W)
        """
        # Stack angles
        angles = torch.stack([primary_angles, secondary_angles], dim=1)  # (B, 2)

        # Sinusoidal encoding
        encoded = self.sinusoidal_encode(angles)  # (B, embedding_dim)

        # Process with MLP
        view_embedding = self.angle_mlp(encoded)  # (B, embedding_dim)

        # Convert to spatial if needed
        if self.output_mode == 'spatial':
            view_embedding = self.to_spatial(view_embedding)

        return view_embedding


class ViewConditionedFeatureFusion(nn.Module):
    """
    Fuses view angle embeddings with SAM 3 image features using FiLM
    """

    def __init__(self, feature_dim=256, fusion_mode='film'):
        """
        Args:
            feature_dim: Dimension of SAM 3 features
            fusion_mode: 'film' or 'cross_attention'
        """
        super().__init__()

        self.fusion_mode = fusion_mode

        if fusion_mode == 'film':
            self.film_gamma = nn.Linear(feature_dim, feature_dim)
            self.film_beta = nn.Linear(feature_dim, feature_dim)

        elif fusion_mode == 'cross_attention':
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.norm = nn.LayerNorm(feature_dim)

    def forward(self, image_features, view_embedding):
        """
        Fuse view embedding with image features

        Args:
            image_features: (B, C, H, W) or (B, N, C)
            view_embedding: (B, C)

        Returns:
            fused_features: Same shape as image_features
        """
        if self.fusion_mode == 'film':
            gamma = self.film_gamma(view_embedding)
            beta = self.film_beta(view_embedding)

            if len(image_features.shape) == 4:
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
                fused_features = gamma * image_features + beta
            else:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
                fused_features = gamma * image_features + beta

        elif self.fusion_mode == 'cross_attention':
            if len(image_features.shape) == 4:
                B, C, H, W = image_features.shape
                img_seq = image_features.view(B, C, H*W).permute(0, 2, 1)
            else:
                img_seq = image_features

            view_query = view_embedding.unsqueeze(1)

            attn_out, _ = self.cross_attn(
                query=view_query,
                key=img_seq,
                value=img_seq
            )

            view_aware_feature = self.norm(attn_out.squeeze(1) + view_embedding)

            if len(image_features.shape) == 4:
                fused_features = image_features + view_aware_feature.unsqueeze(-1).unsqueeze(-1)
            else:
                fused_features = img_seq + view_aware_feature.unsqueeze(1)

        return fused_features


def test_view_encoder():
    """Test view angle encoder"""
    print("=" * 70)
    print("Testing View Angle Encoder V2 (Fixed Frequencies)")
    print("=" * 70)

    # Create encoder
    encoder = ViewAngleEncoder(embedding_dim=256, output_mode='embedding')

    print(f"\nFrequency initialization:")
    print(f"  Min: {encoder.angle_frequencies.min().item():.4f}")
    print(f"  Max: {encoder.angle_frequencies.max().item():.4f}")
    print(f"  Span: {encoder.angle_frequencies.max() / encoder.angle_frequencies.min():.2f}x")

    # Test data
    primary_angles = torch.tensor([32.92, -30.0, 45.0, 0.0])
    secondary_angles = torch.tensor([3.28, 25.0, -15.0, 0.0])

    print(f"\nTest angles:")
    for i, (p, s) in enumerate(zip(primary_angles, secondary_angles)):
        print(f"  {i}: Primary {p.item():6.2f}°, Secondary {s.item():6.2f}°")

    # Encode
    view_embeddings = encoder(primary_angles, secondary_angles)
    print(f"\nView embeddings shape: {view_embeddings.shape}")

    # Check similarity
    cos_sim_01 = torch.nn.functional.cosine_similarity(view_embeddings[0:1], view_embeddings[1:2])
    cos_sim_02 = torch.nn.functional.cosine_similarity(view_embeddings[0:1], view_embeddings[2:3])
    cos_sim_03 = torch.nn.functional.cosine_similarity(view_embeddings[0:1], view_embeddings[3:4])

    print(f"\nCosine similarity:")
    print(f"  32.92° vs -30.0°: {cos_sim_01.item():.6f}")
    print(f"  32.92° vs  45.0°: {cos_sim_02.item():.6f}")
    print(f"  32.92° vs   0.0°: {cos_sim_03.item():.6f}")

    if cos_sim_01.item() < 0.90 and cos_sim_02.item() < 0.95:
        print("\n✓ PASS: Embeddings are DIFFERENT for different angles!")
    else:
        print("\n✗ FAIL: Embeddings are still too similar!")

    # Test fusion
    print("\nTesting Feature Fusion...")
    fusion = ViewConditionedFeatureFusion(feature_dim=256, fusion_mode='film')
    image_features = torch.randn(4, 256, 64, 64)
    fused = fusion(image_features, view_embeddings)
    print(f"Fused features shape: {fused.shape}")

    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)


if __name__ == '__main__':
    test_view_encoder()
