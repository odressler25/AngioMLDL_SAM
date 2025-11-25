"""
View Angle Encoder - FIXED VERSION

BUGS FIXED:
1. Removed LayerNorm that was collapsing all embeddings to identical values
2. Increased frequency initialization from 0.01 to 0.1 for better separation
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

        # Sinusoidal encoding for continuous angles
        # FIXED: Increased initialization from 0.01 to 0.1 for better separation
        self.angle_frequencies = nn.Parameter(
            torch.randn(embedding_dim // 4) * 0.1,  # Was 0.01
            requires_grad=True
        )

        # MLP to process encoded angles
        # FIXED: Removed LayerNorm which was collapsing embeddings
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
        angles_expanded = angles_rad.unsqueeze(-1) * self.angle_frequencies.unsqueeze(0).unsqueeze(0)

        # Apply sin and cos
        sin_encoded = torch.sin(angles_expanded)  # (B, 2, embedding_dim // 4)
        cos_encoded = torch.cos(angles_expanded)  # (B, 2, embedding_dim // 4)

        # Concatenate sin and cos
        encoded = torch.cat([sin_encoded, cos_encoded], dim=-1)  # (B, 2, embedding_dim // 2)

        # Flatten to (B, embedding_dim)
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
    Fuses view angle embeddings with SAM 3 image features

    Two fusion strategies:
    1. Feature-wise Linear Modulation (FiLM) - scales and shifts features
    2. Cross-attention - view embedding attends to image features
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
            # Feature-wise Linear Modulation
            self.film_gamma = nn.Linear(feature_dim, feature_dim)
            self.film_beta = nn.Linear(feature_dim, feature_dim)

        elif fusion_mode == 'cross_attention':
            # Cross-attention: view embedding as query, image features as key/value
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
            image_features: (B, C, H, W) or (B, N, C) image features from SAM 3
            view_embedding: (B, C) view angle embedding

        Returns:
            fused_features: Same shape as image_features
        """
        if self.fusion_mode == 'film':
            # FiLM: gamma * features + beta
            gamma = self.film_gamma(view_embedding)  # (B, C)
            beta = self.film_beta(view_embedding)    # (B, C)

            if len(image_features.shape) == 4:  # (B, C, H, W)
                # Broadcast to spatial dimensions
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
                fused_features = gamma * image_features + beta

            else:  # (B, N, C) sequence format
                gamma = gamma.unsqueeze(1)  # (B, 1, C)
                beta = beta.unsqueeze(1)    # (B, 1, C)
                fused_features = gamma * image_features + beta

        elif self.fusion_mode == 'cross_attention':
            # Cross-attention fusion
            if len(image_features.shape) == 4:  # (B, C, H, W)
                B, C, H, W = image_features.shape
                # Reshape to sequence
                img_seq = image_features.view(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
            else:
                img_seq = image_features  # (B, N, C)

            # View embedding as query
            view_query = view_embedding.unsqueeze(1)  # (B, 1, C)

            # Attend to image features
            attn_out, _ = self.cross_attn(
                query=view_query,
                key=img_seq,
                value=img_seq
            )  # (B, 1, C)

            # Add residual and broadcast back to image shape
            view_aware_feature = self.norm(attn_out.squeeze(1) + view_embedding)  # (B, C)

            if len(image_features.shape) == 4:
                # Broadcast to spatial
                fused_features = image_features + view_aware_feature.unsqueeze(-1).unsqueeze(-1)
            else:
                fused_features = img_seq + view_aware_feature.unsqueeze(1)

        return fused_features


def test_view_encoder():
    """Test view angle encoder"""
    print("Testing FIXED View Angle Encoder...")

    # Create encoder
    encoder = ViewAngleEncoder(embedding_dim=256, output_mode='embedding')

    # Test data
    primary_angles = torch.tensor([32.92, -30.0, 45.0, 0.0])
    secondary_angles = torch.tensor([3.28, 25.0, -15.0, 0.0])

    # Encode
    view_embeddings = encoder(primary_angles, secondary_angles)
    print(f"View embeddings shape: {view_embeddings.shape}")

    # Check similarity
    cos_sim_01 = torch.nn.functional.cosine_similarity(view_embeddings[0:1], view_embeddings[1:2])
    cos_sim_02 = torch.nn.functional.cosine_similarity(view_embeddings[0:1], view_embeddings[2:3])
    print(f"\nCosine similarity (32.92° vs -30°): {cos_sim_01.item():.4f}")
    print(f"Cosine similarity (32.92° vs 45°): {cos_sim_02.item():.4f}")

    if cos_sim_01.item() < 0.99 and cos_sim_02.item() < 0.99:
        print("\nPASS: Embeddings are DIFFERENT for different angles!")
    else:
        print("\nFAIL: Embeddings are still too similar!")

    # Test fusion
    print("\nTesting Feature Fusion...")
    fusion = ViewConditionedFeatureFusion(feature_dim=256, fusion_mode='film')
    image_features = torch.randn(4, 256, 64, 64)
    fused = fusion(image_features, view_embeddings)
    print(f"Fused features shape: {fused.shape}")

    print("\n[DONE] Tests complete!")


if __name__ == '__main__':
    test_view_encoder()
