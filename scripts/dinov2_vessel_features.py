"""
DINOv2 Feature Extraction for Coronary Angiography.

Based on the paper "Leveraging Diffusion Model and Image Foundation Model
for Improved Correspondence Matching in Coronary Angiography"

This script:
1. Extracts DINOv2 features from angiogram images
2. Visualizes features using 3-component PCA (like paper Figure 5)
3. Overlays skeleton and bifurcations to see semantic differentiation
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize, binary_opening, disk as morph_disk
from scipy import ndimage
from scipy.ndimage import label

# Paths
CASS_IMAGES = Path("E:/AngioMLDL_data/coco_cass_segments/train/images")
DEEPSA_MASKS = Path("E:/AngioMLDL_data/deepsa_pseudo_labels")
OUTPUT_DIR = Path("E:/AngioMLDL_data/dinov2_features")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def load_dinov2_model():
    """Load DINOv2 model from torch hub."""
    print("Loading DINOv2 model...")
    # Use the base model with registers (as recommended in the paper)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    model = model.to(DEVICE)
    model.eval()
    print("DINOv2 loaded successfully!")
    return model


def extract_features(model, image, target_size=518):
    """
    Extract DINOv2 features from an image.

    Args:
        model: DINOv2 model
        image: PIL Image or numpy array
        target_size: Size to resize image (must be divisible by 14 for ViT-14)

    Returns:
        features: (H, W, D) feature map
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to target size (divisible by 14)
    orig_size = image.size
    image = image.resize((target_size, target_size), Image.BILINEAR)

    # Convert to tensor and normalize (ImageNet stats)
    img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # Add batch dim

    with torch.no_grad():
        # Get intermediate features (patch tokens)
        features = model.forward_features(img_tensor)
        patch_tokens = features['x_norm_patchtokens']  # (B, N, D)

    # Reshape to spatial grid
    # ViT-14 with 518 input -> 37x37 patches
    B, N, D = patch_tokens.shape
    h = w = int(np.sqrt(N))

    feature_map = patch_tokens.reshape(B, h, w, D)
    feature_map = feature_map.squeeze(0).cpu().numpy()  # (H, W, D)

    return feature_map, orig_size


def pca_features(features, n_components=3):
    """
    Apply PCA to reduce feature dimensions for visualization.

    Args:
        features: (H, W, D) feature map
        n_components: Number of PCA components (3 for RGB visualization)

    Returns:
        pca_result: (H, W, 3) RGB-like visualization
    """
    H, W, D = features.shape

    # Flatten for PCA
    flat_features = features.reshape(-1, D)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(flat_features)

    # Reshape back to spatial
    pca_result = pca_result.reshape(H, W, n_components)

    # Normalize to [0, 1] for visualization
    pca_min = pca_result.min(axis=(0, 1), keepdims=True)
    pca_max = pca_result.max(axis=(0, 1), keepdims=True)
    pca_result = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)

    return pca_result


def clean_vessel_mask(mask):
    """Clean up vessel mask to reduce noise."""
    labeled, num_features = label(mask)
    if num_features == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    max_size = max(sizes) if len(sizes) > 0 else 0
    min_size = max(max_size * 0.01, 100)

    cleaned = np.zeros_like(mask)
    for i, size in enumerate(sizes, 1):
        if size >= min_size:
            cleaned[labeled == i] = 1

    cleaned = binary_opening(cleaned, morph_disk(2))
    return cleaned


def get_skeleton_and_bifurcations(mask):
    """Get skeleton and bifurcation points from vessel mask."""
    cleaned = clean_vessel_mask(mask)
    skeleton = skeletonize(cleaned)

    # Find bifurcations
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)
    bifurcation_mask = (neighbor_count >= 3) & skeleton

    bifurcation_coords = np.where(bifurcation_mask)
    bifurcations = list(zip(bifurcation_coords[0], bifurcation_coords[1]))

    return skeleton, bifurcations, cleaned


def visualize_dinov2_features(image_path, mask_path, output_path, model):
    """
    Create visualization showing DINOv2 features on angiogram.

    Replicates Figure 5 from the paper.
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Load mask
    mask = np.load(mask_path)
    if mask.shape != (img_array.shape[0], img_array.shape[1]):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
        mask = np.array(mask_img) > 127

    # Extract DINOv2 features
    features, orig_size = extract_features(model, img)

    # PCA for visualization
    pca_vis = pca_features(features)

    # Resize PCA visualization to original image size
    pca_vis_resized = np.array(Image.fromarray((pca_vis * 255).astype(np.uint8)).resize(
        orig_size, Image.BILINEAR
    )) / 255.0

    # Get skeleton and bifurcations
    skeleton, bifurcations, cleaned_mask = get_skeleton_and_bifurcations(mask)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Original image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original Angiogram")
    axes[0, 0].axis("off")

    # 2. Vessel mask
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Vessel Mask")
    axes[0, 1].axis("off")

    # 3. DINOv2 PCA (full image)
    axes[0, 2].imshow(pca_vis_resized)
    axes[0, 2].set_title("DINOv2 Features (3-component PCA)")
    axes[0, 2].axis("off")

    # 4. DINOv2 features masked to vessels only
    vessel_features = pca_vis_resized.copy()
    # Darken non-vessel areas
    vessel_features[~mask] = vessel_features[~mask] * 0.2
    axes[1, 0].imshow(vessel_features)
    axes[1, 0].set_title("DINOv2 Features (vessel regions)")
    axes[1, 0].axis("off")

    # 5. Skeleton colored by DINOv2 features
    skeleton_colored = np.zeros((*skeleton.shape, 3))
    skeleton_colored[skeleton] = pca_vis_resized[skeleton]
    # Make it brighter
    skeleton_colored = np.clip(skeleton_colored * 2, 0, 1)
    axes[1, 1].imshow(skeleton_colored)
    axes[1, 1].set_title("Skeleton colored by DINOv2 features")
    axes[1, 1].axis("off")

    # 6. Overlay: image + skeleton colored by features + bifurcations
    overlay = img_array.copy().astype(float) / 255.0

    # Overlay skeleton with DINOv2 colors
    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            if skeleton[y, x]:
                overlay[y, x] = pca_vis_resized[y, x]

    # Mark bifurcations as white circles
    from skimage.draw import disk
    for y, x in bifurcations[:50]:  # Limit to 50 for visibility
        if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
            rr, cc = disk((y, x), 5, shape=skeleton.shape)
            overlay[rr, cc] = [1, 1, 1]  # White

    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f"Overlay: Skeleton + {len(bifurcations)} bifurcations")
    axes[1, 2].axis("off")

    plt.suptitle(f"DINOv2 Feature Analysis: {image_path.name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return features, pca_vis_resized


def analyze_branch_features(features, skeleton, bifurcations, mask):
    """
    Analyze if different branches have different semantic signatures.

    Returns statistics about feature variation along branches.
    """
    # Sample features along skeleton
    skeleton_features = features[skeleton]

    # Compute feature statistics
    mean_feat = skeleton_features.mean(axis=0)
    std_feat = skeleton_features.std(axis=0)

    # Check variance - high variance suggests different branches have different features
    total_variance = std_feat.mean()

    return {
        "n_skeleton_points": skeleton_features.shape[0],
        "n_bifurcations": len(bifurcations),
        "feature_variance": total_variance,
        "feature_dim": features.shape[-1] if len(features.shape) > 2 else 3
    }


def main():
    print("=" * 60)
    print("DINOv2 Feature Analysis for Coronary Angiography")
    print("=" * 60)

    # Load model
    model = load_dinov2_model()

    # Find matching image-mask pairs
    mask_files = list(DEEPSA_MASKS.glob("*_full_vessel_mask.npy"))
    print(f"Found {len(mask_files)} vessel masks")

    # Process samples
    n_samples = 6
    processed = 0

    for mask_path in mask_files[:100]:
        if processed >= n_samples:
            break

        base_name = mask_path.stem.replace("_full_vessel_mask", "")
        image_path = CASS_IMAGES / f"{base_name}.png"

        if not image_path.exists():
            continue

        output_path = OUTPUT_DIR / f"dinov2_{base_name}.png"

        print(f"\nProcessing: {base_name}")

        try:
            features, pca_vis = visualize_dinov2_features(
                image_path, mask_path, output_path, model
            )

            # Load mask for analysis
            mask = np.load(mask_path)
            if mask.shape != pca_vis.shape[:2]:
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_img = mask_img.resize((pca_vis.shape[1], pca_vis.shape[0]), Image.NEAREST)
                mask = np.array(mask_img) > 127

            skeleton, bifurcations, _ = get_skeleton_and_bifurcations(mask)
            stats = analyze_branch_features(pca_vis, skeleton, bifurcations, mask)

            print(f"  Skeleton points: {stats['n_skeleton_points']}")
            print(f"  Bifurcations: {stats['n_bifurcations']}")
            print(f"  Feature variance: {stats['feature_variance']:.4f}")

            processed += 1

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\n{'=' * 60}")
    print(f"Processed {processed} images")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
