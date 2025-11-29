"""
Branch Segmentation Prototype: DINOv2 Features + Filtered Bifurcations

Combines:
1. Major bifurcation detection (filtered top N)
2. DINOv2 semantic features
3. Skeleton-based branch clustering

Goal: Automatically segment vessel branches for CASS classification.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from skimage.morphology import skeletonize, binary_opening, disk as morph_disk
from skimage.draw import disk
from scipy import ndimage
from scipy.ndimage import label, distance_transform_edt
from collections import deque

# Paths
CASS_IMAGES = Path("E:/AngioMLDL_data/coco_cass_segments/train/images")
DEEPSA_MASKS = Path("E:/AngioMLDL_data/deepsa_pseudo_labels")
OUTPUT_DIR = Path("E:/AngioMLDL_data/branch_segmentation")
OUTPUT_DIR.mkdir(exist_ok=True)

# Parameters
MIN_BRANCH_LENGTH = 30
MIN_VESSEL_WIDTH = 3
TOP_N_BIFURCATIONS = 8  # Keep only top 8 major bifurcations
N_BRANCH_CLUSTERS = 5   # Expected number of main branches

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dinov2_model():
    """Load DINOv2 model."""
    print("Loading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    model = model.to(DEVICE)
    model.eval()
    print("DINOv2 loaded!")
    return model


def extract_dinov2_features(model, image, target_size=518):
    """Extract DINOv2 features from image."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    orig_size = image.size
    image = image.resize((target_size, target_size), Image.BILINEAR)

    img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model.forward_features(img_tensor)
        patch_tokens = features['x_norm_patchtokens']

    B, N, D = patch_tokens.shape
    h = w = int(np.sqrt(N))
    feature_map = patch_tokens.reshape(B, h, w, D).squeeze(0).cpu().numpy()

    # Resize to original image size
    feature_map_resized = np.array([
        np.array(Image.fromarray(feature_map[:, :, i]).resize(orig_size, Image.BILINEAR))
        for i in range(D)
    ]).transpose(1, 2, 0)

    return feature_map_resized


def clean_vessel_mask(mask):
    """Clean vessel mask by removing small components."""
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


def get_skeleton(mask):
    """Get skeleton from cleaned mask."""
    cleaned = clean_vessel_mask(mask)
    skeleton = skeletonize(cleaned)
    return skeleton, cleaned


def find_all_bifurcations(skeleton):
    """Find all bifurcation points in skeleton."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)
    bifurcation_mask = (neighbor_count >= 3) & skeleton
    return bifurcation_mask


def cluster_points(points, radius=5):
    """Cluster nearby points and return centroids."""
    if len(points) == 0:
        return []

    points = np.array(points)
    h = max(p[0] for p in points) + 20
    w = max(p[1] for p in points) + 20
    point_mask = np.zeros((h, w), dtype=bool)
    for y, x in points:
        point_mask[y, x] = True

    from skimage.morphology import binary_dilation
    dilated = binary_dilation(point_mask, morph_disk(radius))
    labeled, num_features = label(dilated)

    centroids = []
    for i in range(1, num_features + 1):
        component_points = points[labeled[points[:, 0], points[:, 1]] == i]
        if len(component_points) > 0:
            centroid = component_points.mean(axis=0).astype(int)
            centroids.append(tuple(centroid))

    return centroids


def measure_branch_length(skeleton, start_point, bifurcation_mask, max_steps=200):
    """Measure branch length from a starting point."""
    y, x = start_point
    h, w = skeleton.shape

    visited = set()
    visited.add((y, x))
    queue = deque([(y, x, 0)])

    while queue:
        cy, cx, dist = queue.popleft()
        if dist > max_steps:
            return dist

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                    if skeleton[ny, nx]:
                        visited.add((ny, nx))
                        if bifurcation_mask[ny, nx] and dist > 5:
                            return dist
                        queue.append((ny, nx, dist + 1))

    return len(visited)


def get_branch_directions(skeleton, point, radius=5):
    """Get skeleton pixels in neighborhood of a point."""
    y, x = point
    h, w = skeleton.shape
    pixels = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                pixels.append((ny, nx))
    return pixels


def score_bifurcation(skeleton, point, mask, bifurcation_mask):
    """Score bifurcation by importance (branch lengths * vessel width)."""
    y, x = point

    branch_starts = get_branch_directions(skeleton, point, radius=5)
    if len(branch_starts) < 2:
        return 0

    branch_lengths = []
    for start in branch_starts:
        length = measure_branch_length(skeleton, start, bifurcation_mask)
        branch_lengths.append(length)

    branch_lengths.sort(reverse=True)
    main_branches = branch_lengths[:3]

    if len(main_branches) < 2 or main_branches[1] < MIN_BRANCH_LENGTH:
        return 0

    dist_transform = distance_transform_edt(mask)
    vessel_width = dist_transform[y, x] * 2

    if vessel_width < MIN_VESSEL_WIDTH:
        return 0

    score = main_branches[0] * main_branches[1] * vessel_width
    return score


def find_major_bifurcations(mask, skeleton, top_n=TOP_N_BIFURCATIONS):
    """Find top N major bifurcations by importance score."""
    bifurcation_mask = find_all_bifurcations(skeleton)

    coords = np.where(bifurcation_mask)
    all_points = list(zip(coords[0], coords[1]))
    all_points = cluster_points(all_points, radius=5)

    if len(all_points) == 0:
        return [], []

    scored = []
    for point in all_points:
        score = score_bifurcation(skeleton, point, mask, bifurcation_mask)
        if score > 0:
            scored.append((point, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    major = scored[:top_n]

    major_points = [p for p, s in major]
    scores = [s for p, s in major]

    return major_points, scores


def cluster_skeleton_by_features(skeleton, features, n_clusters=N_BRANCH_CLUSTERS):
    """
    Cluster skeleton points by their DINOv2 features.

    Each cluster represents a semantically distinct branch.
    """
    # Get skeleton point coordinates
    skel_coords = np.where(skeleton)
    skel_points = np.array(list(zip(skel_coords[0], skel_coords[1])))

    if len(skel_points) == 0:
        return None, None

    # Extract features for each skeleton point
    skel_features = features[skel_coords[0], skel_coords[1]]

    # Reduce dimensionality for clustering
    pca = PCA(n_components=10)
    skel_features_pca = pca.fit_transform(skel_features)

    # Cluster using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(skel_features_pca)

    # Create cluster map
    cluster_map = np.zeros(skeleton.shape, dtype=int) - 1  # -1 for non-skeleton
    cluster_map[skel_coords[0], skel_coords[1]] = cluster_labels

    return cluster_map, cluster_labels


def visualize_branch_segmentation(image_path, mask_path, output_path, model):
    """
    Create visualization of branch segmentation.

    Shows:
    1. Original image
    2. DINOv2 PCA features
    3. Clustered branches
    4. Major bifurcations on clustered branches
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Load mask
    mask = np.load(mask_path)
    if mask.shape != img_array.shape[:2]:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
        mask = np.array(mask_img) > 127

    # Get skeleton
    skeleton, cleaned_mask = get_skeleton(mask)

    # Extract DINOv2 features
    features = extract_dinov2_features(model, img)

    # PCA for visualization (3 components -> RGB)
    H, W, D = features.shape
    flat_features = features.reshape(-1, D)
    pca_vis = PCA(n_components=3)
    pca_result = pca_vis.fit_transform(flat_features).reshape(H, W, 3)
    pca_result = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min() + 1e-8)

    # Cluster skeleton by features
    cluster_map, cluster_labels = cluster_skeleton_by_features(skeleton, features)

    # Find major bifurcations
    major_bifs, scores = find_major_bifurcations(cleaned_mask, skeleton)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Original image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original Angiogram")
    axes[0, 0].axis("off")

    # 2. DINOv2 PCA on vessel regions
    vessel_pca = pca_result.copy()
    vessel_pca[~mask] = vessel_pca[~mask] * 0.15
    axes[0, 1].imshow(vessel_pca)
    axes[0, 1].set_title("DINOv2 Features (vessel regions)")
    axes[0, 1].axis("off")

    # 3. Clustered branches (skeleton)
    cluster_colors = np.array([
        [0.2, 0.2, 0.2],   # Background (dark gray)
        [1.0, 0.2, 0.2],   # Cluster 0 - Red
        [0.2, 1.0, 0.2],   # Cluster 1 - Green
        [0.2, 0.2, 1.0],   # Cluster 2 - Blue
        [1.0, 1.0, 0.2],   # Cluster 3 - Yellow
        [1.0, 0.2, 1.0],   # Cluster 4 - Magenta
        [0.2, 1.0, 1.0],   # Cluster 5 - Cyan
        [1.0, 0.6, 0.2],   # Cluster 6 - Orange
        [0.6, 0.2, 1.0],   # Cluster 7 - Purple
    ])

    cluster_vis = np.zeros((*skeleton.shape, 3))
    for i in range(N_BRANCH_CLUSTERS):
        cluster_vis[cluster_map == i] = cluster_colors[i + 1]

    axes[0, 2].imshow(cluster_vis)
    axes[0, 2].set_title(f"Branch Clusters ({N_BRANCH_CLUSTERS} clusters by DINOv2)")
    axes[0, 2].axis("off")

    # 4. Clusters on original image
    overlay = img_array.copy().astype(float) / 255.0
    for i in range(N_BRANCH_CLUSTERS):
        mask_cluster = cluster_map == i
        # Thicken the skeleton for visibility
        from scipy.ndimage import binary_dilation
        mask_thick = binary_dilation(mask_cluster, morph_disk(2))
        overlay[mask_thick] = cluster_colors[i + 1]

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Branch clusters on image")
    axes[1, 0].axis("off")

    # 5. Major bifurcations only
    bif_vis = np.zeros((*skeleton.shape, 3))
    bif_vis[skeleton] = [0.5, 0.5, 0.5]  # Gray skeleton

    bif_colors = [
        [1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0, 1, 0],
        [0, 1, 1], [0, 0.5, 1], [0.5, 0, 1], [1, 0, 1]
    ]

    for i, (y, x) in enumerate(major_bifs):
        if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
            rr, cc = disk((y, x), 10, shape=skeleton.shape)
            bif_vis[rr, cc] = bif_colors[i % len(bif_colors)]

    axes[1, 1].imshow(bif_vis)
    axes[1, 1].set_title(f"Top {len(major_bifs)} Major Bifurcations (filtered)")
    axes[1, 1].axis("off")

    # 6. Final overlay: clusters + major bifurcations
    final_overlay = img_array.copy().astype(float) / 255.0

    # Add vessel tint
    vessel_tint = np.zeros_like(final_overlay)
    vessel_tint[cleaned_mask > 0] = [0, 0.3, 0]
    final_overlay = final_overlay * 0.7 + vessel_tint * 0.3

    # Add colored skeleton
    for i in range(N_BRANCH_CLUSTERS):
        mask_cluster = cluster_map == i
        mask_thick = binary_dilation(mask_cluster, morph_disk(1))
        final_overlay[mask_thick] = cluster_colors[i + 1]

    # Add major bifurcations as numbered markers
    for i, (y, x) in enumerate(major_bifs):
        if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
            rr, cc = disk((y, x), 12, shape=skeleton.shape)
            final_overlay[rr, cc] = [1, 1, 1]  # White circle
            rr2, cc2 = disk((y, x), 8, shape=skeleton.shape)
            final_overlay[rr2, cc2] = bif_colors[i % len(bif_colors)]

    final_overlay = np.clip(final_overlay, 0, 1)
    axes[1, 2].imshow(final_overlay)

    # Add bifurcation number labels
    for i, (y, x) in enumerate(major_bifs):
        axes[1, 2].annotate(
            str(i + 1), (x, y),
            color='white', fontsize=10, fontweight='bold',
            ha='center', va='center'
        )

    axes[1, 2].set_title(f"Branches + {len(major_bifs)} Major Bifurcations")
    axes[1, 2].axis("off")

    plt.suptitle(f"Branch Segmentation: {image_path.name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "n_major_bifurcations": len(major_bifs),
        "n_clusters": N_BRANCH_CLUSTERS,
        "skeleton_points": skeleton.sum()
    }


def main():
    print("=" * 60)
    print("Branch Segmentation Prototype")
    print("DINOv2 Features + Filtered Major Bifurcations")
    print("=" * 60)

    model = load_dinov2_model()

    mask_files = list(DEEPSA_MASKS.glob("*_full_vessel_mask.npy"))
    print(f"Found {len(mask_files)} vessel masks")

    n_samples = 8
    processed = 0

    for mask_path in mask_files[:100]:
        if processed >= n_samples:
            break

        base_name = mask_path.stem.replace("_full_vessel_mask", "")
        image_path = CASS_IMAGES / f"{base_name}.png"

        if not image_path.exists():
            continue

        output_path = OUTPUT_DIR / f"branches_{base_name}.png"

        print(f"\nProcessing: {base_name}")

        try:
            stats = visualize_branch_segmentation(
                image_path, mask_path, output_path, model
            )
            print(f"  Clusters: {stats['n_clusters']}")
            print(f"  Major bifurcations: {stats['n_major_bifurcations']}")
            print(f"  Skeleton points: {stats['skeleton_points']}")
            processed += 1

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 60}")
    print(f"Processed {processed} images")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
