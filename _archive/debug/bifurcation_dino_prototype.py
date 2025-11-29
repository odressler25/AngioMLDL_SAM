"""
Prototype: Bifurcation-aware CASS labeling using DINOv2 features.

CONCEPT:
CASS segments are anatomically defined by bifurcations:
- proximal_lad → mid_lad: at first diagonal bifurcation
- mid_lad → distal_lad: at second diagonal bifurcation
- proximal_rca → mid_rca: at first acute marginal
- etc.

APPROACH:
1. Detect bifurcation points on vessel skeleton
2. Extract DINOv2 patch features around each bifurcation
3. Classify bifurcations by type (main vessel vs side branch)
4. Use bifurcations as natural segment boundaries

This combines:
- Classical CV: precise bifurcation localization (skeleton junctions)
- DINOv2: semantic understanding of vessel appearance
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
import json
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
TRAINING_CSV = Path("E:/AngioMLDL_data/corrected_dataset_training.csv")
IMAGES_DIR = Path("E:/AngioMLDL_data/coco_format_v2/train/images")
OUTPUT_DIR = Path("E:/AngioMLDL_data/bifurcation_dino_prototype")
OUTPUT_DIR.mkdir(exist_ok=True)


def detect_bifurcations(skeleton, vessel_mask, min_branch_length=40, min_branch_width=4, top_n=8):
    """
    Detect MAJOR bifurcation points on skeleton.

    A bifurcation is a pixel with 3+ skeleton neighbors.
    Filter aggressively to keep only anatomically significant bifurcations.

    Criteria for major bifurcation:
    1. At least 2 branches with length >= min_branch_length
    2. At least 2 branches with width >= min_branch_width
    3. Keep only top N by importance score

    Returns: List of (y, x, score) bifurcation coordinates with importance scores
    """
    h, w = skeleton.shape
    bifurcations = []

    # Get distance transform for vessel width
    dist_transform = distance_transform_edt(vessel_mask)

    # Find all junction points (3+ neighbors)
    for y in range(1, h-1):
        for x in range(1, w-1):
            if not skeleton[y, x]:
                continue

            # Count skeleton neighbors
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    if skeleton[y + dy, x + dx]:
                        neighbors += 1

            # Junction = 3+ neighbors
            if neighbors >= 3:
                bifurcations.append((y, x))

    # Cluster nearby bifurcation points (they often form small groups)
    if not bifurcations:
        return []

    clustered = cluster_nearby_points(bifurcations, radius=8)

    # Score each bifurcation by importance
    scored_bifurcations = []
    for by, bx in clustered:
        branch_info = measure_branch_properties(skeleton, (by, bx), clustered, dist_transform)

        if len(branch_info) < 2:
            continue

        # Sort branches by length
        branch_info.sort(key=lambda b: b['length'], reverse=True)

        # Count significant branches
        long_branches = sum(1 for b in branch_info if b['length'] >= min_branch_length)
        wide_branches = sum(1 for b in branch_info if b['width'] >= min_branch_width)

        # Must have at least 2 long AND 2 wide branches
        if long_branches < 2 or wide_branches < 2:
            continue

        # Importance score: sum of (length * width) for top 2 branches
        score = sum(b['length'] * b['width'] for b in branch_info[:2])

        # Bonus for bifurcations with exactly 3 branches (typical anatomical bifurcation)
        if len(branch_info) == 3:
            score *= 1.2

        scored_bifurcations.append((by, bx, score))

    # Sort by score and keep top N
    scored_bifurcations.sort(key=lambda x: x[2], reverse=True)
    top_bifurcations = scored_bifurcations[:top_n]

    # Return just (y, x) tuples
    return [(by, bx) for by, bx, score in top_bifurcations]


def measure_branch_properties(skeleton, bifurcation, all_bifurcations, dist_transform, max_steps=80):
    """
    Measure properties of each branch from a bifurcation.

    Returns list of dicts with 'length', 'width', 'direction' for each branch.
    """
    by, bx = bifurcation
    h, w = skeleton.shape
    bifurcation_set = set(all_bifurcations)

    # Find immediate neighbors (starting points of each branch)
    branch_starts = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = by + dy, bx + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                branch_starts.append((ny, nx, dy, dx))

    branches = []
    for start_y, start_x, init_dy, init_dx in branch_starts:
        visited = {(by, bx), (start_y, start_x)}
        path = [(start_y, start_x)]
        current = (start_y, start_x)

        for _ in range(max_steps):
            cy, cx = current

            # Find next unvisited neighbor
            next_point = None
            for ddy in [-1, 0, 1]:
                for ddx in [-1, 0, 1]:
                    if ddy == 0 and ddx == 0:
                        continue
                    ny, nx = cy + ddy, cx + ddx
                    if 0 <= ny < h and 0 <= nx < w:
                        if skeleton[ny, nx] and (ny, nx) not in visited:
                            next_point = (ny, nx)
                            break
                if next_point:
                    break

            if next_point is None:
                break  # Endpoint

            # Stop if we hit another major bifurcation
            if next_point in bifurcation_set and len(path) > 5:
                break

            visited.add(next_point)
            path.append(next_point)
            current = next_point

        if len(path) < 3:
            continue

        # Calculate average width along branch
        widths = [dist_transform[y, x] for y, x in path]
        avg_width = np.mean(widths)

        # Calculate direction
        path_arr = np.array(path)
        direction = path_arr[-1] - path_arr[0]
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        branches.append({
            'length': len(path),
            'width': avg_width,
            'direction': tuple(direction),
            'initial_dir': (init_dy, init_dx)
        })

    return branches


def cluster_nearby_points(points, radius=5):
    """Cluster nearby points and return centroids."""
    if not points:
        return []

    points = np.array(points)
    used = np.zeros(len(points), dtype=bool)
    clusters = []

    for i, (y, x) in enumerate(points):
        if used[i]:
            continue

        # Find all points within radius
        distances = np.sqrt((points[:, 0] - y)**2 + (points[:, 1] - x)**2)
        in_cluster = distances <= radius

        # Compute centroid
        cluster_points = points[in_cluster]
        centroid = cluster_points.mean(axis=0).astype(int)
        clusters.append(tuple(centroid))

        used[in_cluster] = True

    return clusters


def measure_branch_lengths(skeleton, bifurcation, all_bifurcations, max_steps=100):
    """
    Measure length of each branch from a bifurcation.

    Traces each direction until hitting endpoint, another bifurcation, or max_steps.
    """
    by, bx = bifurcation
    h, w = skeleton.shape

    # Find immediate neighbors (starting points of each branch)
    branch_starts = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = by + dy, bx + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                branch_starts.append((ny, nx, dy, dx))

    # Trace each branch
    branch_lengths = []
    bifurcation_set = set(all_bifurcations)

    for start_y, start_x, init_dy, init_dx in branch_starts:
        visited = {(by, bx), (start_y, start_x)}
        current = (start_y, start_x)
        length = 1

        for _ in range(max_steps):
            cy, cx = current

            # Find next unvisited neighbor
            next_point = None
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if skeleton[ny, nx] and (ny, nx) not in visited:
                            next_point = (ny, nx)
                            break
                if next_point:
                    break

            if next_point is None:
                break  # Endpoint

            # Check if we hit another bifurcation
            if next_point in bifurcation_set:
                break

            visited.add(next_point)
            current = next_point
            length += 1

        branch_lengths.append(length)

    return branch_lengths


def extract_dino_features(image, points, patch_size=64, model=None, device='cuda'):
    """
    Extract DINOv2 features for patches around each point.

    Args:
        image: PIL Image or numpy array
        points: List of (y, x) coordinates
        patch_size: Size of patch to extract around each point
        model: DINOv2 model (loaded if None)
        device: cuda or cpu

    Returns:
        features: (N, feature_dim) array of features
    """
    if model is None:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model = model.to(device)
        model.eval()

    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        image = Image.fromarray(image.astype(np.uint8))

    img_array = np.array(image)
    h, w = img_array.shape[:2]
    half = patch_size // 2

    features = []

    with torch.no_grad():
        for y, x in points:
            # Extract patch (with boundary handling)
            y1 = max(0, y - half)
            y2 = min(h, y + half)
            x1 = max(0, x - half)
            x2 = min(w, x + half)

            patch = img_array[y1:y2, x1:x2]

            # Resize to 224x224 for DINOv2
            patch_pil = Image.fromarray(patch)
            patch_resized = patch_pil.resize((224, 224), Image.BILINEAR)

            # Convert to tensor
            patch_tensor = torch.from_numpy(np.array(patch_resized)).float()
            patch_tensor = patch_tensor.permute(2, 0, 1) / 255.0
            patch_tensor = patch_tensor.unsqueeze(0).to(device)

            # Normalize (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            patch_tensor = (patch_tensor - mean) / std

            # Extract features
            feat = model(patch_tensor)
            features.append(feat.cpu().numpy().squeeze())

    return np.array(features)


def classify_bifurcation_branches(skeleton, bifurcation, vessel_mask, dino_features=None):
    """
    At a bifurcation, classify which branch is the "main" vessel continuation
    vs which is a side branch.

    Uses:
    1. Vessel width (main vessel is usually wider)
    2. Angle (main vessel continues straighter)
    3. DINOv2 features (main vessel has similar features before/after bifurcation)

    Returns:
        dict with 'main_branch' and 'side_branches' directions
    """
    by, bx = bifurcation
    h, w = skeleton.shape

    # Get distance transform for vessel width
    dist_transform = distance_transform_edt(vessel_mask)

    # Find branches
    branches = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = by + dy, bx + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                # Trace a bit to get average direction and width
                direction, avg_width = trace_branch_properties(
                    skeleton, (ny, nx), (by, bx), vessel_mask, dist_transform
                )
                branches.append({
                    'start': (ny, nx),
                    'direction': direction,
                    'width': avg_width,
                    'initial_dir': (dy, dx)
                })

    if len(branches) < 2:
        return {'main_branches': branches, 'side_branches': []}

    # Sort by width (wider = more likely main vessel)
    branches.sort(key=lambda b: b['width'], reverse=True)

    # The two widest branches are likely the main vessel (before and after bifurcation)
    # Narrower branches are side branches
    main_branches = branches[:2]
    side_branches = branches[2:]

    return {
        'main_branches': main_branches,
        'side_branches': side_branches,
        'all_branches': branches
    }


def trace_branch_properties(skeleton, start, origin, vessel_mask, dist_transform, trace_length=20):
    """
    Trace along a branch to measure its properties.

    Returns:
        direction: (dy, dx) average direction
        avg_width: average vessel width along branch
    """
    h, w = skeleton.shape

    visited = {origin, start}
    path = [start]
    current = start

    for _ in range(trace_length):
        cy, cx = current

        # Find next unvisited neighbor
        next_point = None
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] and (ny, nx) not in visited:
                        next_point = (ny, nx)
                        break
            if next_point:
                break

        if next_point is None:
            break

        visited.add(next_point)
        path.append(next_point)
        current = next_point

    if len(path) < 2:
        return (0, 0), 0

    # Calculate direction (from start to end of trace)
    path = np.array(path)
    direction = path[-1] - path[0]
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    # Calculate average width
    widths = [dist_transform[y, x] for y, x in path]
    avg_width = np.mean(widths)

    return tuple(direction), avg_width


def visualize_bifurcations(image, skeleton, bifurcations, vessel_mask, save_path=None):
    """
    Visualize detected MAJOR bifurcations with branch classification.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get distance transform for branch analysis
    dist_transform = distance_transform_edt(vessel_mask)

    # 1. Original image with bifurcation points (numbered)
    axes[0].imshow(image, cmap='gray')
    for i, (by, bx) in enumerate(bifurcations, 1):
        axes[0].plot(bx, by, 'ro', markersize=12)
        axes[0].text(bx + 10, by, str(i), color='yellow', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Major Bifurcations ({len(bifurcations)})')
    axes[0].axis('off')

    # 2. Skeleton with bifurcations highlighted
    skeleton_rgb = np.zeros((*skeleton.shape, 3))
    skeleton_rgb[skeleton > 0] = [0.5, 0.5, 0.5]  # Gray skeleton

    # Color bifurcation regions
    for by, bx in bifurcations:
        # Draw a small circle around bifurcation
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                if dy*dy + dx*dx <= 64:  # Circle of radius 8
                    ny, nx = by + dy, bx + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        skeleton_rgb[ny, nx] = [1, 0, 0]  # Red

    axes[1].imshow(skeleton_rgb)
    axes[1].set_title('Skeleton with Major Bifurcations')
    axes[1].axis('off')

    # 3. Branch classification at each bifurcation
    axes[2].imshow(image, cmap='gray')

    all_clustered = cluster_nearby_points([(by, bx) for by, bx in bifurcations], radius=8)

    for i, (by, bx) in enumerate(bifurcations, 1):
        # Get branch properties
        branch_info = measure_branch_properties(skeleton, (by, bx), all_clustered, dist_transform)

        if not branch_info:
            continue

        # Sort by width (main vessel = wider)
        branch_info.sort(key=lambda b: b['width'], reverse=True)

        # Draw branches - top 2 widest are "main", rest are "side"
        for j, branch in enumerate(branch_info):
            dy, dx = branch['initial_dir']
            length = min(branch['length'], 30)  # Cap arrow length

            if j < 2:  # Main branches (green)
                axes[2].arrow(bx, by, dx*length, dy*length,
                             head_width=6, head_length=4, fc='green', ec='green', linewidth=2)
            else:  # Side branches (yellow)
                axes[2].arrow(bx, by, dx*length, dy*length,
                             head_width=4, head_length=3, fc='yellow', ec='yellow', linewidth=1.5)

        axes[2].plot(bx, by, 'ro', markersize=8)
        axes[2].text(bx + 10, by, str(i), color='cyan', fontsize=10, fontweight='bold')

    # Legend
    main_patch = mpatches.Patch(color='green', label='Main vessel')
    side_patch = mpatches.Patch(color='yellow', label='Side branch')
    axes[2].legend(handles=[main_patch, side_patch], loc='upper right')
    axes[2].set_title('Branch Classification (by width)')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def process_sample(image_path, vessel_mask_path, output_path):
    """
    Process a single sample: detect MAJOR bifurcations, classify branches.
    """
    # Load data
    image = np.array(Image.open(image_path).convert('RGB'))
    vessel_mask = np.load(vessel_mask_path)

    # Resize vessel mask if needed
    if vessel_mask.shape[:2] != image.shape[:2]:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(vessel_mask.astype(np.uint8))
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
        vessel_mask = np.array(mask_pil)

    # Create skeleton
    vessel_binary = (vessel_mask > 0).astype(np.uint8)
    skeleton = skeletonize(vessel_binary > 0)

    # Detect MAJOR bifurcations (filtered by branch length, width, and importance)
    bifurcations = detect_bifurcations(
        skeleton, vessel_binary,
        min_branch_length=40,  # Branches must be at least 40 pixels
        min_branch_width=4,    # Branches must be at least 4 pixels wide
        top_n=6                # Keep only top 6 most important bifurcations
    )

    # Visualize
    visualize_bifurcations(image, skeleton, bifurcations, vessel_binary, output_path)

    return bifurcations


def main():
    """
    Run prototype on sample images.
    """
    import pandas as pd

    print("Bifurcation + DINOv2 Prototype")
    print("=" * 50)

    # Load training data
    df = pd.read_csv(TRAINING_CSV)
    df = df[df['split'] == 'train'].head(10)  # Process first 10 samples

    print(f"Processing {len(df)} samples...")

    for i, (_, row) in enumerate(df.iterrows()):
        deepsa_path = Path(row['deepsa_pseudo_label_path'])
        base_name = deepsa_path.stem.replace('_full_vessel_mask', '')
        img_path = IMAGES_DIR / f"{base_name}.png"

        if not img_path.exists() or not deepsa_path.exists():
            continue

        output_path = OUTPUT_DIR / f"bifurcation_{i+1}.png"

        try:
            bifurcations = process_sample(img_path, deepsa_path, output_path)
            print(f"  [{i+1}] {base_name}: {len(bifurcations)} bifurcations detected")
        except Exception as e:
            print(f"  [{i+1}] {base_name}: Error - {e}")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
