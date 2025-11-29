"""
Prototype: Detect bifurcations from vessel masks using skeletonization.

This uses classical CV (no ML) to find branch points in the vessel tree.
Enhanced with filtering to keep only MAJOR bifurcations.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, binary_dilation, binary_erosion, binary_opening, disk as morph_disk
from skimage.draw import disk
from scipy import ndimage
from scipy.ndimage import label, distance_transform_edt
from collections import deque

# Paths
DEEPSA_MASKS = Path("E:/AngioMLDL_data/deepsa_pseudo_labels")
CASS_IMAGES = Path("E:/AngioMLDL_data/coco_cass_segments/train/images")
OUTPUT_DIR = Path("E:/AngioMLDL_data/bifurcation_detection")
OUTPUT_DIR.mkdir(exist_ok=True)

# Parameters for major bifurcation filtering
MIN_BRANCH_LENGTH = 30  # Minimum pixels a branch must extend
MIN_VESSEL_WIDTH = 3    # Minimum vessel width at bifurcation
TOP_N_BIFURCATIONS = 10  # Keep only top N by importance score


def clean_vessel_mask(mask):
    """Clean up vessel mask to reduce noise."""
    # Remove small isolated regions
    labeled, num_features = label(mask)
    if num_features == 0:
        return mask

    # Find sizes of each component
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))

    # Keep only components larger than 1% of the largest
    max_size = max(sizes) if len(sizes) > 0 else 0
    min_size = max(max_size * 0.01, 100)  # At least 100 pixels or 1% of largest

    cleaned = np.zeros_like(mask)
    for i, size in enumerate(sizes, 1):
        if size >= min_size:
            cleaned[labeled == i] = 1

    # Morphological opening to smooth edges
    cleaned = binary_opening(cleaned, morph_disk(2))

    return cleaned


def measure_branch_length(skeleton, start_point, bifurcation_mask, max_steps=200):
    """
    Measure branch length by tracing from bifurcation until endpoint or another bifurcation.

    Returns length in pixels.
    """
    y, x = start_point
    h, w = skeleton.shape

    visited = set()
    visited.add((y, x))

    # BFS to trace the branch
    queue = deque([(y, x, 0)])  # (y, x, distance)

    while queue:
        cy, cx, dist = queue.popleft()

        if dist > max_steps:
            return dist

        # Check 8-neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx

                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                    if skeleton[ny, nx]:
                        visited.add((ny, nx))

                        # Stop if we hit another bifurcation
                        if bifurcation_mask[ny, nx] and dist > 5:
                            return dist

                        queue.append((ny, nx, dist + 1))

    return len(visited)


def get_branch_directions(skeleton, bifurcation_point, radius=3):
    """Get the directions of branches leaving a bifurcation point."""
    y, x = bifurcation_point
    h, w = skeleton.shape

    # Find skeleton pixels in neighborhood
    branch_pixels = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                branch_pixels.append((ny, nx))

    return branch_pixels


def score_bifurcation(skeleton, bifurcation_point, vessel_mask, bifurcation_mask):
    """
    Score a bifurcation by importance.

    Higher score = more important bifurcation.
    Based on:
    - Branch lengths leaving the bifurcation
    - Vessel width at the bifurcation
    """
    y, x = bifurcation_point

    # Get branch starting points
    branch_starts = get_branch_directions(skeleton, bifurcation_point, radius=5)

    if len(branch_starts) < 2:
        return 0

    # Measure length of each branch
    branch_lengths = []
    for start in branch_starts:
        length = measure_branch_length(skeleton, start, bifurcation_mask)
        branch_lengths.append(length)

    # Sort and get top 2-3 branches (ignore tiny spurs)
    branch_lengths.sort(reverse=True)
    main_branches = branch_lengths[:3]

    # Minimum branch length filter
    if len(main_branches) < 2 or main_branches[1] < MIN_BRANCH_LENGTH:
        return 0

    # Vessel width at bifurcation (using distance transform)
    dist_transform = distance_transform_edt(vessel_mask)
    vessel_width = dist_transform[y, x] * 2  # Diameter

    if vessel_width < MIN_VESSEL_WIDTH:
        return 0

    # Score: product of two longest branches * vessel width
    score = main_branches[0] * main_branches[1] * vessel_width

    return score


def find_bifurcations(vessel_mask, min_branch_length=10):
    """
    Find bifurcation points in a vessel mask using skeletonization.
    Returns ALL bifurcations (unfiltered).
    """
    mask = vessel_mask > 0
    skeleton = skeletonize(mask)

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)
    bifurcation_mask = (neighbor_count >= 3) & skeleton

    bifurcation_coords = np.where(bifurcation_mask)
    bifurcation_points = list(zip(bifurcation_coords[0], bifurcation_coords[1]))
    bifurcation_points = cluster_nearby_points(bifurcation_points, radius=5)

    return skeleton, bifurcation_points, bifurcation_mask


def find_major_bifurcations(vessel_mask, top_n=TOP_N_BIFURCATIONS):
    """
    Find only MAJOR bifurcations by scoring and filtering.

    Returns:
        skeleton: The skeletonized vessel
        major_bifurcations: List of (y, x) coordinates of major bifurcations
        all_bifurcations: List of all bifurcation points (for comparison)
        scores: List of scores for major bifurcations
    """
    # Clean the mask first
    cleaned_mask = clean_vessel_mask(vessel_mask)

    # Get skeleton and all bifurcations
    skeleton, all_bifurcations, bifurcation_mask = find_bifurcations(cleaned_mask)

    if len(all_bifurcations) == 0:
        return skeleton, [], all_bifurcations, []

    # Score each bifurcation
    scored = []
    for bif_point in all_bifurcations:
        score = score_bifurcation(skeleton, bif_point, cleaned_mask, bifurcation_mask)
        if score > 0:
            scored.append((bif_point, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top N
    major = scored[:top_n]

    major_bifurcations = [p for p, s in major]
    scores = [s for p, s in major]

    return skeleton, major_bifurcations, all_bifurcations, scores


def cluster_nearby_points(points, radius=5):
    """Cluster nearby points and return centroids."""
    if len(points) == 0:
        return []

    points = np.array(points)

    if len(points) == 0:
        return []

    h = max(p[0] for p in points) + 20
    w = max(p[1] for p in points) + 20
    point_mask = np.zeros((h, w), dtype=bool)
    for y, x in points:
        point_mask[y, x] = True

    dilated = binary_dilation(point_mask, morph_disk(radius))
    labeled, num_features = label(dilated)

    centroids = []
    for i in range(1, num_features + 1):
        component_points = points[labeled[points[:, 0], points[:, 1]] == i]
        if len(component_points) > 0:
            centroid = component_points.mean(axis=0).astype(int)
            centroids.append(tuple(centroid))

    return centroids


def find_endpoints(skeleton):
    """Find endpoint pixels in skeleton (degree 1 nodes)."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)

    # Endpoints have exactly 1 neighbor
    endpoint_mask = (neighbor_count == 1) & skeleton

    coords = np.where(endpoint_mask)
    return list(zip(coords[0], coords[1]))


def visualize_bifurcations(image_path, mask_path, output_path):
    """
    Create visualization showing:
    - Original image with ALL bifurcations (for comparison)
    - Cleaned vessel mask
    - MAJOR bifurcations only (filtered)
    - Overlay with numbered major bifurcations
    """
    # Load image
    img = np.array(Image.open(image_path).convert("RGB"))

    # Load mask
    mask = np.load(mask_path)

    # Resize mask to match image if needed
    if mask.shape != (img.shape[0], img.shape[1]):
        from PIL import Image as PILImage
        mask_img = PILImage.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((img.shape[1], img.shape[0]), PILImage.NEAREST)
        mask = np.array(mask_img) > 127

    # Find MAJOR bifurcations (filtered)
    skeleton, major_bifurcations, all_bifurcations, scores = find_major_bifurcations(mask)
    endpoints = find_endpoints(skeleton)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # 1. Original with ALL bifurcations (showing the noise problem)
    overlay_all = img.copy().astype(float)
    for y, x in all_bifurcations:
        if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
            rr, cc = disk((y, x), 4, shape=skeleton.shape)
            overlay_all[rr, cc] = [255, 100, 100]  # Light red

    overlay_all = np.clip(overlay_all, 0, 255).astype(np.uint8)
    axes[0, 0].imshow(overlay_all)
    axes[0, 0].set_title(f"ALL bifurcations: {len(all_bifurcations)} (too many!)")
    axes[0, 0].axis("off")

    # 2. Cleaned vessel mask with skeleton
    cleaned_mask = clean_vessel_mask(mask)
    skel_rgb = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    skel_rgb[cleaned_mask > 0] = [50, 50, 50]  # Dark gray for mask
    skel_rgb[skeleton] = [255, 255, 255]  # White skeleton

    axes[0, 1].imshow(skel_rgb)
    axes[0, 1].set_title("Cleaned mask + skeleton")
    axes[0, 1].axis("off")

    # 3. MAJOR bifurcations only
    skel_major = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    skel_major[skeleton] = [100, 100, 100]  # Gray skeleton

    # Draw major bifurcations in bright colors (ranked by importance)
    colors = [
        [255, 0, 0],    # 1st - Red
        [255, 128, 0],  # 2nd - Orange
        [255, 255, 0],  # 3rd - Yellow
        [0, 255, 0],    # 4th - Green
        [0, 255, 255],  # 5th - Cyan
        [0, 128, 255],  # 6th - Light blue
        [128, 0, 255],  # 7th - Purple
        [255, 0, 255],  # 8th - Magenta
        [255, 128, 128],# 9th - Pink
        [128, 255, 128],# 10th - Light green
    ]

    for i, (y, x) in enumerate(major_bifurcations):
        if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
            color = colors[i % len(colors)]
            rr, cc = disk((y, x), 8, shape=skeleton.shape)
            skel_major[rr, cc] = color

    axes[1, 0].imshow(skel_major)
    axes[1, 0].set_title(f"MAJOR bifurcations: {len(major_bifurcations)} (filtered)")
    axes[1, 0].axis("off")

    # 4. Final overlay with numbered bifurcations
    overlay = img.copy().astype(float)

    # Add green tint for vessel mask
    vessel_overlay = np.zeros_like(overlay)
    vessel_overlay[cleaned_mask > 0] = [0, 80, 0]
    overlay = overlay * 0.7 + vessel_overlay * 0.3

    # Draw skeleton in yellow (thin)
    overlay[skeleton] = [200, 200, 0]

    # Draw major bifurcations with numbers
    for i, (y, x) in enumerate(major_bifurcations):
        if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
            color = colors[i % len(colors)]
            rr, cc = disk((y, x), 10, shape=skeleton.shape)
            overlay[rr, cc] = color

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    axes[1, 1].imshow(overlay)

    # Add text labels for bifurcation numbers
    for i, (y, x) in enumerate(major_bifurcations):
        axes[1, 1].annotate(
            str(i + 1),
            (x, y),
            color='white',
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center'
        )

    score_str = ", ".join([f"{s:.0f}" for s in scores[:5]]) if scores else "N/A"
    axes[1, 1].set_title(f"Top {len(major_bifurcations)} major bifurcations (scores: {score_str}...)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return len(major_bifurcations), len(all_bifurcations)


def main():
    print("=" * 60)
    print("Bifurcation Detection Prototype - MAJOR BIFURCATIONS ONLY")
    print("=" * 60)
    print(f"Filtering parameters:")
    print(f"  MIN_BRANCH_LENGTH: {MIN_BRANCH_LENGTH} pixels")
    print(f"  MIN_VESSEL_WIDTH: {MIN_VESSEL_WIDTH} pixels")
    print(f"  TOP_N_BIFURCATIONS: {TOP_N_BIFURCATIONS}")

    # Find matching image-mask pairs
    mask_files = list(DEEPSA_MASKS.glob("*_full_vessel_mask.npy"))
    print(f"\nFound {len(mask_files)} vessel masks")

    # Process a sample of images
    n_samples = 8
    processed = 0
    results = []

    for mask_path in mask_files[:50]:  # Check first 50 to find matches
        if processed >= n_samples:
            break

        # Extract image name from mask name
        base_name = mask_path.stem.replace("_full_vessel_mask", "")
        image_path = CASS_IMAGES / f"{base_name}.png"

        if not image_path.exists():
            continue

        output_path = OUTPUT_DIR / f"major_bifurcation_{base_name}.png"

        print(f"\nProcessing: {base_name}")
        n_major, n_all = visualize_bifurcations(image_path, mask_path, output_path)
        print(f"  ALL bifurcations: {n_all}")
        print(f"  MAJOR bifurcations: {n_major} (filtered)")

        results.append({
            "name": base_name,
            "major": n_major,
            "all": n_all
        })
        processed += 1

    # Create summary
    print(f"\n{'=' * 60}")
    print(f"Processed {processed} images")
    print(f"Results saved to: {OUTPUT_DIR}")

    # Stats
    if results:
        avg_major = np.mean([r["major"] for r in results])
        avg_all = np.mean([r["all"] for r in results])
        print(f"\nBefore filtering: {avg_all:.1f} avg bifurcations")
        print(f"After filtering:  {avg_major:.1f} avg major bifurcations")
        print(f"Reduction: {(1 - avg_major/avg_all)*100:.0f}%")


if __name__ == "__main__":
    main()
