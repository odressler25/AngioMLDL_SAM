"""
CASS Anatomical Labeling: Map bifurcations to segment boundaries.

This script:
1. Finds the vessel origin (proximal end)
2. Traces the main vessel path
3. Orders bifurcations along the path
4. Labels segments according to CASS definitions

CASS Segment Definitions:
- LAD: Proximal (origin→D1), Mid (D1→D2), Distal (after D2)
- RCA: Proximal (origin→1st AM), Mid (1st AM→crux), Distal (at crux)
- LCX: Proximal (origin→OM1), Distal (after OM1)
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize, binary_opening, disk as morph_disk
from skimage.draw import disk
from scipy import ndimage
from scipy.ndimage import label, distance_transform_edt
from collections import deque
import networkx as nx

# Paths
CASS_IMAGES = Path("E:/AngioMLDL_data/coco_cass_segments/train/images")
DEEPSA_MASKS = Path("E:/AngioMLDL_data/deepsa_pseudo_labels")
OUTPUT_DIR = Path("E:/AngioMLDL_data/cass_labeling")
OUTPUT_DIR.mkdir(exist_ok=True)

# Parameters
TOP_N_BIFURCATIONS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CASS Segment definitions
CASS_SEGMENTS = {
    "LAD": {
        "segments": ["Proximal LAD", "Mid LAD", "Distal LAD"],
        "branches": ["D1", "D2"],
        "description": "Proximal: origin to D1, Mid: D1 to D2, Distal: after D2"
    },
    "RCA": {
        "segments": ["Proximal RCA", "Mid RCA", "Distal RCA", "PDA"],
        "branches": ["Acute Marginal", "PDA"],
        "description": "Proximal: origin to 1st AM, Mid: to crux, Distal: at crux"
    },
    "LCX": {
        "segments": ["Proximal LCX", "Distal LCX"],
        "branches": ["OM1", "OM2"],
        "description": "Proximal: origin to OM1, Distal: after OM1"
    },
    "RAMUS": {
        "segments": ["Ramus Intermedius"],
        "branches": [],
        "description": "Intermediate branch between LAD and LCX"
    }
}


def load_dinov2_model():
    """Load DINOv2 model."""
    print("Loading DINOv2...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    model = model.to(DEVICE).eval()
    return model


def extract_dinov2_features(model, image, target_size=518):
    """Extract DINOv2 features."""
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

    feature_map_resized = np.array([
        np.array(Image.fromarray(feature_map[:, :, i]).resize(orig_size, Image.BILINEAR))
        for i in range(D)
    ]).transpose(1, 2, 0)

    return feature_map_resized


def clean_vessel_mask(mask):
    """Clean vessel mask."""
    labeled, num_features = label(mask)
    if num_features == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    max_size = max(sizes) if len(sizes) > 0 else 0
    min_size = max(max_size * 0.02, 200)

    cleaned = np.zeros_like(mask)
    for i, size in enumerate(sizes, 1):
        if size >= min_size:
            cleaned[labeled == i] = 1

    cleaned = binary_opening(cleaned, morph_disk(2))
    return cleaned


def skeleton_to_graph(skeleton):
    """
    Convert skeleton to a graph where:
    - Nodes: bifurcations and endpoints
    - Edges: skeleton paths between nodes
    """
    # Find special points
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)

    # Bifurcations: 3+ neighbors, Endpoints: 1 neighbor
    bifurcations = (neighbor_count >= 3) & skeleton
    endpoints = (neighbor_count == 1) & skeleton

    # Get coordinates
    bif_coords = list(zip(*np.where(bifurcations)))
    end_coords = list(zip(*np.where(endpoints)))

    # Cluster nearby bifurcation points
    bif_coords = cluster_points(bif_coords, radius=8)

    # Create graph
    G = nx.Graph()

    # Add nodes
    all_nodes = []
    for i, (y, x) in enumerate(bif_coords):
        G.add_node(f"bif_{i}", pos=(y, x), type="bifurcation")
        all_nodes.append((y, x, f"bif_{i}"))
    for i, (y, x) in enumerate(end_coords):
        G.add_node(f"end_{i}", pos=(y, x), type="endpoint")
        all_nodes.append((y, x, f"end_{i}"))

    return G, bif_coords, end_coords


def cluster_points(points, radius=5):
    """Cluster nearby points."""
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


def find_vessel_origin(skeleton, mask, image_shape):
    """
    Find the vessel origin (proximal end).

    Heuristics:
    1. Look for catheter tip (usually bright spot near vessel start)
    2. Find endpoint closest to image center-top (common catheter position)
    3. Use the thickest part of the vessel as origin indicator
    """
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)
    endpoints = (neighbor_count == 1) & skeleton

    end_coords = list(zip(*np.where(endpoints)))
    if not end_coords:
        return None

    # Use distance transform to find vessel width at each point
    dist_transform = distance_transform_edt(mask)

    # Score each endpoint: prefer center-top position + thick vessel
    h, w = image_shape[:2]
    center_x = w // 2

    scored_endpoints = []
    for y, x in end_coords:
        # Distance from center-top (catheter usually comes from top-center)
        dist_from_center_top = np.sqrt((x - center_x)**2 + y**2)

        # Vessel width at this point
        vessel_width = dist_transform[y, x]

        # Score: prefer thick vessels near center-top
        # Lower distance + higher width = better
        score = vessel_width * 10 - dist_from_center_top * 0.1
        scored_endpoints.append((y, x, score))

    # Sort by score (higher is better)
    scored_endpoints.sort(key=lambda p: p[2], reverse=True)

    return (scored_endpoints[0][0], scored_endpoints[0][1])


def trace_main_vessel(skeleton, origin, bifurcations):
    """
    Trace the main vessel from origin, passing through major bifurcations.

    Returns ordered list of bifurcations along the main vessel path.
    """
    h, w = skeleton.shape

    # BFS from origin
    visited = set()
    visited.add(origin)

    # Track path and bifurcations encountered
    queue = deque([(origin, [origin], [])])  # (current, path, bifs_on_path)

    all_paths = []

    while queue:
        current, path, bifs_on_path = queue.popleft()
        cy, cx = current

        # Check if current is near a bifurcation
        current_bifs = list(bifs_on_path)
        for bif in bifurcations:
            dist = np.sqrt((cy - bif[0])**2 + (cx - bif[1])**2)
            if dist < 15 and bif not in current_bifs:
                current_bifs.append(bif)

        # Check neighbors
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] and (ny, nx) not in visited:
                        neighbors.append((ny, nx))
                        visited.add((ny, nx))

        if not neighbors:
            # End of path - save it
            all_paths.append((path, current_bifs))
        else:
            for neighbor in neighbors:
                queue.append((neighbor, path + [neighbor], current_bifs))

    # Find the longest path (main vessel)
    if not all_paths:
        return []

    main_path, main_bifs = max(all_paths, key=lambda x: len(x[0]))

    # Order bifurcations by their position along the path
    bif_positions = []
    for bif in main_bifs:
        # Find position along path
        for i, (py, px) in enumerate(main_path):
            dist = np.sqrt((py - bif[0])**2 + (px - bif[1])**2)
            if dist < 15:
                bif_positions.append((bif, i))
                break

    # Sort by position
    bif_positions.sort(key=lambda x: x[1])
    ordered_bifs = [bp[0] for bp in bif_positions]

    return ordered_bifs, main_path


def score_bifurcation(skeleton, point, mask):
    """Score bifurcation importance."""
    y, x = point
    dist_transform = distance_transform_edt(mask)
    vessel_width = dist_transform[y, x] * 2

    # Count branches
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    if skeleton[y, x]:
        neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx]:
                        neighbors += 1
        return vessel_width * neighbors
    return 0


def get_major_bifurcations(skeleton, mask, top_n=TOP_N_BIFURCATIONS):
    """Get top N major bifurcations."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)
    bifurcation_mask = (neighbor_count >= 3) & skeleton

    coords = list(zip(*np.where(bifurcation_mask)))
    coords = cluster_points(coords, radius=8)

    # Score and filter
    scored = []
    for point in coords:
        score = score_bifurcation(skeleton, point, mask)
        if score > 0:
            scored.append((point, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, s in scored[:top_n]]


def infer_vessel_type(filename, features=None):
    """
    Infer vessel type from filename or features.

    In real usage, this would come from:
    - CASS annotation metadata
    - A classifier
    - User input
    """
    filename_lower = filename.lower()

    # RCA and its branches
    if 'rca' in filename_lower or 'pda' in filename_lower or 'rpda' in filename_lower:
        return 'RCA'
    # LAD and its branches (diagonal, septal)
    elif 'lad' in filename_lower or 'diag' in filename_lower or 'd1' in filename_lower or 'd2' in filename_lower:
        return 'LAD'
    # LCX and its branches (obtuse marginal, circumflex system)
    elif 'lcx' in filename_lower or 'om' in filename_lower or '_cs_' in filename_lower or 'llcx' in filename_lower:
        return 'LCX'
    # Ramus intermedius (between LAD and LCX)
    elif 'ramus' in filename_lower:
        return 'RAMUS'
    else:
        return 'UNKNOWN'


def label_segments(ordered_bifs, vessel_type, path_length):
    """
    Label segments based on vessel type and bifurcation positions.

    Returns list of (start_pos, end_pos, segment_name, branch_name)
    """
    segments = []

    if vessel_type == 'LAD':
        # LAD: Proximal (0 to D1), Mid (D1 to D2), Distal (after D2)
        if len(ordered_bifs) >= 2:
            segments.append((0, 0.33, "Proximal LAD", None))
            segments.append((0.33, 0.66, "Mid LAD", "D1"))
            segments.append((0.66, 1.0, "Distal LAD", "D2"))
        elif len(ordered_bifs) == 1:
            segments.append((0, 0.5, "Proximal LAD", None))
            segments.append((0.5, 1.0, "Mid-Distal LAD", "D1"))
        else:
            segments.append((0, 1.0, "LAD", None))

    elif vessel_type == 'RCA':
        # RCA: Proximal, Mid, Distal, PDA
        if len(ordered_bifs) >= 2:
            segments.append((0, 0.33, "Proximal RCA", None))
            segments.append((0.33, 0.66, "Mid RCA", "AM"))
            segments.append((0.66, 1.0, "Distal RCA/PDA", "PDA"))
        elif len(ordered_bifs) == 1:
            segments.append((0, 0.5, "Proximal RCA", None))
            segments.append((0.5, 1.0, "Mid-Distal RCA", "AM"))
        else:
            segments.append((0, 1.0, "RCA", None))

    elif vessel_type == 'LCX':
        # LCX: Proximal, Distal
        if len(ordered_bifs) >= 1:
            segments.append((0, 0.5, "Proximal LCX", None))
            segments.append((0.5, 1.0, "Distal LCX", "OM1"))
        else:
            segments.append((0, 1.0, "LCX", None))

    elif vessel_type == 'RAMUS':
        # Ramus intermedius - typically single segment
        segments.append((0, 1.0, "Ramus Intermedius", None))

    else:
        segments.append((0, 1.0, "Unknown Vessel", None))

    return segments


def visualize_cass_labeling(image_path, mask_path, output_path, model):
    """Create visualization with CASS segment labels."""

    # Load data
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    mask = np.load(mask_path)
    if mask.shape != img_array.shape[:2]:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
        mask = np.array(mask_img) > 127

    # Process
    cleaned_mask = clean_vessel_mask(mask)
    skeleton = skeletonize(cleaned_mask)

    # Get bifurcations
    major_bifs = get_major_bifurcations(skeleton, cleaned_mask)

    # Find origin and trace
    origin = find_vessel_origin(skeleton, cleaned_mask, img_array.shape)

    if origin:
        ordered_bifs, main_path = trace_main_vessel(skeleton, origin, major_bifs)
    else:
        ordered_bifs = major_bifs
        main_path = []

    # Infer vessel type
    vessel_type = infer_vessel_type(image_path.name)

    # Get segment labels
    segments = label_segments(ordered_bifs, vessel_type, len(main_path))

    # Extract DINOv2 features for clustering
    features = extract_dinov2_features(model, img)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1. Original with vessel overlay
    overlay = img_array.copy().astype(float) / 255.0
    vessel_tint = np.zeros_like(overlay)
    vessel_tint[cleaned_mask > 0] = [0, 0.4, 0]
    overlay = overlay * 0.7 + vessel_tint * 0.3

    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title(f"Vessel Type: {vessel_type}")
    axes[0, 0].axis("off")

    # 2. Skeleton with ordered bifurcations
    skel_vis = np.zeros((*skeleton.shape, 3))
    skel_vis[skeleton] = [0.7, 0.7, 0.7]

    # Draw main path if available
    if main_path:
        for py, px in main_path[::5]:  # Every 5th point
            if 0 <= py < skeleton.shape[0] and 0 <= px < skeleton.shape[1]:
                skel_vis[py, px] = [1, 1, 0]  # Yellow for main path

    # Draw origin
    if origin:
        rr, cc = disk(origin, 12, shape=skeleton.shape)
        skel_vis[rr, cc] = [0, 1, 0]  # Green for origin

    # Draw ordered bifurcations with numbers
    bif_colors = [[1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0, 1, 1], [0, 0.5, 1], [0.5, 0, 1]]
    for i, bif in enumerate(ordered_bifs):
        rr, cc = disk(bif, 10, shape=skeleton.shape)
        skel_vis[rr, cc] = bif_colors[i % len(bif_colors)]

    axes[0, 1].imshow(skel_vis)

    # Add number labels
    if origin:
        axes[0, 1].annotate("O", (origin[1], origin[0]), color='white',
                           fontsize=12, fontweight='bold', ha='center', va='center')
    for i, bif in enumerate(ordered_bifs):
        axes[0, 1].annotate(str(i+1), (bif[1], bif[0]), color='white',
                           fontsize=10, fontweight='bold', ha='center', va='center')

    axes[0, 1].set_title(f"Ordered Bifurcations: O=Origin, 1-{len(ordered_bifs)}=Order along vessel")
    axes[0, 1].axis("off")

    # 3. Segment labels on image
    segment_overlay = img_array.copy().astype(float) / 255.0

    # Color code segments along main path
    segment_colors = [
        [1, 0.3, 0.3],   # Proximal - Red
        [0.3, 1, 0.3],   # Mid - Green
        [0.3, 0.3, 1],   # Distal - Blue
    ]

    if main_path and segments:
        path_len = len(main_path)
        for i, (start_pct, end_pct, seg_name, branch) in enumerate(segments):
            start_idx = int(start_pct * path_len)
            end_idx = int(end_pct * path_len)
            color = segment_colors[i % len(segment_colors)]

            for j in range(start_idx, min(end_idx, path_len)):
                py, px = main_path[j]
                # Thicken
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                            segment_overlay[ny, nx] = color

    # Add bifurcation markers
    for i, bif in enumerate(ordered_bifs):
        rr, cc = disk(bif, 8, shape=skeleton.shape)
        segment_overlay[rr, cc] = [1, 1, 1]

    axes[1, 0].imshow(segment_overlay)

    # Add segment labels as text
    segment_text = "\n".join([f"{s[2]}" + (f" (after {s[3]})" if s[3] else "")
                              for s in segments])
    axes[1, 0].text(10, 30, segment_text, color='white', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                   verticalalignment='top')

    axes[1, 0].set_title("CASS Segment Labels")
    axes[1, 0].axis("off")

    # 4. Summary diagram
    axes[1, 1].axis("off")

    # Create text summary
    summary = f"""
CASS SEGMENT ANALYSIS
{'='*40}

Vessel Type: {vessel_type}
Origin Found: {'Yes' if origin else 'No'}
Major Bifurcations: {len(ordered_bifs)}

SEGMENT LABELS:
"""
    for i, (start, end, name, branch) in enumerate(segments):
        branch_str = f" (landmark: {branch})" if branch else ""
        summary += f"\n  {i+1}. {name}{branch_str}"
        summary += f"\n     Position: {start*100:.0f}% - {end*100:.0f}% of vessel"

    summary += f"""

CASS DEFINITION ({vessel_type}):
{CASS_SEGMENTS.get(vessel_type, {}).get('description', 'N/A')}
"""

    axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 1].set_title("Analysis Summary")

    plt.suptitle(f"CASS Anatomical Labeling: {image_path.name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "vessel_type": vessel_type,
        "n_bifurcations": len(ordered_bifs),
        "segments": segments,
        "origin_found": origin is not None
    }


def main():
    print("=" * 60)
    print("CASS Anatomical Labeling")
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

        output_path = OUTPUT_DIR / f"cass_{base_name}.png"

        print(f"\nProcessing: {base_name}")

        try:
            result = visualize_cass_labeling(image_path, mask_path, output_path, model)
            print(f"  Vessel: {result['vessel_type']}")
            print(f"  Bifurcations: {result['n_bifurcations']}")
            print(f"  Segments: {[s[2] for s in result['segments']]}")
            processed += 1

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Processed {processed} images")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
