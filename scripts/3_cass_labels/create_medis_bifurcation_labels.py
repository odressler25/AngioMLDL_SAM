"""
Create CASS segment labels using Medis edges (ground truth) + DeepSA vessel mask.

VERSION 7.0 - BIFURCATION-AWARE TRACING

Builds on v6.0 with bifurcation detection enhancement:
1. Use Medis centerline direction as ground truth for proximal/distal
2. Strict angle constraints for proximal segments (no direction drift)
3. Use viewing angles to validate LAD vs LCX at bifurcation
4. Anatomical knowledge: proximal LAD is straight from LM to mid LAD
5. NEW: Detect major bifurcations and use as segment boundaries/validation

Key insight: CASS segments are anatomically defined BY bifurcations:
- proximal_lad ends at first diagonal bifurcation
- mid_lad ends at second diagonal bifurcation
- proximal_rca ends at first acute marginal, etc.

By detecting bifurcations, we can:
- Stop tracing at natural segment boundaries
- Validate we're on the correct branch at each junction
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json
import pandas as pd
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.draw import polygon as draw_polygon
from pycocotools import mask as mask_utils
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# ============================================================================
# BIFURCATION DETECTION (v7.0 enhancement)
# ============================================================================

def detect_major_bifurcations(skeleton, vessel_mask, min_branch_length=40,
                               min_branch_width=4, top_n=5):
    """
    Detect major bifurcation points on skeleton.

    Filters to keep only anatomically significant bifurcations:
    1. At least 2 branches with length >= min_branch_length
    2. At least 2 branches with width >= min_branch_width
    3. Keep only top N by importance score (length * width)

    Returns: List of (y, x) bifurcation coordinates
    """
    h, w = skeleton.shape
    dist_transform = distance_transform_edt(vessel_mask)

    # Find all junction points (3+ neighbors)
    junctions = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if not skeleton[y, x]:
                continue
            neighbors = sum(1 for dy in [-1,0,1] for dx in [-1,0,1]
                           if (dy != 0 or dx != 0) and skeleton[y+dy, x+dx])
            if neighbors >= 3:
                junctions.append((y, x))

    if not junctions:
        return []

    # Cluster nearby junctions
    junctions = np.array(junctions)
    clustered = []
    used = np.zeros(len(junctions), dtype=bool)

    for i, (y, x) in enumerate(junctions):
        if used[i]:
            continue
        distances = np.sqrt((junctions[:, 0] - y)**2 + (junctions[:, 1] - x)**2)
        in_cluster = distances <= 8
        centroid = junctions[in_cluster].mean(axis=0).astype(int)
        clustered.append(tuple(centroid))
        used[in_cluster] = True

    # Score each bifurcation
    scored = []
    for by, bx in clustered:
        branches = measure_branches(skeleton, (by, bx), clustered, dist_transform)

        if len(branches) < 2:
            continue

        branches.sort(key=lambda b: b['length'], reverse=True)
        long_branches = sum(1 for b in branches if b['length'] >= min_branch_length)
        wide_branches = sum(1 for b in branches if b['width'] >= min_branch_width)

        if long_branches < 2 or wide_branches < 2:
            continue

        score = sum(b['length'] * b['width'] for b in branches[:2])
        if len(branches) == 3:
            score *= 1.2  # Bonus for typical bifurcation

        scored.append((by, bx, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return [(by, bx) for by, bx, _ in scored[:top_n]]


def measure_branches(skeleton, bifurcation, all_bifurcations, dist_transform, max_steps=80):
    """Measure length and width of branches from a bifurcation."""
    by, bx = bifurcation
    h, w = skeleton.shape
    bif_set = set(all_bifurcations)

    # Find branch starts
    starts = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = by + dy, bx + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                starts.append((ny, nx, dy, dx))

    branches = []
    for start_y, start_x, init_dy, init_dx in starts:
        visited = {(by, bx), (start_y, start_x)}
        path = [(start_y, start_x)]
        current = (start_y, start_x)

        for _ in range(max_steps):
            cy, cx = current
            next_pt = None
            for ddy in [-1, 0, 1]:
                for ddx in [-1, 0, 1]:
                    if ddy == 0 and ddx == 0:
                        continue
                    ny, nx = cy + ddy, cx + ddx
                    if 0 <= ny < h and 0 <= nx < w:
                        if skeleton[ny, nx] and (ny, nx) not in visited:
                            next_pt = (ny, nx)
                            break
                if next_pt:
                    break

            if next_pt is None:
                break
            if next_pt in bif_set and len(path) > 5:
                break

            visited.add(next_pt)
            path.append(next_pt)
            current = next_pt

        if len(path) >= 3:
            widths = [dist_transform[y, x] for y, x in path]
            branches.append({
                'length': len(path),
                'width': np.mean(widths),
                'direction': (init_dy, init_dx)
            })

    return branches

# ============================================================================

# Default Paths (can be overridden via CLI)
DEFAULT_TRAINING_CSV = Path("E:/AngioMLDL_data/corrected_dataset_training.csv")
DEFAULT_IMAGES_DIR = Path("E:/AngioMLDL_data/coco_format_v2")
DEFAULT_OUTPUT_DIR = Path("E:/AngioMLDL_data/coco_medis_bifurcation")

# Global paths (set at runtime)
TRAINING_CSV = DEFAULT_TRAINING_CSV
IMAGES_DIR = DEFAULT_IMAGES_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# CASS Categories
CASS_CATEGORIES = [
    {"id": 1, "name": "proximal_rca", "supercategory": "rca"},
    {"id": 2, "name": "mid_rca", "supercategory": "rca"},
    {"id": 3, "name": "distal_rca", "supercategory": "rca"},
    {"id": 4, "name": "pda", "supercategory": "rca"},
    {"id": 5, "name": "proximal_lad", "supercategory": "lad"},
    {"id": 6, "name": "mid_lad", "supercategory": "lad"},
    {"id": 7, "name": "distal_lad", "supercategory": "lad"},
    {"id": 8, "name": "diagonal_1", "supercategory": "lad"},
    {"id": 9, "name": "diagonal_2", "supercategory": "lad"},
    {"id": 10, "name": "proximal_lcx", "supercategory": "lcx"},
    {"id": 11, "name": "distal_lcx", "supercategory": "lcx"},
    {"id": 12, "name": "obtuse_marginal_1", "supercategory": "lcx"},
    {"id": 13, "name": "obtuse_marginal_2", "supercategory": "lcx"},
    {"id": 14, "name": "ramus", "supercategory": "lad"},
    {"id": 15, "name": "left_main", "supercategory": "lm"},
]

CATEGORY_NAME_TO_ID = {c["name"]: c["id"] for c in CASS_CATEGORIES}

# CASS segment chains - anatomical order
CASS_CHAINS = {
    "rca": ["proximal_rca", "mid_rca", "distal_rca", "pda"],
    "lad": ["left_main", "proximal_lad", "mid_lad", "distal_lad"],
    "lcx": ["left_main", "proximal_lcx", "distal_lcx"],
}

# Map CASS numeric codes to category names
CASS_CODE_TO_CATEGORY = {
    1: "proximal_rca", 2: "mid_rca", 3: "distal_rca", 4: "pda",
    12: "proximal_lad", 13: "mid_lad", 14: "distal_lad",
    15: "diagonal_1", 16: "diagonal_2",
    18: "proximal_lcx", 19: "obtuse_marginal_1", 20: "obtuse_marginal_2", 21: "distal_lcx",
    28: "ramus", 5: "left_main",
}


def get_artery_from_segment(segment_name):
    """Get the main artery type from a segment name."""
    if segment_name in ["proximal_rca", "mid_rca", "distal_rca", "pda"]:
        return "rca"
    elif segment_name in ["proximal_lad", "mid_lad", "distal_lad", "diagonal_1", "diagonal_2", "ramus", "left_main"]:
        return "lad"
    elif segment_name in ["proximal_lcx", "distal_lcx", "obtuse_marginal_1", "obtuse_marginal_2"]:
        return "lcx"
    return None


def create_medis_mask(left_edge, right_edge, image_shape):
    """Create mask from Medis left_edge + right_edge polygon."""
    h, w = image_shape[:2]

    if len(left_edge) == 0 or len(right_edge) == 0:
        return None

    left_pts = np.array(left_edge)
    right_pts = np.array(right_edge)[::-1]
    polygon_pts = np.vstack([left_pts, right_pts])

    poly_x = np.clip(polygon_pts[:, 0], 0, w - 1)
    poly_y = np.clip(polygon_pts[:, 1], 0, h - 1)

    rr, cc = draw_polygon(poly_y, poly_x, shape=(h, w))

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[rr, cc] = 1

    return mask


def get_centerline_direction(centerline, num_points=10, end="proximal"):
    """
    Get the direction vector from the centerline endpoints.

    For proximal end: direction points AWAY from the segment (toward ostium)
    For distal end: direction points AWAY from the segment (toward apex/PDA)

    Returns: (dy, dx) normalized direction vector
    """
    if len(centerline) < num_points:
        num_points = len(centerline) // 2

    if end == "proximal":
        # Use first few points, direction from inside toward outside
        pts = np.array(centerline[:num_points])  # [x, y] format
        # Direction: from later point to first point
        direction = pts[0] - pts[-1]
    else:  # distal
        # Use last few points
        pts = np.array(centerline[-num_points:])
        # Direction: from earlier point to last point
        direction = pts[-1] - pts[0]

    # Normalize
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return (0, 0)
    direction = direction / norm

    # Convert from [x, y] to (dy, dx)
    return (direction[1], direction[0])


def find_closest_skeleton_point(skeleton, target_point):
    """Find the skeleton point closest to target_point (y, x)."""
    skeleton_coords = np.array(list(zip(*np.where(skeleton))))
    if len(skeleton_coords) == 0:
        return None

    ty, tx = target_point
    distances = np.sqrt((skeleton_coords[:, 0] - ty)**2 +
                         (skeleton_coords[:, 1] - tx)**2)
    closest_idx = np.argmin(distances)

    return tuple(skeleton_coords[closest_idx])


def trace_with_angle_constraint(skeleton, start_point, initial_direction,
                                 max_angle_deviation=70, max_length=400,
                                 vessel_mask=None, min_vessel_width=2,
                                 strict_mode=False, view_info=None,
                                 bifurcation_points=None):
    """
    Trace along skeleton preferring straight paths and avoiding sharp turns.

    This prevents jumping into branches at bifurcations.

    Args:
        skeleton: Binary skeleton image
        start_point: (y, x) starting point
        initial_direction: (dy, dx) normalized direction to follow
        max_angle_deviation: Maximum angle deviation in degrees (reject sharper turns)
        max_length: Maximum path length in pixels
        vessel_mask: If provided, stop if vessel gets too narrow (entering branch)
        min_vessel_width: Minimum vessel width to continue
        strict_mode: If True, use stricter constraints (for proximal segments):
                     - Tighter angle threshold (45° instead of 70°)
                     - No direction smoothing (maintain initial direction)
        view_info: Dict with 'primary_angle', 'artery_type', 'trace_direction' for validation
        bifurcation_points: List of (y, x) major bifurcation points (v7.0)
                           Used as ADVISORY - tightens angle at bifurcations to stay on main vessel

    Returns:
        List of (y, x) points along the path
    """
    h, w = skeleton.shape
    path = [start_point]
    current = start_point

    # Initial direction (fixed in strict mode)
    initial_dir = np.array(initial_direction, dtype=float)
    if np.linalg.norm(initial_dir) < 1e-6:
        return path
    initial_dir = initial_dir / np.linalg.norm(initial_dir)

    current_dir = initial_dir.copy()

    # In strict mode, use tighter angle constraint
    if strict_mode:
        max_angle_deviation = min(max_angle_deviation, 45)

    visited = set()
    visited.add(start_point)

    # Get vessel width at start for reference
    if vessel_mask is not None:
        dist_transform = distance_transform_edt(vessel_mask)
        start_width = dist_transform[start_point[0], start_point[1]]
    else:
        start_width = None

    # For LAD/LCX bifurcation detection
    # In RAO (primary_angle > 0): LAD is LEFT, LCX is RIGHT
    # In LAO (primary_angle < 0): LAD is RIGHT, LCX is LEFT
    forbidden_direction = None
    if view_info is not None:
        primary_angle = view_info.get('primary_angle', 0)
        artery_type = view_info.get('artery_type', '')
        trace_dir = view_info.get('trace_direction', '')

        # For LAD tracing proximal: don't go toward LCX side
        if artery_type == 'lad' and trace_dir == 'proximal':
            if primary_angle > 0:  # RAO - LCX is on RIGHT
                forbidden_direction = 'right'  # Don't turn right toward LCX
            else:  # LAO - LCX is on LEFT
                forbidden_direction = 'left'  # Don't turn left toward LCX

    while len(path) < max_length:
        cy, cx = current

        # Find all skeleton neighbors not yet visited
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] and (ny, nx) not in visited:
                        neighbors.append((ny, nx, dy, dx))

        if not neighbors:
            break

        # Score each neighbor by how well it continues in current direction
        candidates = []
        for ny, nx, dy, dx in neighbors:
            neighbor_dir = np.array([dy, dx], dtype=float)
            neighbor_dir = neighbor_dir / np.linalg.norm(neighbor_dir)

            # Dot product = cos(angle)
            dot = np.dot(current_dir, neighbor_dir)
            angle_deg = np.degrees(np.arccos(np.clip(dot, -1, 1)))

            # Only consider neighbors within angle threshold
            if angle_deg <= max_angle_deviation:
                # Check forbidden direction for LAD/LCX bifurcation
                is_forbidden = False
                if forbidden_direction is not None:
                    if forbidden_direction == 'right' and dx > 0:
                        # Check if this is a significant turn right
                        cross = initial_dir[0] * dx - initial_dir[1] * dy
                        if cross < -0.3:  # Turning right relative to initial dir
                            is_forbidden = True
                    elif forbidden_direction == 'left' and dx < 0:
                        cross = initial_dir[0] * dx - initial_dir[1] * dy
                        if cross > 0.3:  # Turning left relative to initial dir
                            is_forbidden = True

                if not is_forbidden:
                    candidates.append((ny, nx, dy, dx, dot, angle_deg))

        if not candidates:
            # No valid continuation within angle threshold
            break

        # Choose the straightest path (highest dot product)
        candidates.sort(key=lambda x: x[4], reverse=True)
        ny, nx, dy, dx, dot, angle = candidates[0]

        # Check vessel width if mask provided
        if vessel_mask is not None and start_width is not None:
            current_width = dist_transform[ny, nx]
            # Stop if vessel becomes too narrow (likely entering a branch)
            if current_width < min_vessel_width or current_width < start_width * 0.3:
                break

        path.append((ny, nx))
        visited.add((ny, nx))
        current = (ny, nx)

        # v7.0: Check if we've reached a major bifurcation
        # Use as ADVISORY - validate we're on main vessel, but don't stop
        if bifurcation_points and len(path) > 10:
            for bif_y, bif_x in bifurcation_points:
                dist_to_bif = np.sqrt((ny - bif_y)**2 + (nx - bif_x)**2)
                if dist_to_bif < 8:  # Within 8 pixels of a bifurcation
                    # At a bifurcation - tighten angle constraint temporarily
                    # to ensure we continue on main vessel, not side branch
                    # This is advisory, not a hard stop
                    if not strict_mode:
                        # Temporarily use stricter angle for next few steps
                        max_angle_deviation = min(max_angle_deviation, 50)
                    break

        # Update direction - but NOT in strict mode
        if not strict_mode:
            # Update direction with smoothing (prevents oscillation)
            new_dir = np.array([dy, dx], dtype=float)
            new_dir = new_dir / np.linalg.norm(new_dir)
            current_dir = 0.7 * current_dir + 0.3 * new_dir
            current_dir = current_dir / np.linalg.norm(current_dir)
        # In strict mode, keep using initial_dir (already set above)

    return path


def create_path_mask(vessel_mask, path_points, max_dilation=None):
    """Create a mask covering vessel pixels along a path."""
    if not path_points or len(path_points) < 5:
        return None

    h, w = vessel_mask.shape

    # Create skeleton of path
    path_skeleton = np.zeros((h, w), dtype=np.uint8)
    for py, px in path_points:
        if 0 <= py < h and 0 <= px < w:
            path_skeleton[py, px] = 1

    if path_skeleton.sum() == 0:
        return None

    # Distance from path
    dist_from_path = distance_transform_edt(~path_skeleton.astype(bool))

    # Get vessel width along path
    vessel_dist = distance_transform_edt(vessel_mask)
    path_widths = vessel_dist[path_skeleton > 0]
    if len(path_widths) > 0:
        dilation_radius = np.median(path_widths) * 2.0
        if max_dilation:
            dilation_radius = min(dilation_radius, max_dilation)
    else:
        dilation_radius = 20

    # Mask = vessel pixels within radius of path
    mask = (vessel_mask > 0) & (dist_from_path <= dilation_radius)

    return mask.astype(np.uint8)


def mask_to_rle(mask):
    """Convert binary mask to COCO RLE format."""
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def mask_to_bbox(mask):
    """Get bounding box from mask in COCO format [x, y, width, height]."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def process_single_image(row_dict, image_id, src_images_dir):
    """Process a single image and return annotations."""
    try:
        deepsa_path = Path(row_dict['deepsa_pseudo_label_path'])
        contour_path = Path(row_dict['contours_path'])

        if not deepsa_path.exists() or not contour_path.exists():
            return {"success": False, "error": "Missing files"}

        base_name = deepsa_path.stem.replace('_full_vessel_mask', '')
        img_path = src_images_dir / f"{base_name}.png"

        if not img_path.exists():
            return {"success": False, "error": "Image not found"}

        # Load data
        image = Image.open(img_path).convert('RGB')
        image_array = np.array(image)
        img_h, img_w = image_array.shape[:2]

        vessel_mask = np.load(deepsa_path)

        with open(contour_path) as f:
            contour_data = json.load(f)

        # Resize vessel mask if needed
        if vessel_mask.shape != (img_h, img_w):
            mask_pil = Image.fromarray(vessel_mask.astype(np.uint8))
            mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
            vessel_mask = np.array(mask_pil)

        # Get Medis edges and centerline
        left_edge = contour_data.get('left_edge', [])
        right_edge = contour_data.get('right_edge', [])
        centerline = contour_data.get('centerline', [])

        # Get viewing angles for LAD/LCX validation
        view_angles = contour_data.get('view_angles', {})
        primary_angle = view_angles.get('primary_angle', 0)

        if len(left_edge) == 0 or len(right_edge) == 0:
            return {"success": False, "error": "No Medis edges"}

        if len(centerline) < 10:
            return {"success": False, "error": "Centerline too short"}

        # Get CASS segment from CSV
        cass_code = row_dict.get('cass_segment')
        if pd.isna(cass_code):
            return {"success": False, "error": "No cass_segment in CSV"}

        cass_code = int(cass_code)
        anchor_segment = CASS_CODE_TO_CATEGORY.get(cass_code)

        if anchor_segment is None:
            return {"success": False, "error": f"Unknown CASS code: {cass_code}"}

        artery = get_artery_from_segment(anchor_segment)
        if artery is None:
            return {"success": False, "error": f"Unknown artery for: {anchor_segment}"}

        # Create Medis GT mask from edges
        medis_mask = create_medis_mask(left_edge, right_edge, image_array.shape)

        if medis_mask is None or medis_mask.sum() < 100:
            return {"success": False, "error": "Medis mask too small"}

        # Clean vessel mask and skeletonize
        vessel_binary = (vessel_mask > 0).astype(np.uint8)
        skeleton = skeletonize(vessel_binary > 0)

        # v7.0: Detect major bifurcations for segment boundary detection
        bifurcation_points = detect_major_bifurcations(
            skeleton, vessel_binary,
            min_branch_length=40,
            min_branch_width=4,
            top_n=5
        )

        # === KEY: Use Medis centerline to determine directions ===

        # Proximal end of Medis segment (first centerline point)
        proximal_cl = centerline[0]  # [x, y]
        proximal_pt = (int(proximal_cl[1]), int(proximal_cl[0]))  # (y, x)

        # Distal end of Medis segment (last centerline point)
        distal_cl = centerline[-1]  # [x, y]
        distal_pt = (int(distal_cl[1]), int(distal_cl[0]))  # (y, x)

        # Get direction vectors from centerline
        # Proximal direction: points AWAY from Medis segment toward ostium
        proximal_dir = get_centerline_direction(centerline, num_points=15, end="proximal")

        # Distal direction: points AWAY from Medis segment toward apex/PDA
        distal_dir = get_centerline_direction(centerline, num_points=15, end="distal")

        # Find closest skeleton points to Medis endpoints
        prox_skeleton_pt = find_closest_skeleton_point(skeleton, proximal_pt)
        dist_skeleton_pt = find_closest_skeleton_point(skeleton, distal_pt)

        # Build segment dictionary - anchor is always the Medis mask
        segments = {}
        segments[anchor_segment] = medis_mask

        # Get CASS chain for this artery
        chain = CASS_CHAINS.get(artery, [])

        if anchor_segment in chain:
            anchor_idx = chain.index(anchor_segment)

            # Build view_info for LAD/LCX bifurcation handling
            view_info_proximal = {
                'primary_angle': primary_angle,
                'artery_type': artery,
                'trace_direction': 'proximal'
            }
            view_info_distal = {
                'primary_angle': primary_angle,
                'artery_type': artery,
                'trace_direction': 'distal'
            }

            # === Trace PROXIMAL (toward ostium) ===
            # Use strict_mode for proximal - proximal segments are straight
            if prox_skeleton_pt and proximal_dir != (0, 0) and anchor_idx > 0:
                path_proximal = trace_with_angle_constraint(
                    skeleton, prox_skeleton_pt, proximal_dir,
                    max_angle_deviation=70,
                    max_length=350,
                    vessel_mask=vessel_binary,
                    min_vessel_width=2,
                    strict_mode=True,  # No direction drift for proximal
                    view_info=view_info_proximal,  # LAD/LCX validation
                    bifurcation_points=bifurcation_points  # v7.0: advisory at bifurcations
                )

                if path_proximal and len(path_proximal) > 15:
                    before_mask = create_path_mask(vessel_binary, path_proximal)
                    if before_mask is not None:
                        # Remove overlap with Medis mask
                        before_mask = before_mask & (~medis_mask.astype(bool))
                        before_mask = before_mask.astype(np.uint8)
                        if before_mask.sum() > 100:
                            before_segment = chain[anchor_idx - 1]
                            segments[before_segment] = before_mask

            # === Trace DISTAL (toward apex/PDA) ===
            # Distal can curve more, so use regular mode (not strict)
            if dist_skeleton_pt and distal_dir != (0, 0) and anchor_idx < len(chain) - 1:
                path_distal = trace_with_angle_constraint(
                    skeleton, dist_skeleton_pt, distal_dir,
                    max_angle_deviation=70,
                    max_length=350,
                    vessel_mask=vessel_binary,
                    min_vessel_width=2,
                    strict_mode=False,  # Distal can curve
                    view_info=view_info_distal,
                    bifurcation_points=bifurcation_points  # v7.0: advisory at bifurcations
                )

                if path_distal and len(path_distal) > 15:
                    after_mask = create_path_mask(vessel_binary, path_distal)
                    if after_mask is not None:
                        # Remove overlap with Medis mask
                        after_mask = after_mask & (~medis_mask.astype(bool))
                        after_mask = after_mask.astype(np.uint8)
                        if after_mask.sum() > 100:
                            after_segment = chain[anchor_idx + 1]
                            segments[after_segment] = after_mask

        if not segments:
            return {"success": False, "error": "No valid segments created"}

        # Create annotations
        annotations = []
        for seg_name, seg_mask in segments.items():
            if seg_name not in CATEGORY_NAME_TO_ID:
                continue

            category_id = CATEGORY_NAME_TO_ID[seg_name]

            annotations.append({
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": mask_to_rle(seg_mask),
                "bbox": mask_to_bbox(seg_mask),
                "area": int(seg_mask.sum()),
                "iscrowd": 0,
                "is_anchor": seg_name == anchor_segment
            })

        image_info = {
            "id": image_id,
            "file_name": f"{base_name}.png",
            "width": img_w,
            "height": img_h
        }

        return {
            "success": True,
            "image_info": image_info,
            "annotations": annotations,
            "n_segments": len(annotations),
            "segment_names": list(segments.keys()),
            "anchor_segment": anchor_segment,
            "base_name": base_name
        }

    except Exception as e:
        import traceback
        return {"success": False, "error": f"{str(e)}\n{traceback.format_exc()}"}


def create_dataset(split="train", max_images=None):
    """Create COCO dataset for a split."""

    print(f"\n{'='*70}")
    print(f"Creating Medis+DeepSA CASS Dataset - {split}")
    print(f"Using angle-constrained tracing with Medis centerline direction")
    print(f"{'='*70}\n")

    df = pd.read_csv(TRAINING_CSV)

    if split == "train":
        df_split = df[df["split"] == "train"]
        src_images_dir = IMAGES_DIR / "train" / "images"
    else:
        df_split = df[df["split"].isin(["val", "test"])]
        src_images_dir = IMAGES_DIR / "val" / "images"

    print(f"Found {len(df_split)} samples in {split} split")

    split_dir = OUTPUT_DIR / split
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if max_images:
        df_split = df_split.head(max_images)

    results = []
    stats = defaultdict(int)
    error_details = defaultdict(int)

    for i, (_, row) in enumerate(tqdm(df_split.iterrows(), total=len(df_split), desc="Processing")):
        result = process_single_image(row.to_dict(), i + 1, src_images_dir)
        results.append(result)

        if result["success"]:
            for seg_name in result["segment_names"]:
                stats[seg_name] += 1
        else:
            stats["errors"] += 1
            error_msg = result.get("error", "Unknown")
            error_type = error_msg.split('\n')[0].split(':')[0] if error_msg else "Unknown"
            error_details[error_type] += 1

    # Collect successful results
    images = []
    annotations = []
    ann_id = 1
    total_annotations = 0
    anchor_count = 0

    for result in results:
        if result["success"]:
            images.append(result["image_info"])

            base_name = result["base_name"]
            src_img = src_images_dir / f"{base_name}.png"
            dst_img = images_dir / f"{base_name}.png"
            if src_img.exists() and not dst_img.exists():
                Image.open(src_img).save(dst_img)

            for ann in result["annotations"]:
                ann["id"] = ann_id
                if ann.get("is_anchor", False):
                    anchor_count += 1
                annotations.append(ann)
                ann_id += 1
                total_annotations += 1

    coco_json = {
        "info": {
            "description": f"Medis+DeepSA CASS Dataset - {split}",
            "version": "5.0",
            "year": 2025,
            "date_created": datetime.now().isoformat(),
            "notes": "Angle-constrained tracing using Medis centerline direction"
        },
        "licenses": [],
        "categories": CASS_CATEGORIES,
        "images": images,
        "annotations": annotations
    }

    json_path = split_dir / "annotations.json"
    with open(json_path, "w") as f:
        json.dump(coco_json, f)

    # Print stats
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Images: {len(images)}")
    print(f"Total annotations: {total_annotations}")
    print(f"Anchor (ground truth) annotations: {anchor_count}")
    print(f"Inferred annotations: {total_annotations - anchor_count}")
    print(f"Avg annotations per image: {total_annotations/max(len(images),1):.1f}")
    print(f"Errors: {stats['errors']}")

    if error_details:
        print(f"\nError breakdown:")
        for error_type, count in sorted(error_details.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")

    print(f"\nCategory distribution:")
    for cat in CASS_CATEGORIES:
        count = stats.get(cat['name'], 0)
        if count > 0:
            print(f"  {cat['name']}: {count}")
    print(f"\nSaved to: {json_path}")

    return coco_json


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Create CASS-labeled COCO dataset from Medis contours + DeepSA masks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    parser.add_argument("--max", type=int, default=None, help="Max images to process")

    # Configurable paths
    parser.add_argument("--csv", type=str, default=None,
                        help="Training CSV with image/contour paths (default: built-in)")
    parser.add_argument("--images", type=str, default=None,
                        help="Directory containing source images")
    parser.add_argument("--contours", type=str, default=None,
                        help="Directory containing Medis contour JSON files")
    parser.add_argument("--masks", type=str, default=None,
                        help="Directory containing DeepSA vessel masks")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for COCO dataset")

    args = parser.parse_args()

    # Override global paths if provided
    if args.csv:
        TRAINING_CSV = Path(args.csv)
    if args.images:
        IMAGES_DIR = Path(args.images)
    if args.output:
        OUTPUT_DIR = Path(args.output)

    print(f"Configuration:")
    print(f"  CSV:      {TRAINING_CSV}")
    print(f"  Images:   {IMAGES_DIR}")
    print(f"  Output:   {OUTPUT_DIR}")

    if args.split == "all":
        create_dataset("train", args.max)
        create_dataset("val", args.max)
    else:
        create_dataset(args.split, args.max)
