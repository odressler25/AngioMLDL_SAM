# CASS Segment Labeling Algorithm (v7.0)

**Purpose**: Create CASS segment labels for training SAM3 on coronary artery segmentation.

**Approach**: Use Medis QCA contours as ground truth anchor + DeepSA vessel mask for adjacent segment inference + bifurcation-aware tracing.

---

## Input Data

For each angiography image, we have:

| Data Source | Description |
|-------------|-------------|
| Angiography Image (PNG) | Original X-ray angiogram |
| DeepSA Vessel Mask (NPY) | Full vessel segmentation from DeepSA model |
| Medis Contour JSON | QCA contours with `left_edge`, `right_edge`, `centerline`, `view_angles` |
| CASS Code (CSV) | Numeric segment code (e.g., 2=mid_rca, 13=mid_lad) |

---

## Algorithm Steps

### Step 1: Create Ground Truth (Anchor) Mask

The Medis contour defines the **exact** annotated segment:

```python
# Medis polygon = left_edge + reversed(right_edge)
polygon_points = np.vstack([left_edge, right_edge[::-1]])
anchor_mask = draw_polygon(polygon_points)
```

This gives us pixel-perfect ground truth for ONE segment per image.

---

### Step 2: Extract Direction from Centerline

The Medis centerline (~300-400 points) defines the exact vessel direction:

```python
# Proximal direction: points AWAY from segment toward ostium
# Use first 15 points of centerline
proximal_dir = normalize(centerline[0] - centerline[15])

# Distal direction: points AWAY from segment toward apex/PDA
# Use last 15 points of centerline
distal_dir = normalize(centerline[-1] - centerline[-15])
```

**Key insight**: The centerline coordinates tell us exactly which way is proximal vs distal - no guessing needed.

---

### Step 3: Prepare Skeleton for Tracing

```python
# Binarize and skeletonize the DeepSA vessel mask
vessel_binary = (deepsa_mask > 0).astype(np.uint8)
skeleton = skeletonize(vessel_binary)

# Find skeleton points closest to Medis centerline endpoints
prox_start_point = find_closest_skeleton_point(skeleton, centerline[0])
dist_start_point = find_closest_skeleton_point(skeleton, centerline[-1])
```

---

### Step 3b: Detect Major Bifurcations (v7.0)

```python
# Detect major bifurcation points for advisory guidance
bifurcation_points = detect_major_bifurcations(
    skeleton, vessel_binary,
    min_branch_length=40,   # Branches must extend 40+ pixels
    min_branch_width=4,     # Branches must be 4+ pixels wide
    top_n=5                 # Keep top 5 by importance score
)
```

**Bifurcation filtering criteria:**
1. Junction point with 3+ skeleton neighbors
2. At least 2 branches with length >= 40 pixels
3. At least 2 branches with width >= 4 pixels
4. Scored by `length × width`, top 5 kept

**Usage**: Bifurcations are used as **advisory checkpoints** - when tracing approaches a bifurcation, angle constraints are tightened to ensure we stay on the main vessel rather than jumping into a side branch. This is NOT a hard stop.

---

### Step 4: Trace Adjacent Segments

#### 4a. Proximal Tracing (STRICT MODE)

Proximal segments (e.g., proximal LAD, proximal RCA) are anatomically straight. We use **strict mode**:

```python
path_proximal = trace_with_angle_constraint(
    skeleton=skeleton,
    start_point=prox_start_point,
    initial_direction=proximal_dir,
    max_angle_deviation=45,    # Tight constraint
    strict_mode=True,          # NO direction smoothing
    view_info={
        'primary_angle': view_angles['primary_angle'],
        'artery_type': 'lad',  # or 'rca', 'lcx'
        'trace_direction': 'proximal'
    },
    bifurcation_points=bifurcation_points  # v7.0: advisory at bifurcations
)
```

**Strict mode rules**:
- Maximum 45 degree deviation from initial direction
- **No direction updates** during tracing - maintains Medis centerline direction throughout
- For LAD: forbids turning toward LCX side based on viewing angle

#### 4b. Distal Tracing (REGULAR MODE)

Distal segments can curve more (e.g., distal RCA wrapping around heart). We use **regular mode**:

```python
path_distal = trace_with_angle_constraint(
    skeleton=skeleton,
    start_point=dist_start_point,
    initial_direction=distal_dir,
    max_angle_deviation=70,    # More permissive
    strict_mode=False,         # Allow direction smoothing
    view_info={
        'primary_angle': view_angles['primary_angle'],
        'artery_type': artery,
        'trace_direction': 'distal'
    }
)
```

---

### Step 5: Angle-Constrained Tracing Algorithm

The core tracing algorithm at each skeleton pixel:

```
1. Find all unvisited skeleton neighbors (8-connectivity)

2. For each neighbor:
   - Calculate angle between current_direction and neighbor_direction
   - If angle > max_angle_deviation: reject (too sharp a turn)

3. Apply view angle validation (for LAD at LM bifurcation):
   - If RAO view (primary_angle > 0) and turning right: reject (toward LCX)
   - If LAO view (primary_angle < 0) and turning left: reject (toward LCX)

4. From valid candidates, select the one with smallest angle (straightest path)

5. Update direction:
   - If strict_mode=True: keep original direction unchanged
   - If strict_mode=False: smooth update (70% old + 30% new direction)

6. Stop conditions:
   - No valid neighbors within angle threshold
   - Vessel width drops below threshold (entering narrow branch)
   - Maximum path length reached
```

---

### Step 6: View Angle Validation

The viewing angle determines expected LAD vs LCX position:

```python
# primary_angle interpretation:
#   > 0: RAO (Right Anterior Oblique) - LAD appears on LEFT, LCX on RIGHT
#   < 0: LAO (Left Anterior Oblique) - LAD appears on RIGHT, LCX on LEFT

if artery_type == 'lad' and trace_direction == 'proximal':
    if primary_angle > 0:  # RAO view
        forbidden_direction = 'right'  # Don't turn right toward LCX
    else:  # LAO view
        forbidden_direction = 'left'   # Don't turn left toward LCX
```

This prevents the common error of jumping from LAD into LCX at the left main bifurcation.

---

### Step 7: Create Segment Masks from Traced Paths

```python
def create_path_mask(vessel_mask, path_points):
    # Create skeleton from path points
    path_skeleton = draw_points(path_points)

    # Calculate vessel width along path using distance transform
    vessel_dist = distance_transform_edt(vessel_mask)
    path_widths = vessel_dist[path_points]
    dilation_radius = median(path_widths) * 2.0

    # Dilate path and intersect with vessel mask
    dilated_path = distance_transform_edt(~path_skeleton) <= dilation_radius
    segment_mask = dilated_path & vessel_mask

    return segment_mask
```

---

### Step 8: Assign CASS Labels

CASS segments follow anatomical chains:

```python
CASS_CHAINS = {
    'rca': ['proximal_rca', 'mid_rca', 'distal_rca', 'pda'],
    'lad': ['left_main', 'proximal_lad', 'mid_lad', 'distal_lad'],
    'lcx': ['left_main', 'proximal_lcx', 'distal_lcx']
}

# Find anchor position in chain
anchor_idx = chain.index(anchor_segment)

# Assign labels based on position
if path_proximal:
    proximal_label = chain[anchor_idx - 1]  # Previous in chain
if path_distal:
    distal_label = chain[anchor_idx + 1]    # Next in chain
```

---

### Step 9: Remove Overlaps and Export

```python
# Remove overlap between inferred segments and anchor
proximal_mask = proximal_mask & ~anchor_mask
distal_mask = distal_mask & ~anchor_mask

# Export as COCO format with RLE encoding
annotations = [
    {'category': anchor_segment, 'mask': anchor_mask, 'is_anchor': True},
    {'category': proximal_label, 'mask': proximal_mask, 'is_anchor': False},
    {'category': distal_label, 'mask': distal_mask, 'is_anchor': False},
]
```

---

## Key Design Decisions

| Problem | Solution |
|---------|----------|
| **Direction reversal** (proximal/distal swapped) | Use Medis centerline coordinates to determine direction |
| **LCX jumping at LM bifurcation** | View angle validation + strict mode for proximal tracing |
| **Branch entry** (marginals, PLSA, diagonals) | Angle constraint (45-70 degrees) rejects sharp turns |
| **Aorta spillback** at ostium | Angle constraint prevents U-turns |
| **Proximal LAD is straight** | Strict mode = no direction smoothing, maintains initial direction |
| **Vessel width changes** | Stop tracing when vessel narrows significantly (entering branch) |
| **Staying on main vessel at bifurcations** (v7.0) | Advisory bifurcation detection - tighten angle at junctions |
| **Avoid premature segment termination** (v7.0) | Bifurcations are advisory, not hard stops |

---

## CASS Segment Codes

| Code | Segment | Artery |
|------|---------|--------|
| 1 | proximal_rca | RCA |
| 2 | mid_rca | RCA |
| 3 | distal_rca | RCA |
| 4 | pda | RCA |
| 5 | left_main | LM |
| 12 | proximal_lad | LAD |
| 13 | mid_lad | LAD |
| 14 | distal_lad | LAD |
| 15 | diagonal_1 | LAD |
| 16 | diagonal_2 | LAD |
| 18 | proximal_lcx | LCX |
| 19 | obtuse_marginal_1 | LCX |
| 20 | obtuse_marginal_2 | LCX |
| 21 | distal_lcx | LCX |
| 28 | ramus | LAD |

---

## Output Format

COCO-format JSON with:
- 15 CASS categories
- RLE-encoded segmentation masks
- `is_anchor` flag distinguishing ground truth (Medis) from inferred segments

---

## Usage

```bash
# Generate training dataset
python scripts/create_medis_bifurcation_labels.py --split train

# Generate validation dataset
python scripts/create_medis_bifurcation_labels.py --split val

# Generate both
python scripts/create_medis_bifurcation_labels.py --split all

# Visualize results
python scripts/visualize_medis_dino_labels.py --n 6
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | DINO clustering for segment boundaries |
| 2.0 | - | Bifurcation detection approach |
| 3.0 | - | Medis polygon as ground truth |
| 4.0 | - | Inside-out tracing from Medis mask |
| 5.0 | - | Angle-constrained tracing with centerline direction |
| 6.0 | Nov 2024 | Strict mode for proximal + view angle validation |
| 7.0 | Nov 2024 | **Advisory bifurcation detection** - detects major bifurcations and tightens angle constraints at junctions to stay on main vessel. Not a hard stop. |
