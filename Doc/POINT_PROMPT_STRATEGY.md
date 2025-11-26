# Point Prompt Training Strategy for SAM3 Vessel Segmentation

## Problem Statement

When training SAM3 for coronary vessel segmentation with text prompts ("coronary artery"):
- **AP50 is high** (~94%) - model detects vessels well
- **AP75 is very low** (~4%) - mask boundaries are imprecise
- Text prompt "coronary artery" may be out-of-distribution for SAM3's text encoder (trained on COCO categories)

## Proposed Solution: Centerline Point Prompts

Instead of relying on text prompts, use **geometric point prompts** sampled from vessel centerlines.

### Why This Should Work

1. **SAM3 has full point prompt infrastructure** already implemented
2. **Distance transform sampling** (`center_positive_sample`) naturally finds centerline points
3. **No text encoder dependency** - bypasses the out-of-distribution issue
4. **Better for tubular structures** - points along centerline provide strong guidance
5. **Fixes thin vessel issues** - points don't have degenerate bounding box problems

### Implementation

#### Option A: Add Transform to Config (Easiest)

Add to `train_transforms` in `configs/angiography/vessel_segmentation.yaml`:

```yaml
# After ToTensorAPI transform, add:
- _target_: sam3.train.transforms.point_sampling.RandomGeometricInputsAPI
  num_points: [5, 15]           # Sample 5-15 points per vessel (more for long vessels)
  box_chance: 0.2               # 20% chance to also include box prompt
  point_sample_mode: "centered" # Uses distance transform - finds centerline points!
  box_noise_std: 0.1            # Add noise to boxes for robustness
  geometric_query_str: "visual" # Bypass text encoder
  resample_box_from_mask: true  # Compute box from mask (more accurate)
```

#### Option B: Pre-compute Centerlines in COCO Format

Modify `scripts/convert_to_coco.py` to include centerline points:

```python
from skimage.morphology import skeletonize

def extract_centerline_points(mask, n_points=10):
    """Extract centerline points from vessel mask."""
    # Skeletonize the mask
    skeleton = skeletonize(mask > 0)

    # Get skeleton points
    points = np.argwhere(skeleton)  # (y, x) format

    if len(points) == 0:
        return None

    # Sample n_points evenly along skeleton
    indices = np.linspace(0, len(points)-1, n_points, dtype=int)
    sampled = points[indices]

    # Convert to normalized (x, y, label) format
    h, w = mask.shape
    normalized = []
    for y, x in sampled:
        normalized.append([x/w, y/h, 1])  # 1 = positive label

    return normalized

# In annotation creation:
annotation = {
    'id': idx,
    'image_id': idx,
    'category_id': 1,
    'segmentation': rle,
    'bbox': bbox,
    'area': area,
    'input_points': extract_centerline_points(mask, n_points=10),  # ADD THIS
    'iscrowd': 0,
}
```

### SAM3 Point Sampling Modes

From `sam3/train/transforms/point_sampling.py`:

| Mode | Description | Best For |
|------|-------------|----------|
| `"centered"` | Distance transform - points farthest from edges | **Vessels (centerlines)** |
| `"random_mask"` | Uniform sampling from mask pixels | General objects |
| `"random_box"` | Uniform sampling from bounding box | Objects with background |

### Key Files

- `sam3/train/transforms/point_sampling.py` - Point sampling implementations
- `sam3/model/geometry_encoders.py` - `SequenceGeometryEncoder` encodes points/boxes
- `sam3/train/data/sam3_image_dataset.py` - Dataset supports `input_points` field

### When to Switch to Point Prompts

Monitor these metrics during training:
- If **AP75 plateaus below 20%** after 20-30 epochs → Switch to point prompts
- If **Dice loss plateaus above 0.15** → Switch to point prompts
- If model continues improving → Keep current text prompt approach

### Expected Benefits

1. **Better mask precision** - Points directly indicate vessel location
2. **Faster convergence** - No need to learn text-to-image alignment
3. **More robust** - Works regardless of category name
4. **Natural for vessels** - Centerline sampling matches vessel structure

### Hybrid Approach (Recommended)

Combine point prompts with occasional box prompts:
- **Points**: Provide precise vessel location guidance
- **Boxes**: Add robustness and help with vessel extent

```yaml
num_points: [5, 15]
box_chance: 0.2  # 20% include box
```

## Current Training Status (2025-11-26)

- Epoch 5: AP=0.330, AP50=0.940, AP75=0.044
- Model is learning detection well, mask precision still low
- Continuing training with text prompts to see if AP75 improves
- Will switch to point prompts if AP75 plateaus below 20%

## References

- SAM3 point sampling: `C:/Users/odressler/sam3/sam3/train/transforms/point_sampling.py`
- Geometry encoder: `C:/Users/odressler/sam3/sam3/model/geometry_encoders.py`
- Sam3Processor inference: `C:/Users/odressler/sam3/sam3/model/sam3_image_processor.py`
