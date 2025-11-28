# Vessel Context Training Options

**Goal**: Improve CASS segment classification by providing vessel context from DeepSA pseudo labels.

**Problem**: Training on isolated segment masks loses context of where the segment sits within the full vessel tree.

---

## Option 1: Mask Prompt (Most SAM3-native)

SAM3 supports **mask prompts** as input hints.

**Approach**:
- Input: DeepSA full vessel mask as a "hint" prompt
- Text prompt: CASS segment name (e.g., "rca mid segment")
- Target: CASS segment mask

**Implementation**:
1. For each CASS annotation, find corresponding DeepSA vessel mask (same image)
2. Provide DeepSA mask as `input_mask` prompt in the data loader
3. Model learns: "Given this vessel region, find this specific segment"

**Pros**:
- Native SAM3 capability
- No architecture changes
- Explicit spatial context

**Cons**:
- Need to modify data loader to provide mask prompts
- Requires matching DeepSA masks to CASS images

**Status**: Not attempted

---

## Option 2: Composite Visualization Training

Bake vessel context directly into the training images.

**Approach**:
- Overlay DeepSA vessel mask as semi-transparent tint on the input image
- Keep original CASS segment as target mask
- Model sees vessel highlighted in the image itself

**Implementation**:
1. Load image + DeepSA vessel mask
2. Apply subtle color overlay (e.g., 10-20% opacity) to vessel region
3. Train normally with CASS targets

**Pros**:
- Simple implementation
- No architecture changes
- No data loader changes

**Cons**:
- Modifies input images
- May not generalize to clean images at inference
- Need matching DeepSA masks

**Status**: Not attempted

---

## Option 3: Two-Head Training (Multi-task)

Train both vessel segmentation and segment classification simultaneously.

**Approach**:
- Semantic head: Full vessel segmentation (DeepSA labels)
- Instance head: CASS segment classification
- Shared backbone learns vessel features

**Implementation**:
1. Create dataset with both DeepSA and CASS annotations per image
2. Configure SAM3 for multi-task training
3. Loss = semantic_loss + instance_loss

**Pros**:
- Multi-task learning improves representations
- Backbone learns vessel structure
- Explicit vessel supervision

**Cons**:
- More complex training setup
- Need both annotation types per image
- May require custom collation

**Status**: Not attempted

---

## Option 4: Hierarchical Annotations with Spatial Constraint

Add explicit parent-child relationship between vessel and segment masks.

**Approach**:
- Each CASS annotation includes parent vessel mask
- Custom loss: segment must be geometrically INSIDE parent vessel
- Penalize predictions outside vessel boundary

**Implementation**:
```python
def hierarchical_loss(pred_segment, gt_segment, parent_vessel):
    base_loss = dice_loss(pred_segment, gt_segment)
    # Penalize predictions outside vessel
    outside_vessel = pred_segment * (1 - parent_vessel)
    constraint_loss = outside_vessel.sum()
    return base_loss + lambda * constraint_loss
```

**Pros**:
- Explicit spatial constraint
- Model learns vessel boundaries matter
- Geometrically grounded

**Cons**:
- Requires custom loss function
- Need to modify trainer
- More complex implementation

**Status**: Not attempted

---

## Option 5: Text Prompt Engineering

Improve category names to be more descriptive and hierarchical.

**Approach**:
- Change from "mid_rca" to more descriptive text
- Options:
  - "RCA mid" (vessel-first hierarchy)
  - "mid segment of right coronary artery"
  - "right coronary artery, middle third"

**Implementation**:
1. Update category names in COCO JSON
2. No code changes needed

**Pros**:
- Simplest to implement
- No architecture/training changes
- Can combine with other options

**Cons**:
- May have limited impact
- Depends on text encoder's understanding

**Status**: Not attempted

---

## Option 6: Point Prompts from Vessel Centerline

Use DeepSA vessel to derive point prompts along the vessel.

**Approach**:
- Extract vessel centerline from DeepSA mask (skeletonization)
- Sample points along centerline as prompts
- Text prompt: segment name
- Target: CASS segment mask

**Implementation**:
1. Skeletonize DeepSA vessel mask
2. Sample N points along skeleton
3. Provide as point prompts to SAM3

**Pros**:
- Uses SAM3's point prompt capability
- Provides spatial guidance
- Natural for vessel structures

**Cons**:
- Centerline extraction can be noisy
- Need to handle branching vessels
- Additional preprocessing

**Status**: Not attempted

---

## Recommended Order of Experiments

1. **Option 5** (Text prompts) - Simplest, try "RCA mid" vs "mid RCA" format
2. **Option 1** (Mask prompts) - Most SAM3-native if Option 5 insufficient
3. **Option 3** (Two-head) - If explicit vessel supervision needed
4. **Option 2** (Composite) - Quick hack if others fail
5. **Option 4** (Hierarchical loss) - If spatial constraints needed
6. **Option 6** (Point prompts) - If point-based guidance helps

---

## Data Requirements

| Option | DeepSA Masks | CASS Masks | Code Changes |
|--------|--------------|------------|--------------|
| 1 | Yes | Yes | Data loader |
| 2 | Yes | Yes | Preprocessing |
| 3 | Yes | Yes | Training config |
| 4 | Yes | Yes | Loss function |
| 5 | No | Yes | JSON only |
| 6 | Yes | Yes | Preprocessing + data loader |

---

*Created: 2025-11-27*
