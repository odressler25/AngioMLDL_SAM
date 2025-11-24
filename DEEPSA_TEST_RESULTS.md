# DeepSA Baseline Test Results

## Summary

**CRITICAL FINDING**: DeepSA performs WORSE than SAM 3 for segment-specific detection, but for a good reason!

| Model | Average IoU | Performance |
|-------|-------------|-------------|
| **SAM 3** | 0.372 | Baseline |
| **DeepSA** | 0.030 | **92% worse** |

## Detailed Results

| Case | Vessel | DeepSA IoU | DeepSA Dice | SAM 3 IoU (reference) |
|------|--------|------------|-------------|-----------------------|
| 101-0025 | RCA Mid | 0.043 | 0.083 | 0.194 |
| 101-0086 | LAD Mid | 0.039 | 0.074 | 0.680 |
| 101-0052 | LCX Dist | 0.007 | 0.014 | 0.243 |

## Why DeepSA "Failed"

### The Problem

DeepSA was trained for **GENERAL coronary vessel segmentation**, not **segment-specific detection**.

Looking at the visualization (101-0086_MID_LAD_PRE):
- ✅ **DeepSA prediction (green)**: Segments ALL visible coronary vessels (entire tree)
- ⚠️ **Ground truth (red)**: Only the specific LAD Mid segment being analyzed
- ❌ **Mismatch**: DeepSA segments the whole tree, we need only one segment

### What DeepSA Actually Does

DeepSA is doing **exactly what it was trained for**:
```
Input: Angiography frame
Output: Binary mask of ALL coronary vessels visible
```

Our task is different:
```
Input: Angiography frame + segment identifier
Output: Binary mask of SPECIFIC segment (e.g., "LAD Mid" only)
```

## The Fundamental Mismatch

### DeepSA's Training

From their paper:
- Trained on **FS-CAD dataset** (fine segmentation coronary angiogram dataset)
- Task: **Segment entire coronary tree**
- Achieved Dice 0.828 for **full tree segmentation**

### Our Task

- Segment **ONE specific CASS segment** (1 of 29 segments)
- Need to distinguish:
  - RCA Prox vs RCA Mid vs RCA Dist
  - LAD Prox vs LAD Mid vs LAD Dist
  - LCX vs OM1 vs OM2
- Requires **anatomical segment classification**, not just vessel detection

## Why SAM 3 Performed Better (Despite Being "General Purpose")

SAM 3 did better because we gave it **bbox prompts** that localized the specific segment:

```python
# SAM 3 approach
bbox = get_bbox_from_lesion_mask()  # Localizes JUST the LAD Mid segment
sam3_mask = sam3.segment(frame, bbox=bbox)  # Segments within that bbox
```

DeepSA has no concept of bboxes or segment identifiers - it always segments the full tree.

## Implications

### ✅ What This Means

1. **DeepSA works correctly** - it's segmenting vessels as designed
2. **DeepSA is solving a different task** - full tree vs specific segment
3. **SAM 3's prompting is the key** - bboxes enable segment-specific detection

### ❌ What This Doesn't Mean

1. DeepSA is bad - it's just solving a different problem
2. SAM 3 is inherently better - it's better for **prompted** segmentation
3. DeepSA's 0.828 Dice is wrong - that's for full tree segmentation (which it did)

## Path Forward

### Option 1: DeepSA → Segment Classifier (Two-Stage)

```python
# Stage 1: DeepSA segments full tree
full_tree_mask = deepsa_model(frame)

# Stage 2: Classify which part is which segment
segment_masks = segment_classifier(full_tree_mask, view_angles)
# segment_masks = {
#     'RCA_Prox': mask1,
#     'RCA_Mid': mask2,  # ← This is what we want!
#     'LAD_Prox': mask3,
#     ...
# }
```

**Pros:**
- DeepSA provides high-quality vessel segmentation
- Classifier only needs to partition vessels, not find them

**Cons:**
- Need to train segment classifier
- Two-stage pipeline is more complex

### Option 2: DeepSA → SAM 3 Prompts (Hybrid)

```python
# Stage 1: DeepSA segments full tree
full_tree_mask = deepsa_model(frame)

# Stage 2: Extract bbox for specific segment
# (Using anatomical rules or a lightweight classifier)
segment_bbox = extract_segment_bbox(full_tree_mask, segment_name="LAD Mid")

# Stage 3: SAM 3 refines within that segment
final_mask = sam3.segment(frame, bbox=segment_bbox)
```

**Pros:**
- Leverages DeepSA's vessel knowledge
- SAM 3 handles final refinement
- Simpler than full classifier

**Cons:**
- Still need bbox extraction logic
- Three-stage pipeline

### Option 3: Stick with SAM 3 + LoRA (Recommended)

Given that SAM 3 already outperforms DeepSA for our segment-specific task:

```python
# Just fine-tune SAM 3 with LoRA
model = SAM3WithLoRA()
model.train(800_cases_with_bbox_and_segment_labels)
```

**Pros:**
- SAM 3 already works better (0.372 vs 0.030)
- Prompt-based approach naturally handles segments
- Simpler pipeline
- Can add CASS classification head

**Cons:**
- Doesn't leverage DeepSA's vessel-specific knowledge

## Recommendation

### For Segment-Specific Detection: Use SAM 3

**Reasons:**
1. ✅ **Already performs 12x better** (0.372 vs 0.030 IoU)
2. ✅ **Prompting is natural** for our task (bboxes → specific segments)
3. ✅ **Simpler pipeline** (one model vs multi-stage)
4. ✅ **Can fine-tune** with LoRA on our 800 cases
5. ✅ **Can add segment classification** as additional head

### For Full Tree Segmentation: Use DeepSA

If we ever need to segment the **entire coronary tree** (all vessels):
- DeepSA is perfect for this (Dice 0.828)
- Use cases: vessel extraction, tree reconstruction, etc.

## Updated Strategy

### Week 1: SAM 3 + LoRA Fine-tuning

```bash
# Skip DeepSA for now
# Focus on SAM 3 + LoRA with our 800 segment-labeled cases

python train_sam3_lora.py \
    --cases 800 \
    --epochs 20 \
    --use_bbox_prompts \
    --use_text_prompts (CASS segment names) \
    --multi_task_heads (segment_id, stenosis_pct)
```

**Expected result**: 0.75-0.85 IoU

### Future: Hybrid Pipeline (Optional)

If SAM 3 + LoRA plateaus below 0.75 IoU, try:

```python
# Use DeepSA for coarse vessel localization
coarse_vessels = deepsa(frame)

# Use SAM 3 for segment-specific refinement
refined_segment = sam3(frame, bbox=from_coarse_vessels)
```

## Conclusion

**DeepSA is excellent at what it does** (full tree segmentation), but our task requires **segment-specific detection**.

**SAM 3 with bbox prompting is the right tool** for this job.

DeepSA's strength (segmenting all vessels) is actually a weakness for our use case (segmenting ONE specific segment).

---

**Next Steps**:
1. ✅ Tested DeepSA - confirmed it's not suitable out-of-box
2. ⏭️ Focus on SAM 3 + LoRA fine-tuning
3. ⏭️ Implement multi-task heads (CASS classification + stenosis prediction)
4. ⏭️ Train on 800+ expert-annotated cases
