# SAM3 Full Training Pipeline

**Status:** Planning
**Created:** 2025-11-28
**Branch:** `stage2-cass-native`

---

## Overview

Complete training pipeline for coronary angiography analysis using SAM3, progressing from vessel segmentation to clinical measurements.

```
Stage 1: Vessel Segmentation ✅ (done - 0.8 Dice)
    ↓
Stage 2: Bifurcation Detection → CASS Segments (current)
    ↓
Stage 3+4: Lesion Detection + Measurements (planned)
    ↓
Clinical Report: "57% stenosis in Mid RCA, MLD 1.35mm"
```

---

## Architecture: Hybrid Two-Stage Approach

### Why Hybrid?

| Approach | Pros | Cons |
|----------|------|------|
| Fully Sequential | Easy to debug, modular | Error propagation, slow inference |
| Fully Multi-task | Fast inference, shared features | Hard to train, needs all labels on same images |
| **Hybrid** | Best of both - modular where needed, unified where beneficial | Slightly more complex |

### Our Hybrid Design

**Stage 2: Bifurcation Detection (separate model/head)**
- Different training data (CASS labeling output)
- Spatial reasoning task (landmark detection)
- Rule-based CASS segment inference from bifurcations

**Stage 3+4: Lesion + Measurements (multi-task)**
- Same training data (Medis QCA contours)
- Tightly coupled tasks (measurements depend on lesion location)
- Shared features beneficial

---

## Stage 1: Vessel Segmentation ✅

**Status:** Complete

**Performance:** 0.8 Dice score

**Output:** Binary vessel mask

**Checkpoint:** `checkpoints/phase1_native_format.pth`

**Details:** See `Doc/SAM3_NATIVE_PHASE1_IMPLEMENTATION.md`

---

## Stage 2: Bifurcation Detection

**Status:** In Progress

### Problem Statement

CASS segments are defined by anatomical landmarks (bifurcations), not visual appearance. Direct classification fails because:
- Proximal LAD looks identical to Mid LAD
- Segments are position-relative, not visually distinct

### Solution: Bifurcation-Based Approach

Train model to detect bifurcation points as landmarks, then apply CASS rules to label segments.

### Training Data

**Source:** CASS labeling pipeline output

**Location:** `E:/AngioMLDL_data/cass_labeling/`

**Count:** 747 images processed

**Format:** Per-image JSON with:
- Bifurcation points (x, y coordinates)
- Major bifurcation filtering (scored by branch_length × vessel_width)
- DINOv2 semantic features for branch differentiation
- Auto-generated CASS segment labels

### CASS Segment Definitions

**LAD (Left Anterior Descending):**
- Proximal LAD: Origin to first diagonal (D1)
- Mid LAD: D1 to second diagonal (D2)
- Distal LAD: After D2

**RCA (Right Coronary Artery):**
- Proximal RCA: Origin to first acute marginal (AM)
- Mid RCA: AM to crux (PDA origin)
- Distal RCA: After crux

**LCX (Left Circumflex):**
- Proximal LCX: Origin to first obtuse marginal (OM1)
- Distal LCX: After OM1

### Model Output

- **Bifurcation heatmap:** Probability map of bifurcation locations
- **CASS segments:** Rule-based inference from detected bifurcations

### Key Scripts

```
scripts/cass_anatomical_labeling.py      # Labeling pipeline
scripts/run_cass_labeling_parallel.py    # Batch processing (GPU)
scripts/detect_bifurcations.py           # Bifurcation detection
scripts/dinov2_vessel_features.py        # DINOv2 feature extraction
```

### Documentation

See `Doc/BIFURCATION_BASED_CASS_APPROACH.md` for detailed approach.

---

## Stage 3+4: Lesion Detection + Measurements

**Status:** Planned

### Training Data

**Source:** Medis QCA (Quantitative Coronary Angiography) manual annotations

**Location:** `E:/AngioMLDL_data/corrected_vessel_dataset/`

**Index:** `E:/AngioMLDL_data/corrected_dataset_training.csv`

**Count:** 777 contour files

### Data Structure

**CSV Columns:**
```
patient_id, vessel_pattern, phase, vessel_name, cass_segment,
cine_path, contours_path, vessel_mask_actual_path, frame_index,
MLD_mm, diameter_stenosis_pct, lesion_length_mm,
analyst, analysis_date, split
```

**Contour JSON Structure:**
```json
{
  "segment_name": "RCA Mid PRE",
  "frame_num": 37,
  "centerline": [[x, y], ...],           // 300+ points
  "left_edge": [[x, y], ...],            // Actual vessel wall
  "right_edge": [[x, y], ...],
  "ideal_left_edge": [[x, y], ...],      // Reconstructed healthy reference
  "ideal_right_edge": [[x, y], ...],
  "measurements": {
    "MLD_mm": 1.35,
    "MLD_x_coord": 274.7,                // Exact pixel location
    "MLD_y_coord": 441.9,
    "diameter_stenosis_pct": 57.3,
    "lesion_length_mm": 17.2,
    "lesion_start_x_coord": 285.1,       // Lesion bbox start
    "lesion_start_y_coord": 403.3,
    "lesion_end_x_coord": 271.2,         // Lesion bbox end
    "lesion_end_y_coord": 523.2,
    "interpolated_RVD_mm": 3.16,         // Reference vessel diameter
    "area_stenosis_pct": 81.8,
    "obstruction_position_mm": 21.6,
    "proximal_normal_diameter_mm": 3.33,
    "distal_normal_diameter_mm": 3.25
  },
  "calibration": {
    "scale_factor": 9.82,
    "catheter_diameter_mm": 8.0,
    "imager_pixel_spacing_mm": [0.197, 0.197]
  }
}
```

### Multi-Task Model Design

**Input:**
- Image (angiogram frame)
- Vessel mask (from Stage 1)
- CASS segment context (from Stage 2) - optional conditioning

**Output Heads:**

| Head | Task | Ground Truth | Loss |
|------|------|--------------|------|
| A | Lesion BBox | `lesion_start/end_x/y_coord` | IoU + L1 |
| B | MLD Point | `MLD_x/y_coord` | L2 (heatmap) |
| C | MLD (mm) | `MLD_mm` | L1 regression |
| D | % Stenosis | `diameter_stenosis_pct` | L1 regression |
| E | Lesion Length | `lesion_length_mm` | L1 regression |

**Optional Additional Outputs:**
- Area stenosis %
- Reference vessel diameter
- Proximal/distal normal diameters

### Training Strategy

1. **Stage 3 first:** Train lesion detection (bbox + MLD point)
2. **Add Stage 4:** Fine-tune with measurement regression heads
3. **Joint training:** End-to-end optimization with weighted multi-task loss

### Loss Function

```python
loss = (
    w1 * bbox_loss +           # IoU + L1 for lesion bbox
    w2 * mld_point_loss +      # Focal loss for MLD heatmap
    w3 * mld_mm_loss +         # L1 for MLD measurement
    w4 * stenosis_loss +       # L1 for % stenosis
    w5 * length_loss           # L1 for lesion length
)
```

Weight tuning required to balance spatial vs. measurement tasks.

---

## Data Flow at Inference

```
Input: Angiogram frame
         ↓
    ┌────────────────┐
    │ Stage 1        │
    │ Vessel Segment │ → Binary vessel mask
    └────────────────┘
         ↓
    ┌────────────────┐
    │ Stage 2        │
    │ Bifurcation    │ → Bifurcation points
    │ Detection      │ → CASS segment labels
    └────────────────┘
         ↓
    ┌────────────────┐
    │ Stage 3+4      │
    │ Lesion +       │ → Lesion bounding box
    │ Measurements   │ → MLD location
    └────────────────┘   → MLD (mm)
         ↓               → % stenosis
                         → Lesion length (mm)
    ┌────────────────┐
    │ Clinical       │
    │ Report         │ → "57% stenosis in Mid RCA"
    └────────────────┘   → "MLD: 1.35mm, Length: 17mm"
```

---

## Dataset Summary

| Dataset | Location | Count | Use |
|---------|----------|-------|-----|
| DeepSA pseudo-labels | `E:/AngioMLDL_data/deepsa_pseudo_labels/` | 747 | Stage 1 training |
| CASS labeling output | `E:/AngioMLDL_data/cass_labeling/` | 747 | Stage 2 training |
| Medis QCA contours | `E:/AngioMLDL_data/corrected_vessel_dataset/contours/` | 777 | Stage 3+4 training |
| Training index | `E:/AngioMLDL_data/corrected_dataset_training.csv` | 777 | Data loading |
| Cine sequences | `E:/AngioMLDL_data/corrected_vessel_dataset/cines/` | 777 | Full video (optional) |

---

## Hardware Configuration

**GPUs:** 2x NVIDIA RTX 3090 (24GB VRAM each, 48GB total)

**RAM:** 64GB

**Storage:** E: drive for all training data

**Batch Size Targets:**
- Stage 2: ~24 images per batch (targeting 20GB/GPU)
- Stage 3+4: TBD based on model complexity

---

## Implementation Roadmap

### Phase A: Stage 2 - Bifurcation Detection

1. [ ] Convert CASS labeling output to training format
   - Bifurcation heatmaps (Gaussian blobs at detected points)
   - CASS segment masks (from rule-based labeling)

2. [ ] Design bifurcation detection head for SAM3
   - Heatmap prediction (similar to keypoint detection)
   - Use Phase 1 weights as initialization

3. [ ] Train and evaluate
   - Metric: Bifurcation detection precision/recall
   - Metric: CASS segment classification accuracy

### Phase B: Stage 3+4 - Lesion + Measurements

1. [ ] Create training dataset from Medis contours
   - Extract lesion bboxes from start/end coords
   - Create MLD point heatmaps
   - Normalize measurements

2. [ ] Design multi-task heads
   - Lesion detection head (bbox regression)
   - MLD point head (heatmap)
   - Measurement regression heads

3. [ ] Train with multi-task loss
   - Start with detection only
   - Add measurement heads progressively
   - Tune loss weights

4. [ ] Evaluate
   - Lesion detection: IoU, precision, recall
   - MLD localization: Distance error (pixels, mm)
   - Measurements: MAE, correlation with GT

### Phase C: Integration

1. [ ] End-to-end pipeline testing
2. [ ] Clinical report generation
3. [ ] Comparison with Medis QCA (gold standard)

---

## Success Criteria

| Stage | Metric | Target |
|-------|--------|--------|
| Stage 1 | Dice score | >0.8 ✅ |
| Stage 2 | Bifurcation detection F1 | >0.8 |
| Stage 2 | CASS segment accuracy | >0.85 |
| Stage 3 | Lesion detection IoU | >0.7 |
| Stage 3 | MLD localization error | <5 pixels |
| Stage 4 | % stenosis MAE | <5% |
| Stage 4 | MLD MAE | <0.3mm |
| Stage 4 | Lesion length MAE | <2mm |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Bifurcation detection fails | Fall back to direct CASS classification with Medis GT |
| Multi-task loss imbalance | Uncertainty weighting or manual tuning |
| Measurement regression unstable | Train detection first, freeze, then add regression |
| Domain shift (PRE vs FINAL) | Include both phases in training, possibly as conditioning |

---

## References

- `Doc/BIFURCATION_BASED_CASS_APPROACH.md` - Bifurcation detection approach
- `Doc/STAGE2_CASS_TRAINING.md` - Previous Stage 2 attempts
- `Doc/SAM3_NATIVE_PHASE1_IMPLEMENTATION.md` - Stage 1 implementation
- `Doc/SAM3.md` - SAM3 architecture reference

---

*Last updated: 2025-11-28*
