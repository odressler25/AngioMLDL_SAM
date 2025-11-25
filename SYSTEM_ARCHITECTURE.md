# SAM 3 Coronary Angiography System Architecture

Last updated: 2025-11-24

## Overview

Fine-tuning SAM 3 (Segment Anything Model 3) for coronary angiography analysis. The system learns to segment vessels, classify CASS segments, and detect obstructions.

## 4-Phase Training Plan

| Phase | Goal | Labels | Status |
|-------|------|--------|--------|
| 1 | Full vessel segmentation | DeepSA pseudo-labels | **Training** (Epoch 34, Val Dice 0.80) |
| 2 | CASS segment classification | Medis contours (bboxes) | Ready |
| 3 | Vessel obstruction detection | TBD | Pending |
| 4 | Quantitative measurements | Medis GT masks | Ready (masks aligned correctly) |

## Data Structure

### CSV: `E:\AngioMLDL_data\corrected_dataset_training.csv`

| Column | Description |
|--------|-------------|
| `patient_id` | Patient identifier (e.g., 101-0025) |
| `vessel_pattern` | Vessel location pattern (e.g., MID_LAD, PROX_RCA) |
| `phase` | PRE, FINAL, or EVENT |
| `vessel_name` | LAD, RCA, LCX, PDA, Ramus |
| `cine_path` | Path to cine video (.npy, shape: [frames, H, W, 3]) |
| `frame_index` | Which frame to use for analysis |
| `contours_path` | Medis contours JSON (centerline, edges, view angles) |
| `vessel_mask_actual_path` | Medis GT segment mask (correctly aligned) |
| `deepsa_pseudo_label_path` | DeepSA full vessel mask (512x512, binary) |
| `cass_segment` | CASS segment ID (corrected) |
| `cass_segment_original` | Original CASS segment from data |
| `split` | train/val |

### Dataset Statistics
- Total samples: 748
- Train: 521, Val: 227 (estimated)
- Image resolution: 1016x1016
- DeepSA masks: 512x512 (resized to match)

## CASS Segment Classification

Standard CASS (Coronary Artery Surgery Study) numbering:

### RCA and Branches (1-6)
| ID | Name | Count |
|----|------|-------|
| 1 | Proximal RCA | 42 |
| 2 | Mid RCA | 126 |
| 3 | Distal RCA | 28 |
| 4 | PDA | 38 |

### LAD and Branches (11-17, 29)
| ID | Name | Count |
|----|------|-------|
| 12 | Proximal LAD | 87 |
| 13 | Mid LAD | 192 |
| 14 | Distal LAD | 19 |
| 15 | 1st Diagonal | 55 |
| 16 | 2nd Diagonal | 13 |

### LCX and Branches (18-27)
| ID | Name | Count |
|----|------|-------|
| 18 | Proximal LCX | 26 |
| 19 | Mid/Distal LCX | 65 |
| 20 | OM1 | 29 |
| 21 | OM2 | 14 |

### Other
| ID | Name | Count |
|----|------|-------|
| 28 | Ramus Intermedius | 14 |

**Total: 14 classes, 748 samples**

## Model Architecture

### Phase 1: SAM3DeepSAModel

```
SAM 3 Backbone (840M params, full fine-tuning)
    |
    v
[Image Features] --> Feature Projection (if needed) --> 256 channels
    |
    +-- [View Angles] --> CategoricalViewEncoder --> 256-dim embedding
    |                          |
    v                          v
    ViewConditionedFeatureFusion (FiLM)
    |
    v
Segmentation Head --> 1-channel mask
```

#### CategoricalViewEncoder
- 9 bins per angle (-40 to +40 degrees, 10-degree intervals)
- Separate embeddings for primary (RAO/LAO) and secondary (CRAN/CAUD)
- Output: 256-dim view embedding
- Note: View angles not yet utilized by model (same task regardless of angle)

#### Training Config
- Optimizer: AdamW
- LR: 1e-4 (backbone 0.1x, heads 1x)
- Batch size: 4 per GPU (effective 8)
- Loss: 0.5 * BCE + 0.5 * Dice
- Scheduler: Cosine annealing, 50 epochs

### Phase 2: SAM3CASSModel (extends Phase 1)

```
Phase 1 Model (loaded from checkpoint)
    |
    +-- Segmentation output (maintained)
    |
    +-- Global pooled features + View embedding
            |
            v
        CASS Classifier --> 14 classes
            |
            v
        BBox Regressor --> [x_center, y_center, w, h]
```

#### Training Config
- Loss weights: 30% segmentation, 50% classification, 20% bbox
- LR: 5e-5 (lower, fine-tuning)
- This is where view angles become critical

## Training Infrastructure

### Hardware
- 2x NVIDIA RTX 3090 (24GB each)
- Target: ~20GB VRAM per GPU

### DDP Configuration
- Backend: gloo (Windows, NCCL not available)
- `broadcast_buffers=False` (fixes gloo scalar type error)
- Port: 12355 (Phase 1), 12356 (Phase 2)

## Key Files

| File | Purpose |
|------|---------|
| `train_stage1_deepsa.py` | Phase 1 training script |
| `train_stage2_cass_segments.py` | Phase 2 training script |
| `test_model_during_training.py` | Visualize predictions while training |
| `test_view_angle_impact.py` | Test if model uses view angles |
| `inspect_deepsa_labels.py` | Compare DeepSA vs Medis labels |

## Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `checkpoints/phase1_deepsa_best.pth` | Best Phase 1 model |
| `checkpoints/phase1_deepsa_latest.pth` | Latest Phase 1 model |
| `checkpoints/phase2_cass_best.pth` | Best Phase 2 model (future) |

## Known Issues

### 1. Medis GT Mask Alignment
- Medis GT masks appear misaligned with input images
- Possibly magnification/panning issue during annotation
- **Blocked**: Phase 4 training until resolved
- Model predictions actually align better with vessels than GT

### 2. View Angles Not Utilized in Phase 1
- Embeddings are different for different angles (avg similarity 0.62)
- But predictions don't change with angle (variance 0.00024)
- Expected: Phase 1 task doesn't require angle information
- Phase 2 (CASS classification) will force model to use angles

### 3. Student-Teacher Limitation
- Training on DeepSA pseudo-labels limits performance to DeepSA quality
- To surpass: need corrected labels or leverage additional info (temporal, multi-view)

## Training History

### Phase 1: Full Vessel Segmentation

| Date | Epoch | Train Dice | Val Dice | Notes |
|------|-------|------------|----------|-------|
| 2025-11-24 | 4 | 0.67 | 0.66 | Initial progress |
| 2025-11-24 | 6 | 0.71 | 0.70 | Steady improvement |
| 2025-11-24 | 9 | 0.74 | 0.72 | Current best |

## References

- MedSAM2: Full fine-tuning approach, 12 H100s with DDP
- SAM 3: 840M params, RoPE position encoding
- CASS: Coronary Artery Surgery Study segment numbering (1-29)
