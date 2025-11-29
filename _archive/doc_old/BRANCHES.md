# Branch Documentation

## Overview

This repository uses multiple branches to track different training approaches for SAM3 on coronary angiography data.

---

## Branch: `master`

**Purpose:** Custom training approach (working baseline)

**Status:** Stable - Phase 1 complete

**Key Files:**
- `train_stage1_deepsa.py` - Custom training script with view encoder
- `train_stage2_cass_segments.py` - Custom Stage 2 script (not yet run)

**Results:**
- Phase 1: **0.8 Dice** on full vessel segmentation (epoch 46)
- Checkpoint: `checkpoints/phase1_deepsa_best.pth`

**Approach:**
- Custom model wrapper (`SAM3DeepSAModel`)
- Custom segmentation head
- `CategoricalViewEncoder` for view angles
- DeepSA pseudo-labels as ground truth

---

## Branch: `sam3-native-training`

**Purpose:** SAM3 native training experiments (Phase 1)

**Status:** Archived - Experiments showed limitations

**Key Files:**
- `configs/angiography/vessel_segmentation.yaml` - Native config
- `scripts/convert_to_coco.py` - DeepSA to COCO converter
- `Doc/SAM3_NATIVE_PHASE1_IMPLEMENTATION.md` - Full documentation

**Results:**
- Phase 1 Native: **AP75 = 3.7%** (poor mask precision)
- Detection worked (AP50 = 94%), segmentation failed

**Lessons Learned:**
- DeepSA masks (giant connected blobs) confuse SAM3's instance head
- Need domain-adapted initialization, not vanilla SAM3
- Semantic vs Instance mismatch was the core issue

---

## Branch: `stage2-cass-native` (CURRENT)

**Purpose:** Stage 2 CASS segment classification using SAM3 native training

**Status:** ✅ Training in progress (started 2025-11-27)

**Key Files:**
- `train_sam3_clean.py` - Training launcher with RoPE fix
- `configs/angiography/cass_segmentation.yaml` - Stage 2 native config
- `scripts/create_coco_cass.py` - Medis GT to COCO converter
- `scripts/weight_surgery.py` - Checkpoint format converter
- `checkpoints/phase1_native_format.pth` - Converted checkpoint
- `Doc/STAGE2_CASS_TRAINING.md` - Full documentation

**Approach:**
- SAM3 native training pipeline
- **14 CASS segment categories** (proximal_lad, mid_rca, etc.)
- **Medis GT masks** (discrete segments, not blobs)
- **Initialize from Phase 1 checkpoint** (0.8 Dice, domain-adapted)
- **Resolution 672px** (RoPE regeneration required)
- Text prompts for each segment category

**Data:**
- `E:/AngioMLDL_data/coco_cass_segments/` - COCO format with CASS categories
- Train: 521 samples, Val: 227 samples

**Initial Results (Epoch 0):**
```
Loss=238.61 | Dice=0.6302 | CE=0.0017 | BBox=0.1017 | GIoU=0.4728 | Mask=0.0097
```

**Key Success Factors:**
1. Medis masks are discrete objects (not giant blobs)
2. SAM3's instance head is designed for individual objects
3. Backbone already knows what angiograms look like (from Phase 1)
4. RoPE embeddings regenerated for resolution 672 (gpt5.1's fix)
5. Gloo backend compatibility (float32 conversion)

---

## Branch: `custom-training-backup`

**Purpose:** Backup of early custom training experiments

**Status:** Archived

---

## Summary Table

| Branch | Approach | Phase 1 | Phase 2 | Status |
|--------|----------|---------|---------|--------|
| `master` | Custom | Done (0.8 Dice) | Script ready | Stable |
| `sam3-native-training` | Native | Failed (3.7% AP75) | - | Archived |
| `stage2-cass-native` | Native + Domain Init | Using master's (0.8 Dice) | Training (Dice=0.63 @ epoch 0) | Active |

---

## Merge Strategy

When Stage 2 native training succeeds:
1. Merge `stage2-cass-native` → `master`
2. Archive `sam3-native-training`
3. Continue Stage 3 development on `master`

---

*Last updated: 2025-11-27*
