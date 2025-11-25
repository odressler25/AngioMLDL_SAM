# SAM 3 Coronary Angiography System Architecture

Last updated: 2025-11-25

## IMPORTANT: Read PROJECT_PLAN_SAM3_NATIVE.md First

**This document is now supplementary.** The main project plan is in `PROJECT_PLAN_SAM3_NATIVE.md`, which contains:
- Complete project context and history
- Data source locations and formats
- SAM3 native training approach
- Detailed implementation steps

---

## Overview

Fine-tuning SAM 3 (Segment Anything Model 3) for coronary angiography analysis.

**Key Discovery (Nov 25, 2025)**: We are using the REAL SAM 3 from Meta's paper "SAM 3: Segment Anything with Concepts". The repository includes official training code at `sam3/train/train.py`.

## Git Branches

| Branch | Purpose |
|--------|---------|
| `master` | Stable baseline |
| `custom-training-backup` | Backup of custom training approach |
| `sam3-native-training` | **Current** - SAM3 native training |

## Training Approach Evolution

### Previous: Custom Training (Backup)
- Custom `train_stage1_deepsa.py` with manual SAM3 integration
- CategoricalViewEncoder for view conditioning
- Achieved Val Dice 0.80 at epoch 38/50

### Current: SAM3 Native Training
- Uses `sam3/train/train.py` with Hydra configs
- Text prompts like "coronary artery", "proximal LAD"
- Built-in hard negatives and presence token
- COCO format dataset required

## 4-Phase Training Plan

| Phase | Goal | Labels | Status |
|-------|------|--------|--------|
| 1 | Full vessel segmentation | DeepSA pseudo-labels | **Migrating to SAM3 native** |
| 2 | CASS segment classification | Medis contours | Planned with text prompts |
| 3 | View angle conditioning | - | Experiment design ready |
| 4 | Quantitative measurements | Medis GT masks | Ready (masks confirmed aligned) |

## Data Sources

### Primary CSV
**Path**: `E:\AngioMLDL_data\corrected_dataset_training.csv`

Key columns:
- `frame_index` - Correct frame to extract from DICOM
- `cass_segment` - Corrected CASS segment number
- `deepsa_pseudo_label_path` - Path to DeepSA mask
- `primary_angle`, `secondary_angle` - View angles in degrees

### CASS Segments (Corrected)

| ID | Segment | Count |
|----|---------|-------|
| 1 | Proximal RCA | 42 |
| 2 | Mid RCA | 126 |
| 3 | Distal RCA | 28 |
| 4 | PDA | 38 |
| 12 | Proximal LAD | 87 |
| 13 | Mid LAD | 192 |
| 14 | Distal LAD | 19 |
| 15 | 1st Diagonal | 55 |
| 16 | 2nd Diagonal | 13 |
| 18 | Proximal LCX | 26 |
| 19 | Distal LCX | 65 |
| 20 | OM1 | 29 |
| 21 | OM2 | 14 |
| 28 | Ramus | 14 |

**Total: 14 classes, 748 samples**

## SAM3 Model Architecture

```
SAM3 (~850M parameters)
├── Perception Encoder (PE) backbone
│   ├── Vision encoder (~450M) - ViT-based
│   └── Text encoder (~300M) - BERT-based
├── DETR-style Detector
│   ├── Transformer decoder
│   ├── Presence token (critical for classification)
│   └── Instance queries
└── Segmentation Head
    └── Mask decoder
```

Key components:
- **Presence token**: Distinguishes similar concepts (e.g., "proximal LAD" vs "mid LAD")
- **Hard negatives**: Built-in support for improved classification
- **Text prompting**: Segment based on natural language

## Hardware Configuration

- **GPUs**: 2x NVIDIA RTX 3090 (24GB each)
- **Target VRAM**: ~20GB per GPU
- **Backend**: gloo (Windows) or NCCL (Linux)

## Key Files

| File | Purpose |
|------|---------|
| `PROJECT_PLAN_SAM3_NATIVE.md` | **Main project plan** |
| `train_stage1_deepsa.py` | Custom Phase 1 (backup) |
| `train_stage2_cass_segments.py` | Custom Phase 2 (backup) |
| `checkpoints/phase1_deepsa_best.pth` | Best custom training checkpoint |

## SAM3 Repository

**Location**: `C:\Users\odressler\sam3`

Key paths:
- `sam3/train/train.py` - Main training script
- `sam3/train/configs/` - Example Hydra configs
- `assets/bpe_simple_vocab_16e6.txt.gz` - BPE tokenizer

## Next Steps

See `PROJECT_PLAN_SAM3_NATIVE.md` Section 9 for prioritized implementation steps.

## Training History

### Custom Training (Backup Branch)
| Date | Epoch | Train Dice | Val Dice |
|------|-------|------------|----------|
| 2025-11-24 | 38 | 0.84 | 0.80 |

### SAM3 Native Training
| Date | Phase | Metric | Notes |
|------|-------|--------|-------|
| TBD | 1 | - | Migration in progress |

## Resolved Issues

### 1. Medis GT Mask Alignment - RESOLVED
- **Was**: Thought masks misaligned, Phase 4 blocked
- **Resolution**: Masks are correctly aligned; no scaling needed
- **Status**: Phase 4 ready

### 2. View Angles Not Used in Phase 1 - EXPECTED
- Phase 1 (segmentation) doesn't require view angles
- Phase 2 (CASS classification) will leverage them
- SAM3 text prompts may naturally handle this

### 3. SAM3 Training Infrastructure - AVAILABLE
- **Was**: Assumed only inference code released
- **Resolution**: Full training code at `sam3/train/train.py`
- **Status**: Migrating to native training

## References

- SAM 3 Paper: "SAM 3: Segment Anything with Concepts" (Meta, 2025)
- SAM 3 GitHub: https://github.com/facebookresearch/sam3
- MedSAM2: Reference for medical imaging fine-tuning
- CASS: Coronary Artery Surgery Study segment numbering
