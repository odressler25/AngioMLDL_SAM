# Stage 3: Obstruction Detection Training

## Overview

Fine-tune SAM3 (~840M parameters) for coronary artery obstruction detection, initializing from the Phase 1/2 vessel segmentation checkpoint.

## Dataset

**Location**: `E:\AngioMLDL_data\unified_obstruction_coco\`

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 508    | 502         |
| Val   | 125    | 124         |

**Ground Truth Source**: Medis QCA gold-standard contours
- **Actual contours**: Real vessel boundaries from angiogram
- **Ideal contours**: What the vessel should look like without obstruction
- **Obstruction mask**: `ideal_mask - actual_mask` (areas of stenosis/plaque)

**Category**: Single class - "obstruction" (id=1)

### Dataset Structure
```
E:\AngioMLDL_data\unified_obstruction_coco\
├── train/
│   ├── annotations.json    # COCO format
│   ├── 101-0043_PROX_RCA_PRE.png
│   └── ... (508 images)
├── val/
│   ├── annotations.json    # COCO format
│   └── ... (125 images)
└── train_config.yaml       # Legacy config (not used for SAM3)
```

## Model

- **Architecture**: SAM3 (Segment Anything Model 3) with DETR-based detector
- **Parameters**: ~840M
- **Initialization**: Phase 1/2 vessel segmentation checkpoint
- **Checkpoint**: `E:\AngioMLDL_data\experiments\stage2_bifurcation_v7\checkpoints\checkpoint.pt`

## Training Configuration

**Config file**: `configs/angiography/obstruction_detection.yaml`

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 672x672 | Same as Phase 1/2 |
| Batch size | 2 per GPU | Total 4 with 2 GPUs |
| Epochs | 100 | Extended for fine-tuning |
| LR scale | 0.01 | Half of Phase 1/2 (0.02) for fine-tuning |
| Warmup | 10 epochs | Reduced from 20 |
| GPUs | 2x RTX 3090 | 48GB total VRAM |

### Learning Rates (with lr_scale=0.01)
- Transformer: 8e-6
- Vision backbone: 2.5e-6
- Language backbone: 5e-7

## Training Command

```bash
cd C:/Users/odressler/AngioMLDL_SAM
python training/train_sam3_clean.py -c configs/angiography/obstruction_detection.yaml --num-gpus 2
```

## Output

- **Checkpoints**: `E:\AngioMLDL_data\experiments\obstruction_detection\checkpoints\`
- **TensorBoard**: `E:\AngioMLDL_data\experiments\obstruction_detection\tensorboard\`
- **Logs**: `E:\AngioMLDL_data\experiments\obstruction_detection\logs\`

## Evaluation Metrics

- COCO segmentation AP (primary)
- COCO AP@50, AP@75
- TIDE analysis (optional)

## Data Pipeline

1. **Label Generation**: `scripts/3_training_data/create_unified_obstruction_labels.py`
   - Processes Medis QCA contours from 377 patients
   - Filters by stenosis threshold (>30%)
   - Creates obstruction masks from actual vs ideal contours

2. **COCO Dataset Creation**: `scripts/3_training_data/create_unified_obstruction_coco.py`
   - Converts masks to COCO polygon format
   - Splits 80/20 train/val
   - Copies images to unified directory

## Notes

- Images are directly in `train/` and `val/` directories (not in `images/` subdirectory)
- Uses visual prompts (box-based) via `TextQueryToVisual` transform
- Activation checkpointing enabled for all components to fit in VRAM
- Gloo backend for distributed training (Windows compatible)
