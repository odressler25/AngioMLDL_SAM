# SAM 3 Coronary Angiography Project - Comprehensive Plan

## Document Purpose
This document serves as the authoritative reference for the SAM 3 fine-tuning project for coronary angiography analysis. It contains all critical context, decisions made, and planned next steps. **Read this document first when resuming work on this project.**

---

## 1. Project Overview

### 1.1 Goal
Fine-tune SAM 3 (Segment Anything Model 3) for coronary angiography analysis with the following capabilities:
1. **Vessel segmentation** - Segment coronary arteries from X-ray angiography images
2. **CASS segment classification** - Identify which coronary segment (proximal LAD, mid RCA, etc.)
3. **View angle conditioning** - Account for X-ray projection geometry (RAO/LAO, cranial/caudal)
4. **Lesion/obstruction detection** - Identify stenoses and occlusions

### 1.2 Why SAM 3?
- **Text prompting**: SAM 3 can segment based on text prompts like "proximal LAD"
- **Presence token**: Distinguishes between similar concepts (critical for CASS classification)
- **Proven fine-tuning**: ODinW13 and Roboflow results show 10-shot fine-tuning works
- **MedSAM2 precedent**: SAM/SAM2 successfully adapted for medical imaging

### 1.3 Key Discovery (November 2025)
We ARE using the real SAM 3 from the paper "SAM 3: Segment Anything with Concepts" (Meta, 2025):
- Repository: `C:\Users\odressler\sam3` (github.com/facebookresearch/sam3)
- Version: 0.1.0
- **Training code IS available** at `sam3/train/train.py`
- Uses Hydra configuration management

---

## 2. Data Sources

### 2.1 Primary Dataset CSV
**File**: `E:\AngioMLDL_data\corrected_dataset_training.csv`

| Column | Description |
|--------|-------------|
| `patient_id` | Patient identifier (e.g., "ALL RISE 101-0025") |
| `vessel_pattern` | Vessel + view (e.g., "MID RCA FINAL") |
| `phase` | Procedure phase (e.g., "INDEX BASELINE") |
| `frame_index` | **Correct frame to extract from DICOM** |
| `cass_segment` | CASS segment number (1-29) |
| `cass_segment_original` | Original uncorrected values |
| `primary_angle` | RAO(+)/LAO(-) angle in degrees |
| `secondary_angle` | Cranial(+)/Caudal(-) angle in degrees |
| `deepsa_pseudo_label_path` | Path to DeepSA pseudo-label mask |
| `dicom_path` | Path to source DICOM file |
| `medis_contour_path` | Path to Medis JSON contours |
| `medis_mask_path` | Path to Medis segment mask |

### 2.2 CASS Segment Numbering (Corrected)
**CRITICAL**: Original CSV had incorrect mappings. Corrected values:

| CASS # | Segment Name | Abbreviation |
|--------|-------------|--------------|
| 1 | Proximal RCA | pRCA |
| 2 | Mid RCA | mRCA |
| 3 | Distal RCA | dRCA |
| 4 | PDA (Posterior Descending) | PDA |
| 12 | Proximal LAD | pLAD |
| 13 | Mid LAD | mLAD |
| 14 | Distal LAD | dLAD |
| 15 | First Diagonal | D1 |
| 16 | Second Diagonal | D2 |
| 18 | Proximal LCX | pLCX |
| 19 | Distal LCX | dLCX |
| 20 | First Obtuse Marginal | OM1 |
| 21 | Second Obtuse Marginal | OM2 |
| 28 | Ramus Intermedius | Ramus |

### 2.3 DeepSA Pseudo-Labels
- **Location**: `E:\AngioMLDL_data\deepsa_pseudo_labels\`
- **Format**: 512x512, binary (0/1), uint8 PNG
- **Coverage**: 747/748 samples (100%)
- **Content**: Full coronary vessel tree segmentation
- **Quality**: DeepSA is a specialized coronary segmentation model; pseudo-labels are high quality but not perfect

### 2.4 Medis Ground Truth
- **Contours**: JSON files with `[x, y]` coordinate lists for centerline and boundaries
- **Masks**: PNG files with segment masks (NOT full vessel, just analyzed segment)
- **Alignment**: CONFIRMED CORRECT - No scaling needed between contours and masks
- **Use case**: Phase 4 training and quantitative validation

---

## 3. Current State (as of November 2025)

### 3.1 Git Branches
- `master` - Stable baseline with custom training approach
- `custom-training-backup` - Backup of custom training scripts
- `sam3-native-training` - **Current working branch** for SAM3 native training

### 3.2 Previous Training Results (Custom Approach)
Phase 1 training on DeepSA pseudo-labels achieved:
- **Epoch 38/50**: Val Dice 0.80, Train Dice 0.84
- Model showed slight overfitting (4% gap)
- Checkpoint: `checkpoints/phase1_deepsa_best.pth`

### 3.3 Key Files

| File | Purpose |
|------|---------|
| `train_stage1_deepsa.py` | Custom Phase 1 training (backup) |
| `train_stage2_cass_segments.py` | Custom Phase 2 CASS classification (backup) |
| `test_model_during_training.py` | Visualization during training |
| `test_view_angle_impact.py` | Experiment showing view angles don't affect segmentation |
| `SYSTEM_ARCHITECTURE.md` | Previous system documentation |

---

## 4. Revised Training Plan (SAM3 Native)

### 4.1 Phase 1: Vessel Segmentation with Text Prompts

**Goal**: Train SAM 3 to segment coronary vessels using text prompt "coronary artery"

**Approach**: Use SAM3's native training infrastructure with custom dataset

**Data Format Required**:
```
angio_dataset/
├── train/
│   ├── images/
│   │   ├── 001.png
│   │   └── ...
│   └── annotations.json  # COCO format
└── val/
    ├── images/
    └── annotations.json
```

**COCO Annotation Format**:
```json
{
  "images": [{"id": 1, "file_name": "001.png", "width": 1016, "height": 1016}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": {"counts": "...", "size": [1016, 1016]},  // RLE
      "bbox": [x, y, w, h],
      "area": 12345
    }
  ],
  "categories": [{"id": 1, "name": "coronary artery", "supercategory": "vessel"}]
}
```

**Configuration** (based on `odinw_text_only_train.yaml`):
- Resolution: 1008 (SAM3 default)
- Enable segmentation: True (not just bbox)
- Loss: Include mask loss (dice + focal)
- Text prompt: "coronary artery"

### 4.2 Phase 2: CASS Segment Classification with Text Prompts

**Goal**: Train SAM 3 to segment specific CASS segments using text prompts like "proximal LAD"

**Key Insight from SAM3 Paper**:
- Presence token critical for distinguishing similar concepts
- Hard negatives improve classification dramatically
- SAM3 already supports multi-label scenarios

**Data Format**:
```json
{
  "categories": [
    {"id": 1, "name": "proximal RCA"},
    {"id": 2, "name": "mid RCA"},
    {"id": 3, "name": "distal RCA"},
    {"id": 12, "name": "proximal LAD"},
    {"id": 13, "name": "mid LAD"},
    {"id": 14, "name": "distal LAD"},
    {"id": 15, "name": "first diagonal"},
    {"id": 16, "name": "second diagonal"},
    {"id": 18, "name": "proximal circumflex"},
    {"id": 19, "name": "distal circumflex"},
    {"id": 20, "name": "first obtuse marginal"},
    {"id": 21, "name": "second obtuse marginal"},
    {"id": 28, "name": "ramus intermedius"}
  ]
}
```

**Hard Negatives Strategy**:
For each training sample, include negative prompts for similar segments:
- If image shows "mid LAD", include negatives: "proximal LAD", "distal LAD", "first diagonal"
- SAM3's `include_negatives: true` flag handles this

### 4.3 Phase 3: View Angle Conditioning

**Goal**: Investigate if view angles improve CASS classification

**Approach Options**:
1. **Text prompt augmentation**: "proximal LAD from RAO 30 cranial 20"
2. **Visual exemplars**: Provide reference images from similar angles
3. **Custom conditioning**: Add view angle embeddings (requires model modification)

**Experiment Design**:
- Train with and without view angle information
- Compare CASS classification accuracy
- Hypothesis: View angles help distinguish ambiguous segments

### 4.4 Phase 4: Medis Ground Truth Fine-tuning

**Goal**: Fine-tune on Medis expert masks for quantitative measurements

**Data**: Medis segment masks (confirmed aligned correctly)

**Use Case**:
- Higher precision for QCA (Quantitative Coronary Analysis)
- Stenosis severity measurement
- Reference diameter calculation

---

## 5. Implementation Steps

### Step 1: Prepare COCO-Format Dataset
```python
# create_coco_dataset.py
# Convert CSV + DeepSA labels to COCO format for SAM3 training

# Key tasks:
# 1. Extract frames from DICOMs using frame_index
# 2. Resize DeepSA masks to match image size
# 3. Convert binary masks to RLE format
# 4. Generate COCO JSON with proper category names
# 5. Split into train/val (80/20)
```

### Step 2: Create SAM3 Training Config
```yaml
# configs/angiography/vessel_segmentation.yaml
# Based on odinw_text_only_train.yaml

paths:
  angio_data_root: E:/AngioMLDL_data/coco_format
  experiment_log_dir: C:/Users/odressler/AngioMLDL_SAM/experiments
  bpe_path: C:/Users/odressler/sam3/assets/bpe_simple_vocab_16e6.txt.gz

scratch:
  enable_segmentation: True  # CRITICAL: Enable mask loss
  resolution: 1008
  max_data_epochs: 50
  train_batch_size: 2  # Adjust for 2x RTX 3090

# ... (full config to be created)
```

### Step 3: Run Training
```bash
cd C:\Users\odressler\sam3
python sam3/train/train.py -c configs/angiography/vessel_segmentation.yaml --use-cluster 0 --num-gpus 2
```

### Step 4: Evaluate and Iterate
- Monitor TensorBoard
- Test on held-out samples
- Compare with custom training baseline

---

## 6. Technical Details

### 6.1 SAM3 Model Architecture
```
SAM3 (~850M parameters)
├── Perception Encoder (PE) backbone
│   ├── Vision encoder (~450M) - ViT-based
│   └── Text encoder (~300M) - BERT-based
├── DETR-style Detector
│   ├── Transformer decoder
│   ├── Presence token (classification)
│   └── Instance queries
└── Segmentation Head
    └── Mask decoder (similar to SAM1/SAM2)
```

### 6.2 Key SAM3 Training Components
- **Loss**: IABCE (Instance-Aware Binary Cross Entropy) + Box + GIoU + Mask (Dice + Focal)
- **Matcher**: Hungarian matching for query-to-GT assignment
- **Presence Token**: Predicts if concept exists in image (critical for hard negatives)
- **O2M (One-to-Many)**: Multiple predictions per query for crowded scenes

### 6.3 Hardware Configuration
- **GPUs**: 2x NVIDIA RTX 3090 (24GB each, 48GB total)
- **Target VRAM**: ~20GB per GPU (83% utilization)
- **Backend**: NCCL for multi-GPU (or gloo on Windows if NCCL unavailable)

### 6.4 Windows-Specific Notes
- Use `--use-cluster 0` for local training
- May need `multiprocessing_context: spawn` instead of `forkserver`
- NCCL may not work; fallback to gloo backend

---

## 7. Key Insights and Decisions

### 7.1 Why SAM3 Native Training Over Custom Approach?
1. **Official infrastructure**: Tested on ODinW13, Roboflow - proven to work
2. **Text prompting**: Can use "proximal LAD" directly instead of classifier head
3. **Hard negatives**: Built-in support via `include_negatives: true`
4. **Presence token**: Better classification for similar concepts
5. **Optimization**: Hydra configs, gradient clipping, proper schedulers

### 7.2 DeepSA vs Medis Labels
| Aspect | DeepSA | Medis |
|--------|--------|-------|
| Coverage | Full vessel tree | Only analyzed segment |
| Quality | Pseudo-labels (model output) | Expert ground truth |
| Use | Phase 1 (vessel segmentation) | Phase 4 (quantitative) |
| Alignment | Resized to match | Native resolution |

### 7.3 View Angles
- **Finding**: View angles don't affect segmentation task (same vessel regardless of angle)
- **Hypothesis**: View angles WILL help CASS classification (same segment looks different from different angles)
- **Experiment**: Phase 3 will test this hypothesis

### 7.4 CASS Segment Correction
Original CSV had incorrect mappings. Corrected and saved as:
- `cass_segment` - Corrected values
- `cass_segment_original` - Original (wrong) values

---

## 8. File Locations Summary

### Project Files
```
C:\Users\odressler\AngioMLDL_SAM\
├── PROJECT_PLAN_SAM3_NATIVE.md  # THIS DOCUMENT
├── SYSTEM_ARCHITECTURE.md       # Previous system docs
├── train_stage1_deepsa.py       # Custom Phase 1 (backup)
├── train_stage2_cass_segments.py # Custom Phase 2 (backup)
├── checkpoints/
│   └── phase1_deepsa_best.pth   # Best Phase 1 checkpoint
└── configs/                     # SAM3 training configs (to create)
    └── angiography/
        └── vessel_segmentation.yaml
```

### SAM3 Repository
```
C:\Users\odressler\sam3\
├── sam3/
│   ├── model/              # Model architecture
│   ├── train/
│   │   ├── train.py       # Main training script
│   │   └── configs/       # Example configs
│   └── eval/              # Evaluation scripts
├── README.md              # Installation instructions
└── README_TRAIN.md        # Training instructions
```

### Data Files
```
E:\AngioMLDL_data\
├── corrected_dataset_training.csv  # Master CSV
├── deepsa_pseudo_labels/           # DeepSA masks (512x512)
└── coco_format/                    # To be created
    ├── train/
    └── val/

E:\Angios\                          # Raw DICOM files
└── ALL RISE {patient_id}/
    └── Analysis/PROCEDURES QCA ANALYSIS/{phase}/{vessel_pattern}.dcm
```

---

## 9. Next Steps (Prioritized)

1. **Create COCO dataset converter** (`create_coco_dataset.py`)
   - Extract frames from DICOMs
   - Convert DeepSA masks to RLE
   - Generate COCO JSON

2. **Create SAM3 training config** (`configs/angiography/vessel_segmentation.yaml`)
   - Enable segmentation
   - Set up for 2x RTX 3090
   - Configure text prompts

3. **Test training pipeline**
   - Small subset first (10 images)
   - Verify loss decreasing
   - Check mask outputs

4. **Full Phase 1 training**
   - Train on all 748 samples
   - Compare with custom training baseline (Dice 0.80)

5. **Phase 2: CASS classification**
   - Add CASS segment categories
   - Implement hard negatives
   - Train and evaluate

---

## 10. Contact and Resources

- **SAM3 Paper**: "SAM 3: Segment Anything with Concepts" (Meta, 2025)
- **SAM3 GitHub**: https://github.com/facebookresearch/sam3
- **MedSAM2 Paper**: Reference for medical imaging fine-tuning
- **DeepSA**: Coronary artery segmentation model used for pseudo-labels

---

*Last Updated: November 25, 2025*
*Branch: sam3-native-training*
