# SAM3 Native Training: Phase 1 Implementation Details

## Overview

This document details the implementation of Phase 1 training using SAM3's native training pipeline on the `sam3-native-training` branch. This approach differs from the custom training script (`train_stage1_deepsa.py`) on the `master` branch which achieved 0.8 Dice score.

**Branch:** `sam3-native-training`
**Status:** Experimental - Results were poor (AP75 = 3.7%)

---

## 1. Data Source: DeepSA Pseudo-Labels

### Source
- **DeepSA Model:** Pre-trained vessel segmentation model (`DeepSA/ckpt/fscad_36249.ckpt`)
- **Performance:** Dice 0.828 on full coronary tree segmentation
- **Output:** Full vessel tree masks (all visible vessels, not individual segments)

### Index File
- **Location:** `E:/AngioMLDL_data/deepsa_pseudo_labels/index.csv`
- **Contents:** Mapping of patient cases to their DeepSA mask files
- **Splits:** Train (521), Val (227)

### Mask Characteristics
- **Resolution:** 512x512 (resized from original ~1016x1016)
- **Format:** Binary numpy arrays
- **Content:** Full coronary tree (all vessels visible in frame)

---

## 2. COCO Format Conversion

### Script
`scripts/convert_to_coco.py`

### Process
1. Read `index.csv` to get case metadata
2. Load cine video (`.npy`) and extract frame at `frame_index`
3. Load corresponding DeepSA mask
4. Resize mask to match frame dimensions if needed
5. Encode mask as RLE (Run-Length Encoding) for COCO format
6. Compute bounding box from mask
7. Save image as PNG, annotation as JSON

### Output Structure
```
E:/AngioMLDL_data/coco_format_v2/
├── train/
│   ├── images/           # 512 PNG images
│   └── annotations.json  # COCO format annotations
└── val/
    ├── images/           # 227 PNG images
    └── annotations.json
```

### Category Definition
```json
{
    "categories": [
        {
            "id": 1,
            "name": "coronary artery",
            "supercategory": "vessel"
        }
    ]
}
```

**Note:** Single category "coronary artery" - no distinction between vessel types or CASS segments.

---

## 3. SAM3 Training Configuration

### Config File
`configs/angiography/vessel_segmentation.yaml`

### Key Configuration Decisions

#### 3.1 Resolution
```yaml
scratch:
  resolution: 1008  # SAM3's native resolution
```

#### 3.2 Segmentation Enabled
```yaml
scratch:
  enable_segmentation: True
```

#### 3.3 Loss Functions

**Box Losses:**
```yaml
- _target_: sam3.train.loss.loss_fns.Boxes
  weight_dict:
    loss_bbox: 5.0
    loss_giou: 2.0
```

**Classification Loss (IABCEMdetr):**
```yaml
- _target_: sam3.train.loss.loss_fns.IABCEMdetr
  weight_dict:
    loss_ce: 20.0
    presence_loss: 20.0  # PRESENCE TOKEN LOSS
  use_presence: True     # PRESENCE TOKEN ENABLED
  pos_weight: 5.0
  alpha: 0.25
  gamma: 2
```

**Mask Losses:**
```yaml
- _target_: sam3.train.loss.loss_fns.Masks
  weight_dict:
    loss_mask: 5.0
    loss_dice: 5.0
```

**Semantic Segmentation Loss:**
```yaml
loss_fn_semantic_seg:
  _target_: sam3.train.loss.loss_fns.SemanticSegCriterion
  weight_dict:
    loss_semantic_seg: 5.0
  focal: true
  focal_alpha: 0.6
  focal_gamma: 1.6
  presence_loss: true
```

#### 3.4 Presence Token Configuration

**YES, we configured the presence token:**

From SAM3 documentation (§C.4.1):
> "Stage 3 introduces a presence token (and presence loss) to better model presence of target segments... The presence loss is a binary cross-entropy loss with weight of 20."

Our configuration:
- `presence_weight: 20.0` - Matches SAM3's recommended weight
- `use_presence: True` - Enables presence token during training
- `use_presence_eval: True` - Uses presence during evaluation

The presence token predicts:
```
p(query matches NP) = p(query matches NP | NP appears) × p(NP appears in image)
```

#### 3.5 Matcher Configuration
```yaml
matcher:
  _target_: sam3.train.matcher.BinaryHungarianMatcherV2
  focal: true
  cost_class: 2.0
  cost_bbox: 5.0
  cost_giou: 2.0
  alpha: 0.25
  gamma: 2
```

#### 3.6 Learning Rate (Conservative)
```yaml
scratch:
  lr_scale: 0.02  # Reduced from 0.1 to prevent NaN
  lr_transformer: 1.6e-5      # 8e-4 × 0.02
  lr_vision_backbone: 5e-6    # 2.5e-4 × 0.02
  lr_language_backbone: 1e-6  # 5e-5 × 0.02
```

**Note:** LR was reduced aggressively after NaN issues during initial training.

#### 3.7 Geometry Encoder (for Point Prompts)
```yaml
input_geometry_encoder:
  _target_: sam3.model.geometry_encoders.SequenceGeometryEncoder
  encode_boxes_as_points: False
  points_direct_project: True
  points_pool: True
  points_pos_enc: True
  boxes_direct_project: True
  boxes_pool: True
  boxes_pos_enc: True
  d_model: 256
  num_layers: 3
```

---

## 4. Training Attempts

### Attempt 1: Text Prompts Only
- **Prompt:** "coronary artery"
- **Epochs:** 0-7
- **Results at Epoch 5:**
  - AP: 0.330
  - AP50: 0.940 (94% - good detection)
  - AP75: 0.044 (4.4% - poor mask precision)
  - Dice Loss: 0.2117

**Observation:** Model detects vessels but masks are imprecise (blobby).

### Attempt 2: Point Prompts (Centerline Sampling)

Added `RandomGeometricInputsAPI` transform:
```yaml
- _target_: sam3.train.transforms.point_sampling.RandomGeometricInputsAPI
  num_points: [5, 15]
  box_chance: 0.2
  point_sample_mode: "centered"  # Distance transform for centerline
  geometric_query_str: "visual"  # Bypass text encoder
  resample_box_from_mask: true
```

- **Epochs:** 0-5
- **Results at Epoch 5:**
  - AP: 0.332
  - AP50: 0.941
  - AP75: 0.037 (3.7% - still poor)

**Observation:** Point prompts did not improve mask precision.

---

## 5. What Went Wrong

### 5.1 No Domain Initialization
We started from **vanilla SAM3 weights** instead of initializing from the Stage 1 checkpoint (`phase1_deepsa_best.pth`) that achieved 0.8 Dice on the same data.

The custom approach on `master` branch:
1. Trained SAM3 backbone on DeepSA data
2. Achieved 0.8 Dice after 34 epochs
3. Model learned "what vessels look like"

The SAM3 native approach:
1. Started from pretrained SAM3 (COCO/SA-1B weights)
2. Expected text prompt "coronary artery" to work out-of-box
3. Model could detect but not precisely segment

### 5.2 Single Generic Category
We used a single category "coronary artery" - very different from SAM3's training distribution (COCO objects, everyday items).

The text encoder may not have a strong representation for "coronary artery" since it was trained on general vocabulary.

### 5.3 Missing Multi-Task Learning
The Gemini plan emphasizes:
> "To ensure knowledge is retained between stages, the model is trained on multiple objectives simultaneously."

We only trained on segmentation, missing:
- Classification (which segment is this?)
- View angle encoding (critical for angiography)
- Bounding box regression

### 5.4 Possibly Incorrect Transform Order
Point prompts were added inside `ComposeAPI` after `ToTensorAPI`. The interaction with other transforms may not have been optimal.

---

## 6. What We Did Use Correctly

### 6.1 Presence Token
- Configured with weight 20.0 (matches SAM3 paper)
- Enabled in both training and evaluation
- Used in semantic segmentation loss

### 6.2 Segmentation Losses
- BCE + Dice mask loss (weight 5.0 each)
- Semantic segmentation with focal loss

### 6.3 COCO Format
- Proper RLE encoding
- Correct bounding box computation
- Standard COCO annotation structure

### 6.4 Resolution
- Used SAM3's native 1008x1008 resolution

---

## 7. Comparison: Custom vs Native Training

| Aspect | Custom (master) | Native (sam3-native-training) |
|--------|-----------------|-------------------------------|
| Model Wrapper | SAM3DeepSAModel | Direct SAM3 |
| View Angles | CategoricalViewEncoder | Not used |
| Seg Head | Custom Conv head | SAM3's native decoder |
| Training | Custom loop | SAM3's trainer.py |
| Best Dice | 0.80 | ~0.78 (estimated from losses) |
| Best AP75 | Not measured | 0.044 |
| Checkpoint | phase1_deepsa_best.pth | Not saved (poor results) |

---

## 8. Lessons Learned

1. **Domain adaptation matters:** Can't expect vanilla SAM3 to immediately work on medical imaging.

2. **Progressive learning:** Stage 1 (full vessel) should complete before Stage 2 (specific segments).

3. **Initialization is critical:** Native training should start from domain-adapted checkpoint, not vanilla weights.

4. **Text prompts need context:** "coronary artery" alone may be too generic - specific segment names might work better.

5. **Multi-task helps:** Classification + segmentation together provides richer supervision.

---

## 9. Recommended Next Steps

For Stage 2 with SAM3 native training:

1. **Initialize from Stage 1 checkpoint** (`phase1_deepsa_best.pth`)

2. **Use Medis GT masks** (segment-specific, not full tree)

3. **Create COCO dataset with CASS segment categories:**
   - "proximal_rca", "mid_rca", "distal_rca"
   - "proximal_lad", "mid_lad", "distal_lad"
   - etc. (14 categories)

4. **Let SAM3 learn segment-specific text prompts**

5. **Consider adding view angle information** to category names or as metadata

---

## 10. File References

- **COCO Converter:** `scripts/convert_to_coco.py`
- **Training Config:** `configs/angiography/vessel_segmentation.yaml`
- **Visualization:** `scripts/visualize_predictions.py`
- **Custom Stage 1 (master):** `train_stage1_deepsa.py`
- **SAM3 Documentation:** `Doc/SAM3.md`
- **Project Plan:** `PROPOSED_PROJECT_PLAN.md`

---

*Document created: 2025-11-26*
*Branch: sam3-native-training*
