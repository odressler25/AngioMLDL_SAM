# Stage 1 Integration Plan: DeepSA → SAM 3 Full Vessel Segmentation

## ✅ What We Have

1. **DeepSA pretrained model** (Dice 0.828)
   - Location: `DeepSA/ckpt/fscad_36249.ckpt`
   - Architecture: U-Net (1 channel in, 1 channel out, 32 base filters)
   - Status: ✅ Loaded and tested successfully

2. **Training data** (800+ cases)
   - CSV: `E:/AngioMLDL_data/corrected_dataset_training.csv`
   - Cines: `.npy` files with frame sequences
   - Vessel masks: `vessel_mask_actual_path` (Medis annotations)
   - Status: ✅ Data structure understood

3. **Training script template**
   - File: `train_stage1_vessel_segmentation.py`
   - Features:
     - DeepSA label generation (pseudo-labels)
     - Medis mask loading (ground truth)
     - Training loop structure
     - Validation metrics (Dice, IoU)
   - Status: ✅ Template created, needs SAM 3 integration

## ❓ What We Need: SAM 3 Model

### Critical Question: Which SAM 3 Implementation?

We need to decide on the **SAM 3 model source** before we can proceed:

#### Option A: Official SAM 3 (Meta)
```bash
# If official SAM 3 is available
git clone https://github.com/facebookresearch/sam3
pip install -e sam3
```

**Status**: Need to check if SAM 3 is publicly released
**API**: Unknown until we see the official implementation

#### Option B: SAM 2.1 (Currently Available)
```bash
# SAM 2.1 is available now
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

**Status**: ✅ Available
**Question**: Is SAM 2.1 sufficient, or do we need SAM 3 specifically?

#### Option C: Build on SAM 2.1 + Add Features
```python
# Use SAM 2.1 as base + add our enhancements
from sam2.build_sam import build_sam2
from peft import LoraConfig, get_peft_model

# Add LoRA adapters to SAM 2.1
sam_model = build_sam2(checkpoint="sam2_hiera_large.pt")
lora_config = LoraConfig(...)
sam_with_lora = get_peft_model(sam_model, lora_config)
```

**Status**: ✅ Can implement immediately
**Advantage**: Works with proven SAM 2.1 architecture

---

## 🎯 Recommended Approach: Start with SAM 2.1

Since SAM 3 may not be publicly available yet, let's use **SAM 2.1** as our foundation:

### Why SAM 2.1 is Sufficient

1. ✅ **Proven architecture** (same foundation as SAM 3)
2. ✅ **Prompt-based** (supports bbox, points, masks)
3. ✅ **LoRA compatible** (efficient fine-tuning)
4. ✅ **Available NOW** (can start immediately)

### SAM 2.1 Integration Steps

#### Step 1: Install SAM 2.1

```bash
# Install SAM 2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Download checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

#### Step 2: Integrate into Training Script

```python
# train_stage1_vessel_segmentation.py (updated)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def build_sam_with_lora():
    """Build SAM 2.1 with LoRA adapters"""
    # Load base SAM 2.1
    sam_model = build_sam2(
        config_file="sam2_hiera_l.yaml",
        ckpt_path="checkpoints/sam2_hiera_large.pt",
        device='cuda'
    )

    # Add LoRA adapters
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv", "proj"],  # Transformer attention layers
        lora_dropout=0.05,
        bias="none"
    )

    sam_with_lora = get_peft_model(sam_model, lora_config)

    print(f"SAM 2.1 with LoRA loaded")
    print(f"Trainable params: {sum(p.numel() for p in sam_with_lora.parameters() if p.requires_grad):,}")
    print(f"Total params: {sum(p.numel() for p in sam_with_lora.parameters()):,}")

    return sam_with_lora


# Training loop modification
class Stage1Trainer:
    def train_epoch(self, dataloader, optimizer, device='cuda'):
        self.sam3.train()

        for batch in tqdm(dataloader, desc="Training"):
            images = batch['image'].to(device)
            masks_gt = batch['mask'].to(device)

            # SAM 2.1 forward pass
            # For full image vessel segmentation, use text prompt
            pred_masks = self.sam3(
                images,
                # For Stage 1: segment entire vessel tree
                # No bbox needed (full image)
                multimask_output=False
            )

            # Loss
            loss = self.combined_loss(pred_masks, masks_gt)
            loss.backward()
            optimizer.step()
```

#### Step 3: Adapt for Full Vessel Segmentation

The challenge: SAM is designed for **prompted segmentation**, but we need **full image segmentation** for Stage 1.

**Solution**: Use bbox=entire_image as prompt

```python
def segment_full_vessels(sam_model, image):
    """
    Use SAM to segment full vessel tree

    Strategy: Provide full-image bbox as prompt
    This forces SAM to segment everything in the image
    """
    h, w = image.shape[-2:]

    # Full-image bounding box
    bbox = torch.tensor([0, 0, w, h], dtype=torch.float32)

    # SAM forward pass
    masks = sam_model(
        image,
        box=bbox.unsqueeze(0),  # (1, 4)
        multimask_output=False
    )

    return masks
```

---

## 🔄 Training Strategy Options

### Option 1: DeepSA Pseudo-Labels (Recommended)

**Advantages:**
- ✅ Can use ALL 800+ cases
- ✅ Can use additional unlabeled data from CRF archive
- ✅ DeepSA provides high-quality labels (Dice 0.828)
- ✅ No manual annotation needed

**Process:**
```python
# 1. DeepSA generates labels (one-time preprocessing)
for case in all_cases:
    frame = load_frame(case)
    vessel_mask = deepsa_model(frame)  # Pseudo-label
    save(vessel_mask, f"{case}_deepsa_label.npy")

# 2. SAM learns from DeepSA labels
for case in all_cases:
    frame = load_frame(case)
    deepsa_label = load(f"{case}_deepsa_label.npy")

    sam_pred = sam_model(frame, bbox=full_image)
    loss = dice_loss(sam_pred, deepsa_label)
```

**Expected result:** SAM achieves Dice 0.75-0.80 (slightly lower than DeepSA teacher)

---

### Option 2: Medis Ground Truth (Baseline)

**Advantages:**
- ✅ Real expert annotations (gold standard)
- ✅ No domain shift from teacher model

**Disadvantages:**
- ⚠️ Limited to 800 annotated cases
- ⚠️ Medis masks are SEGMENT-specific, not full tree

**Challenge:** Your Medis masks only cover the analyzed segment (e.g., LAD Mid), not the full vessel tree!

```python
# Your Medis masks look like this:
vessel_mask = load_medis_mask("101-0086_MID_LAD_PRE")
# This is ONLY the LAD Mid segment, not the full coronary tree

# DeepSA segments the full tree:
deepsa_mask = deepsa_model(frame)
# This includes RCA, LAD, LCX, all branches
```

**Solution if using Medis masks:**
You'd need to create full-tree annotations by combining all segments from the same frame. This is complex!

**Recommendation:** Use DeepSA pseudo-labels for Stage 1.

---

### Option 3: Hybrid (Best of Both)

```python
# Use DeepSA for full tree learning
# Use Medis masks for validation only

# Training: DeepSA pseudo-labels
for case in train_cases:
    deepsa_label = generate_deepsa_label(case)
    loss = train_sam(case, deepsa_label)

# Validation: Medis ground truth
for case in val_cases:
    medis_mask = load_medis_mask(case)  # Real expert annotation
    sam_pred = sam_model(case)

    # Segment-specific validation
    segment_dice = dice(sam_pred & medis_bbox, medis_mask)
```

---

## 📋 Next Steps (Choose One)

### Option A: Proceed with SAM 2.1

1. Install SAM 2.1
2. Integrate into training script
3. Generate DeepSA pseudo-labels for all 800 cases
4. Train SAM 2.1 on DeepSA labels
5. Validate on Medis masks

**Timeline:** 3-4 days
**Expected result:** SAM 2.1 with Dice 0.75-0.80 for full vessel segmentation

### Option B: Wait for SAM 3 Release

1. Monitor SAM 3 release
2. Use same training strategy but with SAM 3
3. (Meanwhile, can prep DeepSA pseudo-labels)

**Timeline:** Unknown (when will SAM 3 be released?)

### Option C: Use Existing U-Net from Phase 1

You mentioned having a Phase 1 U-Net already trained. Could we:
1. Use that U-Net as the vessel segmentation module
2. Skip Stage 1 training
3. Move directly to Stage 2 (CASS classification)

**Timeline:** Immediate (if U-Net is already trained)

---

## 🚀 My Recommendation

**Start with Option A (SAM 2.1)** because:

1. ✅ **Available now** - no waiting
2. ✅ **Proven architecture** - SAM 2.1 is robust
3. ✅ **LoRA compatible** - efficient training
4. ✅ **Can migrate to SAM 3** later if needed (same training data/pipeline)
5. ✅ **DeepSA integration ready** - we've tested it successfully

**Action items:**
1. Install SAM 2.1
2. Update `train_stage1_vessel_segmentation.py` with SAM 2.1 API
3. Generate DeepSA pseudo-labels (preprocessing step)
4. Run Stage 1 training on 100 cases (test)
5. Scale to full 800 cases
6. Validate on Medis masks
7. Move to Stage 2 (CASS classification)

---

## ❓ Decision Point

**Which option do you prefer?**

A. Proceed with SAM 2.1 (start immediately)
B. Wait for SAM 3 (prep data in meantime)
C. Use existing Phase 1 U-Net (skip Stage 1)
D. Different approach?

Let me know and I'll implement accordingly!
