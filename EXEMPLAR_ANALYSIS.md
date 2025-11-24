# Can the Examples Folder Images Be Used as SAM 3 Exemplars?

## TL;DR: No, but your expert dataset can

**The Examples folder images**: Educational reference diagrams showing coronary anatomy
**What SAM 3 needs**: Actual angiogram images + pixel-level vessel masks
**What you already have**: 800+ expert cases with BOTH images and masks

---

## What's in the Examples Folder

You have 11 screenshots showing **labeled coronary anatomy** from different views:

### View Angles Covered:
- **LAO 45° CAUDAL 25°** (spider view - shows LCX well)
- **RAO 45° CRANIAL 25°** (shows LAD well)
- **AP 0° CAUDAL 30°** (shows left main bifurcation)
- **LATERAL** views (shows RCA vs left system separation)
- Various other angles

### Vessels Labeled:
- **Left system**: LM, LAD (prox/mid/dist), LCX (prox/dist), Diagonals (D1, D2), OMs (OM1, OM2)
- **Right system**: RCA (prox/mid/dist), Right ventricular branch, Right marginal branch

### Purpose of These Images:
These are **educational/reference materials** showing:
1. What vessels are visible from each view angle
2. Standard naming of coronary segments
3. Expected anatomy in textbook cases

---

## Why Examples Folder Can't Be Used Directly

### SAM 3 Exemplar Learning Requires:

**1. Exemplar Images** (actual angiograms)
```
✅ You have: 800+ real patient angiograms in cines/
❌ Examples folder: Educational diagrams (not your actual data)
```

**2. Exemplar Masks** (binary masks showing vessel pixels)
```
✅ You have: 800+ expert-labeled masks in vessel_masks/
❌ Examples folder: Text labels only (no pixel-level masks)
```

**3. Matched pairs** (same image with and without mask)
```
✅ You have: cines/ + vessel_masks/ are perfectly matched
❌ Examples folder: Labeled diagrams don't have corresponding binary masks
```

### Example of What SAM 3 Needs:

**Exemplar Image:**
```
101-0025_MID_RCA_PRE_cine.npy[0]
→ Raw angiogram showing RCA with contrast
→ Shape: (960, 960)
```

**Exemplar Mask:**
```
101-0025_MID_RCA_PRE_mask.npy
→ Binary mask with vessel pixels = 1
→ Shape: (960, 960)
→ 6,991 vessel pixels
```

**SAM 3 learns:** "Ah, the bright curvy structure in the image corresponds to the mask pixels!"

---

## What the Examples Folder IS Useful For

Even though they can't be SAM 3 exemplars, these reference images are still valuable:

### 1. Validation of View Angle Predictions
Your JSON files have `view_angles` metadata. You can validate:
```python
# If your model predicts: LAO 45° CAUD 25°
# Check against reference: "Should see LCX well, LAD overlaps"
# Sanity check: Does the predicted vessel match expected anatomy?
```

### 2. Understanding Failure Cases
When SAM 3 fails on a case:
```python
# Case: RAO 45° CRAN 25° view
# Reference shows: LAD should be well-separated
# If SAM fails: Likely due to vessel overlap or unusual anatomy
```

### 3. Multi-View Fusion Strategy (Advanced)
If you eventually get multi-view data:
```python
# Reference shows which vessels are visible in each view
# LAO 45° CAUD 25°: Best for LCX
# RAO 45° CRAN 25°: Best for LAD
# Could fuse predictions from optimal views per vessel
```

### 4. Training Data Augmentation Labels
If you want to teach a model "vessel anatomy":
```python
# Extract vessel positions from reference diagrams
# Use as weak supervision for anatomy priors
# E.g., "LAD typically courses down anterior wall"
```

---

## What WILL Work: Your Expert Dataset as Exemplars

You already have everything SAM 3 needs:

### Dataset Structure:
```
E:\AngioMLDL_data\corrected_vessel_dataset\
├── cines/
│   ├── 101-0025_MID_RCA_PRE_cine.npy      ← Exemplar IMAGE
│   ├── 101-0086_MID_LAD_PRE_cine.npy      ← Exemplar IMAGE
│   └── 101-0052_DIST_LCX_PRE_cine.npy     ← Exemplar IMAGE
│
├── vessel_masks/
│   ├── 101-0025_MID_RCA_PRE_mask.npy      ← Exemplar MASK
│   ├── 101-0086_MID_LAD_PRE_mask.npy      ← Exemplar MASK
│   └── 101-0052_DIST_LCX_PRE_mask.npy     ← Exemplar MASK
│
└── contours/
    ├── 101-0025_MID_RCA_PRE_contours.json ← Metadata (view angles, vessel ID)
    ├── 101-0086_MID_LAD_PRE_contours.json
    └── 101-0052_DIST_LCX_PRE_contours.json
```

### How to Use Your Data as Exemplars:

**Step 1: Select diverse exemplar cases**
```python
exemplars = [
    "101-0025_MID_RCA_PRE",   # RCA, RAO view
    "101-0086_MID_LAD_PRE",   # LAD, LAO CRAN view
    "101-0052_DIST_LCX_PRE",  # LCX, LAO CAUD view
    # Add 2-3 more
]
```

**Step 2: Extract bounding boxes from exemplar masks**
```python
for case_id in exemplars:
    mask = np.load(f"{case_id}_mask.npy")

    # Get vessel region
    vessel_pixels = np.argwhere(mask > 0)
    y_min, y_max = vessel_pixels[:, 0].min(), vessel_pixels[:, 0].max()
    x_min, x_max = vessel_pixels[:, 1].min(), vessel_pixels[:, 1].max()

    bbox = [x_min, y_min, x_max, y_max]
    exemplar_bboxes.append(bbox)
```

**Step 3: Use bboxes as geometric prompts on NEW images**
```python
# Test on a new case
test_image = load_image("101-0035_PROX_RCA_PRE")

# Set image in SAM 3
state = processor.set_image(test_image)

# Add geometric prompts from exemplars
for bbox in exemplar_bboxes:
    state = processor.add_geometric_prompt(
        bbox=bbox,
        label="coronary vessel",
        state=state
    )

# Predict
output = processor._forward_grounding(state)
pred_mask = output['masks'][0]

# Evaluate
gt_mask = load_mask("101-0035_PROX_RCA_PRE")
iou = calculate_iou(pred_mask, gt_mask)
```

---

## Expected Results

### If Exemplar Bboxes Work Well (IoU > 0.3):
✅ SAM 3 can learn "coronary vessel" concept from geometric prompts
✅ Proceed to fine-tuning on full dataset
✅ Expect IoU 0.6-0.8 after fine-tuning

### If Exemplar Bboxes Don't Work (IoU < 0.1):
❌ Medical domain too different from SAM 3's training data
❌ Need to fine-tune SAM 3 before few-shot works
❌ Or stick with U-Net (but improve training strategy)

---

## Test Script Ready

I created `test_sam3_exemplar_prompts.py` that:
1. Loads 3 exemplar cases from YOUR expert data
2. Extracts bounding boxes from their masks
3. Uses boxes as geometric prompts for SAM 3
4. Tests on 3 held-out cases
5. Measures IoU/Dice vs ground truth

**Run it:**
```bash
python test_sam3_exemplar_prompts.py
```

**This will answer:** Can SAM 3 learn "coronary vessel" from your expert examples?

---

## Summary Table

| Resource | Can Be SAM 3 Exemplar? | Actual Use |
|----------|----------------------|------------|
| Examples folder screenshots | ❌ No (no pixel masks) | Reference for view angles |
| Your expert cines/ | ✅ Yes (exemplar images) | Feed to SAM 3 |
| Your expert vessel_masks/ | ✅ Yes (exemplar masks) | Extract bboxes for prompts |
| Your RPCA pseudo_labels/ | ✅ Yes (noisy but usable) | Pre-training augmentation |
| Your contours/ JSON | ✅ Yes (metadata) | View angles, vessel IDs |

**Bottom line:** You don't need the Examples folder for SAM 3. Your 800 expert cases ARE the exemplars!
