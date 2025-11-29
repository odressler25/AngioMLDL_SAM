# Complete Dataset Analysis

## Summary

You have a **RICH, multi-level dataset** that's perfect for training a sophisticated QCA system!

---

## Dataset Breakdown

### 1. Expert-Labeled Data (800+ cases)

**Location:** `E:\AngioMLDL_data\corrected_vessel_dataset\`

**What you have per case:**

#### A. Cine Videos
- File: `cines/{patient}_{vessel}_{phase}_cine.npy`
- Example: 66 frames × 960×960 pixels
- Normalized float32 [0, 1]

#### B. Vessel Masks
- File: `vessel_masks/{patient}_{vessel}_{phase}_mask.npy`
- Binary mask (0/1) at 960×960 or 1016×1016
- **Covers entire functional segment** (lesion + healthy proximal/distal)
- Example: 6,991 pixels (~0.76% of image)

#### C. Contour JSON Files
- File: `contours/{patient}_{vessel}_{phase}_contours.json`
- **Contains:**
  - `centerline`: ~325 points tracing vessel path
  - `left_edge`: ~307 points (vessel left boundary)
  - `right_edge`: ~347 points (vessel right boundary)
  - `ideal_left_edge`, `ideal_right_edge`: Smoothed versions
  - `measurements`:
    - **MLD_mm**: e.g., 1.35 mm
    - **diameter_stenosis_pct**: e.g., 57.3%
    - **lesion_length_mm**: e.g., 17.24 mm
    - **segment_length_mm**: e.g., 43.44 mm (total segment)
    - **proximal_normal_diameter_mm**: e.g., 3.33 mm
    - **distal_normal_diameter_mm**: e.g., 3.25 mm
    - **interpolated_RVD_mm**: e.g., 3.16 mm (reference)
    - **Pixel coordinates** for:
      - MLD location: (274.7, 441.9)
      - Lesion start: (285.1, 403.3)
      - Lesion end: (271.2, 523.2)
      - Centerline indices for all key points
  - `calibration`:
    - `scale_factor`: e.g., 9.815 px/mm
    - `catheter_diameter_mm`: 8.0 mm
  - `view_angles`:
    - `primary_angle`: LAO/RAO
    - `secondary_angle`: CRAN/CAUD
  - `analyst`, `analysis_date`

---

### 2. RPCA Pseudo-Labels (769 cases)

**Location:** `E:\AngioMLDL_data\rpca_pseudo_labels_corrected\`

**What you have:**
- Files: `pseudo_label_0000.npy` to `pseudo_label_0768.npy`
- Binary masks (0/1) at 1016×1016
- **Much larger coverage** than expert masks
  - Example: 325,643 pixels (~31.55% of image)
  - **~40x more pixels** than expert labels!

**What this means:**
- RPCA detected **all visible vessels** in the image (not just the target lesion segment)
- Includes:
  - Main vessel branches (RCA + side branches, or LAD + diagonals, etc.)
  - Background vessels
  - Possibly some noise/artifacts

**Quality:** "A bit noisy but vessel contours are visible"
- Good enough for pre-training / weak supervision
- Not precise enough for clinical measurements

---

## Key Insights

### 1. Expert Labels Cover **Functional Segments**, Not Just Lesions

Example from 101-0025_MID_RCA_PRE:
- **Total segment**: 325 centerline points = 43.44 mm
- **Lesion portion**: Points 116-243 (39.1% of segment)
- **Healthy portions**:
  - Proximal: Points 0-116 (35.7% of segment)
  - Distal: Points 243-325 (25.2% of segment)

**This is PERFECT for training!** You have:
- ✅ Full context (healthy + diseased tissue)
- ✅ Reference diameter measurements from healthy regions
- ✅ Complete QCA pipeline training data

### 2. RPCA Labels Provide Weak Supervision at Scale

**Expert labels:** ~7,000 pixels per case (targeted segment)
**RPCA labels:** ~325,000 pixels per case (all vessels)

**Use cases:**
- Pre-train vessel detection on 769 RPCA cases
- Learn general "where are blood vessels" patterns
- Transfer to expert-labeled cases for precision

### 3. You Have Multi-Task Learning Opportunities

Your JSON data supports training a model to predict:

| Task | Label Source | Use Case |
|------|-------------|----------|
| **Vessel segmentation** | Masks + contours | Find vessel boundaries |
| **Vessel classification** | Segment name | RCA vs LAD vs LCX |
| **CASS localization** | Measurements | Which segment? |
| **Lesion detection** | Lesion start/end coords | Where is stenosis? |
| **MLD localization** | MLD coords | Pinpoint narrowest point |
| **Reference region ID** | Proximal/distal measurements | Find healthy tissue |
| **Stenosis severity** | Stenosis % | Regression target |
| **View angle prediction** | View angles | Learn anatomy-view mapping |

---

## Recommended Training Strategy

### Phase 1: Pre-training on RPCA (Vessel Detection)
```
Train a U-Net on 769 RPCA pseudo-labels
Goal: Learn "where are vessels" in angiograms
Output: Coarse vessel segmentation
```

### Phase 2: Fine-tuning on Expert Labels (Precision QCA)
```
Fine-tune on 800 expert cases with multi-task objectives:
1. Vessel mask segmentation (precise boundaries)
2. Centerline regression (predict path)
3. MLD localization (heatmap)
4. Vessel classification (RCA/LAD/LCX)
5. CASS segment classification
6. Stenosis severity regression
```

### Phase 3: SAM 3 Refinement (Boundary Polish)
```
Use trained model to predict centerline points
→ Feed to fine-tuned SAM 3
→ Get ultra-precise boundaries
→ Extract geometric measurements
```

---

## Why SAM 3 Zero-Shot Failed

**Expected:** SAM 3 trained on natural images (cars, cats, people)
**Your data:** Medical X-rays of contrast-filled coronary vessels

**The gap:**
- Different visual domain (X-ray vs color photos)
- Specialized anatomy (coronary tree vs common objects)
- Text prompts don't transfer ("coronary artery" means nothing to SAM 3)

**Solution:** Fine-tune SAM 3 on your data!

---

## Next Steps - Three Options

### Option A: Improve Your Existing U-Net
**If U-Net is already working reasonably well:**
1. Add RPCA pre-training if not already doing it
2. Implement multi-task learning for all JSON measurements
3. Add geometric post-processing for QCA
4. **Skip SAM 3** - you don't need it!

**Timeline:** 1-2 weeks
**Best for:** Production system, fastest path to deployment

---

### Option B: U-Net + Fine-Tuned SAM 3 Hybrid
**If you want best-possible boundary precision:**
1. Keep your U-Net for vessel detection + classification
2. Fine-tune SAM 3 on expert masks (using centerlines as point prompts)
3. U-Net → centerline points → SAM 3 → refined mask → QCA

**Timeline:** 2-3 weeks
**Best for:** Research quality, publication-worthy results

---

### Option C: Fully Automated SAM 3 Pipeline
**If you want to explore SAM 3 capabilities:**
1. Train centerline predictor on expert data
2. Fine-tune SAM 3 on expert + RPCA data
3. Centerline predictor → SAM 3 → QCA

**Timeline:** 3-4 weeks
**Best for:** Learning experience, potentially better generalization

---

## My Recommendation

Given that you **already have a U-Net in training**, I'd suggest:

**Start with Option A** (improve U-Net):
- Add RPCA pre-training if not using it
- Implement full multi-task learning from your rich JSON labels
- See how far you can get with pure U-Net

**Then evaluate:**
- If U-Net boundary precision is limiting → try Option B (SAM 3 refinement)
- If U-Net fails to generalize to new cases → try Option C (SAM 3 pipeline)
- If U-Net works well → ship it, you're done!

---

## Questions for You

1. **What's your current U-Net performance?**
   - IoU/Dice on held-out test set?
   - What are the failure modes?

2. **Are you already using RPCA for pre-training?**
   - Or just expert labels?

3. **What's the bottleneck you're trying to solve?**
   - Segmentation accuracy?
   - Generalization to new cases?
   - Speed/efficiency?
   - Something else?

Let me know and I can give more specific guidance!
