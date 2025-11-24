# Complete Understanding: Your Data & SAM 3 Training Strategy

## What You Actually Have

### 1. Data Structure (CORRECTED)

**From CSV**: `E:\AngioMLDL_data\corrected_dataset_training.csv`

Each row contains:
```
- patient_id: e.g., 101-0025
- vessel_pattern: MID_RCA, PROX_LAD, DIST_LCX (NON-NORMALIZED)
- phase: PRE/POST/FINAL (before/after/final angioplasty)
- vessel_name: RCA, LAD, LCX, PDA (main vessel type)
- cine_path: Full cine sequence (e.g., 66 frames)
- contours_path: JSON with measurements
- vessel_mask_actual_path: Mask of analyzed segment
- frame_index: THE CONTRAST-FILLED FRAME experts used
- cass_segment: NORMALIZED CASS ID (1-29) ✅ THIS IS KEY!
- MLD_mm: Minimum lumen diameter
- diameter_stenosis_pct: Stenosis percentage
- lesion_length_mm: Lesion length
```

**Example**: `101-0025_MID_RCA_PRE`
- Frame 37 out of 66 total frames
- CASS segment: **2** (normalized!)
- Vessel pattern: "MID_RCA" (non-normalized, tech-written)
- MLD: 1.35mm
- Stenosis: 57.3%
- Mask: 6,991 pixels of the MID RCA segment

### 2. What the Mask Represents

**The mask is the ENTIRE ANALYZED SEGMENT**, not just the stenotic portion:
- Includes proximal normal region
- Includes the lesion (stenotic area)
- Includes distal normal region
- Typically spans one CASS segment (sometimes overlaps 2)

From measurements:
```json
{
  "segment_start": proximal end coordinates,
  "lesion_start": where stenosis begins,
  "MLD_position": maximum obstruction point,
  "lesion_end": where stenosis ends,
  "segment_end": distal end coordinates
}
```

So the mask contains ~40-50mm of vessel, with the lesion being ~15-20mm of that.

### 3. DICOM vs NPY Files

**Your Question**: Can we use DICOM directly?

**Answer**:
- **NPY files** contain the same pixel data as DICOM, just pre-extracted
- **SAM 3** doesn't care about DICOM format - it needs pixel arrays
- **Advantage of NPY**: Already normalized, no DICOM parsing overhead
- **You can use DICOM if needed**, but would need to:
  ```python
  import pydicom
  dcm = pydicom.dcmread("file.dcm")
  pixels = dcm.pixel_array  # Same as NPY
  ```

**Recommendation**: Stick with NPY for training efficiency.

### 4. Viewing Angle Problem

**Critical Issue**: The SAME CASS segment looks DIFFERENT from different angles!

From JSON, you have viewing angles:
```json
"view_angles": {
  "primary_angle": "RAO 30",
  "secondary_angle": "CAUDAL 25"
}
```

**SAM 3 MUST learn**:
- CASS 2 (MID RCA) in RAO 30° looks like THIS
- CASS 2 (MID RCA) in LAO 45° looks like THAT
- Same vessel, different appearance

**Solution**: Include view angle as additional prompt!

## SAM-VMNet Paper: What They Did

From the paper you provided, their approach was:

### Architecture:
1. **VM-UNet** (coarse segmentation) → generates bbox prompts
2. **MedSAM** (fine segmentation) → uses bbox to extract features
3. **Fusion**: Combine both feature streams
4. **Output**: High-quality vessel segmentation

### Performance:
- Dataset: 1,500 coronary angiography images (ARCADE)
- mIoU: 63.03%
- Dice: 77.33%
- Specificity: 99.33%

### Key Insight:
They used SAM for **vessel segmentation**, not **stenosis detection** or **CASS classification**.

## Your Goal vs. SAM-VMNet

| Aspect | SAM-VMNet (Paper) | Your Goal |
|--------|-------------------|-----------|
| **Task** | Segment entire vessel tree | Detect & locate lesions |
| **Output** | Vessel mask (any vessel) | CASS segment ID + stenosis % |
| **Input** | Single frame | Frame from cine (with temporal context) |
| **Prompts** | Bbox from coarse segmentation | Bbox + CASS name + view angle |
| **Classification** | No | Yes - 29 CASS segments |
| **Clinical use** | Visualization | Quantitative stenosis analysis (QCA) |

## Proposed Approach: SAM 3 for Lesion Detection

### Architecture Overview

```
Input: Angiography frame (from cine at frame_index)
       ↓
┌─────────────────────────────────────────┐
│  Stage 1: Lesion Proposal Network       │
│  (Light CNN or your existing R(2+1)D)   │
│                                          │
│  Inputs:  - Frame                        │
│           - View angle embeddings        │
│                                          │
│  Outputs: - Coarse bbox proposals        │
│           - Vessel type (RCA/LAD/LCX)    │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│  Stage 2: SAM 3 with LoRA               │
│  (Fine-tuned for coronary segments)     │
│                                          │
│  Prompts: - Bbox from Stage 1            │
│           - Text: "CASS segment 2"       │
│           - View: "RAO 30 CAUDAL 25"     │
│                                          │
│  Frozen:  MedSAM encoder                 │
│  Trained: LoRA adapters + decoder        │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│  Stage 3: Multi-Task Heads              │
│                                          │
│  From SAM 3 features, predict:           │
│  1. Segment mask (IoU loss)              │
│  2. CASS segment ID (CE loss)            │
│  3. Stenosis % (regression)              │
│  4. MLD (regression)                     │
└─────────────────────────────────────────┐
       ↓
  Output: {
    "cass_segment": 2,
    "stenosis_pct": 57.3,
    "mld_mm": 1.35,
    "mask": binary_mask,
    "confidence": 0.89
  }
```

### Training Data Format

```python
{
    # Image
    'image': cine[frame_index],  # The frame with contrast
    'cine': cine,  # Full sequence for context (optional)

    # Ground truth
    'mask': vessel_mask,  # Expert-traced segment
    'cass_segment': 2,  # CASS segment number (1-29)
    'vessel_name': 'RCA',  # Main vessel

    # Prompts for SAM 3
    'bbox': [x, y, w, h],  # From mask or proposal network
    'text_prompt': 'CASS segment 2 in RCA',
    'view_angle': 'RAO 30 CAUDAL 25',

    # Measurements (for multi-task learning)
    'stenosis_pct': 57.3,
    'mld_mm': 1.35,
    'lesion_length_mm': 17.24,

    # Metadata
    'patient_id': '101-0025',
    'phase': 'PRE',
}
```

### Why This is Better Than SAM-VMNet

1. **Task-specific**: Designed for lesion detection, not general segmentation
2. **Clinical output**: Directly predicts CASS segment + measurements
3. **View-aware**: Incorporates viewing angle information
4. **Efficient**: Uses LoRA instead of full fine-tuning
5. **Leverages your data**: 800+ expert cases with measurements

## Implementation Strategy

### Phase 1: Baseline (2-3 days)
```python
# Test current SAM 3 with bbox + CASS name
for case in test_cases:
    frame = load_frame(case, frame_index)
    bbox = get_bbox_from_mask(mask)
    cass_name = f"CASS segment {cass_id}"

    # Test combined prompting
    mask_pred = sam3.segment(frame, bbox=bbox, text=cass_name)
    iou = calculate_iou(mask_pred, mask_gt)
```

### Phase 2: Add View Angle (1 week)
```python
# Encode view angle as text
view_prompt = f"RAO {angle1} CAUDAL {angle2}"
combined_prompt = f"{cass_name} in {vessel_name}, {view_prompt}"

mask_pred = sam3.segment(frame, bbox=bbox, text=combined_prompt)
```

### Phase 3: LoRA Fine-tuning (1-2 weeks)
```python
from peft import LoraConfig, get_peft_model

# Add LoRA to SAM 3
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.1,
)

sam3_lora = get_peft_model(sam3_base, lora_config)

# Train on 800+ cases
for epoch in range(20):
    for batch in dataloader:
        # Forward
        masks_pred = sam3_lora(
            images=batch['image'],
            bboxes=batch['bbox'],
            texts=batch['text_prompt']
        )

        # Loss
        loss = dice_loss(masks_pred, batch['mask'])

        # Backward (only LoRA parameters update)
        loss.backward()
        optimizer.step()
```

### Phase 4: Multi-Task Heads (1 week)
```python
class SAM3_QCA(nn.Module):
    def __init__(self, sam3_lora):
        self.sam3 = sam3_lora

        # Add task heads
        self.cass_classifier = nn.Linear(256, 29)  # 29 CASS segments
        self.stenosis_regressor = nn.Linear(256, 1)
        self.mld_regressor = nn.Linear(256, 1)

    def forward(self, image, bbox, text):
        # SAM 3 features
        features, mask = self.sam3(image, bbox, text)

        # Multi-task predictions
        cass_id = self.cass_classifier(features)
        stenosis = self.stenosis_regressor(features)
        mld = self.mld_regressor(features)

        return mask, cass_id, stenosis, mld
```

## Expected Results

Based on SAM-VMNet paper + your domain:

| Metric | Baseline (no training) | With LoRA | With Multi-task |
|--------|------------------------|-----------|-----------------|
| Segmentation IoU | 0.4-0.5 | 0.7-0.8 | 0.75-0.85 |
| CASS Classification | - | - | 85-92% |
| Stenosis MAE | - | - | ±8-12% |
| MLD MAE | - | - | ±0.2-0.3mm |

## Key Advantages of Your Approach

1. **You have CASS segment labels** (normalized!)
2. **You have exact measurements** (stenosis %, MLD, lesion length)
3. **You have view angles** (critical for appearance variation)
4. **You have 800+ expert cases** (enough for LoRA fine-tuning)
5. **Multi-phase data** (PRE/POST/FINAL) - can learn intervention effects

## Next Steps

1. **Verify my understanding** with you
2. **Test baseline** SAM 3 with bbox + CASS prompts
3. **Implement view angle encoding**
4. **Set up LoRA training pipeline**
5. **Add multi-task heads for CASS/stenosis prediction**

## Critical Questions

1. **DICOM preference**: Do you want to load from DICOM or stick with NPY?
2. **View angle format**: Are view angles consistently in JSON? Any missing?
3. **CASS segment distribution**: How balanced are the 29 CASS segments?
4. **Temporal context**: Should we use surrounding frames or just the single frame?
5. **Inference goal**: Detect lesions automatically or refine given bboxes?