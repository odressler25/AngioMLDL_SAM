# Phase 3: End-to-End Coronary Angiography Analysis

## Vision

**Input:** DICOM cine angiography run
**Output:**
1. Best frame selection (diastolic, optimal contrast)
2. CASS segment identification with bounding boxes
3. Obstruction localization (MLD position)
4. Quantitative measurements (MLD, %DS, lesion length)

All without manual prompts or "crutches."

---

## Current State (Phase 2 Complete)

### What We Have
- **Segmentation Model:** SAM3 fine-tuned on CASS segments
  - AP50 = 53% on held-out patients (0% overlap)
  - 15 CASS segment classes
  - Works with box prompts

- **Training Data:**
  - 512 training images, 180 patients
  - 227 validation images, 79 patients
  - COCO format with RLE masks

- **Medis Ground Truth:**
  - 777 contour files with measurements
  - Actual + ideal contours
  - Full QCA measurements (MLD, %DS, RVD, etc.)

### Current Limitations
- Requires box prompt (not fully automatic)
- Single frame input (no frame selection)
- No measurement output (segmentation only)
- PNG input (not direct DICOM)

---

## Data Pipeline: Source to Training

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     SOURCE: E:/Angios/                          │
│  274 patients, structure per patient:                           │
│  ALL RISE 101-XXXX/                                             │
│    ├── Analysis/PROCEDURES QCA ANALYSIS/INDEX BASELINE/         │
│    │     ├── MID_LAD_PRE.dcm    (DICOM + Medis metadata)       │
│    │     ├── MID_LAD_PRE.pdf    (Medis report)                 │
│    │     └── ...                                                │
│    ├── CRF/                      (Case report forms)            │
│    └── measurement_frames/       (Extracted PNGs)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 1: DICOM EXTRACTION                            │
│  Scripts (in C:/Users/odressler/AngioMLDL/scripts/data_preparation/): │
│    - extract_medis_complete.py       (main extraction)          │
│    - extract_vessel_contours_from_dicom.py                      │
│    - decode_medis_annotations.py                                │
│    - decode_medis_sequences.py                                  │
│    - extract_qca_measurements_summary.py                        │
│  Output:                                                         │
│    - PNG images (best frame selected by Medis)                  │
│    - Contour JSON files (centerline, edges, ideal edges)        │
│    - Measurements (MLD, %DS, lesion length, etc.)               │
│    - Metadata CSV (patient info, viewing angles, calibration)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 2: VESSEL SEGMENTATION                         │
│  Script: DeepSA/demo.py                                          │
│  Input: PNG images                                               │
│  Output: NPY vessel masks (full coronary tree)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 3: CASS LABELING                               │
│  Script: scripts/create_medis_bifurcation_labels.py             │
│  Input:                                                          │
│    - PNG images                                                  │
│    - NPY vessel masks                                           │
│    - Contour JSON files (Medis centerlines + edges)             │
│    - Metadata CSV (CASS codes, viewing angles)                  │
│  Output: COCO annotations with RLE masks                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 4: SAM3 TRAINING                               │
│  Script: train_sam3_clean.py                                     │
│  Config: configs/angiography/cass_medis_bifurcation_v7.yaml     │
│  Input: COCO dataset                                             │
│  Output: Trained segmentation model                             │
└─────────────────────────────────────────────────────────────────┘
```

### Essential Scripts (Keep)

| Script | Purpose | Phase |
|--------|---------|-------|
| `extract_dicom_data.py` | Extract images + Medis data from DICOM | Data Prep |
| `DeepSA/demo.py` | Generate vessel masks | Data Prep |
| `scripts/create_medis_bifurcation_labels.py` | Create CASS COCO annotations | Data Prep |
| `train_sam3_clean.py` | Train SAM3 model | Training |
| `patch_coco_writer.py` | Fix COCO area calculation | Training |
| `scripts/visualize_medis_dino_labels.py` | Visualize labels | Debug |
| `scripts/visualize_predictions.py` | Visualize model output | Debug |

### Scripts to Archive/Remove

All `test_*.py`, `debug_*.py`, `inspect_*.py` files and intermediate experiment scripts.

---

## Phase 3A: Measurement Model

### Medis Data Structure

Each contour JSON file contains:

```python
{
    # Segment identification
    "segment_name": "Mid LAD PRE",
    "frame_num": 20,

    # Contours (pixel coordinates)
    "centerline": [[x, y], ...],        # ~100-400 points
    "left_edge": [[x, y], ...],         # Actual vessel boundary
    "right_edge": [[x, y], ...],        # Actual vessel boundary
    "ideal_left_edge": [[x, y], ...],   # Reference (healthy) boundary
    "ideal_right_edge": [[x, y], ...],  # Reference (healthy) boundary

    # Key measurements
    "measurements": {
        # Stenosis metrics
        "diameter_stenosis_pct": 46.3,      # PRIMARY TARGET
        "area_stenosis_pct": 71.2,
        "MLD_mm": 1.09,                     # Minimum lumen diameter
        "interpolated_RVD_mm": 2.03,        # Reference vessel diameter

        # Location (pixel coordinates)
        "MLD_x_coord": 239.9,
        "MLD_y_coord": 149.5,
        "MLD_centerline_index": 44,

        # Lesion extent
        "lesion_length_mm": 9.22,
        "lesion_start_x_coord": 231.7,
        "lesion_start_y_coord": 131.8,
        "lesion_end_x_coord": 242.9,
        "lesion_end_y_coord": 172.7,

        # Reference segments
        "proximal_normal_diameter_mm": 2.26,
        "distal_normal_diameter_mm": 1.85,
        "segment_length_mm": 20.9
    },

    # Calibration
    "calibration": {
        "scale_factor": 3.38,               # pixels per mm
        "catheter_diameter_mm": 8.0,
        "imager_pixel_spacing_mm": [0.308, 0.308]
    },

    # View
    "view_angles": {
        "primary_angle": 8.9,   # RAO(+) / LAO(-)
        "secondary_angle": 24.3 # CRA(+) / CAU(-)
    }
}
```

### Measurement Model Architecture

**Option A: Regression Head on SAM3**
```
SAM3 Encoder → Mask Decoder → Segmentation
                    ↓
              Measurement Head → MLD, %DS, lesion_length
```

**Option B: Separate Measurement Model (Recommended)**
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Input Image    │ ──► │ SAM3 Segmentation│ ──► │ Cropped Region  │
│  + Box Prompt   │     │     Model        │     │ + Mask          │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Measurement    │
                                                 │  Model          │
                                                 │  (CNN + MLP)    │
                                                 └────────┬────────┘
                                                          │
                                                          ▼
                                        ┌─────────────────────────────────┐
                                        │ Output:                         │
                                        │  - MLD location (x, y)          │
                                        │  - MLD value (mm)               │
                                        │  - %DS                          │
                                        │  - Lesion boundaries            │
                                        └─────────────────────────────────┘
```

### Training Data for Measurement Model

From the 777 contour files, we can create:
- **Regression targets:** MLD_mm, diameter_stenosis_pct, lesion_length_mm
- **Localization targets:** MLD_x_coord, MLD_y_coord, lesion boundaries
- **Reference data:** ideal contours for diameter calculation

---

## Phase 3B: Frame Selection

### Current State
- Training uses "best frame" selected by Medis analyst
- Frame number stored in contour JSON (`frame_num`)

### Adding Frame Selection

**Option 1: Heuristic-Based (Simple)**
- Select frame at ~70% of cardiac cycle (diastole)
- Use ECG signal from DICOM if available
- Filter by contrast presence (vessel intensity)

**Option 2: Learned Frame Selection (Better)**
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  DICOM Cine     │ ──► │ Frame Quality    │ ──► │ Best Frame      │
│  (all frames)   │     │ Classifier       │     │ Selection       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

Training data: Use Medis `frame_num` as ground truth for which frame is "best."

### Benefits of Full Cine Training
1. **Motion understanding:** Model learns what's static (vessels) vs. moving (heart wall)
2. **Contrast timing:** Model learns optimal contrast phase
3. **Noise robustness:** Model sees variation across frames

**Architecture consideration:** This would require either:
- 3D convolutions (video model)
- Frame-wise processing with temporal aggregation
- Or simpler: frame quality scoring network

---

## Phase 3C: Direct DICOM Input

### Question: Retrain or Not?

**Current situation:**
- Model trained on PNG images
- PNG extracted from DICOM with specific preprocessing
- Viewing angle used in LABELING, not model input

**Answer: No retraining needed for basic inference IF:**
1. Same preprocessing applied to DICOM → tensor
2. Viewing angle extracted from DICOM headers (not model input currently)

**To add viewing angle as model input (future):**
- Would require architecture change
- Would require retraining
- Benefit: Model could use viewing angle for better predictions

### DICOM Integration Plan

```python
# Pseudocode for DICOM inference
def process_dicom(dicom_path):
    # 1. Read DICOM
    dcm = pydicom.dcmread(dicom_path)

    # 2. Extract pixel data
    frames = dcm.pixel_array  # [num_frames, H, W] or [H, W]

    # 3. Extract metadata
    view_angles = extract_view_angles(dcm)  # From DICOM headers
    calibration = extract_calibration(dcm)  # Pixel spacing

    # 4. Frame selection (if cine)
    if len(frames.shape) == 3:
        best_frame_idx = select_best_frame(frames)
        frame = frames[best_frame_idx]
    else:
        frame = frames

    # 5. Preprocess (match training)
    img_tensor = preprocess(frame)  # Resize, normalize to [0.5, 0.5, 0.5]

    # 6. Run segmentation model
    segments = segmentation_model(img_tensor)

    # 7. Run measurement model
    for segment in segments:
        measurements = measurement_model(img_tensor, segment.mask)
        # Convert pixels to mm using calibration
        measurements_mm = apply_calibration(measurements, calibration)

    return segments, measurements_mm
```

---

## End-to-End Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: DICOM CINE RUN                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │  Extract Metadata   │         │  Frame Selection    │
        │  - View angles      │         │  Model              │
        │  - Calibration      │         │  (or heuristic)     │
        └─────────────────────┘         └──────────┬──────────┘
                    │                              │
                    │                              ▼
                    │                   ┌─────────────────────┐
                    │                   │  Best Frame Image   │
                    │                   └──────────┬──────────┘
                    │                              │
                    │                              ▼
                    │                   ┌─────────────────────┐
                    │                   │  Vessel Detection   │
                    │                   │  (DeepSA or SAM3)   │
                    │                   └──────────┬──────────┘
                    │                              │
                    │                              ▼
                    │                   ┌─────────────────────┐
                    │                   │  CASS Segment       │
                    │                   │  Classification     │
                    │                   │  (SAM3 Stage 2)     │
                    │                   └──────────┬──────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
                        ┌─────────────────────┐
                        │  Measurement Model  │
                        │  - MLD location     │
                        │  - %DS calculation  │
                        │  - Lesion extent    │
                        └──────────┬──────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                      │
│  - CASS segment ID (e.g., "Mid LAD")                                   │
│  - Segmentation mask                                                    │
│  - MLD: 1.09 mm at position (240, 149)                                 │
│  - Diameter Stenosis: 46.3%                                            │
│  - Lesion Length: 9.2 mm                                               │
│  - Reference Vessel Diameter: 2.03 mm                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Recommended Implementation Order

### Phase 3A: Measurement Model (2-3 weeks)
1. Create measurement dataset from Medis contours
2. Design measurement model architecture
3. Train on cropped vessel regions
4. Validate against Medis ground truth

### Phase 3B: Automatic Detection (1-2 weeks)
1. Replace box prompts with automatic segment proposals
2. Use DeepSA output to generate candidate regions
3. Classify each region as CASS segment

### Phase 3C: Frame Selection (1 week)
1. Build frame quality dataset using Medis frame_num
2. Train simple frame classifier
3. Integrate into pipeline

### Phase 3D: DICOM Integration (1 week)
1. Create DICOM loading pipeline
2. Extract all metadata (view angles, calibration)
3. End-to-end testing

### Phase 3E: Validation & Refinement (ongoing)
1. Test on completely new patients
2. Clinical validation
3. Performance optimization

---

## Strategic Questions & Answers

### Q1: Switch to DICOM directly or apply as last step?

**Answer: Apply as last step - no retraining needed.**

The model processes pixel values. Whether pixels come from PNG or DICOM doesn't matter IF:
- Same preprocessing (resize to 672, normalize to [0.5, 0.5, 0.5])
- Same bit depth handling

**Implementation:**
```python
import pydicom
import numpy as np

def dicom_to_tensor(dicom_path, frame_idx=None):
    dcm = pydicom.dcmread(dicom_path)
    pixels = dcm.pixel_array  # [frames, H, W] or [H, W]

    # Select frame
    if len(pixels.shape) == 3:
        frame = pixels[frame_idx] if frame_idx else pixels[0]
    else:
        frame = pixels

    # Normalize to 0-255 if needed
    if frame.max() > 255:
        frame = (frame / frame.max() * 255).astype(np.uint8)

    # Convert to RGB (angiograms are grayscale)
    rgb = np.stack([frame, frame, frame], axis=-1)

    # Apply same preprocessing as training
    # ... resize, normalize to [0.5, 0.5, 0.5]
    return tensor
```

### Q2: Add full cine runs for frame selection training?

**Answer: Yes, this would improve the model significantly.**

**Benefits:**
1. Model learns contrast timing (when vessels are best visible)
2. Model distinguishes moving structures (heart wall) from static (vessels)
3. Model becomes robust to motion blur
4. Enables automatic "best frame" selection

**Implementation options:**

| Option | Complexity | Benefit |
|--------|------------|---------|
| A. Separate frame classifier | Low | Quick to implement |
| B. 3D conv on short clips | Medium | Learns temporal patterns |
| C. Video transformer | High | Best quality, most data needed |

**Recommended: Option A first, then B if needed.**

**Training data:** Medis `frame_num` field = ground truth for best frame.

### Q3: How many more patients would help?

**Current:** 180 train / 79 val patients
**Available:** 274 total patients in E:/Angios

**Recommendation:**
- Use remaining ~15 patients as held-out test set (never touched)
- If more data available, adding to 500+ patients would significantly help
- Diminishing returns likely after 1000 patients

### Q4: Use ideal contours in training?

**Answer: Yes, critical for measurement model.**

The ideal contours represent "healthy" vessel diameter. This enables:
- **%DS calculation:** (ideal - actual) / ideal × 100
- **Reference training:** Model learns what "normal" looks like
- **Lesion localization:** Where actual deviates from ideal

### Q5: Strategy for Medis measurements in training?

**Two-phase approach:**

**Phase 1: Segmentation (DONE)**
- Input: image + box prompt
- Output: segment mask
- Training data: Medis contour polygons

**Phase 2: Measurement (NEXT)**
- Input: image + segment mask (from Phase 1)
- Output: MLD location, %DS, lesion extent
- Training data: Medis measurements JSON

**Why separate?**
- Easier to debug
- Can validate segmentation before adding measurements
- Measurement model can be smaller/faster

---

## Questions Still Open

1. **Data availability:** Can we get more than 274 patients?
2. **Clinical validation:** Who will validate the measurements?
3. **Deployment target:** Real-time cath lab or batch research?
4. **Regulatory:** Any FDA/CE requirements to consider?

---

## File Structure (After Cleanup)

```
AngioMLDL_SAM/
├── configs/
│   └── angiography/
│       └── cass_medis_bifurcation_v7.yaml
├── scripts/
│   ├── data_preparation/
│   │   ├── extract_dicom_data.py       # Step 1: DICOM extraction
│   │   ├── generate_vessel_masks.py     # Step 2: DeepSA inference
│   │   └── create_cass_labels.py        # Step 3: COCO creation
│   ├── training/
│   │   └── train_sam3.py                # Step 4: Training
│   ├── inference/
│   │   ├── segment_cass.py              # Run segmentation
│   │   └── measure_stenosis.py          # Run measurements
│   └── visualization/
│       ├── visualize_labels.py
│       └── visualize_predictions.py
├── models/
│   └── measurement_head.py              # Phase 3A
├── docs/
│   ├── cass_labeling_algorithm.md
│   └── phase3_plan.md
├── DeepSA/                              # Vessel segmentation
└── checkpoints/                         # Trained models
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| Nov 2024 | 1.0 | Initial Phase 3 plan |
