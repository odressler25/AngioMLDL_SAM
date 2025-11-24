# Three-Stage Training Strategy: DeepSA → SAM 3 → Stenosis Detection

## Why This Approach is Correct

### The Fundamental Insight

**You're absolutely right** - SAM 3 needs to understand **vessel anatomy first** before it can:
1. Locate specific CASS segments within the tree
2. Detect obstructions/stenosis within those segments
3. Measure stenosis accurately

**DeepSA's full vessel segmentation isn't "just for fun"** - it's the **foundational knowledge** that enables everything else!

## The Hierarchy of Tasks

```
Level 1 (Foundation): WHERE ARE THE VESSELS?
├─> Full coronary tree segmentation
├─> Vessel vs background
└─> [DeepSA excels at this - Dice 0.828]

Level 2 (Anatomy): WHICH VESSEL IS THIS?
├─> CASS segment classification (1-29)
├─> RCA Prox vs Mid vs Dist
├─> LAD, LCX, diagonal, OM branches
└─> View-angle invariant recognition

Level 3 (Pathology): WHERE IS THE STENOSIS?
├─> Obstruction detection within segment
├─> Stenosis percentage measurement
├─> MLD measurement
└─> Lesion length measurement
```

**You can't do Level 2 or 3 without Level 1!**

## Three-Stage Training Pipeline

### Stage 1: Knowledge Transfer (DeepSA → SAM 3)

**Goal**: Teach SAM 3 to segment **full coronary tree** (all vessels)

**Method**: Knowledge distillation from DeepSA

```python
# Stage 1: Full Vessel Segmentation
# Use DeepSA as teacher, SAM 3 as student

class Stage1_FullTreeDistillation:
    """
    Transfer DeepSA's vessel segmentation knowledge to SAM 3
    """

    def __init__(self):
        # Teacher: DeepSA (frozen)
        self.teacher = load_deepsa_pretrained("ckpt/fscad_36249.ckpt")
        self.teacher.eval()

        # Student: SAM 3 (trainable with LoRA)
        self.student = build_sam3_with_lora()

    def train_stage1(self, unlabeled_crf_archive, num_cases=20000):
        """
        Train SAM 3 to replicate DeepSA's vessel segmentation

        Args:
            unlabeled_crf_archive: 20K-50K angiography cases (no labels needed!)
            num_cases: How many to use for distillation

        Expected result: SAM 3 learns to segment full coronary tree
        """

        for frame in tqdm(unlabeled_crf_archive[:num_cases]):
            # Teacher generates vessel mask
            with torch.no_grad():
                teacher_mask = self.teacher(frame)  # Full tree segmentation

            # Student learns to replicate
            # Use bbox=entire_image to force full-frame segmentation
            student_mask = self.student(
                frame,
                bbox=[0, 0, width, height]  # Full image
            )

            # Distillation loss: student matches teacher
            loss = F.binary_cross_entropy(student_mask, teacher_mask)

            loss.backward()
            optimizer.step()

        print("Stage 1 complete: SAM 3 can now segment full coronary tree!")
        return self.student

# Training
stage1_model = Stage1_FullTreeDistillation()
sam3_with_vessel_knowledge = stage1_model.train_stage1(
    unlabeled_crf_archive=all_20K_cases,
    num_cases=20000
)

# Expected: SAM 3 achieves Dice 0.75-0.80 for full tree segmentation
```

**Key advantages:**
- ✅ Uses **unlabeled data** (your 20K-50K CRF archive)
- ✅ DeepSA does the heavy lifting (generates pseudo-labels)
- ✅ SAM 3 learns vessel anatomy without human annotation
- ✅ Foundation for Stages 2 & 3

---

### Stage 2: CASS Segment Classification

**Goal**: Given a vessel tree, classify each part by CASS segment (1-29)

**Method**: Fine-tune on our 800+ expert-annotated CASS segments

```python
# Stage 2: CASS Segment Classification
# Now SAM 3 knows WHERE vessels are, teach it WHICH segment is which

class Stage2_CASSClassification:
    """
    Train SAM 3 to classify CASS segments within the vessel tree
    """

    def __init__(self, stage1_model):
        # Start from Stage 1 model (already knows vessels)
        self.model = stage1_model

        # Add CASS classification head
        self.cass_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 29)  # 29 CASS segments
        )

    def train_stage2(self, expert_annotated_cases):
        """
        Train CASS segment classification

        Args:
            expert_annotated_cases: Our 800+ Medis-annotated segments
                Each case has:
                - Frame
                - CASS segment label (1-29)
                - Segment mask
                - View angles

        Expected result: SAM 3 can identify which CASS segment is which
        """

        for case in expert_annotated_cases:
            frame = case['frame']
            cass_label = case['cass_segment']  # e.g., 2 = RCA Mid
            segment_mask = case['mask']  # Expert-traced segment
            view_angles = case['view_angles']

            # Step 1: Get full vessel tree (from Stage 1 knowledge)
            full_tree = self.model(frame, bbox=[0, 0, W, H])

            # Step 2: Get features for this segment
            # Use bbox from expert mask to localize segment
            segment_bbox = get_bbox_from_mask(segment_mask)
            segment_features = self.model.get_features(
                frame,
                bbox=segment_bbox,
                view_angles=view_angles  # CRITICAL: view-angle conditioning
            )

            # Step 3: Classify CASS segment
            predicted_cass = self.cass_classifier(segment_features)

            # Multi-task loss
            loss = (
                F.cross_entropy(predicted_cass, cass_label)  # CASS classification
                + F.binary_cross_entropy(segment_mask_pred, segment_mask)  # Segmentation
            )

            loss.backward()
            optimizer.step()

        print("Stage 2 complete: SAM 3 can now classify CASS segments!")
        return self.model

# Training
stage2_model = Stage2_CASSClassification(sam3_with_vessel_knowledge)
sam3_with_cass = stage2_model.train_stage2(
    expert_annotated_cases=our_800_medis_cases
)

# Expected: 85-92% CASS classification accuracy
```

**Key features:**
- ✅ **View-angle invariant**: Train on multiple angles (RAO/LAO/Cranial/Caudal)
- ✅ **Anatomical knowledge**: Learns RCA vs LAD vs LCX distinctions
- ✅ **Spatial relationships**: Learns segment topology (Prox → Mid → Dist)

---

### Stage 3: Stenosis Detection & Measurement

**Goal**: Within a CASS segment, detect stenosis and measure severity

**Method**: Add stenosis detection module (inspired by SAM-VMNet)

```python
# Stage 3: Stenosis Detection and Measurement
# Now SAM 3 knows vessels AND segments, teach it to find obstructions

class Stage3_StenosisDetection:
    """
    Add stenosis detection and measurement capabilities

    Could integrate SAM-VMNet's MATLAB stenosis module
    """

    def __init__(self, stage2_model):
        self.model = stage2_model

        # Add stenosis detection heads
        self.stenosis_detector = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)  # Stenosis heatmap
        )

        self.stenosis_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # [stenosis_%, MLD_mm, lesion_length_mm]
        )

    def train_stage3(self, expert_annotated_cases):
        """
        Train stenosis detection and measurement

        Uses our 800+ cases with:
        - Segment masks
        - Stenosis measurements (%, MLD, lesion length)
        - Lesion location coordinates
        """

        for case in expert_annotated_cases:
            frame = case['frame']
            cass_segment = case['cass_segment']
            segment_mask = case['mask']

            # Stenosis measurements (from Medis QCA)
            stenosis_pct = case['diameter_stenosis_pct']
            mld_mm = case['MLD_mm']
            lesion_length = case['lesion_length_mm']
            lesion_coords = case['lesion_coordinates']  # From JSON

            # Step 1: Segment the vessel (from Stage 1)
            vessel_mask = self.model(frame, bbox=full_image)

            # Step 2: Classify CASS segment (from Stage 2)
            segment_bbox = get_bbox_from_mask(segment_mask)
            predicted_cass, segment_features = self.model(
                frame,
                bbox=segment_bbox,
                return_features=True
            )

            # Step 3: Detect stenosis within segment
            stenosis_heatmap = self.stenosis_detector(segment_features)

            # Step 4: Measure stenosis
            measurements = self.stenosis_regressor(
                segment_features.mean(dim=(2,3))  # Global pool
            )
            pred_stenosis_pct, pred_mld, pred_length = measurements

            # Multi-task loss
            loss = (
                F.mse_loss(stenosis_heatmap, create_stenosis_target(lesion_coords))
                + F.mse_loss(pred_stenosis_pct, stenosis_pct)
                + F.mse_loss(pred_mld, mld_mm)
                + F.mse_loss(pred_length, lesion_length)
            )

            loss.backward()
            optimizer.step()

        print("Stage 3 complete: SAM 3 can detect and measure stenosis!")
        return self.model

# Training
stage3_model = Stage3_StenosisDetection(sam3_with_cass)
sam3_final = stage3_model.train_stage3(
    expert_annotated_cases=our_800_medis_cases
)

# Expected:
# - Stenosis detection: 0.75-0.85 IoU
# - Stenosis %: ±8-12% MAE
# - MLD: ±0.2-0.3mm MAE
```

**Option**: Use SAM-VMNet's MATLAB stenosis module
- They have a proven stenosis detection algorithm
- Could integrate as post-processing step
- OR: Replicate in Python and add as Stage 3 module

---

## Complete Pipeline Overview

```
Input: Angiography Frame
       ↓
┌──────────────────────────────────────────────────────────┐
│ Stage 1: Full Vessel Segmentation (DeepSA → SAM 3)     │
│ Output: Binary mask of entire coronary tree             │
│ Trained on: 20K unlabeled cases (DeepSA pseudo-labels)  │
└──────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────┐
│ Stage 2: CASS Segment Classification                    │
│ Output: CASS segment ID (1-29) + segment mask           │
│ Trained on: 800 expert-annotated cases                  │
│ Features: View-angle invariant                          │
└──────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────┐
│ Stage 3: Stenosis Detection & Measurement               │
│ Output: Stenosis %, MLD, lesion length, location        │
│ Trained on: 800 cases with QCA measurements             │
│ Optional: SAM-VMNet MATLAB module integration           │
└──────────────────────────────────────────────────────────┘
       ↓
Output: Complete QCA Analysis
```

## Why This Works

### Stage 1 Foundation

Without knowing **WHERE vessels are**, SAM 3 can't:
- ❌ Distinguish vessels from background artifacts
- ❌ Handle low-contrast regions
- ❌ Separate overlapping vessels
- ❌ Track vessels across viewing angles

**DeepSA provides this foundation** (Dice 0.828 on full tree)

### Stage 2 Builds On Foundation

With vessel knowledge, SAM 3 can now:
- ✅ Parse the vessel tree into anatomical segments
- ✅ Learn spatial relationships (RCA proximal → mid → distal)
- ✅ Handle view-angle variations (same segment looks different in RAO vs LAO)

### Stage 3 Requires Both

Stenosis detection needs:
- ✅ **Stage 1**: Know where the vessel is
- ✅ **Stage 2**: Know which segment to analyze
- ✅ **Stage 3**: Find the narrowing within that segment

## Data Requirements

| Stage | Data Needed | Volume | Source |
|-------|-------------|--------|--------|
| **Stage 1** | Angiography frames (no labels!) | 20,000 | CRF archive + DeepSA pseudo-labels |
| **Stage 2** | CASS segment annotations | 800+ | Medis QCA (our dataset) |
| **Stage 3** | Stenosis measurements | 800+ | Medis QCA (same dataset) |

**Key insight**: We can use 20K unlabeled cases for Stage 1, then fine-tune on our 800 labeled cases for Stages 2 & 3!

## Expected Performance

### Stage 1: Full Tree Segmentation
- **Expected Dice**: 0.75-0.80 (vs DeepSA's 0.828)
- **Training time**: 1-2 days (20K cases)
- **Inference**: <1s per frame

### Stage 2: CASS Classification
- **Expected accuracy**: 85-92%
- **Training time**: 4-6 hours (800 cases)
- **Challenge**: View-angle invariance

### Stage 3: Stenosis Measurement
- **Expected stenosis % MAE**: ±8-12%
- **Expected MLD MAE**: ±0.2-0.3mm
- **Training time**: 4-6 hours (800 cases)
- **Optional**: SAM-VMNet MATLAB module

## Implementation Plan

### Week 1-2: Stage 1 Implementation

```bash
# Implement knowledge distillation from DeepSA
python train_stage1_distillation.py \
    --teacher_model DeepSA/ckpt/fscad_36249.ckpt \
    --student_model sam3 \
    --num_cases 20000 \
    --epochs 30
```

### Week 3: Stage 2 Implementation

```bash
# Add CASS classification head
python train_stage2_cass.py \
    --pretrained stage1_model.pth \
    --data corrected_dataset_training.csv \
    --num_cases 800 \
    --epochs 20
```

### Week 4: Stage 3 Implementation

```bash
# Add stenosis detection
python train_stage3_stenosis.py \
    --pretrained stage2_model.pth \
    --data corrected_dataset_training.csv \
    --num_cases 800 \
    --epochs 20
```

**OR** integrate SAM-VMNet MATLAB module:

```bash
# Extract MATLAB stenosis module from SAM-VMNet
# Translate to Python or use MATLAB engine
python integrate_samvmnet_stenosis.py
```

## SAM-VMNet Stenosis Module

From the SAM-VMNet paper, they have a **stenosis detection module in MATLAB**:

### What It Does
1. Takes vessel segmentation as input
2. Computes vessel diameter along centerline
3. Detects narrowings (stenosis)
4. Measures stenosis percentage

### Integration Options

**Option A**: Use their MATLAB code directly
```python
import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Use SAM-VMNet stenosis module
stenosis_results = eng.detect_stenosis(
    vessel_mask,  # From our Stage 2
    centerline,
    nargout=1
)
```

**Option B**: Reimplement in Python
```python
# Replicate their algorithm in Python
def detect_stenosis_samvmnet(vessel_mask):
    # 1. Extract centerline (skeletonization)
    centerline = skeletonize(vessel_mask)

    # 2. Compute diameter at each centerline point
    # (Distance transform method)
    diameters = compute_vessel_diameter(vessel_mask, centerline)

    # 3. Find reference diameter (healthy section)
    ref_diameter = np.percentile(diameters, 90)

    # 4. Detect stenosis (narrowing > 30%)
    stenosis_pct = (ref_diameter - diameters.min()) / ref_diameter * 100

    return stenosis_pct, diameters
```

## Next Steps

Would you like me to:

1. **Implement Stage 1 training script** (DeepSA → SAM 3 distillation)?
2. **Create the full 3-stage training pipeline**?
3. **Extract SAM-VMNet's stenosis module** for Stage 3?
4. **All of the above** in sequence?

This is the correct approach - thank you for the course correction!
