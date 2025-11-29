# DeepSA + SAM 3 Integration Strategy

## Why DeepSA is Our Best Knowledge Transfer Source

Based on SOTA analysis, DeepSA has **critical advantages**:

1. ✅ **Self-supervised pretraining on 58K+ angiograms** (exactly our domain!)
2. ✅ **Achieves Dice 0.828 with only 40 fine-tuning samples** (vs 0.657 baseline)
3. ✅ **Public pretrained weights available** (Google Drive)
4. ✅ **Simple U-Net architecture** (4.3M params, easy to understand/modify)
5. ✅ **Designed for angiography** (vs MedSAM's general medical imaging)

## Three-Stage Integration Strategy

```
Stage 1: DeepSA Baseline Testing
├─> Download pretrained DeepSA weights
├─> Test on our 3 cases (RCA, LAD, LCX)
├─> Measure Dice/IoU performance
└─> Compare with SAM 3 baseline (0.372 IoU)

Stage 2: Hybrid DeepSA → SAM 3 Pipeline
├─> DeepSA generates coarse vessel masks
├─> Extract point/bbox prompts from DeepSA output
├─> SAM 3 refines with prompts
└─> Expected: Combine strengths of both models

Stage 3: Self-Supervised Pretraining on OUR Data
├─> We have 20,000-50,000 cases in CRF archive
├─> Replicate DeepSA's self-supervised approach
├─> Pretrain on ALL unlabeled data
├─> Fine-tune on 800+ Medis annotations
└─> Expected: SOTA performance tailored to our data
```

## Stage 1: Test DeepSA Pretrained Model

### Installation

```bash
# Clone DeepSA repository
git clone https://github.com/newfyu/DeepSA.git
cd DeepSA

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights
# From Google Drive link in their README:
# - deepsa_pretrained.pth (self-supervised on 58K images)
# - deepsa_finetuned_fs_cad.pth (fine-tuned on FS-CAD dataset)
# - deepsa_finetuned_xcad.pth (fine-tuned on XCAD dataset)
```

### Test Script

```python
# test_deepsa_baseline.py

import sys
sys.path.append("DeepSA/")  # Add DeepSA to path

import torch
import cv2
import numpy as np
from pathlib import Path
from deepsa.model import DeepSA  # Their model class
from deepsa.utils import preprocess_frame

def test_deepsa_on_our_cases():
    """
    Test pretrained DeepSA on our 3 angiography test cases
    """

    # Load pretrained DeepSA
    print("Loading DeepSA pretrained model...")
    model = DeepSA.load_pretrained(
        weights_path="DeepSA/checkpoints/deepsa_finetuned_xcad.pth"
    )
    model.eval()
    model.cuda()

    # Our test cases (same as SAM 3 testing)
    test_cases = [
        {
            'name': '101-0025_MID_RCA_PRE',
            'cine_dir': 'E:/AngioMLDL_data/corrected_vessel_dataset/cines/101-0025_MID_RCA_PRE',
            'json_path': 'E:/AngioMLDL_data/corrected_vessel_dataset/contours/101-0025_MID_RCA_PRE_contours.json',
            'frame_num': 37  # From JSON
        },
        {
            'name': '101-0086_MID_LAD_PRE',
            'cine_dir': 'E:/AngioMLDL_data/corrected_vessel_dataset/cines/101-0086_MID_LAD_PRE',
            'json_path': 'E:/AngioMLDL_data/corrected_vessel_dataset/contours/101-0086_MID_LAD_PRE_contours.json',
            'frame_num': 30
        },
        {
            'name': '101-0052_DIST_LCX_PRE',
            'cine_dir': 'E:/AngioMLDL_data/corrected_vessel_dataset/cines/101-0052_DIST_LCX_PRE',
            'json_path': 'E:/AngioMLDL_data/corrected_vessel_dataset/contours/101-0052_DIST_LCX_PRE_contours.json',
            'frame_num': 40
        }
    ]

    results = []

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {case['name']}")
        print(f"Frame: {case['frame_num']}")

        # Load frame
        frame_path = Path(case['cine_dir']) / f"frame_{case['frame_num']:04d}.png"
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

        if frame is None:
            print(f"ERROR: Could not load {frame_path}")
            continue

        # Preprocess for DeepSA
        # (Specific preprocessing depends on DeepSA's requirements)
        frame_tensor = preprocess_frame(frame)
        frame_tensor = frame_tensor.cuda()

        # Predict with DeepSA
        with torch.no_grad():
            prediction = model(frame_tensor)
            pred_mask = (prediction > 0.5).cpu().numpy().astype(np.uint8)

        # Load ground truth mask
        gt_mask = load_ground_truth_mask(case['json_path'])

        # Compute metrics
        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)

        print(f"DeepSA IoU:  {iou:.3f}")
        print(f"DeepSA Dice: {dice:.3f}")

        # Save visualization
        save_comparison(frame, pred_mask, gt_mask,
                       f"deepsa_results/{case['name']}_deepsa.png")

        results.append({
            'case': case['name'],
            'iou': iou,
            'dice': dice
        })

    # Summary
    print(f"\n{'='*60}")
    print("DEEPSA BASELINE SUMMARY")
    print(f"{'='*60}")
    avg_iou = np.mean([r['iou'] for r in results])
    avg_dice = np.mean([r['dice'] for r in results])
    print(f"Average IoU:  {avg_iou:.3f}")
    print(f"Average Dice: {avg_dice:.3f}")
    print(f"\nComparison with SAM 3 baseline:")
    print(f"SAM 3 IoU:   0.372")
    print(f"DeepSA IoU:  {avg_iou:.3f}")
    print(f"Improvement: {(avg_iou - 0.372) / 0.372 * 100:+.1f}%")

    return results

def load_ground_truth_mask(json_path):
    """Load ground truth mask from JSON contours"""
    import json

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create mask from contours (similar to json_to_masks.py)
    # ... implementation ...

    return gt_mask

def compute_iou(pred, gt):
    """Compute Intersection over Union"""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-6)

def compute_dice(pred, gt):
    """Compute Dice coefficient"""
    intersection = np.logical_and(pred, gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-6)

if __name__ == '__main__':
    results = test_deepsa_on_our_cases()
```

## Stage 2: Hybrid DeepSA → SAM 3 Pipeline

**Key Insight**: Use DeepSA's angiography-specific knowledge to guide SAM 3.

### Approach A: DeepSA Generates Point Prompts

```python
# deepsa_to_sam3_points.py

def deepsa_to_sam3_point_pipeline(frame):
    """
    DeepSA generates coarse mask → Sample points → SAM 3 refines

    Expected benefit: Combine DeepSA's vessel knowledge with SAM 3's
    general segmentation power
    """

    # Step 1: DeepSA coarse segmentation
    deepsa_mask = deepsa_model.predict(frame)  # Dice ~0.75-0.82

    # Step 2: Sample points from DeepSA mask
    # Strategy: Sample along vessel centerline + edges
    centerline_points = extract_centerline_points(deepsa_mask, n=5)
    edge_points = extract_edge_points(deepsa_mask, n=5)
    all_points = np.vstack([centerline_points, edge_points])

    # Label: 1 = foreground (vessel), 0 = background
    point_labels = np.ones(len(all_points))

    # Step 3: SAM 3 refinement
    sam3_mask = sam3_model.segment(
        image=frame,
        point_coords=all_points,
        point_labels=point_labels
    )

    return {
        'deepsa_mask': deepsa_mask,
        'sam3_mask': sam3_mask,
        'points': all_points
    }

def extract_centerline_points(mask, n=5):
    """
    Extract n points along vessel centerline

    Uses skeletonization to find centerline
    """
    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask)
    skeleton_coords = np.argwhere(skeleton > 0)

    # Sample n evenly-spaced points along skeleton
    if len(skeleton_coords) < n:
        return skeleton_coords

    indices = np.linspace(0, len(skeleton_coords)-1, n, dtype=int)
    sampled_points = skeleton_coords[indices]

    return sampled_points

def extract_edge_points(mask, n=5):
    """
    Extract n points along vessel edges

    Useful for helping SAM 3 understand vessel boundaries
    """
    from skimage.morphology import binary_erosion, binary_dilation

    # Edge = dilation - erosion
    dilated = binary_dilation(mask)
    eroded = binary_erosion(mask)
    edge = dilated & ~eroded

    edge_coords = np.argwhere(edge > 0)

    if len(edge_coords) < n:
        return edge_coords

    indices = np.linspace(0, len(edge_coords)-1, n, dtype=int)
    sampled_points = edge_coords[indices]

    return sampled_points
```

### Approach B: DeepSA Generates Bounding Box

```python
# deepsa_to_sam3_bbox.py

def deepsa_to_sam3_bbox_pipeline(frame):
    """
    DeepSA generates coarse mask → Compute tight bbox → SAM 3 segments
    """

    # Step 1: DeepSA coarse segmentation
    deepsa_mask = deepsa_model.predict(frame)

    # Step 2: Extract tight bounding box
    coords = np.argwhere(deepsa_mask > 0)
    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add 10% padding
    h, w = frame.shape[:2]
    padding = 0.1
    y_pad = int((y_max - y_min) * padding)
    x_pad = int((x_max - x_min) * padding)

    bbox = [
        max(0, x_min - x_pad),
        max(0, y_min - y_pad),
        min(w, x_max + x_pad),
        min(h, y_max + y_pad)
    ]

    # Step 3: SAM 3 refinement
    sam3_mask = sam3_model.segment(
        image=frame,
        bbox=bbox
    )

    return {
        'deepsa_mask': deepsa_mask,
        'sam3_mask': sam3_mask,
        'bbox': bbox
    }
```

### Approach C: Knowledge Distillation

```python
# deepsa_distill_to_sam3.py

class DistillationTrainer:
    """
    Train SAM 3 to mimic DeepSA's predictions

    Advantage: SAM 3 learns vessel-specific features from DeepSA
    """

    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model  # DeepSA (frozen)
        self.student = student_model  # SAM 3 (trainable with LoRA)

    def distillation_loss(self, student_output, teacher_output, ground_truth):
        """
        Combined loss:
        - Learn from ground truth (supervised)
        - Learn from teacher predictions (distillation)
        """

        # Supervised loss (student vs ground truth)
        loss_gt = F.binary_cross_entropy(student_output, ground_truth)

        # Distillation loss (student vs teacher)
        loss_distill = F.mse_loss(student_output, teacher_output)

        # Combined (weighted)
        alpha = 0.7  # Weight for ground truth
        beta = 0.3   # Weight for distillation

        total_loss = alpha * loss_gt + beta * loss_distill

        return total_loss

    def train_epoch(self, dataloader):
        """Train one epoch with distillation"""

        for batch in dataloader:
            frames, masks_gt = batch

            # Teacher predictions (frozen, no grad)
            with torch.no_grad():
                teacher_preds = self.teacher(frames)

            # Student predictions (trainable)
            student_preds = self.student(frames)

            # Distillation loss
            loss = self.distillation_loss(student_preds, teacher_preds, masks_gt)

            # Backprop
            loss.backward()
            optimizer.step()
```

## Stage 3: Self-Supervised Pretraining on OUR Data

**Big Opportunity**: We have 20,000-50,000 cases in CRF archive - MORE than DeepSA used!

### DeepSA's Self-Supervised Approach

DeepSA uses **CycleGAN-based Image-to-Image (I2I) translation**:

```
Pretext Task: Learn to translate between different angiography conditions

Frame A (RAO 30°) ──[Generator]──> Synthesized Frame B' (LAO 45°)
                                             ↓
                                    [Discriminator]
                                             ↓
                                   Compare with Real Frame B (LAO 45°)

Loss: CycleGAN loss (adversarial + cycle consistency)

Result: Generator learns vessel structure features WITHOUT annotations!
```

### Replicate on OUR Data

```python
# self_supervised_pretrain.py

class SelfSupervisedPretrainer:
    """
    Replicate DeepSA's self-supervised pretraining on our 20K-50K cases

    Uses CycleGAN to learn vessel features from unlabeled data
    """

    def __init__(self):
        # Generator: U-Net (same as DeepSA)
        self.generator = UNet(in_channels=1, out_channels=1)

        # Discriminator: PatchGAN
        self.discriminator = PatchGANDiscriminator()

    def pretrain_on_crf_archive(self, crf_dir, num_cases=20000):
        """
        Pretrain on ALL unlabeled CRF cases

        Args:
            crf_dir: Directory with 20K-50K CRF angiography cases
            num_cases: How many to use for pretraining
        """

        print(f"Loading {num_cases} unlabeled angiography cases...")

        # Load pairs of frames from different viewing angles
        # (or temporal pairs from cine sequences)
        dataset = AngiographyPairDataset(crf_dir, num_cases)

        # CycleGAN training
        print("Starting self-supervised pretraining...")
        for epoch in range(100):  # DeepSA used 100 epochs
            for frame_A, frame_B in dataset:
                # Forward cycle: A → B' → A''
                fake_B = self.generator(frame_A)
                reconstructed_A = self.generator(fake_B)

                # Losses
                loss_adversarial = self.discriminator_loss(fake_B, frame_B)
                loss_cycle = F.l1_loss(reconstructed_A, frame_A)
                loss_identity = F.l1_loss(self.generator(frame_B), frame_B)

                # Total loss
                loss = loss_adversarial + 10 * loss_cycle + 5 * loss_identity
                loss.backward()
                optimizer.step()

        # Save pretrained generator (U-Net)
        torch.save(self.generator.state_dict(),
                   "checkpoints/our_selfsupervised_pretrained.pth")

        print("Self-supervised pretraining complete!")
        print("U-Net has learned vessel features from 20K+ unlabeled cases")

    def finetune_on_medis_annotations(self, medis_cases):
        """
        Fine-tune pretrained U-Net on our 800+ Medis annotations

        Expected: Achieve Dice 0.82+ (similar to DeepSA)
        """

        # Load pretrained weights
        self.generator.load_state_dict(
            torch.load("checkpoints/our_selfsupervised_pretrained.pth")
        )

        # Fine-tune with supervised learning
        dataset = MedisAnnotationDataset(medis_cases)

        for epoch in range(20):
            for frame, mask_gt in dataset:
                pred_mask = self.generator(frame)
                loss = F.binary_cross_entropy(pred_mask, mask_gt)
                loss.backward()
                optimizer.step()

        print("Fine-tuning complete!")
        return self.generator

class AngiographyPairDataset:
    """
    Create pairs of frames for self-supervised learning

    Strategies:
    1. Different viewing angles (RAO/LAO pairs)
    2. Temporal pairs (frame t and t+5 from same cine)
    3. Contrast-enhanced vs non-enhanced
    """

    def __init__(self, crf_dir, num_cases):
        self.pairs = self.create_pairs(crf_dir, num_cases)

    def create_pairs(self, crf_dir, num_cases):
        """
        Scan CRF archive and create frame pairs

        Example: For each case, pair different viewing angles
        """
        pairs = []

        for case_dir in Path(crf_dir).glob("*"):
            # Find frames with different view angles
            rao_frames = list(case_dir.glob("*RAO*.dcm"))
            lao_frames = list(case_dir.glob("*LAO*.dcm"))

            # Pair them up
            for rao, lao in zip(rao_frames, lao_frames):
                pairs.append((rao, lao))

            if len(pairs) >= num_cases:
                break

        return pairs

    def __getitem__(self, idx):
        frame_A_path, frame_B_path = self.pairs[idx]
        frame_A = load_dicom_frame(frame_A_path)
        frame_B = load_dicom_frame(frame_B_path)
        return frame_A, frame_B
```

## Implementation Timeline

### Week 1: Test DeepSA Baseline
```bash
# Day 1-2: Setup
git clone https://github.com/newfyu/DeepSA.git
pip install -r requirements.txt
# Download pretrained weights

# Day 3: Test on our 3 cases
python test_deepsa_baseline.py

# Day 4-5: Compare with SAM 3
# Expected: DeepSA 0.6-0.8 IoU vs SAM 3 0.37 IoU
```

### Week 2: Hybrid Pipeline
```bash
# Day 1-2: Implement DeepSA → SAM 3 point prompts
python deepsa_to_sam3_points.py

# Day 3-4: Implement DeepSA → SAM 3 bbox
python deepsa_to_sam3_bbox.py

# Day 5: Compare all approaches
# - DeepSA alone
# - SAM 3 alone
# - DeepSA → SAM 3 (hybrid)
```

### Week 3-4: Fine-tune on Our 800 Cases
```bash
# Option A: Fine-tune DeepSA
python finetune_deepsa.py --cases 800

# Option B: Distillation to SAM 3
python deepsa_distill_to_sam3.py

# Expected: 0.75-0.85 IoU
```

### Month 2-3 (Optional): Self-Supervised Pretraining
```bash
# Massive opportunity: Use all 20K-50K CRF cases
python self_supervised_pretrain.py \
    --crf_archive /path/to/crf/archive \
    --num_cases 20000 \
    --epochs 100

# Then fine-tune
python finetune_on_medis.py --cases 800

# Expected: SOTA performance (Dice 0.85-0.90+)
```

## Comparison Matrix

| Approach | Expected IoU | Training Time | Inference Speed | Pros | Cons |
|----------|-------------|---------------|-----------------|------|------|
| **SAM 3 (baseline)** | 0.37 | 0h (pretrained) | ~1s/frame | General purpose | No vessel knowledge |
| **DeepSA (pretrained)** | 0.65-0.75 | 0h (pretrained) | ~0.3s/frame | Vessel-specific | May need fine-tuning |
| **DeepSA → SAM 3 (hybrid)** | 0.70-0.80 | 0h | ~1.3s/frame | Best of both | Slower |
| **DeepSA fine-tuned** | 0.75-0.85 | 2-3h | ~0.3s/frame | SOTA-level | Requires training |
| **SAM 3 + LoRA** | 0.75-0.85 | 2-4h | ~1s/frame | Latest arch | No vessel prior |
| **Self-supervised + fine-tune** | 0.85-0.90+ | 2-3 weeks | ~0.3s/frame | Ultimate performance | Long training |

## Decision Tree

```
Start Here: Do we have DeepSA weights?
│
├─ YES → Test DeepSA on 3 cases
│   │
│   ├─ IoU > 0.7? → Use DeepSA directly, fine-tune on 800 cases
│   │
│   └─ IoU 0.5-0.7? → Try hybrid DeepSA → SAM 3
│
└─ NO → Start with SAM 3 baseline
    │
    ├─ SAM 3 IoU > 0.6? → Fine-tune SAM 3 with LoRA
    │
    └─ SAM 3 IoU < 0.6? → Wait for DeepSA or try SAM-VMNet
```

## Key Takeaways

1. **DeepSA is our best knowledge transfer source** because:
   - Pretrained on 58K+ angiograms (our exact domain)
   - Achieves 0.828 Dice with minimal fine-tuning
   - Public weights available

2. **Hybrid approach is promising**:
   - DeepSA provides vessel-specific coarse mask
   - SAM 3 refines with superior segmentation architecture
   - Combines domain knowledge + general purpose power

3. **Ultimate opportunity**:
   - We have 20K-50K CRF cases (MORE than DeepSA used!)
   - Could replicate their self-supervised pretraining
   - Expected result: SOTA performance on OUR data

4. **Practical recommendation**:
   - Week 1: Test DeepSA pretrained
   - Week 2: Try hybrid pipeline
   - Week 3-4: Fine-tune winner on 800 Medis cases
   - Month 2+ (optional): Self-supervised pretraining on full archive

## Next Steps

Would you like me to:

1. **Create test_deepsa_baseline.py** and try it on your 3 cases?
2. **Download DeepSA** and set it up for testing?
3. **Implement the hybrid pipeline** (DeepSA → SAM 3)?
4. **Focus on SAM 3 fine-tuning** instead (if DeepSA weights unavailable)?
