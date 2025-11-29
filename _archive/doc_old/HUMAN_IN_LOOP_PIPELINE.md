# Human-in-the-Loop Pipeline for SAM 3

## Overview

Implement an iterative training pipeline where SAM 3 assists with annotation, and expert corrections improve the model. This dramatically reduces annotation time while improving model performance.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Round 1: Bootstrap (200 cases - already annotated)         │
├─────────────────────────────────────────────────────────────┤
│ 1. Load expert annotations (your existing 200+ cases)      │
│ 2. Train SAM 3 + LoRA on this initial set                  │
│ 3. Evaluate on validation set                              │
│ 4. Expected: 0.5-0.6 IoU (bootstrap model)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Round 2: Assisted Annotation (400 new cases)               │
├─────────────────────────────────────────────────────────────┤
│ 1. Model pre-segments vessels (bbox from lesion detection) │
│ 2. Expert reviews predictions in annotation tool           │
│ 3. Expert corrects mistakes (45-65% faster than manual)    │
│ 4. Fine-tune on 200 original + 400 corrected               │
│ 5. Expected: 0.65-0.75 IoU                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Round 3: Refinement (200 final cases)                      │
├─────────────────────────────────────────────────────────────┤
│ 1. Round 2 model pre-segments (much better now)            │
│ 2. Expert reviews (65-85% faster)                          │
│ 3. Expert corrects remaining mistakes                      │
│ 4. Final fine-tune on all 800 cases                        │
│ 5. Expected: 0.75-0.85 IoU (production model)              │
└─────────────────────────────────────────────────────────────┘
```

## Components Needed

### 1. Annotation Interface

We need a simple tool for experts to review and correct SAM 3 predictions.

**Options:**

**Option A: 3D Slicer Plugin** (recommended if you have 3D volumes)
- MedSAM has a Slicer plugin we can adapt
- Full medical imaging interface
- Supports DICOM
- Learning curve required

**Option B: Custom Python/Qt Interface** (recommended for 2D angiography)
```python
# Simple annotation tool
class HumanInLoopAnnotator:
    """
    GUI for reviewing SAM 3 predictions and correcting them
    """
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def review_case(self, case_id):
        # 1. Load image + bbox
        image, bbox, ground_truth = self.load_case(case_id)

        # 2. Get SAM 3 prediction
        prediction = self.model.segment(image, bbox)

        # 3. Show side-by-side: prediction vs ground truth
        self.display(image, prediction, ground_truth)

        # 4. Expert corrects prediction
        # Options: adjust mask, refine bbox, accept as-is
        corrected_mask = self.expert_correction_ui()

        # 5. Save corrected annotation
        self.save_correction(case_id, corrected_mask)

    def batch_review(self, case_ids):
        """Review multiple cases sequentially"""
        for case_id in case_ids:
            self.review_case(case_id)

    def compute_correction_stats(self):
        """Track how much time was saved"""
        return {
            'cases_reviewed': ...,
            'avg_correction_time': ...,
            'vs_manual_annotation': ...
        }
```

**Option C: Label Studio** (fastest to deploy)
- Open-source annotation platform
- SAM integration available
- Web-based interface
- Zero custom code needed
- https://labelstud.io/

**Option D: CVAT (Computer Vision Annotation Tool)**
- Open-source
- SAM 2 integration already exists
- Can adapt for SAM 3
- https://github.com/opencv/cvat

### 2. Training Pipeline

```python
# train_human_in_loop.py

class HumanInLoopTrainer:
    """
    Manages iterative training rounds with human corrections
    """

    def __init__(self, base_model="sam3"):
        self.base_model = load_sam3()
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv", "proj"],
            lora_dropout=0.05
        )

    def round_1_bootstrap(self, initial_cases):
        """
        Round 1: Train on existing expert annotations

        Args:
            initial_cases: 200 pre-annotated cases
        """
        print("Round 1: Bootstrap training...")

        # Load your existing 200+ annotated cases
        dataset = CoronaryLesionDataset(
            csv_path="corrected_dataset_training.csv",
            cases=initial_cases
        )

        # Train SAM 3 + LoRA
        model_r1 = self.train_lora(
            dataset=dataset,
            epochs=20,
            batch_size=8,
            lr=1e-4
        )

        # Evaluate
        val_iou = self.evaluate(model_r1)
        print(f"Round 1 IoU: {val_iou:.3f}")

        return model_r1

    def round_2_assisted_annotation(self, model_r1, new_cases):
        """
        Round 2: Use Round 1 model to assist annotation

        Args:
            model_r1: Model from Round 1
            new_cases: 400 new cases to annotate
        """
        print("Round 2: Assisted annotation...")

        # 1. Generate predictions for new cases
        predictions = []
        for case in tqdm(new_cases, desc="Pre-segmenting"):
            image, bbox = load_case(case)
            pred_mask = model_r1.segment(image, bbox)
            predictions.append({
                'case_id': case,
                'image': image,
                'bbox': bbox,
                'prediction': pred_mask
            })

        # 2. Expert reviews and corrects predictions
        print("Waiting for expert corrections...")
        print("Use annotation tool to review and correct predictions")
        print(f"Expected time: ~{len(new_cases) * 2} minutes")
        print("(vs ~{len(new_cases) * 5} minutes for manual annotation)")

        # This is where expert uses the annotation tool
        corrected_cases = self.wait_for_corrections(predictions)

        # 3. Fine-tune on original + corrected
        combined_dataset = CoronaryLesionDataset(
            cases=initial_cases + corrected_cases
        )

        model_r2 = self.train_lora(
            base_model=model_r1,  # Continue from Round 1
            dataset=combined_dataset,
            epochs=15,
            batch_size=8,
            lr=5e-5  # Lower LR for fine-tuning
        )

        # Evaluate
        val_iou = self.evaluate(model_r2)
        print(f"Round 2 IoU: {val_iou:.3f}")

        return model_r2, corrected_cases

    def round_3_refinement(self, model_r2, final_cases):
        """
        Round 3: Final refinement round

        Args:
            model_r2: Model from Round 2
            final_cases: 200 final cases
        """
        print("Round 3: Final refinement...")

        # Similar to Round 2 but with better model
        # Expert correction time should be 65-85% faster now

        predictions = []
        for case in tqdm(final_cases, desc="Pre-segmenting"):
            image, bbox = load_case(case)
            pred_mask = model_r2.segment(image, bbox)
            predictions.append({
                'case_id': case,
                'prediction': pred_mask
            })

        corrected_cases = self.wait_for_corrections(predictions)

        # Final training on ALL 800 cases
        all_cases = initial_cases + round2_cases + corrected_cases
        final_dataset = CoronaryLesionDataset(cases=all_cases)

        model_final = self.train_lora(
            base_model=model_r2,
            dataset=final_dataset,
            epochs=10,
            batch_size=8,
            lr=1e-5  # Very low LR for final refinement
        )

        # Final evaluation
        val_iou = self.evaluate(model_final)
        print(f"Final IoU: {val_iou:.3f}")

        return model_final

    def wait_for_corrections(self, predictions):
        """
        Save predictions and wait for expert corrections

        In practice, this would:
        1. Export predictions to annotation tool
        2. Expert reviews in tool
        3. Import corrected annotations
        """
        # Export predictions
        export_for_annotation(predictions, "predictions_for_review/")

        input("Press Enter after corrections are complete...")

        # Import corrected annotations
        corrected = import_corrections("corrected_annotations/")
        return corrected

    def evaluate(self, model):
        """Evaluate on validation set"""
        val_dataset = CoronaryLesionDataset(
            csv_path="corrected_dataset_validation.csv"
        )

        ious = []
        for image, bbox, mask_gt in val_dataset:
            pred_mask = model.segment(image, bbox)
            iou = compute_iou(pred_mask, mask_gt)
            ious.append(iou)

        return np.mean(ious)
```

### 3. Annotation Tool (Option B - Custom PyQt)

```python
# annotation_tool.py

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QSlider)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class SAM3AnnotationTool(QMainWindow):
    """
    Simple tool for reviewing SAM 3 predictions and correcting them
    """

    def __init__(self, predictions_dir, output_dir):
        super().__init__()
        self.predictions_dir = predictions_dir
        self.output_dir = output_dir
        self.current_idx = 0
        self.cases = self.load_cases()

        self.init_ui()
        self.load_current_case()

    def init_ui(self):
        self.setWindowTitle("SAM 3 Human-in-the-Loop Annotation")
        self.setGeometry(100, 100, 1600, 900)

        # Main layout
        main_layout = QVBoxLayout()

        # Image display (side-by-side)
        image_layout = QHBoxLayout()

        # Original + prediction overlay
        self.prediction_label = QLabel()
        self.prediction_label.setFixedSize(800, 800)
        image_layout.addWidget(self.prediction_label)

        # Corrected mask (editable)
        self.corrected_label = QLabel()
        self.corrected_label.setFixedSize(800, 800)
        image_layout.addWidget(self.corrected_label)

        main_layout.addLayout(image_layout)

        # Controls
        controls = QHBoxLayout()

        # Accept prediction as-is
        self.accept_btn = QPushButton("Accept (A)")
        self.accept_btn.clicked.connect(self.accept_prediction)
        controls.addWidget(self.accept_btn)

        # Refine mask
        self.refine_btn = QPushButton("Refine (R)")
        self.refine_btn.clicked.connect(self.refine_mask)
        controls.addWidget(self.refine_btn)

        # Reject and re-annotate
        self.reject_btn = QPushButton("Reject (D)")
        self.reject_btn.clicked.connect(self.reject_prediction)
        controls.addWidget(self.reject_btn)

        # Next case
        self.next_btn = QPushButton("Next (→)")
        self.next_btn.clicked.connect(self.next_case)
        controls.addWidget(self.next_btn)

        # Previous case
        self.prev_btn = QPushButton("Previous (←)")
        self.prev_btn.clicked.connect(self.prev_case)
        controls.addWidget(self.prev_btn)

        main_layout.addLayout(controls)

        # Progress
        self.progress_label = QLabel()
        main_layout.addWidget(self.progress_label)

        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_current_case(self):
        """Load current case for review"""
        case = self.cases[self.current_idx]

        # Load image, prediction, ground truth (if exists)
        self.image = cv2.imread(case['image_path'])
        self.prediction = np.load(case['prediction_path'])

        # Display
        self.display_prediction()
        self.update_progress()

    def display_prediction(self):
        """Show prediction overlay on image"""
        # Overlay prediction on image (green mask)
        overlay = self.image.copy()
        overlay[self.prediction > 0] = [0, 255, 0]
        blended = cv2.addWeighted(self.image, 0.7, overlay, 0.3, 0)

        # Convert to QPixmap
        height, width, channel = blended.shape
        bytes_per_line = 3 * width
        q_img = QImage(blended.data, width, height, bytes_per_line,
                       QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)

        self.prediction_label.setPixmap(pixmap.scaled(
            self.prediction_label.size(), Qt.KeepAspectRatio))

    def accept_prediction(self):
        """Accept prediction as-is (no correction needed)"""
        # Save prediction as corrected annotation
        case = self.cases[self.current_idx]
        output_path = f"{self.output_dir}/{case['case_id']}_corrected.npy"
        np.save(output_path, self.prediction)

        print(f"✓ Accepted: {case['case_id']}")
        self.next_case()

    def refine_mask(self):
        """Open mask refinement tool"""
        # Could open a drawing tool here
        # For simplicity, we'll skip this in template
        print("Refinement tool (to be implemented)")

    def reject_prediction(self):
        """Reject prediction - flag for manual annotation"""
        case = self.cases[self.current_idx]
        # Flag for manual annotation
        with open(f"{self.output_dir}/rejected_cases.txt", "a") as f:
            f.write(f"{case['case_id']}\n")

        print(f"✗ Rejected: {case['case_id']}")
        self.next_case()

    def next_case(self):
        """Load next case"""
        if self.current_idx < len(self.cases) - 1:
            self.current_idx += 1
            self.load_current_case()
        else:
            print("All cases reviewed!")
            self.close()

    def prev_case(self):
        """Load previous case"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_case()

    def update_progress(self):
        """Update progress label"""
        self.progress_label.setText(
            f"Case {self.current_idx + 1} / {len(self.cases)} - "
            f"{self.cases[self.current_idx]['case_id']}"
        )

    def load_cases(self):
        """Load all cases for review"""
        # Load predictions directory
        import glob
        prediction_files = glob.glob(f"{self.predictions_dir}/*.npy")

        cases = []
        for pred_path in prediction_files:
            case_id = pred_path.split('/')[-1].replace('_prediction.npy', '')
            image_path = pred_path.replace('_prediction.npy', '.png')

            cases.append({
                'case_id': case_id,
                'prediction_path': pred_path,
                'image_path': image_path
            })

        return cases

# Usage
if __name__ == '__main__':
    app = QApplication(sys.argv)
    tool = SAM3AnnotationTool(
        predictions_dir="predictions_for_review/",
        output_dir="corrected_annotations/"
    )
    tool.show()
    sys.exit(app.exec_())
```

## Usage Workflow

### Step 1: Bootstrap Training
```bash
python train_human_in_loop.py --round 1 \
    --initial_cases corrected_dataset_training.csv \
    --num_cases 200
```

### Step 2: Generate Predictions for Round 2
```bash
python generate_predictions.py \
    --model checkpoints/round1_model.pth \
    --cases round2_cases.csv \
    --output predictions_for_review/
```

### Step 3: Expert Review
```bash
python annotation_tool.py \
    --predictions predictions_for_review/ \
    --output corrected_annotations/
```

### Step 4: Round 2 Training
```bash
python train_human_in_loop.py --round 2 \
    --base_model checkpoints/round1_model.pth \
    --corrected corrected_annotations/ \
    --num_cases 400
```

### Step 5: Repeat for Round 3
```bash
# Generate predictions with Round 2 model
python generate_predictions.py \
    --model checkpoints/round2_model.pth \
    --cases round3_cases.csv \
    --output predictions_round3/

# Expert review
python annotation_tool.py \
    --predictions predictions_round3/ \
    --output corrected_round3/

# Final training
python train_human_in_loop.py --round 3 \
    --base_model checkpoints/round2_model.pth \
    --corrected corrected_round3/ \
    --num_cases 200
```

## Expected Timeline

| Round | Cases | Annotation Time | Training Time | Total |
|-------|-------|----------------|---------------|-------|
| 1 | 200 (done) | 0h (already annotated) | 2-3h | 2-3h |
| 2 | 400 | ~13h (vs 33h manual) | 2-3h | 15-16h |
| 3 | 200 | ~3h (vs 17h manual) | 2h | 5h |
| **Total** | **800** | **~16h** | **~7h** | **~23h** |

**vs. Manual Annotation**: 800 cases × 5 min/case = **67 hours**

**Time Saved**: ~44 hours (65% reduction)

## Metrics to Track

```python
class AnnotationMetrics:
    """Track efficiency of human-in-the-loop process"""

    def __init__(self):
        self.round_stats = []

    def track_round(self, round_num, corrections):
        """
        Track statistics for each round

        Args:
            round_num: 1, 2, or 3
            corrections: List of correction events
        """
        stats = {
            'round': round_num,
            'total_cases': len(corrections),
            'accepted': sum(c['action'] == 'accept' for c in corrections),
            'refined': sum(c['action'] == 'refine' for c in corrections),
            'rejected': sum(c['action'] == 'reject' for c in corrections),
            'avg_time_per_case': np.mean([c['time_seconds'] for c in corrections]),
            'acceptance_rate': sum(c['action'] == 'accept' for c in corrections) / len(corrections)
        }

        self.round_stats.append(stats)

        print(f"\n=== Round {round_num} Statistics ===")
        print(f"Total cases: {stats['total_cases']}")
        print(f"Accepted: {stats['accepted']} ({stats['acceptance_rate']:.1%})")
        print(f"Refined: {stats['refined']}")
        print(f"Rejected: {stats['rejected']}")
        print(f"Avg time: {stats['avg_time_per_case']:.1f}s per case")
        print(f"Time saved: {self.compute_time_saved(stats):.1f} hours")

    def compute_time_saved(self, stats):
        """Compute time saved vs manual annotation"""
        manual_time = stats['total_cases'] * 5 * 60  # 5 min per case
        actual_time = stats['total_cases'] * stats['avg_time_per_case']
        time_saved_hours = (manual_time - actual_time) / 3600
        return time_saved_hours
```

## For Echo Dataset

Since you mentioned having thousands of annotated echo images, you can run a **parallel pipeline**:

```python
# Separate pipeline for echo data
class EchoHumanInLoop(HumanInLoopTrainer):
    """
    Specialized for cardiac echo
    """

    def __init__(self):
        # Use MedSAM2 for echo instead of SAM 3
        # MedSAM2 was pre-trained on 3,583 echo videos!
        self.base_model = load_medsam2()

    def round_1_bootstrap(self, echo_cases):
        """
        For echo, MedSAM2 already performs well (96% DSC)
        Just need to fine-tune for your specific echo data
        """
        print("Echo Round 1: Fine-tuning MedSAM2...")
        # Should get 90%+ IoU immediately

# Usage for echo
echo_trainer = EchoHumanInLoop()
echo_model = echo_trainer.round_1_bootstrap(echo_cases=1000)
# Expected: 90%+ IoU out of box (vs 50-60% for angiography)
```

## Summary

**Yes, we can definitely implement human-in-the-loop for SAM 3!**

Key components:
1. ✅ **Iterative training script** (train_human_in_loop.py)
2. ⚠️ **Annotation tool** (choose from: Label Studio, CVAT, or custom PyQt)
3. ✅ **Metrics tracking** (acceptance rate, time saved, IoU improvement)

Expected benefits:
- **65-85% reduction** in annotation time
- **Progressive improvement** in model quality
- **800 annotated cases** in ~23 hours total (vs 67 hours manual)

For echo dataset:
- Use **MedSAM2** instead (pre-trained on echo)
- Expected **90%+ IoU immediately**
- Much faster annotation process

Would you like me to:
1. Create the complete training pipeline code?
2. Set up Label Studio for annotation?
3. Create the custom PyQt annotation tool?
4. Start with echo dataset first (easier due to MedSAM2)?
