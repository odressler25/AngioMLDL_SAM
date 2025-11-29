# SAM 3 for Coronary Angiography Lesion Detection

Training SAM 3 (Segment Anything Model 3) with LoRA for automated lesion detection and CASS segment classification in coronary angiography.

## Project Goal

Develop an AI system that can:
1. **Detect** coronary artery lesions in angiography images
2. **Classify** lesions by CASS segment (1-29)
3. **Measure** stenosis percentage and minimum lumen diameter (MLD)
4. **Account for** viewing angle variations (RAO/LAO/Cranial/Caudal)

This enables automated Quantitative Coronary Angiography (QCA) analysis.

## Approach

### Based on SAM-VMNet (2024)
We adapt the SAM-VMNet architecture which combines:
- MedSAM (medical foundation model) for feature extraction
- VM-UNet (Vision Mamba) for efficient segmentation
- LoRA fine-tuning for domain adaptation

### Our Enhancements
1. **Task-specific**: Lesion detection + CASS classification (not just segmentation)
2. **View-aware**: Incorporates viewing angle information
3. **Multi-task**: Predicts mask + CASS segment + stenosis % + MLD
4. **Efficient**: LoRA adapters (~100MB) instead of full fine-tuning

## Dataset

- **800+ expert-annotated cases** from clinical QCA analysis
- Each case includes:
  - Cine sequence (40-100 frames)
  - Frame with optimal contrast filling
  - Expert-traced vessel segment mask
  - CASS segment label (1-29)
  - Stenosis measurements (%, MLD, lesion length)
  - Viewing angles (RAO/LAO, Cranial/Caudal)

## Architecture

```
Input Frame + View Angle
       ↓
┌─────────────────────────┐
│ Stage 1: Bbox Proposal  │
│ (Coarse localization)   │
└─────────────────────────┘
       ↓
┌─────────────────────────┐
│ Stage 2: SAM 3 + LoRA   │
│ Prompts:                │
│  - Bbox                 │
│  - "CASS segment 2"     │
│  - "RAO 30 CAUDAL 25"   │
└─────────────────────────┘
       ↓
┌─────────────────────────┐
│ Stage 3: Multi-Task     │
│  - Segment mask         │
│  - CASS ID (1-29)       │
│  - Stenosis %           │
│  - MLD (mm)             │
└─────────────────────────┘
```

## Key Files

### Testing Scripts
- `test_sam3_correct_frames.py` - Test on contrast-filled frames
- `test_combined_prompts.py` - Test bbox + text prompting
- `test_sam3_bbox.py` - Bbox-only baseline

### Training
- `train_sam3_lora.py` - LoRA fine-tuning template
- `SAM3_TRAINING_STRATEGY.md` - Detailed training strategy

### Documentation
- `CORRECTED_UNDERSTANDING.md` - Complete data structure documentation
- `DATASET_ANALYSIS.md` - Dataset statistics
- `SAM3_EXEMPLAR_STRATEGY.md` - Few-shot learning approach

### Utilities
- `json_to_masks.py` - Convert JSON contours to masks
- `check_sam3_api.py` - Verify SAM 3 API

## Results (Preliminary)

Testing on 3 cases with correct contrast-filled frames:

| Case | Vessel | Frame | Bbox IoU | Text IoU |
|------|--------|-------|----------|----------|
| 101-0025 | RCA Mid | 37 | 0.391 | 0.000 |
| 101-0086 | LAD Mid | 30 | TBD | TBD |
| 101-0052 | LCX Dist | 40 | TBD | TBD |

**Key Finding**: SAM 3 works reasonably well with bbox prompts when vessels are visible (contrast-filled frames). Text-only prompts fail without fine-tuning.

## Expected Performance (After Training)

Based on SAM-VMNet paper + our enhancements:

| Metric | Target |
|--------|--------|
| Segmentation IoU | 0.75-0.85 |
| CASS Classification Acc | 85-92% |
| Stenosis % MAE | ±8-12% |
| MLD MAE | ±0.2-0.3mm |

## Hardware Requirements

- **GPU**: 2x NVIDIA RTX 3090 (24GB each)
- **Training time**: 2-4 hours for LoRA (20 epochs)
- **Inference**: <100ms per frame

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/AngioMLDL_SAM.git
cd AngioMLDL_SAM

# Install dependencies
pip install torch torchvision
pip install peft transformers
pip install numpy pillow opencv-python matplotlib tqdm

# Install SAM 3 (from official repo)
# Follow SAM 3 installation instructions
```

## Usage

### 1. Test SAM 3 on your data
```bash
python test_sam3_correct_frames.py
```

### 2. Test combined prompting
```bash
python test_combined_prompts.py
```

### 3. Train with LoRA (coming soon)
```bash
python train_sam3_lora.py --epochs 20 --batch_size 8
```

## References

1. **SAM-VMNet** (Zeng et al., 2024): "Deep Neural Networks For Coronary Angiography Vessel Segmentation"
   - arXiv:2406.00492
   - Combined MedSAM + VM-UNet
   - Achieved 63% mIoU on ARCADE dataset

2. **MedSAM** (Ma et al., 2024): "Segment Anything in Medical Images"
   - Nature Communications
   - Medical adaptation of SAM

3. **LoRA** (Hu et al., 2021): "Low-Rank Adaptation of Large Language Models"
   - Efficient fine-tuning method
   - 1-10% of parameters trained

## Related Work

See `SOTA_VESSEL_SEGMENTATION_2024-2025.md` in parent directory for comprehensive review of:
- DeepSA (self-supervised pretraining)
- TVS-Net (temporal vessel segmentation)
- DeepDiscern (CASS segment classification)
- ResAttNet (attention-based segmentation)

## License

[Add your license here]

## Citation

```bibtex
@misc{angioml_sam3,
  title={SAM 3 for Coronary Angiography Lesion Detection},
  author={[Your Name]},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/AngioMLDL_SAM}
}
```

## Contact

[Your contact information]

## Acknowledgments

- SAM 3 team at Meta AI
- SAM-VMNet authors for architectural inspiration
- Clinical experts for QCA annotations
