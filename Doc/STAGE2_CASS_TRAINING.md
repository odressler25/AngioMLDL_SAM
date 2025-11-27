# Stage 2: CASS Segment Classification Training

**Status:** ✅ Training Successfully Started (2025-11-27)

**Branch:** `sam3-native-training`

---

## Overview

Stage 2 trains SAM3 to classify 14 CASS coronary artery segments using:
- **SAM3 native training pipeline** (not custom wrapper)
- **Phase 1 domain-adapted weights** (0.8 Dice on vessel segmentation)
- **Medis GT masks** (discrete segment instances, not blobs)
- **Resolution 672px** (reduced from 1008 for memory constraints)

---

## Dataset

**Source:** Medis manual annotations (ground truth)

**Format:** COCO with 14 CASS categories

**Location:** `E:/AngioMLDL_data/coco_cass_segments/`

**Categories:**
```
1. proximal_rca      8. distal_lcx
2. mid_rca           9. obtuse_marginal_1
3. distal_rca       10. obtuse_marginal_2
4. pda              11. obtuse_marginal_3
5. proximal_lad     12. left_main
6. mid_lad          13. ramus
7. distal_lad       14. proximal_lcx
```

**Splits:**
- Train: 521 samples
- Val: 227 samples

---

## Key Files

```
train_sam3_clean.py                          # Training launcher
configs/angiography/cass_segmentation.yaml   # Stage 2 config
checkpoints/phase1_native_format.pth         # Domain-adapted init weights
scripts/weight_surgery.py                    # Checkpoint converter
scripts/create_coco_cass.py                  # Dataset creator
```

---

## Training Command

```powershell
python train_sam3_clean.py -c configs/angiography/cass_segmentation.yaml --num-gpus 2
```

---

## Technical Challenges & Solutions

### Challenge 1: RoPE Embedding Resizing

**Problem:** Training at resolution 672 (not default 1008) requires resizing Rotary Position Embeddings (RoPE). Initial approaches failed with assertion errors.

**Failed Approaches:**
1. ❌ Interpolating RoPE embeddings (breaks mathematical structure)
2. ❌ Resizing after DDP wrapping (buffer modifications don't propagate)
3. ❌ Simple regeneration without module detection (resizes wrong modules)

**Solution (by gpt5.1):**
```python
def resize_rope_embeddings(model, target_resolution, patch_size=14):
    """Regenerate RoPE embeddings using ViTDet formula."""

    def resize_module(module, window_context=None):
        # Track window size for windowed attention modules
        current_window = getattr(module, "window_size", None) or window_context

        # Only resize modules with:
        # 1. freqs_cis buffer (RoPE embeddings)
        # 2. input_size attribute (spatial RoPE users)
        # 3. Non-windowed attention (window_size == 0 or None)
        if (hasattr(module, "freqs_cis") and
            isinstance(module.freqs_cis, torch.Tensor) and
            getattr(module, "input_size", None) is not None):

            uses_window = current_window is not None and current_window > 0
            if not uses_window:
                # Detect CLS token offset
                cls_offset = 1 if getattr(module, "cls_token", False) else 0
                seq_len = module.freqs_cis.shape[0] - cls_offset

                if seq_len > 0:
                    grid_candidate = int(math.sqrt(seq_len))
                    if grid_candidate * grid_candidate == seq_len:
                        # Regenerate using ViTDet formula
                        new_freqs = precompute_freqs_cis(head_dim, target_grid)

                        # Prepend zero frequencies for CLS token if needed
                        if cls_offset:
                            cls_freqs = torch.polar(torch.ones(...), torch.zeros(...))
                            new_freqs = torch.cat([cls_freqs, new_freqs], dim=0)

                        module.register_buffer("freqs_cis", new_freqs, persistent=False)
```

**Key Insights:**
1. **Regenerate, don't interpolate** - RoPE has specific frequency structure
2. **Check `input_size`** - Only spatial attention modules need resizing
3. **Skip windowed attention** - Local attention doesn't use global grid
4. **Handle CLS tokens** - Prepend zero frequencies for class tokens
5. **Resize during checkpoint loading** - Before DDP wrapping

### Challenge 2: Windows Gloo Backend Limitations

**Problem:** Windows uses Gloo backend which doesn't support:
- bfloat16 parameters
- Complex buffers in collective operations

**Solution:**
```python
# Convert bfloat16 → float32
if bf16_params > 0:
    self.model = self.model.float()

# Disable buffer broadcasting for complex freqs_cis
DDP(model, broadcast_buffers=False, find_unused_parameters=True)
```

### Challenge 3: Checkpoint Format Mismatch

**Problem:** Phase 1 checkpoint has custom format (keys prefixed with `sam3.`), native trainer expects different format.

**Solution:** Weight surgery script (see `scripts/weight_surgery.py`)
```python
# Strip 'sam3.' prefix from backbone keys
native_state_dict = {key[5:]: value for key, value in custom_state_dict.items()
                     if key.startswith('sam3.')}

# Discard custom heads (seg_head, view_encoder, etc.)
# Save as 'model' key for native trainer
torch.save({'model': native_state_dict, ...}, output_path)
```

---

## Training Configuration

**Resolution:** 672px (48x48 grid with patch size 14)

**Batch Size:** 1 per GPU (memory constraints)

**GPUs:** 2x RTX 3090

**Precision:** float32 (Gloo requirement)

**Learning Rate:** From config (native trainer's schedule)

**Epochs:** From config

**Initialization:** Phase 1 checkpoint (`phase1_native_format.pth`)

---

## Expected Results

**Epoch 0 Baseline:**
```
Loss=238.61 | BBox=0.1017 | GIoU=0.4728 | CE=0.0017 | Mask=0.0097 | Dice=0.6302
```

**Target Performance:**
- Dice > 0.75 (segment segmentation)
- AP75 > 50% (instance detection at 0.75 IoU)
- Accurate CASS category classification

**Why This Should Work:**
1. ✅ **Domain-adapted backbone** (Phase 1: 0.8 Dice on angiograms)
2. ✅ **Discrete object masks** (Medis GT, not giant blobs)
3. ✅ **SAM3's instance head** (designed for object instances)
4. ✅ **Text prompts** (14 segment category names)
5. ✅ **Presence token** (multi-class detection support)

---

## Monitoring Training

**Experiment Directory:**
```
experiments/sam3_native_cass_672/
├── checkpoints/
│   └── checkpoint.pt        # Latest checkpoint
├── config.yaml              # Saved config
└── logs/                    # Training logs
```

**Key Metrics to Watch:**
- `Dice` - Segmentation quality (target: >0.75)
- `CE` - Classification loss (14 categories)
- `BBox/GIoU` - Detection quality
- `Mask` - Mask prediction loss

**Validation:**
```bash
# Check latest metrics
tail -f experiments/sam3_native_cass_672/logs/train.log

# Visualize predictions
python scripts/visualize_predictions.py --checkpoint experiments/sam3_native_cass_672/checkpoints/checkpoint.pt
```

---

## Comparison: Custom vs Native Training

| Aspect | Custom (Phase 1) | Native (Phase 2) |
|--------|------------------|------------------|
| **Architecture** | Custom wrapper + seg_head | SAM3 native |
| **Data Format** | PyTorch Dataset | COCO JSON |
| **Masks** | DeepSA blobs | Medis GT segments |
| **Categories** | 1 (vessel) | 14 (CASS) |
| **Result** | 0.8 Dice | TBD |
| **Use Case** | Domain adaptation | CASS classification |

**Lesson Learned:** DeepSA blobs (giant connected regions) confused SAM3's instance head. Medis GT provides proper discrete objects that match SAM3's architecture.

---

## Next Steps

1. ✅ Monitor training to convergence
2. ⏳ Evaluate on validation set (AP metrics)
3. ⏳ Visualize predictions per CASS category
4. ⏳ Compare with Phase 1 custom training
5. ⏳ Consider Stage 3: Stenosis detection

---

## Acknowledgments

- **RoPE fix:** Solved by gpt5.1 (sophisticated module detection)
- **Weight surgery:** Custom → Native format conversion
- **Dataset:** Medis manual annotations

---

*Last updated: 2025-11-27*
