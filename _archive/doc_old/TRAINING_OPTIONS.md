# Training Options Summary - Ready to Run

## Current Status

✓ **Both scripts are ready and working**
✓ **Dataset loading fixed** (521 train, 111 val samples)
✓ **RoPE issue solved** (1008×1008 resolution)

## Option 1: Frozen SAM 3 (Recommended for Overnight)

**Script**: `train_stage1_frozen_sam3.py`

**Pros**:
- Uses **both RTX 3090 GPUs** (DataParallel works)
- Faster training (~2x speed with both GPUs)
- More stable (no LoRA complexity)
- Proven to work in all tests

**Cons**:
- No SAM 3 domain adaptation
- Only trains view encoder + seg head (~5M params)

**Expected**:
- ~15-20GB per GPU with batch_size=16
- Training time: ~6-8 hours for 30 epochs
- Dice: 0.60-0.70 expected

**To Run**:
```bash
python train_stage1_frozen_sam3.py
```

---

## Option 2: LoRA + Single GPU

**Script**: `train_stage1_lora_fixed.py`

**Pros**:
- SAM 3 domain adaptation via LoRA (~3.1M LoRA params)
- Better performance potential (LoRA adapts to angiography)

**Cons**:
- **Only uses 1 GPU** (DataParallel + LoRA = RoPE error)
- Slower training (~2x slower, only 1 GPU)
- Less tested in long runs

**Expected**:
- ~12-15GB on GPU 0, GPU 1 idle
- Training time: ~12-16 hours for 30 epochs
- Dice: 0.65-0.75 expected (better than frozen)

**To Run**:
```bash
python train_stage1_lora_fixed.py
```

---

## Key Issues Solved

1. **RoPE Assertion Fixed**: SAM 3 needs 1008×1008, not 1024×1024
2. **Dataset Loading Fixed**: Properly loads from CSV with view angles
3. **Image Resizing Fixed**: All images resized to 1008×1008 for batching
4. **LoRA Device Fixed**: LoRA layers moved to correct GPU device

## Remaining Issue

**LoRA + DataParallel = RoPE Error**:
- Root cause unknown (likely LoRA wrapper interaction with DataParallel)
- Workaround: Disable DataParallel for LoRA version
- Alternative: Use frozen SAM 3 with DataParallel

---

## My Recommendation

**For overnight**: Run **Option 1 (Frozen SAM 3)** because:
1. Uses both GPUs (2x faster)
2. More stable and proven
3. Get results in the morning
4. Can always add LoRA later if needed

**For testing**: Run **Option 2 (LoRA)** if you want to see:
1. How much LoRA helps performance
2. If single-GPU LoRA training is stable overnight

---

## Monitoring Commands

```bash
# Watch GPU usage
nvidia-smi -l 1

# Check training progress (if checkpoints are being saved)
dir checkpoints

# View training output
# (will show loss/dice scores every epoch)
```

---

## Expected Output

Both scripts will print:
- Epoch progress bars with loss/dice
- Best model saved to `checkpoints/`
- GPU memory usage after each epoch

Example:
```
Epoch 1/30
  Train Loss: 0.6543, Train Dice: 0.3457
  Val Loss: 0.6234, Val Dice: 0.3766
  -> Saved best model (Dice: 0.3766)
  Max GPU memory: 18.23 GB
```

---

## Files Overview

- `train_stage1_frozen_sam3.py` - Frozen SAM 3 + Both GPUs ✓
- `train_stage1_lora_fixed.py` - LoRA + Single GPU ✓
- `E:\AngioMLDL_data\corrected_dataset_training.csv` - Dataset (748 samples) ✓

**All ready to run when you want!**
