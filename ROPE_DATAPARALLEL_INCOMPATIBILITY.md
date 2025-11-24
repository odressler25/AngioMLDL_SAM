# SAM 3 RoPE + DataParallel Incompatibility

## The Real Problem

**SAM 3's RoPE (Rotary Position Encoding) is fundamentally incompatible with PyTorch's DataParallel.**

This affects:
- ✗ Frozen SAM 3 with DataParallel
- ✗ SAM 3 with LoRA + DataParallel
- ✗ SAM 3 with manual LoRA + DataParallel
- ✗ SAM 3 with `peft` LoRA + DataParallel

## Why It Fails

When DataParallel replicates the model across GPUs:
1. The RoPE cache (`freqs_cis`) is a registered buffer in SAM 3's attention modules
2. DataParallel tries to replicate this buffer to each GPU
3. Something in the replication process causes a shape mismatch
4. The assertion fails: `assert freqs_cis.shape == (x.shape[-2], x.shape[-1])`

## Evidence

Both scripts fail with the **exact same error**:
- `train_stage1_frozen_sam3.py` (no LoRA at all) - **FAILS**
- `train_stage1_lora_fixed.py` (with LoRA) - **FAILS**

Error location: `sam3/model/vitdet.py:63` in `reshape_for_broadcast`

## Solutions

### Option 1: No DataParallel (Recommended)
**File**: `train_stage1_frozen_no_dataparallel.py`

- SAM 3 backbone runs on GPU 0 only
- Trainable parts (view encoder, seg head) run on GPU 0
- No DataParallel, single GPU training
- **Downside**: Slower training (~12-16 hours instead of 6-8)
- **Upside**: Actually works

### Option 2: Manual Batch Splitting
- Extract SAM 3 features in a loop, splitting batch manually
- Send half to GPU 0, half to GPU 1
- Concatenate results
- Use DataParallel only for trainable parts
- **Downside**: Complex implementation
- **Upside**: Uses both GPUs for SAM 3

### Option 3: DistributedDataParallel (DDP)
- Use PyTorch's DDP instead of DataParallel
- Might handle RoPE buffers better
- **Downside**: Requires multi-process launch script
- **Upside**: Better than DataParallel in general

### Option 4: Train Without SAM 3
- Use DeepSA or another baseline
- Don't use SAM 3 at all
- **Downside**: Defeats the purpose
- **Upside**: No RoPE issues

## What About `peft`?

**Answer**: `peft` is already installed (version 0.15.2) and there's no dependency conflict.

The problem was **never about the LoRA implementation** (manual vs `peft`). The problem is **DataParallel + SAM 3's RoPE**.

Using `peft` instead of manual LoRA won't fix this - the RoPE error happens in SAM 3's attention layers before LoRA even gets involved.

## Recommendation

**Use `train_stage1_frozen_no_dataparallel.py`** for now:
- It will work reliably
- Training takes overnight (12-16 hours) on single GPU
- You'll get baseline results
- Can investigate DDP or manual splitting later

## Future Investigation

If you want to use both GPUs:
1. Try DistributedDataParallel instead of DataParallel
2. Or manually split batches and call SAM 3 twice (once per GPU)
3. Or contact Meta about SAM 3 + DataParallel compatibility

But for now, single-GPU training is the path of least resistance.
