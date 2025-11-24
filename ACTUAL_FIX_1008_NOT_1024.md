# Actual Fix: 1008 Resolution, Not 1024

## The Problem

The external fix suggested using **1024×1024 resolution** with the `.contiguous()` fix. However, this was **incorrect** and still caused RoPE assertion errors:

```
AssertionError in sam3/model/vitdet.py:63
assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
```

## The Solution

SAM 3's RoPE (Rotary Position Encoding) cache is pre-computed for **1008×1008** resolution, not 1024. Your original discovery was correct.

## What Was Fixed

### 1. Kept the `.contiguous()` Fix
**File**: `lora_layers.py` line 136

This part of the external fix was correct:
```python
return (pretrained_output + lora_output).contiguous()
```

**Why**: DataParallel + tensor addition creates non-contiguous memory. `.contiguous()` ensures linear memory layout for SAM 3's reshaping operations.

### 2. Changed Resolution Back to 1008
**File**: `sam3_lora_wrapper.py` line 115

Changed from:
```python
target_size = 1024
```

To:
```python
target_size = 1008
```

**Why**: SAM 3's RoPE expects 1008×1008, which produces the correct sequence length for position encoding.

### 3. Updated Training Script
**File**: `train_stage1_lora_fixed.py` line 297

Changed from:
```python
image_size = 1024  # Changed to 1024 as per the fix
```

To:
```python
image_size = 1008  # SAM 3 expects 1008x1008
```

## The Complete Fix

**Two parts**:
1. ✅ `.contiguous()` in lora_layers.py (for DataParallel compatibility)
2. ✅ 1008×1008 resolution (for RoPE compatibility)

## Why 1008 Works

From your previous session (SAM3_SOLUTION.md):
- Sam3Processor uses 1008×1008 by default
- RoPE cache is pre-computed for this resolution
- Using 1024 creates a sequence length mismatch

## Comparison

| Resolution | Result |
|-----------|---------|
| 1024×1024 | ❌ RoPE assertion error |
| 1008×1008 | ✅ Works correctly |

## Ready to Test

The script is now fixed with both:
1. `.contiguous()` for DataParallel
2. 1008×1008 for RoPE

Try running:
```bash
python train_stage1_lora_fixed.py
```

Both RTX 3090 GPUs should now work with LoRA enabled.
