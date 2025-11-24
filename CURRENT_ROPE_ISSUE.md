# Current RoPE Issue - Status

## Problem

Even with both fixes applied:
1. ✅ `.contiguous()` in `lora_layers.py` line 136
2. ✅ 1008×1008 resolution in wrapper and training script

We're **still** getting the RoPE assertion error:
```
File "C:\Users\odressler\sam3\sam3\model\vitdet.py", line 63, in reshape_for_broadcast
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
AssertionError
```

## What This Means

The RoPE (Rotary Position Encoding) cache shape doesn't match the Q/K tensor shapes being passed to it. This happens deep inside SAM 3's attention mechanism.

## Possible Causes

1. **LoRA wrapper changes tensor behavior**: When LoRA adds its output to the pretrained output, even with `.contiguous()`, something about the tensor shape/stride might be wrong for RoPE.

2. **DataParallel interaction**: The error happens when DataParallel splits the batch across GPUs. Maybe the LoRA-wrapped layers don't replicate correctly.

3. **Resolution still wrong**: Maybe 1008 isn't actually correct, or the patch size calculation is different with LoRA.

4. **Attention layer incompatibility**: The LoRA wrapper might be changing how `self.qkv(x)` behaves in the attention layer, causing the reshape operation to produce wrong dimensions.

## Next Steps to Debug

### Test 1: LoRA Without DataParallel
Run: `python test_lora_dataparallel_minimal.py`

This will test if LoRA works at all with 1008 resolution (without DataParallel).

**If this fails**: The issue is LoRA + SAM 3 fundamentally incompatible, not a DataParallel issue.
**If this succeeds**: The issue is specifically LoRA + DataParallel interaction.

### Test 2: Different Resolutions
Run: `python test_rope_debug.py`

This will test SAM 3 (without LoRA) at different resolutions to find what actually works.

## Alternative Approaches

If LoRA + DataParallel continues to fail, consider:

1. **Use the frozen SAM 3 version**: `train_stage1_frozen_sam3.py` works perfectly
   - Still uses both GPUs
   - Still has view conditioning
   - Just doesn't adapt SAM 3 with LoRA

2. **Try single-GPU LoRA training**: Disable DataParallel, use smaller batch size
   - Will be slower (12-16 hours instead of 6-8)
   - But LoRA might work without DataParallel

3. **Different LoRA implementation**: Try the official `peft` library instead of manual LoRA
   - Might handle DataParallel better
   - Risk: dependency conflicts you wanted to avoid

## Current Files

- `train_stage1_lora_fixed.py` - LoRA version (currently failing)
- `train_stage1_frozen_sam3.py` - Frozen version (works, ready to use)
- `test_lora_dataparallel_minimal.py` - New test to isolate the issue
- `test_rope_debug.py` - Test different resolutions

## Recommendation

Before spending more time debugging, **run the frozen version overnight** to get results while we investigate the LoRA issue. The frozen version:
- ✅ Uses both GPUs
- ✅ Has view conditioning
- ✅ Works reliably
- ❌ Doesn't adapt SAM 3 to your domain

This way you'll have baseline results even if LoRA debugging takes longer.
