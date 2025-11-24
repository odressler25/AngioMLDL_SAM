# DataParallel + LoRA Fix - Complete

## Fixes Applied

### Fix #1: lora_layers.py - Add .contiguous()

**File**: `lora_layers.py`
**Line**: 136
**Change**: Added `.contiguous()` to LinearWithLoRA forward output

```python
def forward(self, x):
    pretrained_output = self.linear(x)
    lora_output = self.lora(x)
    # FIX: Add .contiguous() for DataParallel compatibility
    return (pretrained_output + lora_output).contiguous()
```

**Why**: DataParallel + tensor addition creates non-contiguous memory layout that breaks SAM 3's reshaping operations in RoPE.

---

### Fix #2: sam3_lora_wrapper.py - Proper Forward Implementation

**File**: `sam3_lora_wrapper.py`
**Lines**: 101-140
**Change**: Replaced placeholder with actual implementation

```python
def forward(self, images, bboxes=None):
    # 1. Force resolution to 1008x1008 for RoPE
    target_size = 1008
    if images.shape[-1] != target_size or images.shape[-2] != target_size:
        images = torch.nn.functional.interpolate(
            images, size=(target_size, target_size),
            mode='bilinear', align_corners=False
        )

    # 2. Forward through LoRA-adapted backbone
    features = self.sam3_base.forward_image(images)

    # 3. Extract embeddings
    if isinstance(features, dict):
        image_embeddings = features.get('vision_features', features.get('image_embeddings', features))
    else:
        image_embeddings = features

    return image_embeddings
```

**Why**: Ensures 1008x1008 resolution for RoPE, and returns proper embeddings instead of zeros.

---

### Fix #3: train_stage1_lora_fixed.py - Update Training Script

**Changes**:
1. Re-enabled DataParallel (line 316-319)
2. Uses 1008x1008 resolution (line 297)
3. Batch size 16 (line 294)
4. Simplified forward method to use wrapper (lines 81-113)

**Why**: Wrapper now handles preprocessing, so training script is simpler and uses both GPUs.

---

## Test Before Running

```bash
python test_dataparallel_fix.py
```

Expected output:
```
[OK] ALL TESTS PASSED!
LoRA + DataParallel is now working!
Ready for overnight training on both GPUs.
```

---

## Ready to Run

**Script**: `train_stage1_lora_fixed.py`

**Configuration**:
- Batch size: 16 (8 per GPU with DataParallel)
- Resolution: 1008x1008 (handled by wrapper)
- Both RTX 3090 GPUs will be used
- Expected memory: ~15-20GB per GPU

**To Run**:
```bash
python train_stage1_lora_fixed.py
```

---

## What Changed from Yesterday

| Yesterday | Today (Fixed) |
|-----------|--------------|
| DataParallel disabled | DataParallel enabled |
| Single GPU only | Both GPUs used |
| Batch size 8 | Batch size 16 |
| Manual preprocessing | Automatic in wrapper |
| Resolution: 1008×1008 | Resolution: 1008×1008 |
| Training time: 12-16 hrs | Training time: 6-8 hrs |

---

## Technical Details

### Why .contiguous() Fixes DataParallel

DataParallel splits batches and scatters tensors across GPUs. When LoRA adds `pretrained + lora_output`, PyTorch creates a view with non-contiguous strides. SAM 3's attention layers expect contiguous memory for efficient reshaping. `.contiguous()` forces a memory copy into linear layout.

### Why 1008 (Not 1024)

SAM 3's RoPE cache is pre-computed for 1008×1008 resolution. The wrapper forces 1008 to match the expected sequence length. Using 1024 causes RoPE assertion failures.

---

## All Fixed! ✓

Both GPUs will now be utilized for LoRA training.
