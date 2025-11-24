# LoRA vs Full Fine-tuning: Complete 3-Stage Analysis

## The Big Picture Question

Since we need to train SAM 3 through **3 sequential stages**, the choice of LoRA vs full fine-tuning has **compounding effects**:

```
Stage 1: Full Vessel Segmentation
    ↓ (transfer weights)
Stage 2: CASS Segment Classification
    ↓ (transfer weights)
Stage 3: Stenosis Detection
    ↓
Final Model
```

Each stage builds on the previous, so we need to consider:
1. **Weight transfer between stages** - can we easily transfer?
2. **Catastrophic forgetting** - does Stage 2 forget Stage 1?
3. **Training efficiency** - time and GPU memory
4. **Final performance** - which achieves better clinical accuracy?

---

## Option A: LoRA for All 3 Stages ⭐ RECOMMENDED

### Architecture

```python
# Stage 1: Base SAM 3 + LoRA_1 (vessel segmentation)
sam3_base [frozen, 840M params]
  ↓
lora_adapters_stage1 [trainable, ~10M params]
  ↓
Output: Full vessel tree masks

# Stage 2: Keep LoRA_1 + Add LoRA_2 (CASS classification)
sam3_base [frozen, 840M params]
  ↓
lora_adapters_stage1 [frozen, keeps vessel knowledge]
  ↓
lora_adapters_stage2 [trainable, ~10M params for CASS]
  ↓
Output: CASS segment masks + IDs

# Stage 3: Keep LoRA_1 + LoRA_2 + Add heads (stenosis)
sam3_base [frozen, 840M params]
  ↓
lora_adapters_stage1 [frozen]
  ↓
lora_adapters_stage2 [frozen]
  ↓
stenosis_heads [trainable, ~5M params]
  ↓
Output: Stenosis %, MLD, lesion location
```

### Pros ✅

#### 1. **No Catastrophic Forgetting**
```python
# Stage 1 knowledge is preserved in LoRA_1 (frozen)
# Stage 2 adds new knowledge in LoRA_2 without destroying Stage 1
# Stage 3 only adds heads, doesn't touch segmentation

# Result: Model remembers EVERYTHING it learned
vessel_knowledge = lora_stage1  # Preserved
cass_knowledge = lora_stage2    # Preserved
stenosis_knowledge = heads_stage3  # New
```

**Clinical benefit**: Final model can still do full vessel segmentation (Stage 1) AND CASS classification (Stage 2) AND stenosis detection (Stage 3).

#### 2. **Modular & Composable**
```python
# Want just vessel segmentation? Load Stage 1 only
sam3 + lora_stage1 → vessel masks

# Want vessel + CASS? Load Stages 1+2
sam3 + lora_stage1 + lora_stage2 → CASS segments

# Want full pipeline? Load all 3
sam3 + lora_stage1 + lora_stage2 + heads_stage3 → full QCA
```

**Research benefit**: Can publish each stage independently, test ablations easily.

#### 3. **Efficient Training**
```
Stage 1: Train 10M params (LoRA_1) - ~2 hours
Stage 2: Train 10M params (LoRA_2) - ~2 hours
Stage 3: Train 5M params (heads) - ~1 hour
Total: ~5 hours, 25M trainable params
```

vs full fine-tuning:
```
Stage 1: Train 840M params - ~20 hours
Stage 2: Train 840M params - ~20 hours
Stage 3: Train 840M params - ~20 hours
Total: ~60 hours, 840M params each stage
```

**12x faster training!**

#### 4. **GPU Memory Efficient**
```python
# LoRA: Only need gradients for adapters
Memory = model_activations + lora_gradients
        ≈ 8GB + 2GB = 10GB per GPU

# Full fine-tuning: Gradients for entire model
Memory = model_activations + all_gradients
        ≈ 8GB + 16GB = 24GB per GPU

# Your 2x RTX 3090 (24GB each):
LoRA: Can use batch_size = 16-32 ✅
Full: Can use batch_size = 4-8 ⚠️
```

**Better GPU utilization with LoRA!**

#### 5. **Easy Model Versioning**
```
sam3_base.pth                    [840MB, shared]
├─ lora_stage1_v1.pth           [40MB]
├─ lora_stage1_v2.pth           [40MB]  ← Try different hyperparameters
├─ lora_stage2_cass.pth         [40MB]
└─ stenosis_heads.pth           [20MB]

Total storage for 5 versions: 840MB + 5×100MB = 1.34GB
```

vs full fine-tuning:
```
sam3_stage1_v1.pth              [840MB]
sam3_stage1_v2.pth              [840MB]
sam3_stage2_v1.pth              [840MB]
sam3_stage3_final.pth           [840MB]

Total: 3.36GB for just 4 versions
```

**Easy to experiment with LoRA!**

#### 6. **Preserves SAM 3's General Abilities**
```python
# SAM 3 base model still knows how to segment ANYTHING
# LoRA just adds "coronary vessel expertise"

# Example: Can still segment other organs if needed
liver_mask = sam3(liver_image, bbox=liver_bbox)  # Still works!
vessel_mask = sam3(angio_image) + lora_stage1   # Enhanced for vessels
```

**Benefit**: If you later want to segment other structures (catheter, stent, etc.), base model still has that capability.

---

### Cons ⚠️

#### 1. **Slightly Lower Peak Performance (Maybe)**
```
Reported in literature:
- LoRA: 95-98% of full fine-tuning performance
- Full fine-tuning: 100% (by definition)

Practical difference for your task:
- LoRA: Dice 0.78-0.82
- Full: Dice 0.80-0.84

Difference: ~2% Dice (likely not clinically significant)
```

**Counterpoint**: Many papers show LoRA MATCHES full fine-tuning on domain-specific tasks. The 2% gap may not materialize.

#### 2. **Complexity in Managing Multiple LoRA Modules**
```python
# Need to carefully manage which LoRAs are active

# Stage 2 training:
lora_stage1.requires_grad = False  # Freeze
lora_stage2.requires_grad = True   # Train

# Stage 3 training:
lora_stage1.requires_grad = False  # Freeze
lora_stage2.requires_grad = False  # Freeze
stenosis_heads.requires_grad = True  # Train

# Easy to make mistakes!
```

**Mitigation**: Use `peft` library which handles this automatically.

#### 3. **Need to Find Optimal Rank (r)**
```python
# LoRA hyperparameter: rank 'r'
lora_config = LoraConfig(
    r=16,  # ← Need to tune this
    lora_alpha=32,
    ...
)

# Too small (r=4): Underfits, poor performance
# Too large (r=64): Overfits, loses efficiency
# Just right (r=16-32): Best tradeoff

# Need to experiment to find optimal r
```

**Mitigation**: Start with r=16 (standard), adjust if needed.

---

## Option B: Full Fine-tuning All 3 Stages

### Architecture

```python
# Stage 1: Fine-tune entire SAM 3
sam3_base [all 840M params trainable]
  ↓
Output: Full vessel tree masks
Save as: sam3_stage1.pth [840MB]

# Stage 2: Load Stage 1 → Fine-tune again
sam3_stage1 [all 840M params trainable]
  ↓
Output: CASS segment masks + IDs
Save as: sam3_stage2.pth [840MB]

# Stage 3: Load Stage 2 → Fine-tune again
sam3_stage2 [all 840M params trainable]
  ↓
Output: Stenosis %, MLD
Save as: sam3_stage3.pth [840MB]
```

### Pros ✅

#### 1. **Maximum Flexibility**
Every parameter can adapt to the task. No architectural constraints from LoRA.

#### 2. **Potentially Highest Performance**
If LoRA does have a 2% gap, full fine-tuning avoids it.

#### 3. **Simpler Training Code**
```python
# No need to manage LoRA modules
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# Just train everything
```

---

### Cons ❌

#### 1. **Catastrophic Forgetting Risk**
```python
# Stage 2 training might destroy Stage 1 knowledge
# Example:
#   Stage 1: Learned to segment vessels perfectly (Dice 0.85)
#   Stage 2: Focus on CASS classification
#   Result: Vessel segmentation degrades (Dice drops to 0.75)

# Why? All 840M params are modified, old knowledge can be overwritten
```

**Mitigation**: Use very low learning rate (1e-6), careful regularization.

**But**: Even with mitigation, some forgetting is inevitable.

#### 2. **Training Time: 12x Slower**
```
Stage 1: 20 hours (840M params)
Stage 2: 20 hours (840M params)
Stage 3: 20 hours (840M params)
Total: 60 hours

vs LoRA: 5 hours total
```

**Reality check**: 60 hours = 2.5 days of GPU time!

#### 3. **GPU Memory: Requires Smaller Batches**
```
Your 2x RTX 3090 (24GB each):

Full fine-tuning batch size: 4-8
LoRA batch size: 16-32

Smaller batches → slower convergence → need more epochs
```

**Compounding effect**: Slower + smaller batches = even longer training.

#### 4. **Storage: 3x More Disk Space**
```
LoRA: 840MB (base) + 100MB (adapters) = 940MB
Full: 840MB × 3 stages = 2.52GB

For 5 experiments:
LoRA: 1.34GB
Full: 12.6GB
```

#### 5. **Can't Reuse Base Model for Other Tasks**
```python
# After Stage 3, SAM 3 is ONLY good for coronary QCA
# Lost general segmentation ability

# Want to segment liver? Need to start from original SAM 3 again
# LoRA preserves this flexibility
```

---

## Head-to-Head Comparison

| Aspect | LoRA ⭐ | Full Fine-tuning |
|--------|---------|------------------|
| **Training Time** | 5 hours | 60 hours |
| **GPU Memory** | 10GB/GPU | 24GB/GPU |
| **Batch Size** | 16-32 | 4-8 |
| **Catastrophic Forgetting** | ❌ None (frozen stages) | ⚠️ High risk |
| **Peak Performance** | ~98% of full | 100% |
| **Storage** | 940MB | 2.52GB |
| **Modularity** | ✅ Can load stages independently | ❌ All-or-nothing |
| **Experimentation** | ✅ Fast, cheap | ❌ Slow, expensive |
| **General ability preservation** | ✅ Yes | ❌ No |
| **Code complexity** | ⚠️ Moderate (manage LoRA) | ✅ Simple |

---

## Clinical Perspective: Does 2% Dice Matter?

### Dice 0.78 (LoRA) vs Dice 0.80 (Full Fine-tuning)

**In practice**:
```
Dice 0.78: IoU ≈ 0.64 → Very good clinical performance
Dice 0.80: IoU ≈ 0.67 → Slightly better

For stenosis measurement:
- Both achieve ±10-12% stenosis accuracy (clinical standard)
- Both achieve ±0.2-0.3mm MLD accuracy

Cardiologist can't tell the difference!
```

**More important factors**:
1. View-angle robustness (same segment, different angles)
2. Stenosis measurement accuracy
3. False positive rate (flagging normal vessels as diseased)

These depend MORE on training data quality than LoRA vs full fine-tuning.

---

## Practical Recommendation: LoRA Wins

### Why LoRA is Better for Your Use Case

#### 1. **3-Stage Pipeline is Perfect for LoRA**
```python
# Stage 1: sam3 + lora_vessel → learn vessel anatomy
# Stage 2: sam3 + lora_vessel [frozen] + lora_cass → add CASS knowledge
# Stage 3: sam3 + lora_vessel + lora_cass [both frozen] + heads → add stenosis

# Each stage builds on previous WITHOUT destroying it
# This is EXACTLY what LoRA was designed for!
```

#### 2. **Iterative Development**
```python
# Week 1: Train Stage 1, test vessel segmentation
if vessel_dice < 0.75:
    # Retrain Stage 1 with different hyperparameters
    # Only 2 hours to retrain!

# Week 2: Train Stage 2, test CASS classification
if cass_accuracy < 0.85:
    # Retrain Stage 2
    # Stage 1 knowledge is safe (frozen LoRA)

# Week 3: Train Stage 3, test stenosis
# Again, Stages 1 & 2 are safe
```

**With full fine-tuning**: Each mistake requires 20 hours to fix!

#### 3. **Your Hardware is Ideal for LoRA**
```
2x RTX 3090 (24GB each) = 48GB total

LoRA training:
- Batch size 32 across 2 GPUs (16 per GPU)
- Full 20GB/GPU utilization ✅
- 2-3 hour training per stage

Full fine-tuning:
- Batch size 8 across 2 GPUs (4 per GPU)
- Only 12GB/GPU utilization ⚠️
- 20 hour training per stage
```

**LoRA uses your hardware more efficiently!**

#### 4. **Research & Publication**
```python
# Can publish/share modular components:
"DeepSA → SAM 3 Knowledge Transfer via LoRA" (Stage 1)
"View-Angle Invariant CASS Classification" (Stage 2)
"Automated Stenosis Quantification" (Stage 3)

# Each is independently valuable!
```

---

## Final Recommendation: LoRA with Safety Net

### The Plan

**Primary approach: LoRA**
- Use LoRA (r=16) for all 3 stages
- Expected performance: Dice 0.78-0.82 for vessels, 85-92% CASS accuracy
- Training time: ~5 hours total
- Flexibility: Can experiment rapidly

**Safety net: Full fine-tuning Stage 3 ONLY (if needed)**
```python
# If Stage 3 (stenosis) performance is insufficient:

# Option: Unfreeze everything for Stage 3 final training
sam3_base.requires_grad = True  # Unfreeze
lora_stage1.requires_grad = True
lora_stage2.requires_grad = True

# Do gentle full fine-tuning (very low LR)
# This gives LoRA's speed for Stages 1-2, full tuning for final polish
```

### Implementation

```python
# Stage 1: LoRA for vessel segmentation
lora_config_stage1 = LoraConfig(
    r=16,  # Start conservative
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05
)

# Stage 2: LoRA for CASS classification
lora_config_stage2 = LoraConfig(
    r=16,  # Same rank as Stage 1
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],  # Add k_proj
    lora_dropout=0.05
)

# Stage 3: Just add heads (no new LoRA needed)
stenosis_heads = StenosisDetectionHeads(...)
```

---

## Bottom Line

**Use LoRA** because:
1. ✅ **12x faster** (5 hours vs 60 hours)
2. ✅ **No catastrophic forgetting** (each stage preserved)
3. ✅ **Better GPU utilization** (batch_size 32 vs 8)
4. ✅ **Modular** (can publish/share stages)
5. ✅ **Likely same clinical performance** (±2% Dice is negligible)
6. ✅ **Faster iteration** (fix mistakes in hours, not days)

**When to consider full fine-tuning:**
- ❌ Never for Stages 1-2 (LoRA is clearly better)
- ⚠️ Maybe for Stage 3 IF LoRA underperforms AND you have time
- ⚠️ Only if you need that last 1-2% performance AND can afford 20+ hours

**My strong recommendation: Go with LoRA.**

You can always add full fine-tuning later as a "polish" step if needed, but start with LoRA for speed and modularity.

---

## Next Steps

Ready to implement LoRA?

1. Install `peft` library
2. Add LoRA config to `train_stage1_vessel_segmentation.py`
3. Run Stage 1 training (~2 hours)
4. Evaluate and iterate

Should I proceed with LoRA implementation?
