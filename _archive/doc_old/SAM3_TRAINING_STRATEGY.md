# SAM 3 Training Strategy for Coronary Angiography

## Key Clarification: Your Data Structure

**Your bounding boxes are LESION segments, not full vessels!**
- Each bbox contains a **specific CASS segment with stenosis**
- Example: "101-0025_MID_RCA_PRE" = Mid segment of RCA with a lesion
- The mask shows only the ~6,000 pixels of that specific segment
- This is actually MORE valuable than full vessel masks!

## Training Approaches for SAM 3

### 1. LoRA (Low-Rank Adaptation) - RECOMMENDED ✅
**What it is:** Adds small trainable matrices to frozen SAM 3 weights

**Pros:**
- Only trains ~1-10% of parameters
- Preserves SAM 3's general segmentation ability
- Fast training (few hours on 2x RTX 3090)
- Small checkpoint files (~50-200MB)
- Can swap different LoRAs for different vessels

**Implementation:**
```python
from peft import LoraConfig, get_peft_model

# Freeze SAM 3 base model
for param in sam3_model.parameters():
    param.requires_grad = False

# Add LoRA layers
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # attention layers
    lora_dropout=0.1,
)
model = get_peft_model(sam3_model, lora_config)

# Training data structure
train_data = {
    'image': cine[frame_num],
    'bbox': normalized_bbox,
    'text': 'MID_RCA stenosis',  # Lesion-specific!
    'mask': lesion_segment_mask,
    'metadata': {
        'vessel': 'RCA',
        'segment': 'MID',
        'stenosis_pct': 57.3,
        'mld': 1.35
    }
}
```

### 2. Prompt Tuning / Adapters
**What it is:** Trains only the prompt encoder while freezing the image encoder

**Pros:**
- Even fewer parameters than LoRA
- Learns to map "RCA lesion" → correct visual features
- Very fast training

**Cons:**
- Limited expressiveness
- May not capture fine vessel details

### 3. Full Fine-tuning
**What it is:** Updates all SAM 3 parameters

**Pros:**
- Maximum adaptation to medical domain
- Best potential performance

**Cons:**
- Risk of catastrophic forgetting
- Needs lots of data (you have 800+ cases ✅)
- Slow and expensive
- Large checkpoints (>2GB)

### 4. Knowledge Distillation
**What it is:** Train a smaller student model using SAM 3 as teacher

**Use case:**
- If you need faster inference
- Mobile/edge deployment

---

## Recommended Approach: Lesion-Specific LoRA

Since your data is **lesion segments**, not full vessels, here's the optimal strategy:

### Stage 1: Lesion Detection LoRA
Train SAM 3 to recognize **stenotic segments** specifically:

```python
# Training examples
examples = [
    {
        'prompt': 'proximal RCA stenosis',
        'bbox': [x, y, w, h],  # Around lesion
        'target': lesion_mask,   # Just the stenotic segment
        'stenosis_pct': 75.2
    },
    {
        'prompt': 'mid LAD stenosis',
        'bbox': [x, y, w, h],
        'target': lesion_mask,
        'stenosis_pct': 82.1
    }
]
```

### Stage 2: Multi-Task Learning
Since you have rich metadata, train for multiple objectives:

1. **Segmentation**: Pixel-wise vessel mask
2. **Classification**: Which CASS segment (1-29)
3. **Regression**: Stenosis %, MLD, reference diameter

```python
class SAM3QCA(nn.Module):
    def __init__(self, sam3_model):
        super().__init__()
        self.sam3 = sam3_model

        # Add task-specific heads
        self.cass_classifier = nn.Linear(256, 29)  # 29 CASS segments
        self.stenosis_regressor = nn.Linear(256, 1)  # Stenosis %
        self.mld_regressor = nn.Linear(256, 1)  # MLD in mm

    def forward(self, image, bbox, text):
        # Get SAM 3 features
        features, mask = self.sam3(image, bbox, text)

        # Multi-task predictions
        cass_segment = self.cass_classifier(features)
        stenosis = self.stenosis_regressor(features)
        mld = self.mld_regressor(features)

        return mask, cass_segment, stenosis, mld
```

### Stage 3: Vessel-Specific LoRAs
Train separate LoRA for each vessel type:

- **RCA_LoRA**: Specializes in RCA lesions
- **LAD_LoRA**: Specializes in LAD lesions
- **LCX_LoRA**: Specializes in LCX lesions

At inference, select LoRA based on view angle or initial classification.

---

## Why This Works for Your Use Case

Your data is actually PERFECT for LoRA training because:

1. **Focused task**: Segment stenotic lesions (not entire vessels)
2. **Clear bbox supervision**: Each lesion has tight bounds
3. **Rich labels**: Vessel name + CASS segment + measurements
4. **Clinical relevance**: Direct path to QCA measurements

## Implementation Timeline

### Week 1: Basic LoRA Training
```bash
pip install peft transformers
python train_sam3_lora.py \
    --data_path E:/AngioMLDL_data/corrected_vessel_dataset \
    --lora_rank 16 \
    --epochs 10 \
    --lr 1e-4
```

### Week 2: Multi-task heads
Add stenosis/MLD prediction on top of segmentation

### Week 3: Vessel-specific models
Train RCA vs LAD vs LCX specialists

### Week 4: Inference pipeline
1. Detect vessel type from view angle
2. Load appropriate LoRA
3. Find lesions with bbox proposals
4. Segment + measure

---

## Expected Results

With LoRA fine-tuning on your lesion data:
- **Segmentation IoU**: 0.6-0.8 (up from 0.2)
- **CASS classification**: 85-95% accuracy
- **Stenosis % error**: ±5-10%
- **Inference time**: <100ms per frame

## Code to Get Started

```python
# Install requirements
pip install transformers peft torch torchvision

# Prepare dataset
dataset = []
for case in cases:
    dataset.append({
        'image': load_frame(case, frame_num),
        'bbox': get_lesion_bbox(mask),
        'text': f"{vessel} {segment} stenosis",
        'mask': lesion_mask,
        'stenosis': stenosis_pct
    })

# Train LoRA
from train_sam3_lora import train_lora
model = train_lora(
    sam3_model,
    dataset,
    lora_rank=16,
    learning_rate=1e-4,
    batch_size=4,
    epochs=10
)

# Save LoRA weights only (~100MB)
model.save_pretrained("sam3_coronary_lora")
```

## Key Insight

Your "limitation" (only lesion segments) is actually a **strength**:
- More focused task = easier to learn
- Tight bboxes = less ambiguity
- Direct clinical application = measure stenosis

You don't need SAM 3 to segment entire vessels - just the parts that matter clinically!