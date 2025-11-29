# Training Data Flow - What the Model Actually Sees

## Step-by-Step Breakdown

### 1. Data Loading (Dataset.__getitem__)

**From CSV row**:
```
cine_path: E:/AngioMLDL_data/.../patient_123_vessel_LAD_cine.npy
mask_path: E:/AngioMLDL_data/.../patient_123_vessel_LAD_mask.npy
contours_path: E:/AngioMLDL_data/.../patient_123_contours.json
frame_index: 42
```

**Loaded data**:
- **Cine video**: (T, H, W) or (T, H, W, C) - e.g., (120, 512, 512) - 120 frames
- **Extract single frame**: `frame = cine[42]` → (512, 512) grayscale
- **Convert to RGB**: Stack 3 times → (512, 512, 3)
- **Normalize**: Divide by 255 → values in [0, 1]
- **Resize**: Interpolate to (1008, 1008, 3) for consistent batching

**Vessel mask**:
- Load from .npy file: (512, 512) binary mask
- Normalize to [0, 1]
- Resize to (1008, 1008)

**View angles from JSON**:
```json
{
  "view_angles": {
    "primary": -30.0,     // LAO/RAO angle (degrees)
    "secondary": 25.0     // Cranial/Caudal angle (degrees)
  }
}
```

**Returns**:
- `image`: (3, 1008, 1008) tensor in [0, 1]
- `mask`: (1008, 1008) tensor in [0, 1] (1=vessel, 0=background)
- `primary_angle`: scalar tensor (-30.0)
- `secondary_angle`: scalar tensor (25.0)

---

### 2. DataLoader Batching

**Input**: 16 individual samples
**Output**: Batched tensors
- `images`: **(16, 3, 1008, 1008)** - batch of 16 angiography frames
- `masks`: **(16, 1008, 1008)** - batch of 16 vessel masks
- `primary_angles`: **(16,)** - 16 LAO/RAO angles
- `secondary_angles`: **(16,)** - 16 Cranial/Caudal angles

---

### 3. Model Forward Pass

#### 3a. View Angle Encoding

**Input**:
- `primary_angles`: (16,) e.g., [-30, 45, 0, -15, ...]
- `secondary_angles`: (16,) e.g., [25, 10, -20, 30, ...]

**Processing**:
```python
# Combine into (16, 2)
angles = torch.stack([primary_angles, secondary_angles], dim=1)

# Sinusoidal encoding
angles_rad = angles * (π / 180)  # Convert to radians
# Apply learned frequencies
encoded = sin/cos encoding with learnable frequencies

# MLP processing
view_embedding = MLP(encoded)  # (16, 256)
```

**Output**: `view_embedding` **(16, 256)** - learned representation of view angles

---

#### 3b. SAM 3 Feature Extraction (Frozen)

**Input**: `images` (16, 3, 1008, 1008) in [0, 1]

**Processing**:
```python
# 1. Normalize for SAM 3
images_normalized = (images - 0.5) / 0.5  # Range: [-1, 1]

# 2. Extract features (FROZEN, no gradients)
with torch.no_grad():
    features = sam3.backbone.forward_image(images_normalized)
    # SAM 3 backbone: ViT-based transformer
    # Patch size: 16x16
    # Tokens: (1008/16)^2 = 3969 tokens
    # Feature dim: varies (e.g., 1024)
```

**Output**: `features`
- Raw: **(16, 3969, 1024)** - (batch, tokens, channels)
- Reshaped: **(16, 1024, 63, 63)** - (batch, channels, H, W)

---

#### 3c. Feature Projection (Trainable)

**Input**: `features` (16, 1024, 63, 63)

**Processing**:
```python
# Project from 1024 → 256 channels
features = Conv2d(1024, 256)(features)
```

**Output**: `features` **(16, 256, 63, 63)**

---

#### 3d. Feature Fusion (Trainable - FiLM)

**Input**:
- `features`: (16, 256, 63, 63) - SAM features
- `view_embedding`: (16, 256) - view angle encoding

**Processing** (FiLM - Feature-wise Linear Modulation):
```python
# Predict scale and shift from view embedding
gamma = Linear(view_embedding)  # (16, 256) → (16, 256)
beta = Linear(view_embedding)   # (16, 256) → (16, 256)

# Expand for spatial dimensions
gamma = gamma[:, :, None, None]  # (16, 256, 1, 1)
beta = beta[:, :, None, None]    # (16, 256, 1, 1)

# Modulate features: scale + shift
fused = gamma * features + beta  # (16, 256, 63, 63)
```

**Purpose**: View angles **condition** the SAM features
- Example: If view is RAO 30°, gamma might amplify channels sensitive to right coronary
- If view is Cranial, beta might shift features to expect vessels higher in frame

**Output**: `fused_features` **(16, 256, 63, 63)** - view-conditioned features

---

#### 3e. Segmentation Head (Trainable)

**Input**: `fused_features` (16, 256, 63, 63)

**Processing**:
```python
# Conv layers to decode features to mask
x = Conv2d(256, 128, 3x3)(fused_features)  # (16, 128, 63, 63)
x = BatchNorm2d(128)(x)
x = ReLU(x)

x = Conv2d(128, 64, 3x3)(x)  # (16, 64, 63, 63)
x = BatchNorm2d(64)(x)
x = ReLU(x)

mask_logits = Conv2d(64, 1, 1x1)(x)  # (16, 1, 63, 63)

# Upsample to original resolution
mask_logits = interpolate(mask_logits, 1008x1008)  # (16, 1, 1008, 1008)
mask_logits = squeeze(mask_logits)  # (16, 1008, 1008)
```

**Output**: `mask_logits` **(16, 1008, 1008)** - raw logits (not yet probabilities)

---

### 4. Loss Computation

**Input**:
- `mask_logits`: (16, 1008, 1008) - model predictions (logits)
- `masks`: (16, 1008, 1008) - ground truth masks [0, 1]

**Processing**:
```python
# 1. BCE Loss (handles class imbalance)
bce = binary_cross_entropy_with_logits(mask_logits, masks)
# Compares each pixel: how far is prediction from ground truth?

# 2. Dice Loss (optimizes overlap)
pred_probs = sigmoid(mask_logits)  # Convert logits to [0, 1]
intersection = (pred_probs * masks).sum()
dice = 1 - (2 * intersection) / (pred_probs.sum() + masks.sum())

# 3. Combined Loss
loss = 0.5 * bce + 0.5 * dice
```

**What the model learns**:
- **From BCE**: Each pixel should be correctly classified (vessel vs background)
- **From Dice**: The overall predicted vessel region should overlap with true vessels

---

## What the Model Learns from This

### Trainable Components (0.6M params):

1. **View Encoder** (0.13M):
   - Learns: "When angle is RAO 30°, expect vessels on right side"
   - Learns: "Cranial views show different vessel curves than caudal"
   - Output: Embedding that captures view-specific expectations

2. **Feature Fusion** (0.13M):
   - Learns: "How to modulate SAM features based on view angle"
   - Example: γ amplifies channels, β shifts baseline for view-specific patterns

3. **Feature Projection** (~0.07M):
   - Learns: "Which of SAM's 1024 features are useful for vessels"
   - Projects high-dim SAM features to task-specific 256-dim

4. **Segmentation Head** (0.37M):
   - Learns: "How to decode fused features into vessel masks"
   - Learns: "What spatial patterns indicate vessels vs background"

### Frozen Component (840M params):

- **SAM 3 Backbone**: Extracts general visual features
  - NOT learning anything (frozen)
  - Provides foundation features that trainable parts adapt

---

## Example Training Iteration

**Batch 1, Sample 5**:
```
Input image: LAD vessel, RAO 30°, Cranial 25°
  - Shows left anterior descending artery
  - View from right anterior oblique
  - Angled from above (cranial)

View encoder:
  - Sees angles (30, 25)
  - Outputs embedding: [0.2, -0.4, 0.7, ..., 0.1] (256 values)
  - This embedding means: "expect LAD on left-center, coursing down-left"

SAM 3 features:
  - Extracts general patterns: edges, textures, shapes
  - Doesn't "know" it's looking at vessels
  - Features: abstract high-level representations

Feature fusion:
  - Uses view embedding to condition SAM features
  - γ amplifies channels that respond to leftward curves
  - β shifts baseline to expect vessels in upper-left quadrant

Segmentation head:
  - Decodes fused features
  - Predicts: "Vessel pixels at (200, 300) to (400, 700)"

Ground truth:
  - True vessel pixels: (205, 295) to (405, 695)

Loss:
  - BCE: Penalizes each misclassified pixel
  - Dice: Penalizes incomplete overlap
  - Gradients flow back through trainable parts
  - View encoder learns: "For RAO 30°, this embedding works better"
  - Fusion learns: "Scale these channels more for this view"
  - Seg head learns: "These feature patterns mean vessel"
```

---

## Key Limitation

**SAM 3 is frozen** - it cannot learn vessel-specific features:
- SAM was trained on natural images (photos, not medical)
- Its features are generic (edges, objects, textures)
- It has NO knowledge of vessels, angiography, or medical imaging

**The 0.6M trainable params must**:
- Take generic SAM features
- Adapt them using view information
- Decode them into vessel masks

**This is hard!** That's why LoRA would help - it would let SAM learn vessel-specific features.

---

## Why Dice is Low (6%)

**Possible reasons**:

1. **Class imbalance**: Vessels are ~5% of pixels
   - Model learns: "Always predict background" → 95% accuracy, 0% Dice
   - Fix: BCE loss helps (just added)

2. **SAM features not useful**: Generic features don't capture vessels
   - SAM sees "curvy line" not "coronary artery"
   - Fix: Need LoRA to adapt SAM, or different backbone

3. **Too few trainable params**: 0.6M params to bridge huge gap
   - SAM outputs 1024-dim abstract features
   - Need to decode to vessel-specific 1008×1008 masks
   - Fix: Larger segmentation head, or LoRA

4. **View conditioning not working**: Maybe view angles aren't informative
   - Or FiLM modulation isn't the right fusion method
   - Fix: Try cross-attention fusion, or ignore views entirely

---

## Next Steps

**If Dice stays low (<20%) after BCE fix**:
1. SAM 3 frozen features genuinely don't work for vessels
2. Need to either:
   - Add LoRA to adapt SAM (blocked by DataParallel bug)
   - Use different backbone (ResNet, MedSAM, etc.)
   - Train on DeepSA pseudo-labels with simpler model
