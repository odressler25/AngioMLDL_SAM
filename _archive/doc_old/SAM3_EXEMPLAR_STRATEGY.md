# SAM 3 Exemplar-Based Strategy for Coronary Vessel Segmentation

## TL;DR - Why This Could Work

SAM 3's new **exemplar-based learning** is PERFECT for your use case:
- ❌ Text prompts failed (SAM 3 doesn't know "coronary artery")
- ✅ But SAM 3 can learn from **visual examples** (your expert masks!)
- ✅ Can segment **ALL instances** (separate RCA, LAD, LCX, not one blob)
- ✅ No need to train U-Net from scratch

---

## SAM 3's Key New Features (From Model Card)

### 1. Exemplar-Based Prompting
Instead of text-only, you can **show SAM 3 examples**:
```python
# Show what coronary vessels LOOK like
exemplar_masks = [rca_mask, lad_mask, lcx_mask]
sam3.segment_by_exemplar(new_image, exemplars=exemplar_masks)
```

### 2. Exhaustive Instance Segmentation
Segment **ALL instances** of a concept:
```python
# Not just one vessel → ALL vessels separately
output = sam3.segment_all_instances(
    image=angiogram,
    concept="coronary vessel",
    exemplars=[example_masks]
)
# Returns: {RCA: mask1, LAD: mask2, diagonal: mask3, ...}
```

### 3. Massive Concept Dictionary
- Trained on 4 million concepts
- 270K unique concepts in benchmark
- But... "coronary artery" might not be in there (medical domain)

---

## Strategy: Few-Shot Learning with Your Expert Data

Instead of zero-shot (text only), use **few-shot** (text + examples):

### Phase 1: Test Few-Shot Prompting (1 day)

```python
# 1. Pick 5-10 diverse expert-labeled cases as exemplars
exemplar_cases = [
    "101-0025_MID_RCA_PRE",   # RCA example
    "101-0086_MID_LAD_PRE",   # LAD example
    "101-0052_DIST_LCX_PRE",  # LCX example
    # Add 2-3 more for diversity
]

# 2. Load their masks
exemplar_masks = []
exemplar_images = []
for case in exemplar_cases:
    mask = np.load(f"{case}_mask.npy")
    image = np.load(f"{case}_cine.npy")[frame_idx]
    exemplar_masks.append(mask)
    exemplar_images.append(image)

# 3. Test on NEW cases
test_image = load_new_angiogram()

# Method A: Geometric prompts (boxes around vessels)
for mask in exemplar_masks:
    # Get bounding box of vessel from exemplar mask
    box = get_bbox_from_mask(mask)
    state = processor.add_geometric_prompt(box, label="vessel", state)

# Method B: Direct mask exemplars (if API supports)
output = processor.segment_with_exemplars(
    image=test_image,
    exemplar_images=exemplar_images,
    exemplar_masks=exemplar_masks,
    text_prompt="coronary vessel"  # Now has visual context!
)
```

**Test on 20-30 held-out cases:**
- Calculate IoU/Dice vs expert ground truth
- See if exemplars help SAM 3 "understand" coronary vessels

---

### Phase 2: Fine-Tune SAM 3 (If Few-Shot Works)

If Phase 1 shows promise (IoU > 0.3), fine-tune SAM 3:

```python
# Fine-tuning loop
for epoch in range(num_epochs):
    for case in training_cases:
        image = load_image(case)
        gt_mask = load_mask(case)
        centerline = load_centerline(case)

        # Sample point prompts along centerline
        points = sample_points_from_centerline(centerline, n=5)

        # Forward pass
        pred_masks = sam3.predict(image, points=points)

        # Loss: Compare predicted masks to ground truth
        loss = dice_loss(pred_masks, gt_mask)
        loss.backward()
        optimizer.step()
```

**Training split:**
- Train: 640 cases (80%)
- Val: 80 cases (10%)
- Test: 80 cases (10%)

**Expected improvement:**
- Zero-shot (text only): IoU = 0.00 ❌
- Few-shot (text + exemplars): IoU = 0.3-0.5 ?
- Fine-tuned: IoU = 0.6-0.8 ✅

---

## Alternative: SAM 3 + Your Expert Data Pipeline

Combine SAM 3's strengths with your rich annotations:

### Step 1: Automatic Centerline → Point Prompts
```python
# Use your expert centerlines to train a lightweight point predictor
class CenterlinePredictor(nn.Module):
    # Simple CNN: angiogram → centerline heatmap
    # Train on 800 expert cases

# At inference:
pred_centerline = centerline_predictor(new_image)
points = sample_along_predicted_centerline(pred_centerline)
```

### Step 2: Points → SAM 3 Segmentation
```python
# Feed predicted points to fine-tuned SAM 3
masks = sam3.predict(new_image, points=points)
```

### Step 3: Geometric QCA from Masks
```python
# Extract measurements (your existing pipeline)
centerline = skeletonize(masks)
diameters = measure_diameters(masks, centerline)
mld = np.min(diameters)
# ... etc
```

---

## Checking SAM 3's "Dictionary"

### What Concepts Does SAM 3 Know?

SAM 3 doesn't have a public "dictionary" API, but you can test concepts:

```python
# Test if SAM 3 recognizes medical concepts
test_prompts = [
    # Anatomy
    "artery", "vein", "blood vessel", "capillary",

    # Medical imaging
    "x-ray", "angiogram", "contrast agent",

    # Cardiac
    "heart", "coronary", "cardiac vessel",

    # Pathology
    "stenosis", "blockage", "plaque", "lesion",
]

for prompt in test_prompts:
    output = sam3.predict(test_image, text_prompt=prompt)
    score = output['scores'].max() if len(output['scores']) > 0 else 0
    print(f"{prompt}: confidence={score:.3f}, detected={len(output['masks'])}")
```

**Expected results:**
- General terms ("artery", "blood vessel"): Might work partially
- Medical terms ("coronary", "stenosis"): Probably low confidence
- Specific anatomy ("RCA", "LAD"): Almost certainly unknown

**Why:** SAM 3 was trained on web images (photos, illustrations), not medical data.

---

## SAM 3 API Methods (What's Available)

From `Sam3Processor`:

```python
processor = Sam3Processor(model)

# 1. Set image for processing
state = processor.set_image(pil_image)

# 2. Text prompts (what we tried - failed for coronary)
output = processor.set_text_prompt(state, prompt="coronary artery")

# 3. Geometric prompts (boxes, points - THIS IS KEY!)
state = processor.add_geometric_prompt(
    bbox=[x1, y1, x2, y2],  # Bounding box
    label="vessel",
    state=state
)

# 4. Set confidence threshold
processor.set_confidence_threshold(0.5)

# 5. Reset prompts
processor.reset_all_prompts(state)

# 6. Batch processing
states = processor.set_image_batch([img1, img2, img3])
```

**Most promising for you: `add_geometric_prompt`**
- Give SAM 3 boxes/points from your expert centerlines
- Much more concrete than text
- Can be combined with text for better results

---

## Recommended Experiment (Quick Test)

### Test Script: Exemplar + Geometric Prompts

```python
# test_sam3_exemplar_prompts.py

# Load 3 exemplar cases (diverse vessels)
exemplars = [
    ("101-0025_MID_RCA_PRE", "RCA"),
    ("101-0086_MID_LAD_PRE", "LAD"),
    ("101-0052_DIST_LCX_PRE", "LCX"),
]

# For each exemplar, get vessel bounding box
exemplar_boxes = []
for case_id, vessel in exemplars:
    mask = np.load(f"vessel_masks/{case_id}_mask.npy")
    box = get_tight_bbox(mask)  # [x1, y1, x2, y2]
    exemplar_boxes.append(box)

# Test on new cases
test_cases = get_test_cases(n=20)

for test_case in test_cases:
    image = load_test_image(test_case)
    gt_mask = load_test_mask(test_case)

    # Set image
    state = processor.set_image(image)

    # Add geometric prompts based on exemplar boxes
    # (Use similar locations/sizes from exemplars)
    for box in exemplar_boxes:
        state = processor.add_geometric_prompt(
            bbox=box,
            label="coronary vessel",
            state=state
        )

    # Get predictions
    output = processor._forward_grounding(state)

    # Evaluate
    pred_mask = combine_masks(output['masks'])
    iou = calculate_iou(pred_mask, gt_mask)

    print(f"{test_case}: IoU={iou:.3f}")
```

**If this works better than zero-shot:**
→ Exemplar-based approach is promising!
→ Proceed with fine-tuning

**If still fails:**
→ SAM 3 needs fine-tuning on medical data
→ Or stick with custom U-Net (but improve training strategy)

---

## Next Steps

1. **Quick test** (today): Run few-shot with exemplars
   - Use `add_geometric_prompt` with boxes from expert masks
   - Test on 10-20 cases
   - Measure IoU improvement vs zero-shot

2. **If promising** (this week):
   - Fine-tune SAM 3 on your 800 expert cases
   - Use centerline points as prompts
   - Train for 10-20 epochs

3. **If still not working** (pivot):
   - Go back to U-Net
   - But use better training strategy:
     - Pre-train on RPCA (769 cases)
     - Add heavy augmentation
     - Multi-task learning (vessel + classification + measurements)
     - Use centerline loss (not just mask IoU)

---

## Key Insight

SAM 3 CAN'T understand "coronary artery" from text alone.
BUT it CAN learn from visual examples (exemplars + geometric prompts).

The question is: **Can it generalize from your 800 expert examples to new cases?**

Only one way to find out: **Test it!**

Want me to create the exemplar test script?
