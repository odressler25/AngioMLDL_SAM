# Future Experiments & Alternative Approaches

## MedSAM2 (To Test Later)

**Repository**: https://github.com/bowang-lab/MedSAM2
**Paper**: `Doc/2504.03600v1.pdf`

### Why MedSAM2 is Interesting

MedSAM2 is specifically designed for medical imaging and may offer advantages over general SAM 3:

1. **Medical-specific training**: Pre-trained on diverse medical imaging datasets
2. **Domain adaptation**: Better understanding of medical image characteristics
3. **Potential benefits for angiography**: May handle low-contrast vessels better

### When to Test MedSAM2

Test MedSAM2 **after** we establish SAM 3 baseline:

1. **First**: Complete SAM 3 testing and LoRA training
2. **Establish baseline**: Get SAM 3 performance metrics (IoU, CASS accuracy)
3. **Then compare**: Test MedSAM2 with same protocol
4. **Decision**: Choose best model for production

### Comparison Protocol

When testing MedSAM2, use identical setup for fair comparison:

```python
# Same test cases
test_cases = [
    "101-0025_MID_RCA_PRE",   # RCA
    "101-0086_MID_LAD_PRE",   # LAD
    "101-0052_DIST_LCX_PRE",  # LCX
]

# Same prompting strategy
prompts = {
    'bbox': get_bbox_from_mask(mask),
    'text': f"CASS segment {cass_id}",
    'view': f"{view_angles}"
}

# Measure same metrics
metrics = {
    'segmentation_iou': ...,
    'cass_classification_acc': ...,
    'stenosis_mae': ...,
    'mld_mae': ...,
    'inference_time': ...
}
```

### Expected Comparison

| Aspect | SAM 3 | MedSAM2 | Notes |
|--------|-------|---------|-------|
| **Pre-training** | Natural images | Medical images | MedSAM2 advantage |
| **Model size** | Larger | Similar/Smaller | TBD |
| **Inference speed** | Fast | TBD | To measure |
| **Out-of-box performance** | Unknown | Potentially better | Test first |
| **LoRA compatibility** | Yes | TBD | Check compatibility |
| **Medical domain knowledge** | No | Yes | MedSAM2 advantage |

## Other Future Experiments

### 1. Temporal Context

Currently testing single frames. Future: Use temporal information from cine sequences.

```python
# Option A: Optical flow between frames
flow = compute_optical_flow(frame_t, frame_t+1)
features = combine(spatial_features, flow_features)

# Option B: 3D convolutions on frame sequences
frames_sequence = cine[frame_num-2:frame_num+3]  # 5 frames
features_3d = conv3d(frames_sequence)

# Option C: Transformer on frame embeddings
frame_embeddings = [encode(frame) for frame in frames]
temporal_features = temporal_transformer(frame_embeddings)
```

**Hypothesis**: Temporal info helps distinguish vessels from background by tracking contrast flow.

### 2. Self-Supervised Pre-training

Following DeepSA approach (Nature 2024):

```python
# Stage 1: Self-supervised on ALL 800+ cines (no labels needed)
# Learn contrast flow dynamics, vessel appearance patterns

# Stage 2: Fine-tune on labeled subset
# Transfer learned features to CASS classification
```

**Advantage**: Use full dataset, not just labeled portion.

### 3. Multi-View Fusion

Combine predictions from different viewing angles:

```python
# Get predictions from multiple views
pred_RAO = model(image_RAO, view="RAO 30")
pred_LAO = model(image_LAO, view="LAO 45")

# Fuse predictions
final_pred = weighted_fusion([pred_RAO, pred_LAO])
```

**Hypothesis**: Multiple views reduce ambiguity, especially for LCX.

### 4. Uncertainty Quantification

Add Monte Carlo dropout or ensembles for confidence estimates:

```python
predictions = []
for _ in range(10):
    pred = model_with_dropout(image)
    predictions.append(pred)

mean_pred = np.mean(predictions, axis=0)
uncertainty = np.std(predictions, axis=0)

# Flag high-uncertainty regions for expert review
if uncertainty.max() > threshold:
    flag_for_review = True
```

**Clinical value**: Know when model is uncertain → human review needed.

### 5. Active Learning

Prioritize which cases to annotate next:

```python
# Train on initial 100 cases
model = train(initial_cases)

# Find most informative unannotated cases
unlabeled_pool = all_cases - annotated_cases
scores = []
for case in unlabeled_pool:
    uncertainty = model.predict_uncertainty(case)
    diversity = compute_diversity(case, annotated_cases)
    scores.append(uncertainty * diversity)

# Annotate top 20 most informative
next_to_annotate = sorted(unlabeled_pool, key=lambda c: scores[c])[:20]
```

**Benefit**: Reach high performance with fewer annotations.

### 6. Explainability

Add attention visualization and feature attribution:

```python
# Grad-CAM for understanding model focus
attention_map = grad_cam(model, image)

# SHAP values for feature importance
shap_values = shap.explain(model, image)

# Overlay on original image
visualize_explanation(image, attention_map, shap_values)
```

**Clinical requirement**: Physicians need to understand WHY model made prediction.

### 7. Weakly-Supervised Learning

Train with image-level labels instead of pixel-level masks:

```python
# Only need: "This image has RCA lesion" (no mask)
# Model learns to localize lesion automatically

# Class Activation Maps (CAM)
lesion_location = cam(model, image, label="RCA_lesion")
```

**Advantage**: Cheaper annotation (image labels vs. pixel masks).

## Testing Priority

### Phase 1 (Current): SAM 3 Baseline
- [x] Test on correct contrast frames
- [ ] Test combined bbox + text prompts
- [ ] Establish baseline metrics
- [ ] Train LoRA on 800 cases
- [ ] Evaluate on test set

### Phase 2: MedSAM2 Comparison
- [ ] Install MedSAM2
- [ ] Test on same cases as SAM 3
- [ ] Compare performance metrics
- [ ] Decide which to use for production
- [ ] If MedSAM2 wins, train LoRA on MedSAM2

### Phase 3: Enhancements
- [ ] Add temporal context (if needed)
- [ ] Multi-view fusion (if available)
- [ ] Uncertainty quantification
- [ ] Explainability features

### Phase 4: Production
- [ ] Optimize inference speed
- [ ] Deploy model
- [ ] Clinical validation study
- [ ] Iterate based on feedback

## Decision Points

### When to switch from SAM 3 to MedSAM2?

Switch if MedSAM2 shows:
- **>5% better IoU** on segmentation
- **>10% better accuracy** on CASS classification
- **Comparable or better inference speed**
- **Equal or better LoRA compatibility**

### When to add temporal context?

Add temporal if:
- **Single-frame IoU < 0.7** (need more information)
- **Significant performance variance** between frames
- **Contrast timing is critical** for your use case

### When to try self-supervised pre-training?

Try if:
- **Labeled data is limited** (< 500 cases)
- **Performance plateaus** with supervised learning
- **You have large unlabeled dataset** (which you do!)

## Resources

### Papers to Read (Priority Order)

1. **MedSAM2** (2025) - `Doc/2504.03600v1.pdf` ⭐ Next!
2. **SAM-VMNet** (2024) - `Doc/2406.00492v1.pdf` ✅ Read
3. **DeepSA** (2024) - Self-supervised pretraining
4. **TVS-Net** (2025) - Temporal vessel segmentation

### Code Repositories

1. **MedSAM2**: https://github.com/bowang-lab/MedSAM2
2. **SAM 3**: (official repo)
3. **DeepSA**: https://github.com/newfyu/DeepSA
4. **SAM-VMNet**: https://github.com/baixianghuang/SAM-VMNet

## Notes

- Keep all experiments reproducible (fixed random seeds)
- Document all hyperparameters
- Save all model checkpoints
- Track experiments in wandb or tensorboard
- Always compare against SAM 3 baseline

---

**Current Focus**: SAM 3 baseline establishment
**Next Step**: MedSAM2 comparison
**Future**: Enhancements based on clinical needs
