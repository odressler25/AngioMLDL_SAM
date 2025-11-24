"""
Test SAM 3 with COMBINED bbox + text prompts to teach vessel recognition.

Strategy:
1. Give SAM 3 the bounding box AND vessel name together
2. Test if this improves segmentation quality
3. Build training examples: "This box contains RCA", "This box contains LAD", etc.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple

# Import SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh


def get_vessel_bbox_xywh(mask: np.ndarray, padding: int = 20):
    """Get bounding box in (x, y, w, h) format."""
    vessel_pixels = np.argwhere(mask > 0)
    if len(vessel_pixels) == 0:
        return None

    y_coords = vessel_pixels[:, 0]
    x_coords = vessel_pixels[:, 1]

    y_min = max(0, y_coords.min() - padding)
    y_max = min(mask.shape[0], y_coords.max() + padding)
    x_min = max(0, x_coords.min() - padding)
    x_max = min(mask.shape[1], x_coords.max() + padding)

    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def normalize_bbox(bbox_cxcywh, img_width, img_height):
    """Normalize bbox to [0, 1] range."""
    return [
        bbox_cxcywh[0] / img_width,
        bbox_cxcywh[1] / img_height,
        bbox_cxcywh[2] / img_width,
        bbox_cxcywh[3] / img_height
    ]


def get_vessel_name_variants(segment_name: str) -> List[str]:
    """
    Generate different text descriptions for a vessel segment.

    Args:
        segment_name: e.g., "RCA Mid PRE"

    Returns:
        List of text descriptions to try
    """
    # Parse vessel info
    vessel_type = None
    location = None

    if "RCA" in segment_name:
        vessel_type = "RCA"
        full_name = "right coronary artery"
    elif "LAD" in segment_name:
        vessel_type = "LAD"
        full_name = "left anterior descending artery"
    elif "LCX" in segment_name or "LCx" in segment_name:
        vessel_type = "LCX"
        full_name = "left circumflex artery"
    elif "OM" in segment_name:
        vessel_type = "OM"
        full_name = "obtuse marginal branch"
    elif "DIAG" in segment_name:
        vessel_type = "DIAG"
        full_name = "diagonal branch"
    else:
        vessel_type = segment_name.split()[0]
        full_name = "coronary vessel"

    # Get location
    if "PROX" in segment_name:
        location = "proximal"
    elif "MID" in segment_name:
        location = "mid"
    elif "DIST" in segment_name:
        location = "distal"

    # Generate variants
    variants = []

    # Basic vessel name
    variants.append(vessel_type)
    variants.append(full_name)

    # With location
    if location:
        variants.append(f"{location} {vessel_type}")
        variants.append(f"{location} {full_name}")

    # Clinical descriptions
    variants.append(f"coronary artery segment {vessel_type}")
    variants.append("contrast-filled coronary vessel")
    variants.append(f"vessel segment labeled {vessel_type}")

    # Training-style prompts (what we want SAM to learn)
    variants.append(f"this is {vessel_type}")
    variants.append(f"the {vessel_type} vessel")

    return variants


def test_combined_prompting(processor, case_id: str, base_path: Path) -> Dict:
    """
    Test different prompting strategies on one case.
    """
    print(f"\n{'='*60}")
    print(f"Testing {case_id}")
    print(f"{'='*60}")

    # Load metadata
    json_path = base_path / "contours" / f"{case_id}_contours.json"
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    frame_num = metadata.get('frame_num', 0)
    segment_name = metadata.get('segment_name', 'Unknown')

    print(f"Vessel: {segment_name}, Frame: {frame_num}")

    # Load correct frame
    cine_path = base_path / "cines" / f"{case_id}_cine.npy"
    cine = np.load(cine_path)
    image = cine[min(frame_num, len(cine)-1)]

    # Convert to PIL
    image_uint8 = (image * 255).astype(np.uint8)
    image_rgb = np.stack([image_uint8] * 3, axis=-1)
    pil_image = Image.fromarray(image_rgb)
    width, height = pil_image.size

    # Load ground truth
    mask_path = base_path / "vessel_masks" / f"{case_id}_mask.npy"
    gt_mask = np.load(mask_path)

    # Get bbox
    bbox_xywh = get_vessel_bbox_xywh(gt_mask, padding=20)
    if bbox_xywh is None:
        print("  No vessel pixels!")
        return {}

    # Convert and normalize bbox
    bbox_xywh_tensor = torch.tensor(bbox_xywh).view(-1, 4)
    bbox_cxcywh = box_xywh_to_cxcywh(bbox_xywh_tensor).flatten().tolist()
    norm_box = normalize_bbox(bbox_cxcywh, width, height)

    # Get text variants
    text_variants = get_vessel_name_variants(segment_name)

    results = {}

    # Test 1: Bbox only (baseline)
    print("\n1. Bbox only:")
    inference_state = processor.set_image(pil_image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.add_geometric_prompt(
        state=inference_state,
        box=norm_box,
        label=True
    )

    masks = inference_state.get("masks", [])
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    if len(masks) > 0:
        pred_mask = masks[0]
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask.astype(np.float32),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0).sum()
        union = np.logical_or(pred_mask > 0.5, gt_mask > 0).sum()
        iou = float(intersection / union) if union > 0 else 0.0

        results['bbox_only'] = {
            'iou': iou,
            'pixels': (pred_mask > 0.5).sum()
        }
        print(f"  IoU: {iou:.3f}")

    # Test 2: Combined bbox + text for each variant
    print("\n2. Combined bbox + text:")
    best_combined_iou = 0
    best_text = ""

    for text in text_variants[:3]:  # Test top 3 variants
        # Reset and add bbox
        processor.reset_all_prompts(inference_state)
        inference_state = processor.add_geometric_prompt(
            state=inference_state,
            box=norm_box,
            label=True
        )

        # Also add text prompt
        try:
            # Try adding text on top of bbox
            inference_state = processor.set_text_prompt(
                state=inference_state,
                prompt=text
            )
        except:
            # If that doesn't work, just use the bbox result
            pass

        masks = inference_state.get("masks", [])
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        if len(masks) > 0:
            pred_mask = masks[0]
            if len(pred_mask.shape) == 3:
                pred_mask = pred_mask[0]

            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.float32),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0).sum()
            union = np.logical_or(pred_mask > 0.5, gt_mask > 0).sum()
            iou = float(intersection / union) if union > 0 else 0.0

            print(f"  '{text}': IoU = {iou:.3f}")

            if iou > best_combined_iou:
                best_combined_iou = iou
                best_text = text

    results['best_combined'] = {
        'iou': best_combined_iou,
        'text': best_text
    }

    # Test 3: Text only with vessel name (for comparison)
    print("\n3. Text only (vessel name):")
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(
        state=inference_state,
        prompt=f"{segment_name.split()[0]} coronary artery"
    )

    masks = inference_state.get("masks", [])
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    text_only_iou = 0
    if len(masks) > 0:
        pred_mask = masks[0]
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask.astype(np.float32),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0).sum()
        union = np.logical_or(pred_mask > 0.5, gt_mask > 0).sum()
        text_only_iou = float(intersection / union) if union > 0 else 0.0

    results['text_only'] = {'iou': text_only_iou}
    print(f"  IoU: {text_only_iou:.3f}")

    return results


def create_training_strategy(results: Dict[str, Dict]) -> None:
    """
    Propose a training strategy based on results.
    """
    print("\n" + "="*60)
    print("PROPOSED TRAINING STRATEGY")
    print("="*60)

    # Analyze results
    bbox_ious = [r.get('bbox_only', {}).get('iou', 0) for r in results.values()]
    combined_ious = [r.get('best_combined', {}).get('iou', 0) for r in results.values()]
    text_ious = [r.get('text_only', {}).get('iou', 0) for r in results.values()]

    avg_bbox = np.mean(bbox_ious)
    avg_combined = np.mean(combined_ious)
    avg_text = np.mean(text_ious)

    print(f"\nAverage IoUs:")
    print(f"  Bbox only:     {avg_bbox:.3f}")
    print(f"  Combined:      {avg_combined:.3f}")
    print(f"  Text only:     {avg_text:.3f}")

    print("\n📊 ANALYSIS:")
    if avg_bbox > 0.5:
        print("✅ Bbox prompts work well - SAM 3 can see vessels when told where to look")
    elif avg_bbox > 0.2:
        print("⚠️ Bbox prompts partially work - SAM 3 sees something but not precise")
    else:
        print("❌ Bbox prompts fail - SAM 3 doesn't understand vessel appearance")

    if avg_combined > avg_bbox:
        print("✅ Combined prompting improves results - text helps refine segmentation")
    else:
        print("⚠️ Text doesn't add value - bbox alone is sufficient")

    print("\n🎯 RECOMMENDED APPROACH:")

    if avg_bbox > 0.3:
        print("""
1. **Few-shot fine-tuning with bbox supervision**
   - Use your 800+ expert cases as training data
   - Input: Image + bbox + vessel name
   - Target: Expert mask
   - This teaches SAM 3: "RCA looks like THIS in THIS box"

2. **Training data structure:**
   ```python
   {
     'image': cine[frame_num],
     'bbox': normalized_bbox,
     'text': 'RCA coronary artery',
     'mask': expert_mask
   }
   ```

3. **Progressive training:**
   - Stage 1: Train with bbox only (since it works)
   - Stage 2: Add vessel names to learn associations
   - Stage 3: Gradually reduce bbox size to force learning

4. **Inference strategy:**
   - Use a simple CNN to predict approximate bbox
   - Feed bbox + vessel type to SAM 3
   - Get precise segmentation
        """)
    else:
        print("""
SAM 3 struggles even with bbox - need different approach:
1. Consider domain-specific fine-tuning from scratch
2. Use U-Net with better training strategy
3. Investigate why SAM 3 can't see vessels (contrast? resolution?)
        """)

    print("\n💾 NEXT STEPS:")
    print("1. Prepare training dataset with all 800+ cases")
    print("2. Extract frame_num, bbox, vessel name for each")
    print("3. Fine-tune SAM 3 with bbox + text supervision")
    print("4. Test if it learns vessel-specific features")


def main():
    """
    Test combined prompting strategies.
    """
    base_path = Path(r"E:\AngioMLDL_data\corrected_vessel_dataset")

    test_cases = [
        "101-0025_MID_RCA_PRE",   # RCA
        "101-0086_MID_LAD_PRE",   # LAD
        "101-0052_DIST_LCX_PRE",  # LCX
    ]

    # Load SAM 3
    print("Loading SAM 3...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    print("SAM 3 loaded!")

    # Test each case
    all_results = {}

    for case_id in test_cases:
        try:
            results = test_combined_prompting(processor, case_id, base_path)
            all_results[case_id] = results
        except Exception as e:
            print(f"Error on {case_id}: {e}")
            import traceback
            traceback.print_exc()

    # Propose training strategy
    create_training_strategy(all_results)


if __name__ == '__main__':
    main()