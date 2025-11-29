"""
Patch for SAM3 COCO Writer.

This file provides a patched version of `prepare_for_coco_segmentation`
that correctly calculates the absolute area of segmentation masks.
The original implementation in `sam3` incorrectly normalized the area
by the image size, causing all masks to be treated as "small" (area < 1)
and resulting in 0.0 AP during COCO evaluation.

This patch is applied dynamically in `train_sam3_clean.py`.
"""

import torch
import pycocotools.mask as mask_utils
from sam3.train.masks_ops import rle_encode
from sam3.eval.coco_eval_offline import convert_to_xywh

@torch.no_grad()
def prepare_for_coco_segmentation_patched(self, predictions):
    """
    Convert predictions to COCO segmentation format.
    PATCHED: Removes area normalization (division by h*w).
    """
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        boxes = None
        if "boxes" in prediction:
            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            assert len(boxes) == len(scores)

        if "masks_rle" in prediction:
            rles = prediction["masks_rle"]
            areas = []
            for rle in rles:
                cur_area = mask_utils.area(rle)
                # PATCH: Use absolute area and force float for JSON serialization
                areas.append(float(cur_area))
        else:
            masks = prediction["masks"]
            masks = masks > 0.5
            
            # PATCH: Use absolute area
            areas = masks.flatten(1).sum(1)
            areas = areas.tolist()

            rles = rle_encode(masks.squeeze(1))

            # Memory cleanup
            del masks
            del prediction["masks"]

        assert len(areas) == len(rles) == len(scores)

        for k, rle in enumerate(rles):
            payload = {
                "image_id": original_id,
                "category_id": int(labels[k]),
                "segmentation": rle,
                "score": float(scores[k]),
                "area": float(areas[k]),
            }
            if boxes is not None:
                payload["bbox"] = [float(x) for x in boxes[k]]

            coco_results.append(payload)

    return coco_results
