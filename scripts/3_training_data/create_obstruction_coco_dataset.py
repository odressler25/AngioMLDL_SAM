"""
Create COCO Dataset for Obstruction Detection Training

Converts the Medis QCA obstruction labels to COCO format for SAM 3 training.

Category: "obstruction" (id=1)
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from datetime import datetime
from tqdm import tqdm
import random


def mask_to_polygons(mask, min_area=10):
    """Convert binary mask to COCO polygon format."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        # Skip small contours
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 3:  # Need at least 3 points for polygon
            # Flatten to [x1, y1, x2, y2, ...]
            polygon = approx.flatten().tolist()
            if len(polygon) >= 6:  # At least 3 points (6 coordinates)
                polygons.append(polygon)

    return polygons


def mask_to_bbox(mask):
    """Get bounding box from mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # COCO format: [x, y, width, height]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def create_coco_dataset(obstruction_dir, images_dir, output_dir, train_ratio=0.8):
    """
    Create COCO format dataset for obstruction detection.

    Args:
        obstruction_dir: Directory with obstruction masks (*_obstruction.png)
        images_dir: Directory with original images
        output_dir: Output directory for COCO dataset
        train_ratio: Ratio of training samples (default 0.8)
    """
    obstruction_dir = Path(obstruction_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    # Load summary to get metadata
    summary_path = obstruction_dir / "obstruction_labels_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    print(f"Found {len(summary)} obstruction labels")

    # Shuffle and split
    random.seed(42)
    random.shuffle(summary)
    split_idx = int(len(summary) * train_ratio)
    train_samples = summary[:split_idx]
    val_samples = summary[split_idx:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # COCO structure
    def create_coco_structure():
        return {
            "info": {
                "description": "Coronary Obstruction Detection Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "Medis QCA Labels",
                "date_created": datetime.now().strftime("%Y-%m-%d")
            },
            "licenses": [],
            "categories": [
                {
                    "id": 1,
                    "name": "obstruction",
                    "supercategory": "lesion"
                }
            ],
            "images": [],
            "annotations": []
        }

    train_coco = create_coco_structure()
    val_coco = create_coco_structure()

    def process_samples(samples, coco_data, output_subdir, split_name):
        image_id = 0
        annotation_id = 0

        for sample in tqdm(samples, desc=f"Processing {split_name}"):
            image_name = sample['image']
            mask_path = Path(sample['mask_path'])

            # Find corresponding image
            image_path = images_dir / f"{image_name}.png"
            if not image_path.exists():
                print(f"  Image not found: {image_path.name}")
                continue

            if not mask_path.exists():
                print(f"  Mask not found: {mask_path.name}")
                continue

            # Load image and mask
            image = Image.open(image_path)
            mask = np.array(Image.open(mask_path)) > 127

            # Get image dimensions
            width, height = image.size

            # Copy image to output directory
            output_image_path = output_subdir / f"{image_name}.png"
            image.save(output_image_path)

            # Add image to COCO
            coco_data["images"].append({
                "id": image_id,
                "file_name": f"{image_name}.png",
                "width": width,
                "height": height,
                "stenosis_pct": sample.get('stenosis_pct', 0),
                "mld_mm": sample.get('mld_mm', 0),
                "lesion_length_mm": sample.get('lesion_length_mm', 0)
            })

            # Convert mask to polygons
            polygons = mask_to_polygons(mask)
            bbox = mask_to_bbox(mask)

            if polygons and bbox:
                area = int(mask.sum())

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # obstruction
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "stenosis_pct": sample.get('stenosis_pct', 0),
                    "mld_mm": sample.get('mld_mm', 0)
                })
                annotation_id += 1

            image_id += 1

        return image_id, annotation_id

    # Process train and val
    train_images, train_anns = process_samples(train_samples, train_coco, train_dir, "train")
    val_images, val_anns = process_samples(val_samples, val_coco, val_dir, "val")

    # Save COCO annotations
    train_json_path = output_dir / "train_obstruction.json"
    val_json_path = output_dir / "val_obstruction.json"

    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)

    with open(val_json_path, 'w') as f:
        json.dump(val_coco, f, indent=2)

    print(f"\nDataset created:")
    print(f"  Train: {train_images} images, {train_anns} annotations")
    print(f"  Val: {val_images} images, {val_anns} annotations")
    print(f"  Train JSON: {train_json_path}")
    print(f"  Val JSON: {val_json_path}")

    # Create training config
    config = {
        "dataset_name": "obstruction",
        "train_json": str(train_json_path),
        "val_json": str(val_json_path),
        "train_images": str(train_dir),
        "val_images": str(val_dir),
        "num_classes": 1,
        "class_names": ["obstruction"],
        "num_train": train_images,
        "num_val": val_images,
        "image_size": 512,
        "concept_prompt": "obstruction"
    }

    config_path = output_dir / "dataset_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Config: {config_path}")

    return config


def main():
    print("=" * 60)
    print("Creating COCO Dataset for Obstruction Detection")
    print("=" * 60)

    obstruction_dir = Path(r"E:\AngioMLDL_data\batch2\obstruction_labels")
    images_dir = Path(r"E:\AngioMLDL_data\batch2\images")
    output_dir = Path(r"E:\AngioMLDL_data\batch2\coco_obstruction")

    config = create_coco_dataset(obstruction_dir, images_dir, output_dir)

    print("\n" + "=" * 60)
    print("COCO dataset creation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
