"""
Create Unified COCO Dataset for Obstruction Detection Training

Combines obstruction labels from ALL 377 patients with their original images.
Creates train/val COCO format for SAM3 training.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from datetime import datetime
from tqdm import tqdm
import random
import shutil


def mask_to_polygons(mask, min_area=10):
    """Convert binary mask to COCO polygon format."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 3:
            polygon = approx.flatten().tolist()
            if len(polygon) >= 6:
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

    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def find_image(image_name, image_dirs):
    """Find image in multiple directories."""
    for img_dir in image_dirs:
        # Try different extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            path = img_dir / f"{image_name}{ext}"
            if path.exists():
                return path
    return None


def main():
    print("=" * 70)
    print("Creating Unified COCO Dataset for Obstruction Detection")
    print("=" * 70)

    # Paths
    obstruction_labels_dir = Path(r"E:\AngioMLDL_data\unified_obstruction_labels")
    summary_path = obstruction_labels_dir / "obstruction_labels_summary.json"

    # Image directories (multiple locations where PNGs exist)
    image_dirs = [
        Path(r"E:\AngioMLDL_data\coco_medis_bifurcation\train\images"),
        Path(r"E:\AngioMLDL_data\coco_medis_bifurcation\val\images"),
        Path(r"E:\AngioMLDL_data\batch2\images"),
    ]

    output_dir = Path(r"E:\AngioMLDL_data\unified_obstruction_coco")
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    # Load summary
    with open(summary_path) as f:
        summary = json.load(f)

    labels = summary['labels']
    print(f"Found {len(labels)} obstruction labels")

    # Check which image directories exist
    existing_dirs = [d for d in image_dirs if d.exists()]
    print(f"Image directories: {[str(d) for d in existing_dirs]}")

    # Shuffle and split (80/20)
    random.seed(42)
    random.shuffle(labels)
    split_idx = int(len(labels) * 0.8)
    train_samples = labels[:split_idx]
    val_samples = labels[split_idx:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # COCO structure
    def create_coco_structure():
        return {
            "info": {
                "description": "Unified Coronary Obstruction Detection Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "Medis QCA Gold-Standard Labels",
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
        missing_images = 0
        missing_masks = 0

        for sample in tqdm(samples, desc=f"Processing {split_name}"):
            image_name = sample['image_name']
            mask_path = Path(sample['mask_path'])

            # Find image
            image_path = find_image(image_name, existing_dirs)
            if image_path is None:
                missing_images += 1
                continue

            if not mask_path.exists():
                missing_masks += 1
                continue

            # Load image and mask
            image = Image.open(image_path)
            mask = np.array(Image.open(mask_path)) > 127

            width, height = image.size

            # Copy image to output directory
            output_image_name = f"{image_name}.png"
            output_image_path = output_subdir / output_image_name
            image.save(output_image_path)

            # Add image to COCO
            coco_data["images"].append({
                "id": image_id,
                "file_name": output_image_name,
                "width": width,
                "height": height,
                "stenosis_pct": sample.get('stenosis_pct', 0),
                "mld_mm": sample.get('mld_mm', 0),
                "source": sample.get('source', 'unknown')
            })

            # Convert mask to polygons
            polygons = mask_to_polygons(mask)
            bbox = mask_to_bbox(mask)

            if polygons and bbox:
                area = int(mask.sum())

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "stenosis_pct": sample.get('stenosis_pct', 0),
                    "mld_mm": sample.get('mld_mm', 0)
                })
                annotation_id += 1

            image_id += 1

        if missing_images > 0:
            print(f"  Warning: {missing_images} images not found")
        if missing_masks > 0:
            print(f"  Warning: {missing_masks} masks not found")

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
        "dataset": {
            "name": "unified_obstruction",
            "train_json": str(train_json_path),
            "val_json": str(val_json_path),
            "train_images": str(train_dir),
            "val_images": str(val_dir),
            "num_classes": 1,
            "class_names": ["obstruction"]
        },
        "statistics": {
            "num_train": train_images,
            "num_val": val_images,
            "total_patients": 377,
            "stenosis_threshold": 30
        },
        "concept": {
            "enabled": True,
            "prompt": "obstruction",
            "alternative_prompts": ["stenosis", "narrowing", "lesion", "blockage"]
        },
        "model": {
            "name": "sam3_multihead",
            "image_size": 512,
            "phase1_checkpoint": "E:/AngioMLDL_data/experiments/stage2_bifurcation_v7/checkpoints/checkpoint.pt",
            "freeze_backbone": True
        },
        "training": {
            "batch_size": 16,
            "num_epochs": 50,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 5,
            "min_lr": 1e-5
        },
        "output": {
            "checkpoint_dir": "E:/AngioMLDL_data/experiments/obstruction_detection/checkpoints",
            "log_dir": "E:/AngioMLDL_data/experiments/obstruction_detection/logs"
        }
    }

    config_path = output_dir / "train_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  Config: {config_path}")

    print("\n" + "=" * 70)
    print("COCO dataset creation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
