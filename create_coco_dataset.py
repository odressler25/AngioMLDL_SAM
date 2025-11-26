"""
Create COCO-format dataset for SAM3 training from coronary angiography data.

This script converts:
- Cine video frames (from .npy files, extracted using frame_index)
- DeepSA pseudo-label masks (512x512 binary)

Into COCO format with:
- Images in train/ and val/ folders
- annotations.json with RLE-encoded masks

Usage:
    python create_coco_dataset.py --output-dir E:/AngioMLDL_data/coco_format
"""

import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_util
from tqdm import tqdm


def extract_frame_from_cine(cine_path: str, frame_index: int) -> np.ndarray:
    """Extract a specific frame from a cine .npy file."""
    try:
        # Load cine video (shape: [frames, H, W, 3] or [frames, H, W])
        cine = np.load(cine_path)

        # Handle frame index
        if frame_index >= len(cine):
            frame_index = len(cine) - 1

        frame = cine[frame_index]

        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)

        # Ensure 3 channels (RGB)
        if len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 1:
            frame = np.concatenate([frame] * 3, axis=-1)

        return frame
    except Exception as e:
        print(f"Error extracting frame from {cine_path}: {e}")
        return None


def load_and_resize_mask(mask_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Load DeepSA mask (.npy or image) and resize to match image size."""
    try:
        # Handle .npy files (DeepSA format)
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path, allow_pickle=True)
        else:
            mask = np.array(Image.open(mask_path))

        # Resize if needed (target_size is (height, width))
        if mask.shape[:2] != target_size:
            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.resize((target_size[1], target_size[0]), Image.NEAREST)
            mask = np.array(mask_pil)

        # Ensure binary
        mask = (mask > 0).astype(np.uint8)

        return mask
    except Exception as e:
        print(f"Error loading mask from {mask_path}: {e}")
        return None


def mask_to_rle(binary_mask: np.ndarray) -> Dict:
    """Convert binary mask to COCO RLE format."""
    # Ensure Fortran order for pycocotools
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(mask_fortran)
    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def compute_bbox_from_mask(binary_mask: np.ndarray) -> List[float]:
    """Compute bounding box from binary mask in COCO format [x, y, width, height]."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def process_sample(args: Tuple) -> Dict:
    """Process a single sample (for multiprocessing)."""
    idx, row, output_dir, split, global_idx = args

    try:
        # Extract paths from row
        cine_path = row['cine_path']
        frame_index = int(row['frame_index'])
        mask_path = row['deepsa_pseudo_label_path']

        # Check if files exist
        if not os.path.exists(cine_path):
            return {'status': 'error', 'reason': f'Cine not found: {cine_path}'}
        if not os.path.exists(mask_path):
            return {'status': 'error', 'reason': f'Mask not found: {mask_path}'}

        # Extract frame from cine
        frame = extract_frame_from_cine(cine_path, frame_index)
        if frame is None:
            return {'status': 'error', 'reason': 'Failed to extract frame'}

        # Get image dimensions
        height, width = frame.shape[:2]

        # Load and resize mask
        mask = load_and_resize_mask(mask_path, (height, width))
        if mask is None:
            return {'status': 'error', 'reason': 'Failed to load mask'}

        # Check if mask has any content
        if mask.sum() == 0:
            return {'status': 'error', 'reason': 'Empty mask'}

        # Save image
        image_filename = f"{global_idx:06d}.png"
        image_path = os.path.join(output_dir, split, 'images', image_filename)
        Image.fromarray(frame).save(image_path)

        # Convert mask to RLE
        rle = mask_to_rle(mask)
        bbox = compute_bbox_from_mask(mask)
        area = float(mask.sum())

        return {
            'status': 'success',
            'image_id': global_idx,
            'file_name': image_filename,
            'width': width,
            'height': height,
            'rle': rle,
            'bbox': bbox,
            'area': area,
            'split': split
        }

    except Exception as e:
        import traceback
        return {'status': 'error', 'reason': f'{str(e)}\n{traceback.format_exc()}'}


def create_coco_dataset(
    csv_path: str,
    output_dir: str,
    num_workers: int = None,
    category_name: str = "coronary artery"
):
    """
    Create COCO-format dataset from CSV and DeepSA masks.

    Args:
        csv_path: Path to the corrected dataset CSV
        output_dir: Output directory for COCO dataset
        num_workers: Number of parallel workers (default: cpu_count - 2)
        category_name: Text prompt for the category
    """
    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # Filter samples with valid DeepSA paths
    df = df[df['deepsa_pseudo_label_path'].notna()].reset_index(drop=True)
    print(f"Samples with DeepSA labels: {len(df)}")

    # Use existing split column
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    # If no split column or all same, create one
    if len(train_df) == 0 or len(val_df) == 0:
        print("No train/val split found in CSV, creating 85/15 split...")
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)

    # Process samples
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    print(f"Processing with {num_workers} workers...")

    # Prepare arguments for multiprocessing with global indices
    train_args = [(i, row, output_dir, 'train', i) for i, row in train_df.iterrows()]
    val_args = [(i, row, output_dir, 'val', i + len(train_df)) for i, row in val_df.iterrows()]
    all_args = train_args + val_args

    # Process in parallel
    results = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap(process_sample, all_args), total=len(all_args)):
            results.append(result)

    # Separate results by split
    train_results = [r for r in results if r.get('split') == 'train' and r['status'] == 'success']
    val_results = [r for r in results if r.get('split') == 'val' and r['status'] == 'success']

    # Count errors
    errors = [r for r in results if r['status'] == 'error']
    print(f"\nSuccessfully processed: {len(train_results)} train, {len(val_results)} val")
    print(f"Errors: {len(errors)}")
    if errors[:3]:
        print("Sample errors:")
        for err in errors[:3]:
            print(f"  - {err['reason'][:200]}")

    # Create COCO annotations
    category = {
        "id": 1,
        "name": category_name,
        "supercategory": "vessel"
    }

    for split, split_results in [('train', train_results), ('val', val_results)]:
        images = []
        annotations = []

        for i, r in enumerate(split_results):
            # Image entry
            images.append({
                "id": r['image_id'],
                "file_name": r['file_name'],
                "width": r['width'],
                "height": r['height']
            })

            # Annotation entry
            annotations.append({
                "id": i,
                "image_id": r['image_id'],
                "category_id": 1,
                "segmentation": r['rle'],
                "bbox": r['bbox'],
                "area": r['area'],
                "iscrowd": 0
            })

        coco_json = {
            "images": images,
            "annotations": annotations,
            "categories": [category]
        }

        # Save JSON
        json_path = os.path.join(output_dir, split, 'annotations.json')
        with open(json_path, 'w') as f:
            json.dump(coco_json, f, indent=2)
        print(f"Saved {split} annotations to {json_path} ({len(images)} images, {len(annotations)} annotations)")

    print("\nDataset creation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Category: '{category_name}'")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Create COCO dataset for SAM3 training')
    parser.add_argument('--csv-path', type=str,
                        default='E:/AngioMLDL_data/corrected_dataset_training.csv',
                        help='Path to the dataset CSV')
    parser.add_argument('--output-dir', type=str,
                        default='E:/AngioMLDL_data/coco_format',
                        help='Output directory for COCO dataset')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--category-name', type=str, default='coronary artery',
                        help='Category name (text prompt)')

    args = parser.parse_args()

    create_coco_dataset(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        category_name=args.category_name
    )


if __name__ == '__main__':
    main()
