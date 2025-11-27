"""
Create COCO dataset from Medis GT masks with CASS segment categories.

This script converts the Medis expert-annotated segment masks into COCO format
with individual CASS segment categories. Unlike DeepSA (one giant blob),
Medis masks are discrete segment masks - perfect for SAM3's instance head.

Categories:
- 14 CASS segments (proximal_rca, mid_lad, etc.)
- Each image has exactly ONE segment annotated
- Clean instance segmentation task

Usage:
    python scripts/create_coco_cass.py
    python scripts/create_coco_cass.py --output-dir E:/AngioMLDL_data/coco_cass_segments
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_util
from tqdm import tqdm


# CASS Segment Categories
# Using CASS IDs as category IDs for direct mapping
CASS_CATEGORIES = [
    {"id": 1, "name": "proximal_rca", "supercategory": "rca"},
    {"id": 2, "name": "mid_rca", "supercategory": "rca"},
    {"id": 3, "name": "distal_rca", "supercategory": "rca"},
    {"id": 4, "name": "pda", "supercategory": "rca"},
    {"id": 12, "name": "proximal_lad", "supercategory": "lad"},
    {"id": 13, "name": "mid_lad", "supercategory": "lad"},
    {"id": 14, "name": "distal_lad", "supercategory": "lad"},
    {"id": 15, "name": "d1", "supercategory": "lad"},
    {"id": 16, "name": "d2", "supercategory": "lad"},
    {"id": 18, "name": "proximal_lcx", "supercategory": "lcx"},
    {"id": 19, "name": "distal_lcx", "supercategory": "lcx"},
    {"id": 20, "name": "om1", "supercategory": "lcx"},
    {"id": 21, "name": "om2", "supercategory": "lcx"},
    {"id": 28, "name": "ramus", "supercategory": "other"},
]

# Valid CASS segment IDs
VALID_CASS_IDS = {cat["id"] for cat in CASS_CATEGORIES}

# CASS ID to name mapping for logging
CASS_NAMES = {cat["id"]: cat["name"] for cat in CASS_CATEGORIES}


def process_single_case(args):
    """Process a single case - extract frame, load Medis mask, create annotation."""
    row, output_images_dir, target_size = args

    try:
        case_id = f"{row['patient_id']}_{row['vessel_pattern']}_{row['phase']}"
        cass_segment = int(row['cass_segment'])

        # Skip if not a valid CASS segment
        if cass_segment not in VALID_CASS_IDS:
            return None, f"Invalid CASS segment {cass_segment} for {case_id}"

        # Load cine and extract frame
        cine_path = row['cine_path']
        frame_idx = int(row['frame_index'])

        if not os.path.exists(cine_path):
            return None, f"Cine not found: {cine_path}"

        cine = np.load(cine_path)
        if frame_idx >= len(cine):
            frame_idx = len(cine) - 1

        frame = cine[frame_idx]

        # Load Medis GT mask (segment-specific, not full tree)
        mask_path = row['vessel_mask_actual_path']
        if not os.path.exists(mask_path):
            return None, f"Medis mask not found: {mask_path}"

        mask = np.load(mask_path)

        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.uint8)

        # Check if mask is empty
        if mask.sum() == 0:
            return None, f"Empty mask for {case_id}"

        # Get original dimensions
        orig_h, orig_w = frame.shape[:2] if len(frame.shape) > 2 else frame.shape
        mask_h, mask_w = mask.shape

        # Resize mask to match frame if needed
        if (mask_h, mask_w) != (orig_h, orig_w):
            from skimage.transform import resize
            mask = resize(mask.astype(float), (orig_h, orig_w), order=0, preserve_range=True)
            mask = (mask > 0.5).astype(np.uint8)

        # Resize frame and mask to target size if specified
        if target_size is not None and (orig_h, orig_w) != (target_size, target_size):
            from skimage.transform import resize
            frame = resize(frame, (target_size, target_size), preserve_range=True)
            mask = resize(mask.astype(float), (target_size, target_size), order=0, preserve_range=True)
            mask = (mask > 0.5).astype(np.uint8)
            final_h, final_w = target_size, target_size
        else:
            final_h, final_w = orig_h, orig_w

        # Convert frame to uint8 if needed
        if frame.dtype != np.uint8:
            frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)

        # Convert to RGB if grayscale
        if len(frame.shape) == 2:
            frame = np.stack([frame, frame, frame], axis=-1)

        # Save image
        img_filename = f"{case_id}.png"
        img_path = output_images_dir / img_filename
        Image.fromarray(frame).save(img_path)

        # Create RLE encoding for mask
        mask_fortran = np.asfortranarray(mask)
        rle = mask_util.encode(mask_fortran)
        rle['counts'] = rle['counts'].decode('utf-8')

        # Calculate bounding box [x, y, width, height]
        rows_with_mask = np.any(mask, axis=1)
        cols_with_mask = np.any(mask, axis=0)

        y_min, y_max = np.where(rows_with_mask)[0][[0, -1]]
        x_min, x_max = np.where(cols_with_mask)[0][[0, -1]]
        bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]

        # Calculate area
        area = int(mask.sum())

        result = {
            'case_id': case_id,
            'img_filename': img_filename,
            'width': final_w,
            'height': final_h,
            'segmentation': rle,
            'bbox': bbox,
            'area': area,
            'cass_segment': cass_segment,
            'split': row['split'],
        }

        return result, None

    except Exception as e:
        return None, f"Error processing {row.get('patient_id', 'unknown')}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Create COCO dataset from Medis GT masks')
    parser.add_argument('--csv-path', type=str,
                        default='E:/AngioMLDL_data/corrected_dataset_training.csv',
                        help='Path to the corrected dataset CSV')
    parser.add_argument('--output-dir', type=str,
                        default='E:/AngioMLDL_data/coco_cass_segments',
                        help='Output directory for COCO dataset')
    parser.add_argument('--target-size', type=int, default=None,
                        help='Resize all images to this size (e.g., 1024). None keeps original.')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count - 2)')
    args = parser.parse_args()

    print("=" * 70)
    print("Medis GT Masks -> COCO Format (CASS Segment Categories)")
    print("=" * 70)

    # Load CSV
    print(f"\nLoading: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"  Total rows: {len(df)}")

    # Filter to valid CASS segments
    df = df[df['cass_segment'].isin(VALID_CASS_IDS)].reset_index(drop=True)
    print(f"  Valid CASS segments: {len(df)}")

    # Show distribution
    print("\n  CASS Segment Distribution:")
    segment_counts = df['cass_segment'].value_counts().sort_index()
    for cass_id, count in segment_counts.items():
        name = CASS_NAMES.get(cass_id, 'unknown')
        print(f"    {cass_id:2d} ({name:15s}): {count:4d}")

    # Split counts
    print(f"\n  Train: {len(df[df['split'] == 'train'])}")
    print(f"  Val: {len(df[df['split'] == 'val'])}")
    print(f"  Test: {len(df[df['split'] == 'test'])}")

    # Create output directories
    output_dir = Path(args.output_dir)
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'val']:
        print(f"\n{'=' * 70}")
        print(f"Processing {split} split...")
        print("=" * 70)

        # Filter by split (combine val and test into val)
        if split == 'val':
            split_df = df[df['split'].isin(['val', 'test'])]
        else:
            split_df = df[df['split'] == split]

        print(f"  Cases to process: {len(split_df)}")

        if len(split_df) == 0:
            print(f"  No cases for {split}, skipping...")
            continue

        output_images_dir = output_dir / split / 'images'

        # Prepare arguments for parallel processing
        process_args = [
            (row, output_images_dir, args.target_size)
            for _, row in split_df.iterrows()
        ]

        # Process in parallel
        num_workers = args.num_workers or max(1, cpu_count() - 2)
        print(f"  Using {num_workers} workers...")

        results = []
        errors = []

        with Pool(processes=num_workers) as pool:
            for result, error in tqdm(
                pool.imap(process_single_case, process_args),
                total=len(process_args),
                desc=f"  Processing {split}"
            ):
                if error:
                    errors.append(error)
                elif result:
                    results.append(result)

        if errors:
            print(f"\n  Errors ({len(errors)}):")
            for e in errors[:5]:
                print(f"    - {e}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")

        print(f"  Successfully processed: {len(results)}")

        # Create COCO format
        coco_data = {
            'info': {
                'description': 'Coronary Angiography CASS Segment Dataset (Medis GT)',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'AngioMLDL',
                'date_created': datetime.now().strftime('%Y-%m-%d'),
                'notes': 'Individual CASS segment masks from Medis expert annotations'
            },
            'licenses': [],
            'categories': CASS_CATEGORIES,
            'images': [],
            'annotations': []
        }

        # Track segment distribution in this split
        split_segment_counts = {}

        for idx, result in enumerate(results):
            # Image entry
            coco_data['images'].append({
                'id': idx,
                'file_name': result['img_filename'],
                'width': result['width'],
                'height': result['height'],
            })

            # Annotation entry
            cass_segment = result['cass_segment']
            coco_data['annotations'].append({
                'id': idx,
                'image_id': idx,
                'category_id': cass_segment,  # Use CASS ID directly as category ID
                'segmentation': result['segmentation'],
                'bbox': result['bbox'],
                'area': result['area'],
                'iscrowd': 0,
            })

            # Count segments
            split_segment_counts[cass_segment] = split_segment_counts.get(cass_segment, 0) + 1

        # Save COCO JSON
        ann_path = output_dir / split / 'annotations.json'
        with open(ann_path, 'w') as f:
            json.dump(coco_data, f)

        print(f"\n  Saved: {ann_path}")
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")
        print(f"  Categories: {len(CASS_CATEGORIES)}")

        print(f"\n  Segment distribution in {split}:")
        for cass_id in sorted(split_segment_counts.keys()):
            name = CASS_NAMES.get(cass_id, 'unknown')
            count = split_segment_counts[cass_id]
            print(f"    {cass_id:2d} ({name:15s}): {count:4d}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nCategories ({len(CASS_CATEGORIES)} CASS segments):")
    for cat in CASS_CATEGORIES:
        print(f"  {cat['id']:2d}: {cat['name']}")
    print("\nTo use with SAM3 training, update your config:")
    print(f"  angio_data_root: {output_dir}")


if __name__ == '__main__':
    main()
