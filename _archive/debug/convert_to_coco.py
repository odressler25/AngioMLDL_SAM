"""
Convert DeepSA pseudo labels to COCO format for SAM3 training.

This script:
1. Reads the index.csv to get case metadata
2. Extracts the correct frame from each cine using frame_index
3. Loads the corresponding DeepSA mask
4. Creates COCO format annotations with RLE encoding
5. Splits into train/val based on the split column

Usage:
    python scripts/convert_to_coco.py
    python scripts/convert_to_coco.py --output-dir E:/AngioMLDL_data/coco_format_v2
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


def process_single_case(args):
    """Process a single case - extract frame and create annotation."""
    row, output_images_dir, target_size = args

    try:
        case_id = f"{row['patient_id']}_{row['vessel_pattern']}_{row['phase']}"

        # Load cine and extract frame
        cine_path = row['cine_path']
        frame_idx = int(row['frame_index'])

        if not os.path.exists(cine_path):
            return None, f"Cine not found: {cine_path}"

        cine = np.load(cine_path)
        if frame_idx >= len(cine):
            return None, f"Frame index {frame_idx} out of range for {case_id}"

        frame = cine[frame_idx]

        # Load DeepSA mask
        deepsa_path = row['deepsa_pseudo_label_path']
        if not os.path.exists(deepsa_path):
            return None, f"DeepSA mask not found: {deepsa_path}"

        mask = np.load(deepsa_path)

        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.uint8)

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
        rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string for JSON

        # Calculate bounding box [x, y, width, height]
        rows_with_mask = np.any(mask, axis=1)
        cols_with_mask = np.any(mask, axis=0)

        if not rows_with_mask.any() or not cols_with_mask.any():
            return None, f"Empty mask for {case_id}"

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
            'split': row['split'],
        }

        return result, None

    except Exception as e:
        return None, f"Error processing {row.get('patient_id', 'unknown')}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Convert DeepSA labels to COCO format')
    parser.add_argument('--index-csv', type=str,
                        default='E:/AngioMLDL_data/deepsa_pseudo_labels/index.csv',
                        help='Path to index CSV file')
    parser.add_argument('--output-dir', type=str,
                        default='E:/AngioMLDL_data/coco_format_v2',
                        help='Output directory for COCO dataset')
    parser.add_argument('--target-size', type=int, default=None,
                        help='Resize all images to this size (e.g., 1024). None keeps original.')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count - 2)')
    args = parser.parse_args()

    print("=" * 60)
    print("DeepSA to COCO Converter")
    print("=" * 60)

    # Load index
    print(f"\nLoading index: {args.index_csv}")
    df = pd.read_csv(args.index_csv)
    print(f"  Total cases: {len(df)}")
    print(f"  Train: {len(df[df['split'] == 'train'])}")
    print(f"  Val: {len(df[df['split'] == 'val'])}")
    print(f"  Test: {len(df[df['split'] == 'test'])}")

    # Create output directories
    output_dir = Path(args.output_dir)
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'val']:
        print(f"\n{'=' * 60}")
        print(f"Processing {split} split...")
        print("=" * 60)

        # Filter by split (combine val and test into val for simplicity)
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
                'description': 'Coronary Angiography Vessel Segmentation Dataset (DeepSA)',
                'version': '2.0',
                'year': datetime.now().year,
                'contributor': 'AngioMLDL',
                'date_created': datetime.now().strftime('%Y-%m-%d'),
            },
            'licenses': [],
            'categories': [
                {
                    'id': 1,
                    'name': 'coronary artery',
                    'supercategory': 'vessel'
                }
            ],
            'images': [],
            'annotations': []
        }

        for idx, result in enumerate(results):
            # Image entry
            coco_data['images'].append({
                'id': idx,
                'file_name': result['img_filename'],
                'width': result['width'],
                'height': result['height'],
            })

            # Annotation entry
            coco_data['annotations'].append({
                'id': idx,
                'image_id': idx,
                'category_id': 1,
                'segmentation': result['segmentation'],
                'bbox': result['bbox'],
                'area': result['area'],
                'iscrowd': 0,
            })

        # Save COCO JSON
        ann_path = output_dir / split / 'annotations.json'
        with open(ann_path, 'w') as f:
            json.dump(coco_data, f)

        print(f"  Saved: {ann_path}")
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nTo use with SAM3 training, update your config:")
    print(f"  angio_data_root: {output_dir}")


if __name__ == '__main__':
    main()
