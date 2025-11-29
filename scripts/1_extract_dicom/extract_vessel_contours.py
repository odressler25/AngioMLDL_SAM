#!/usr/bin/env python3
"""
Extract Vessel Contours from DICOM
===================================

Extracts vessel contours and measurements from QANGIOXA_BRACHY private tags in DICOMs.

Usage:
    python extract_vessel_contours.py --source E:/Angios --output E:/AngioMLDL_data/contours
    python extract_vessel_contours.py --source E:/Angios_new --output E:/AngioMLDL_data/batch2/contours --dry-run

Output structure:
    output/
    ├── images/           # PNG frames at lesion location
    ├── contours/         # JSON files with vessel coordinates + measurements
    └── dataset.csv       # Summary of all extractions
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def has_vessel_measurements(dcm):
    """Check if DICOM has QANGIOXA_BRACHY vessel measurements."""
    try:
        if (0x7917, 0x1099) not in dcm:
            return False
        brachy_tag = dcm[0x7917, 0x1099]
        if not brachy_tag.value or len(brachy_tag.value) == 0:
            return False
        item = brachy_tag.value[0]
        if (0x7919, 0x1099) not in item:
            return False
        return True
    except:
        return False


def extract_vessel_data(dcm, dcm_path):
    """Extract vessel contour and measurement data from DICOM."""
    try:
        brachy_tag = dcm[0x7917, 0x1099]
        item = brachy_tag.value[0]

        # Frame number where vessel was analyzed
        frame_num = int(item[0x7919, 0x1021].value)

        # Get segment data
        seg_item = item[0x7919, 0x1099].value[0]

        # Vessel segment name (e.g., "Mid LAD PRE")
        segment_name = str(seg_item[0x7921, 0x1022].value)

        # Extract contours (pixel coordinates)
        centerline = None
        left_edge = None
        right_edge = None
        ideal_left_edge = None
        ideal_right_edge = None

        # Centerline
        if (0x7921, 0x1034) in seg_item:
            arr = np.array(seg_item[0x7921, 0x1034].value).reshape(-1, 2)
            centerline = arr.tolist()

        # Left edge (actual vessel boundary)
        if (0x7921, 0x1051) in seg_item:
            arr = np.array(seg_item[0x7921, 0x1051].value).reshape(-1, 2)
            left_edge = arr.tolist()

        # Right edge (actual vessel boundary)
        if (0x7921, 0x1052) in seg_item:
            arr = np.array(seg_item[0x7921, 0x1052].value).reshape(-1, 2)
            right_edge = arr.tolist()

        # Ideal left edge (for %DS calculation - healthy vessel reference)
        if (0x7921, 0x1053) in seg_item:
            arr = np.array(seg_item[0x7921, 0x1053].value).reshape(-1, 2)
            ideal_left_edge = arr.tolist()

        # Ideal right edge
        if (0x7921, 0x1054) in seg_item:
            arr = np.array(seg_item[0x7921, 0x1054].value).reshape(-1, 2)
            ideal_right_edge = arr.tolist()

        # Extract clinical measurements
        measurements = {}
        measurement_tags = {
            'MLD_mm': (0x7921, 0x1041),
            'interpolated_RVD_mm': (0x7921, 0x1042),
            'proximal_RVD_mm': (0x7921, 0x1043),
            'distal_RVD_mm': (0x7921, 0x1044),
            'diameter_stenosis_pct': (0x7921, 0x1045),
            'area_stenosis_pct': (0x7921, 0x1046),
            'lesion_length_mm': (0x7921, 0x1048),
            'segment_length_mm': (0x7921, 0x1049),
        }

        for key, tag in measurement_tags.items():
            if tag in seg_item:
                measurements[key] = float(seg_item[tag].value)

        # MLD location (x, y coordinates)
        if (0x7921, 0x1061) in seg_item:
            mld_coords = seg_item[0x7921, 0x1061].value
            if len(mld_coords) >= 2:
                measurements['MLD_x_coord'] = float(mld_coords[0])
                measurements['MLD_y_coord'] = float(mld_coords[1])

        # Calibration info
        calibration = {}
        if (0x7919, 0x1031) in item:
            calibration['scale_factor'] = float(item[0x7919, 0x1031].value)
        if (0x7919, 0x1033) in item:
            calibration['catheter_size_fr'] = float(item[0x7919, 0x1033].value)

        # Analyst info
        analyst = str(seg_item[0x7921, 0x1026].value) if (0x7921, 0x1026) in seg_item else 'Unknown'
        analysis_date = str(seg_item[0x7921, 0x1024].value) if (0x7921, 0x1024) in seg_item else 'Unknown'

        return {
            'segment_name': segment_name,
            'frame_num': frame_num,
            'centerline': centerline,
            'left_edge': left_edge,
            'right_edge': right_edge,
            'ideal_left_edge': ideal_left_edge,
            'ideal_right_edge': ideal_right_edge,
            'measurements': measurements,
            'calibration': calibration,
            'analyst': analyst,
            'analysis_date': analysis_date,
            'dcm_path': str(dcm_path)
        }

    except Exception as e:
        return None


def extract_frame_image(dcm, frame_num):
    """Extract a specific frame from DICOM as numpy array."""
    try:
        if hasattr(dcm, 'NumberOfFrames') and dcm.NumberOfFrames > 1:
            frames = dcm.pixel_array
            if frame_num >= len(frames):
                frame_num = len(frames) - 1
            frame = frames[frame_num]
        else:
            frame = dcm.pixel_array

        # Normalize to 0-255
        frame = frame.astype(np.float32)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frame = (frame * 255).astype(np.uint8)

        return frame

    except Exception as e:
        return None


def process_single_dicom(args):
    """Process a single DICOM file (for multiprocessing)."""
    dcm_path, output_dir = args

    try:
        dcm = pydicom.dcmread(dcm_path)

        if not has_vessel_measurements(dcm):
            return None

        data = extract_vessel_data(dcm, dcm_path)
        if data is None:
            return None

        # Extract patient ID from path
        case_id = "UNKNOWN"
        for part in dcm_path.parts:
            if part.startswith("ALL RISE"):
                case_id = part.replace("ALL RISE ", "")
                break

        # Create filename
        segment_safe = data['segment_name'].replace(' ', '_').replace('/', '_')
        filename = f"{case_id}_{segment_safe}"

        # Extract and save frame image
        frame = extract_frame_image(dcm, data['frame_num'])
        if frame is not None:
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            img = Image.fromarray(frame)
            img.save(images_dir / f"{filename}.png")
            data['image_file'] = f"{filename}.png"
            data['image_size'] = frame.shape

        # Save contour JSON
        contours_dir = output_dir / "contours"
        contours_dir.mkdir(parents=True, exist_ok=True)

        json_path = contours_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Return summary for CSV
        return {
            'case_id': case_id,
            'segment_name': data['segment_name'],
            'frame_num': data['frame_num'],
            'image_file': data.get('image_file'),
            'contour_file': f"{filename}.json",
            'analyst': data['analyst'],
            'analysis_date': data['analysis_date'],
            **data.get('measurements', {}),
            **{f"cal_{k}": v for k, v in data.get('calibration', {}).items()}
        }

    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract vessel contours from DICOM files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Source directory with patient DICOMs (e.g., E:/Angios)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for extracted data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only first 10 files"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=max(1, cpu_count() - 2),
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    print("=" * 70)
    print("EXTRACT VESSEL CONTOURS FROM DICOM")
    print("=" * 70)
    print(f"Source:  {source_dir}")
    print(f"Output:  {output_dir}")
    print(f"Workers: {args.workers}")

    # Find all DICOM files
    dcm_pattern = 'ALL RISE */Analysis/PROCEDURES QCA ANALYSIS/INDEX BASELINE/*.dcm'
    dcm_files = list(source_dir.glob(dcm_pattern))

    print(f"\nFound {len(dcm_files)} DICOM files")

    if len(dcm_files) == 0:
        print("[ERROR] No DICOM files found matching pattern")
        print(f"  Pattern: {dcm_pattern}")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] Processing first 10 files only")
        dcm_files = dcm_files[:10]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    task_args = [(dcm_path, output_dir) for dcm_path in dcm_files]

    results = []
    with Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap(process_single_dicom, task_args),
                           total=len(task_args), desc="Extracting"):
            if result is not None:
                results.append(result)

    # Save summary CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = output_dir / "dataset.csv"
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"Extracted: {len(results)} vessel measurements")
        print(f"Patients:  {df['case_id'].nunique()}")
        print(f"\nOutput:")
        print(f"  Images:   {output_dir / 'images'}")
        print(f"  Contours: {output_dir / 'contours'}")
        print(f"  CSV:      {csv_path}")

        if 'diameter_stenosis_pct' in df.columns:
            print(f"\nStenosis distribution:")
            print(f"  Mean: {df['diameter_stenosis_pct'].mean():.1f}%")
            print(f"  Min:  {df['diameter_stenosis_pct'].min():.1f}%")
            print(f"  Max:  {df['diameter_stenosis_pct'].max():.1f}%")
    else:
        print("\n[WARNING] No vessel measurements extracted!")
        sys.exit(1)


if __name__ == "__main__":
    main()
