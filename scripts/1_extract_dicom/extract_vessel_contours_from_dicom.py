"""
Extract Vessel Contours Directly from DICOM Private Tags
==========================================================

Extracts vessel measurements from QANGIOXA_BRACHY private tags in DICOMs.
No longer relies on JSON files - all data comes from DICOM.

For each vessel measurement found:
- Extract frame number with lesion
- Extract full cine sequence
- Extract vessel left edge, right edge, centerline
- Extract clinical measurements (MLD, stenosis, etc.)
- Save preprocessed data

Output: E:/AngioMLDL_data/vessel_contours_dataset/
"""

import pandas as pd
import numpy as np
import pydicom
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from sklearn.model_selection import GroupShuffleSplit
import warnings
warnings.filterwarnings('ignore')


def has_vessel_measurements(dcm):
    """
    Check if DICOM has QANGIOXA_BRACHY vessel measurements.

    Returns: True if vessel measurements exist, False otherwise
    """
    try:
        # Check for QANGIOXA_BRACHY tag
        if (0x7917, 0x1099) not in dcm:
            return False

        brachy_tag = dcm[0x7917, 0x1099]
        if not brachy_tag.value or len(brachy_tag.value) == 0:
            return False

        # Check if first item has segment data
        item = brachy_tag.value[0]
        if (0x7919, 0x1099) not in item:
            return False

        return True
    except:
        return False


def extract_vessel_measurement(dcm, dcm_path):
    """
    Extract vessel contour and clinical measurement data from DICOM.

    Returns: dict with vessel data, or None if extraction fails
    """
    try:
        # Get BRACHY data
        brachy_tag = dcm[0x7917, 0x1099]
        item = brachy_tag.value[0]

        # Frame number where vessel was analyzed
        frame_num = item[0x7919, 0x1021].value

        # Get segment data
        seg_item = item[0x7919, 0x1099].value[0]

        # Vessel segment name
        segment_name = seg_item[0x7921, 0x1022].value

        # Extract contours (pixel coordinates)
        centerline = None
        left_edge = None
        right_edge = None

        # Tag 0x1034: Centerline
        if (0x7921, 0x1034) in seg_item:
            centerline_array = seg_item[0x7921, 0x1034].value
            centerline = np.array(centerline_array).reshape(-1, 2)

        # Tag 0x1051: Left edge
        if (0x7921, 0x1051) in seg_item:
            left_array = seg_item[0x7921, 0x1051].value
            left_edge = np.array(left_array).reshape(-1, 2)

        # Tag 0x1052: Right edge
        if (0x7921, 0x1052) in seg_item:
            right_array = seg_item[0x7921, 0x1052].value
            right_edge = np.array(right_array).reshape(-1, 2)

        # Extract clinical measurements
        measurements = {}
        measurement_tags = {
            'mld_mm': (0x7921, 0x1041),           # Minimum Lumen Diameter
            'ref_diameter_mm': (0x7921, 0x1042),  # Reference Diameter
            'prox_ref_mm': (0x7921, 0x1043),      # Proximal Reference
            'dist_ref_mm': (0x7921, 0x1044),      # Distal Reference
            'pct_diameter_stenosis': (0x7921, 0x1045),  # % Diameter Stenosis
            'pct_area_stenosis': (0x7921, 0x1046),      # % Area Stenosis
            'lesion_length_mm': (0x7921, 0x1048),       # Lesion Length
            'segment_length_mm': (0x7921, 0x1049),      # Segment Length
        }

        for key, tag in measurement_tags.items():
            if tag in seg_item:
                measurements[key] = float(seg_item[tag].value)
            else:
                measurements[key] = None

        # Stenosis location indices
        stenosis_indices = None
        if (0x7921, 0x1069) in seg_item:
            stenosis_indices = seg_item[0x7921, 0x1069].value

        # Analyst info
        analyst = seg_item[0x7921, 0x1026].value if (0x7921, 0x1026) in seg_item else 'Unknown'
        analysis_date = seg_item[0x7921, 0x1024].value if (0x7921, 0x1024) in seg_item else 'Unknown'

        return {
            'segment_name': segment_name,
            'frame_num': frame_num,
            'centerline': centerline,
            'left_edge': left_edge,
            'right_edge': right_edge,
            'measurements': measurements,
            'stenosis_indices': stenosis_indices,
            'analyst': analyst,
            'analysis_date': analysis_date,
            'dcm_path': str(dcm_path)
        }

    except Exception as e:
        print(f"Error extracting vessel data: {e}")
        return None


def create_vessel_mask(left_edge, right_edge, target_size=1536):
    """
    Create binary vessel mask from left and right edge contours.

    Args:
        left_edge: (N, 2) array of left edge coordinates
        right_edge: (M, 2) array of right edge coordinates
        target_size: Output mask size

    Returns:
        mask: (H, W) binary numpy array
    """
    try:
        # Combine edges to form closed polygon
        polygon = np.vstack([left_edge, right_edge[::-1]])

        # Find bounding box
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)
        orig_size = max(x_max - x_min, y_max - y_min)

        # Scale to target size
        scale = target_size / orig_size
        scaled_polygon = [(int((x - x_min) * scale), int((y - y_min) * scale))
                         for x, y in polygon]

        # Create mask
        mask = Image.new('L', (target_size, target_size), 0)
        ImageDraw.Draw(mask).polygon(scaled_polygon, outline=1, fill=1)

        return np.array(mask, dtype=np.uint8)

    except Exception as e:
        print(f"Error creating mask: {e}")
        return None


def load_dicom_cine(dcm_path):
    """Load full DICOM cine sequence."""
    try:
        dcm = pydicom.dcmread(dcm_path)

        if hasattr(dcm, 'NumberOfFrames') and dcm.NumberOfFrames > 1:
            frames = dcm.pixel_array
            num_frames = dcm.NumberOfFrames
        else:
            frames = dcm.pixel_array[np.newaxis, ...]
            num_frames = 1

        # Normalize to [0, 1]
        frames = frames.astype(np.float32)
        frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8)

        return frames, num_frames

    except Exception as e:
        print(f"Error loading DICOM: {e}")
        return None, 0


def resize_frame(frame, target_size=1536):
    """Resize frame to target size."""
    h, w = frame.shape
    frame_uint8 = (frame * 255).astype(np.uint8)
    img = Image.fromarray(frame_uint8)
    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return np.array(img_resized, dtype=np.float32) / 255.0


def scan_dicoms_for_vessel_data(base_dir='E:/Angios'):
    """
    Scan all DICOMs in E:/Angios for vessel measurements.

    Returns: list of dicts with vessel data
    """
    print("="*80)
    print("SCANNING DICOMS FOR VESSEL MEASUREMENTS")
    print("="*80)

    base_dir = Path(base_dir)

    # Find all DICOM files
    dcm_pattern = 'ALL RISE */Analysis/PROCEDURES QCA ANALYSIS/INDEX BASELINE/*.dcm'
    dcm_files = list(base_dir.glob(dcm_pattern))

    print(f"\nFound {len(dcm_files)} DICOM files")

    vessel_data = []

    for dcm_path in tqdm(dcm_files, desc="Scanning DICOMs"):
        try:
            dcm = pydicom.dcmread(dcm_path)

            # Check if this DICOM has vessel measurements
            if not has_vessel_measurements(dcm):
                continue

            # Extract vessel data
            data = extract_vessel_measurement(dcm, dcm_path)
            if data is not None:
                # Extract case ID from path
                case_dir = dcm_path.parents[3]
                case_id = case_dir.name.replace('ALL RISE ', '')
                data['case_id'] = case_id

                vessel_data.append(data)

        except Exception as e:
            continue

    print(f"\nFound {len(vessel_data)} DICOMs with vessel measurements")

    return vessel_data


def process_vessel_dataset(vessel_data, output_dir, dry_run=False):
    """
    Process vessel data and save to output directory.

    Args:
        vessel_data: List of vessel measurement dicts
        output_dir: Output directory path
        dry_run: If True, only process first 5 samples
    """
    print("\n" + "="*80)
    print("PROCESSING VESSEL DATASET")
    print("="*80)

    if dry_run:
        print("\nDRY RUN MODE: Processing first 5 samples")
        vessel_data = vessel_data[:5]

    output_dir = Path(output_dir)
    cines_dir = output_dir / 'cines'
    frames_dir = output_dir / 'frames'
    masks_dir = output_dir / 'masks'

    for d in [cines_dir, frames_dir, masks_dir]:
        d.mkdir(exist_ok=True, parents=True)

    results = []
    success_count = 0
    failed_count = 0

    for data in tqdm(vessel_data, desc="Processing"):
        case_id = data['case_id']
        segment_name = data['segment_name']
        frame_num = data['frame_num']
        dcm_path = data['dcm_path']

        # Create safe filename
        filename = f"{case_id}_{segment_name.replace(' ', '_')}"

        try:
            # Load full cine
            frames, num_frames = load_dicom_cine(dcm_path)
            if frames is None:
                raise ValueError("Failed to load DICOM")

            # Validate frame number
            if frame_num >= num_frames:
                frame_num = num_frames - 1

            # Extract lesion frame
            lesion_frame = frames[frame_num]
            lesion_frame_resized = resize_frame(lesion_frame, 1536)

            # Create vessel mask from edges
            left_edge = data['left_edge']
            right_edge = data['right_edge']

            if left_edge is None or right_edge is None:
                raise ValueError("Missing edge contours")

            mask = create_vessel_mask(left_edge, right_edge, 1536)
            if mask is None:
                raise ValueError("Failed to create mask")

            # Save files
            cine_path = cines_dir / f"{filename}_cine.npy"
            frame_path = frames_dir / f"{filename}_frame.npy"
            mask_path = masks_dir / f"{filename}_mask.npy"

            np.save(cine_path, frames)
            np.save(frame_path, lesion_frame_resized)
            np.save(mask_path, mask)

            # Record result
            results.append({
                'case_id': case_id,
                'segment_name': segment_name,
                'cine_preprocessed_path': str(cine_path),
                'lesion_frame_path': str(frame_path),
                'vessel_mask_path': str(mask_path),
                'lesion_frame_num': frame_num,
                'num_frames': num_frames,
                'original_dcm_path': dcm_path,
                'analyst': data['analyst'],
                'analysis_date': data['analysis_date'],
                **data['measurements']
            })

            success_count += 1

        except Exception as e:
            print(f"\nFailed: {case_id} {segment_name} - {e}")
            failed_count += 1
            continue

    # Create dataset CSV
    df = pd.DataFrame(results)

    # Assign splits
    if len(df) > 0:
        df = assign_splits(df)

    output_csv = output_dir / 'dataset.csv'
    df.to_csv(output_csv, index=False)

    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Successfully processed: {success_count}/{len(vessel_data)}")
    print(f"Failed: {failed_count}/{len(vessel_data)}")
    print(f"\nDataset CSV saved: {output_csv}")
    print(f"  Total rows: {len(df)}")

    if len(df) > 0:
        print("\nSplit breakdown:")
        for split in ['train', 'val', 'test']:
            count = (df['split'] == split).sum()
            print(f"  {split}: {count} samples")

    return df


def assign_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Assign train/val/test splits by patient."""
    df['patient_id'] = df['case_id'].str.split('-').str[0]

    unique_patients = df['patient_id'].unique()
    n_patients = len(unique_patients)

    if n_patients < 3:
        # Simple random split for small datasets
        n_samples = len(df)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        np.random.seed(random_state)
        indices = np.random.permutation(n_samples)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        df['split'] = 'train'
        df.iloc[val_idx, df.columns.get_loc('split')] = 'val'
        df.iloc[test_idx, df.columns.get_loc('split')] = 'test'
    else:
        # Grouped split by patient
        gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
        train_idx, temp_idx = next(gss1.split(df, groups=df['patient_id']))

        val_size = val_ratio / (val_ratio + test_ratio)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size, random_state=random_state)
        temp_df = df.iloc[temp_idx]
        val_idx_temp, test_idx_temp = next(gss2.split(temp_df, groups=temp_df['patient_id']))

        val_idx = temp_idx[val_idx_temp]
        test_idx = temp_idx[test_idx_temp]

        df['split'] = 'train'
        df.loc[val_idx, 'split'] = 'val'
        df.loc[test_idx, 'split'] = 'test'

    return df


def main():
    import sys

    angios_dir = r'E:\Angios'
    output_dir = r'E:\AngioMLDL_data\vessel_contours_dataset'

    dry_run_only = '--dry-run' in sys.argv

    print("Configuration:")
    print(f"  Angios dir: {angios_dir}")
    print(f"  Output dir: {output_dir}")

    # Scan for vessel data
    vessel_data = scan_dicoms_for_vessel_data(angios_dir)

    if len(vessel_data) == 0:
        print("\nNo vessel measurements found in DICOMs!")
        return

    # Process dataset
    if dry_run_only:
        print("\nDRY RUN MODE")
        df = process_vessel_dataset(vessel_data, output_dir, dry_run=True)
        print("\nDry run complete. Run without --dry-run to process all samples.")
    else:
        print(f"\nProcessing {len(vessel_data)} vessel measurements...")
        df = process_vessel_dataset(vessel_data, output_dir, dry_run=False)
        print("\nDataset ready!")


if __name__ == '__main__':
    main()
