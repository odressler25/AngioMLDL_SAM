"""
Create Unified Obstruction Labels from ALL Medis QCA Data

Processes BOTH datasets:
- Original 277 patients: E:\AngioMLDL_data\corrected_vessel_dataset\contours
- Batch2 100 patients: E:\AngioMLDL_data\batch2\contours

PRE images only (before treatment) - clear obstruction ground truth

Output: Unified obstruction labels for training
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_medis_contours(json_path):
    """Load Medis QCA contour data."""
    with open(json_path) as f:
        return json.load(f)


def contours_to_polygon(left_edge, right_edge):
    """Convert left/right edge points to a closed polygon."""
    left_points = np.array(left_edge)
    right_points = np.array(right_edge)[::-1]
    return np.vstack([left_points, right_points])


def create_vessel_mask(left_edge, right_edge, image_size):
    """Create binary mask from vessel contours."""
    h, w = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    polygon = contours_to_polygon(left_edge, right_edge).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def create_obstruction_mask(actual_left, actual_right, ideal_left, ideal_right, image_size):
    """Create obstruction mask from actual vs ideal contours."""
    actual_mask = create_vessel_mask(actual_left, actual_right, image_size)
    ideal_mask = create_vessel_mask(ideal_left, ideal_right, image_size)
    obstruction_mask = (ideal_mask > 0) & (actual_mask == 0)
    return obstruction_mask.astype(np.uint8), actual_mask, ideal_mask


def process_single_contour(args):
    """Process a single contour file (for multiprocessing)."""
    contour_path, output_dir, min_stenosis_pct = args

    try:
        contour_data = load_medis_contours(contour_path)

        # Check required fields
        if 'ideal_left_edge' not in contour_data or 'ideal_right_edge' not in contour_data:
            return None, f"No ideal contours: {contour_path.name}"

        # Get measurements
        measurements = contour_data.get('measurements', {})
        stenosis_pct = measurements.get('diameter_stenosis_pct', 0)

        # Filter by stenosis threshold
        if stenosis_pct < min_stenosis_pct:
            return None, f"Stenosis too low ({stenosis_pct:.1f}%): {contour_path.name}"

        # Get image size
        image_size = contour_data.get('image_size', [512, 512])
        h, w = image_size[1], image_size[0]  # JSON stores as [width, height]

        # Create obstruction mask
        obstruction_mask, actual_mask, ideal_mask = create_obstruction_mask(
            contour_data['left_edge'],
            contour_data['right_edge'],
            contour_data['ideal_left_edge'],
            contour_data['ideal_right_edge'],
            (h, w)
        )

        obstruction_area = int(obstruction_mask.sum())
        if obstruction_area < 10:  # Skip if obstruction too small
            return None, f"Obstruction too small ({obstruction_area}px): {contour_path.name}"

        # Save mask
        base_name = contour_path.stem.replace('_contours', '')
        mask_path = output_dir / f"{base_name}_obstruction.png"
        Image.fromarray((obstruction_mask * 255).astype(np.uint8)).save(mask_path)

        # Return result
        result = {
            'contour_file': contour_path.name,
            'image_name': base_name,
            'stenosis_pct': stenosis_pct,
            'mld_mm': measurements.get('MLD_mm', 0),
            'lesion_length_mm': measurements.get('lesion_length_mm', 0),
            'obstruction_pixels': obstruction_area,
            'mask_path': str(mask_path),
            'image_size': [w, h],
            'mld_x': measurements.get('MLD_x_coord', 0),
            'mld_y': measurements.get('MLD_y_coord', 0),
            'view_angles': contour_data.get('view_angles', {}),
            'source': str(contour_path.parent.parent.name)  # dataset name
        }

        return result, None

    except Exception as e:
        return None, f"Error processing {contour_path.name}: {e}"


def process_dataset(contours_dir, output_dir, min_stenosis_pct=30, dataset_name=""):
    """Process all PRE contours in a dataset."""
    contours_dir = Path(contours_dir)

    # Find PRE contour files only
    pre_contours = list(contours_dir.glob("*_PRE_contours.json"))
    pre_contours += list(contours_dir.glob("*_PRE_v*_contours.json"))

    print(f"\n{dataset_name}: Found {len(pre_contours)} PRE contour files")

    # Prepare args for multiprocessing
    args_list = [(p, output_dir, min_stenosis_pct) for p in pre_contours]

    # Process with multiprocessing
    num_workers = max(1, cpu_count() - 2)
    results = []
    errors = []

    with Pool(processes=num_workers) as pool:
        for result, error in tqdm(
            pool.imap(process_single_contour, args_list),
            total=len(args_list),
            desc=f"Processing {dataset_name}"
        ):
            if result:
                results.append(result)
            if error:
                errors.append(error)

    return results, errors


def main():
    print("=" * 70)
    print("Creating Unified Obstruction Labels from ALL Medis QCA Data")
    print("PRE images only (before treatment)")
    print("=" * 70)

    # Dataset locations
    datasets = [
        {
            'name': 'Original (277 patients)',
            'contours': Path(r"E:\AngioMLDL_data\corrected_vessel_dataset\contours"),
        },
        {
            'name': 'Batch2 (100 patients)',
            'contours': Path(r"E:\AngioMLDL_data\batch2\contours"),
        }
    ]

    # Output directory
    output_dir = Path(r"E:\AngioMLDL_data\unified_obstruction_labels")
    output_dir.mkdir(exist_ok=True)

    # Parameters
    min_stenosis_pct = 30  # Minimum stenosis to include

    all_results = []
    all_errors = []

    for dataset in datasets:
        if not dataset['contours'].exists():
            print(f"\nSkipping {dataset['name']} - contours not found")
            continue

        results, errors = process_dataset(
            dataset['contours'],
            output_dir,
            min_stenosis_pct,
            dataset['name']
        )

        all_results.extend(results)
        all_errors.extend(errors)

        print(f"  Processed: {len(results)} obstruction labels")
        print(f"  Skipped: {len(errors)}")

    # Save summary
    summary = {
        'total_labels': len(all_results),
        'min_stenosis_pct': min_stenosis_pct,
        'labels': all_results
    }

    summary_path = output_dir / "obstruction_labels_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total obstruction labels: {len(all_results)}")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_path}")

    # Stenosis distribution
    if all_results:
        stenosis_values = [r['stenosis_pct'] for r in all_results]
        print(f"\nStenosis distribution:")
        print(f"  30-50%: {sum(1 for s in stenosis_values if 30 <= s < 50)}")
        print(f"  50-70%: {sum(1 for s in stenosis_values if 50 <= s < 70)}")
        print(f"  70-90%: {sum(1 for s in stenosis_values if 70 <= s < 90)}")
        print(f"  >90%:   {sum(1 for s in stenosis_values if s >= 90)}")

    print("\n" + "=" * 70)
    print("Obstruction label generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
