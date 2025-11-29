"""
Create Obstruction Training Labels from Medis QCA Data

Uses the difference between actual and ideal contours to create
ground truth obstruction masks for training SAM 3.

Medis QCA provides:
- left_edge / right_edge: Actual vessel contours (with stenosis)
- ideal_left_edge / ideal_right_edge: Reference contours (without stenosis)
- The DIFFERENCE = obstruction region

This creates gold-standard training data for teaching SAM 3
what "obstruction" means in coronary angiography.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cv2
from tqdm import tqdm


def load_medis_contours(json_path):
    """Load Medis QCA contour data."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def contours_to_polygon(left_edge, right_edge):
    """Convert left/right edge points to a closed polygon."""
    # Left edge goes forward, right edge goes backward
    left_points = np.array(left_edge)  # (N, 2) as [x, y]
    right_points = np.array(right_edge)[::-1]  # Reverse for closed polygon

    polygon = np.vstack([left_points, right_points])
    return polygon


def create_vessel_mask(left_edge, right_edge, image_size):
    """Create binary mask from vessel contours."""
    h, w = image_size
    mask = np.zeros((h, w), dtype=np.uint8)

    polygon = contours_to_polygon(left_edge, right_edge)
    polygon = polygon.astype(np.int32)

    # Fill polygon
    cv2.fillPoly(mask, [polygon], 1)

    return mask


def create_obstruction_mask(actual_left, actual_right, ideal_left, ideal_right, image_size):
    """
    Create obstruction mask from the difference between actual and ideal contours.

    The obstruction is where the vessel is narrower than it should be:
    Obstruction = Ideal vessel mask - Actual vessel mask
    """
    # Create actual vessel mask (with stenosis - narrower)
    actual_mask = create_vessel_mask(actual_left, actual_right, image_size)

    # Create ideal vessel mask (without stenosis - wider)
    ideal_mask = create_vessel_mask(ideal_left, ideal_right, image_size)

    # Obstruction = where ideal is vessel but actual is not
    # This represents the "missing" vessel area due to stenosis
    obstruction_mask = (ideal_mask > 0) & (actual_mask == 0)

    return obstruction_mask.astype(np.uint8), actual_mask, ideal_mask


def visualize_obstruction(image_path, contour_data, output_path=None):
    """Visualize the obstruction detection."""
    # Load image
    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]

    # Get contours
    actual_left = contour_data['left_edge']
    actual_right = contour_data['right_edge']
    ideal_left = contour_data['ideal_left_edge']
    ideal_right = contour_data['ideal_right_edge']
    centerline = contour_data['centerline']

    # Create masks
    obstruction_mask, actual_mask, ideal_mask = create_obstruction_mask(
        actual_left, actual_right, ideal_left, ideal_right, (h, w)
    )

    # Get measurements
    measurements = contour_data.get('measurements', {})
    stenosis_pct = measurements.get('diameter_stenosis_pct', 0)
    mld_mm = measurements.get('MLD_mm', 0)
    lesion_length = measurements.get('lesion_length_mm', 0)
    mld_x = measurements.get('MLD_x_coord', 0)
    mld_y = measurements.get('MLD_y_coord', 0)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top left: Original image with contours
    ax1 = axes[0, 0]
    ax1.imshow(image, cmap='gray')

    # Draw actual contours (red)
    actual_poly = contours_to_polygon(actual_left, actual_right)
    ax1.plot(actual_poly[:, 0], actual_poly[:, 1], 'r-', linewidth=1, alpha=0.7, label='Actual')

    # Draw ideal contours (green)
    ideal_poly = contours_to_polygon(ideal_left, ideal_right)
    ax1.plot(ideal_poly[:, 0], ideal_poly[:, 1], 'g--', linewidth=1, alpha=0.7, label='Ideal/Reference')

    # Draw centerline (cyan)
    cx = [p[0] for p in centerline]
    cy = [p[1] for p in centerline]
    ax1.plot(cx, cy, 'c-', linewidth=1, alpha=0.7, label='Centerline')

    # Mark MLD location (stenosis point)
    if mld_x and mld_y:
        ax1.scatter([mld_x], [mld_y], c='yellow', s=100, marker='*', zorder=5, label=f'MLD ({mld_mm:.2f}mm)')

    ax1.set_title(f"Medis QCA Contours\n{stenosis_pct:.1f}% Stenosis", fontsize=12)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axis('off')

    # Top right: Actual vs Ideal overlay
    ax2 = axes[0, 1]
    ax2.imshow(image, cmap='gray')

    # Overlay ideal (green, semi-transparent)
    ideal_overlay = np.zeros((*ideal_mask.shape, 4))
    ideal_overlay[ideal_mask > 0] = [0, 1, 0, 0.3]  # Green
    ax2.imshow(ideal_overlay)

    # Overlay actual (red, semi-transparent)
    actual_overlay = np.zeros((*actual_mask.shape, 4))
    actual_overlay[actual_mask > 0] = [1, 0, 0, 0.3]  # Red
    ax2.imshow(actual_overlay)

    ax2.set_title("Actual (red) vs Ideal (green) Vessel", fontsize=12)
    ax2.axis('off')

    # Bottom left: Obstruction mask
    ax3 = axes[1, 0]
    ax3.imshow(image, cmap='gray')

    # Overlay obstruction (yellow)
    obstruction_overlay = np.zeros((*obstruction_mask.shape, 4))
    obstruction_overlay[obstruction_mask > 0] = [1, 1, 0, 0.7]  # Yellow
    ax3.imshow(obstruction_overlay)

    # Mark stenosis location
    if mld_x and mld_y:
        ax3.scatter([mld_x], [mld_y], c='red', s=100, marker='x', linewidths=3, zorder=5)

    obstruction_area = obstruction_mask.sum()
    ax3.set_title(f"Obstruction Mask (Training Label)\nArea: {obstruction_area} pixels", fontsize=12)
    ax3.axis('off')

    # Bottom right: Measurements summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    Medis QCA Measurements
    ----------------------

    Diameter Stenosis: {stenosis_pct:.1f}%
    Area Stenosis: {measurements.get('area_stenosis_pct', 0):.1f}%

    MLD (Minimum Lumen Diameter): {mld_mm:.2f} mm
    Reference Vessel Diameter: {measurements.get('interpolated_RVD_mm', 0):.2f} mm

    Lesion Length: {lesion_length:.2f} mm
    Segment Length: {measurements.get('segment_length_mm', 0):.2f} mm

    Stenosis Location: ({mld_x:.1f}, {mld_y:.1f})
    Position along segment: {measurements.get('obstruction_position_mm', 0):.2f} mm

    Proximal Normal Diameter: {measurements.get('proximal_normal_diameter_mm', 0):.2f} mm
    Distal Normal Diameter: {measurements.get('distal_normal_diameter_mm', 0):.2f} mm

    View Angles:
      Primary: {contour_data.get('view_angles', {}).get('primary_angle', 0):.1f}°
      Secondary: {contour_data.get('view_angles', {}).get('secondary_angle', 0):.1f}°
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title("QCA Measurements", fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def process_all_pre_cases(contours_dir, images_dir, output_dir):
    """
    Process all PRE-procedure cases to create obstruction training labels.
    Only PRE cases (before intervention) show the stenosis clearly.
    """
    contours_dir = Path(contours_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find all PRE contour files
    pre_contours = list(contours_dir.glob("*_PRE_contours.json"))
    pre_contours += list(contours_dir.glob("*_PRE_v*_contours.json"))

    print(f"Found {len(pre_contours)} PRE-procedure contour files")

    results = []

    for contour_path in tqdm(pre_contours, desc="Processing PRE cases"):
        try:
            # Load contour data
            contour_data = load_medis_contours(contour_path)

            # Find corresponding image
            base_name = contour_path.stem.replace('_contours', '')
            image_path = images_dir / f"{base_name}.png"

            if not image_path.exists():
                print(f"  Image not found: {image_path.name}")
                continue

            # Check if has ideal contours
            if 'ideal_left_edge' not in contour_data or 'ideal_right_edge' not in contour_data:
                print(f"  No ideal contours: {contour_path.name}")
                continue

            # Get measurements
            measurements = contour_data.get('measurements', {})
            stenosis_pct = measurements.get('diameter_stenosis_pct', 0)

            # Only process cases with significant stenosis (>30%)
            if stenosis_pct < 30:
                print(f"  Stenosis too low ({stenosis_pct:.1f}%): {contour_path.name}")
                continue

            # Load image to get size
            image = np.array(Image.open(image_path))
            h, w = image.shape[:2]

            # Create obstruction mask
            obstruction_mask, actual_mask, ideal_mask = create_obstruction_mask(
                contour_data['left_edge'],
                contour_data['right_edge'],
                contour_data['ideal_left_edge'],
                contour_data['ideal_right_edge'],
                (h, w)
            )

            # Save obstruction mask
            mask_path = output_dir / f"{base_name}_obstruction.png"
            Image.fromarray((obstruction_mask * 255).astype(np.uint8)).save(mask_path)

            # Save visualization
            viz_path = output_dir / f"{base_name}_visualization.png"
            visualize_obstruction(image_path, contour_data, viz_path)

            results.append({
                'image': base_name,
                'stenosis_pct': stenosis_pct,
                'mld_mm': measurements.get('MLD_mm', 0),
                'lesion_length_mm': measurements.get('lesion_length_mm', 0),
                'obstruction_pixels': int(obstruction_mask.sum()),
                'mask_path': str(mask_path),
                'viz_path': str(viz_path)
            })

            print(f"  {base_name}: {stenosis_pct:.1f}% stenosis, {obstruction_mask.sum()} px obstruction")

        except Exception as e:
            print(f"  Error processing {contour_path.name}: {e}")
            continue

    # Save summary
    summary_path = output_dir / "obstruction_labels_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessed {len(results)} cases")
    print(f"Summary saved to: {summary_path}")

    return results


def main():
    print("=" * 60)
    print("Creating Obstruction Training Labels from Medis QCA")
    print("=" * 60)

    contours_dir = Path(r"E:\AngioMLDL_data\batch2\contours")
    images_dir = Path(r"E:\AngioMLDL_data\batch2\images")
    output_dir = Path(r"E:\AngioMLDL_data\batch2\obstruction_labels")

    # Process a single example first for visualization
    print("\nProcessing example case...")
    example_contour = contours_dir / "108-0070_MID_RCA_PRE_contours.json"
    example_image = images_dir / "108-0070_MID_RCA_PRE.png"

    if example_contour.exists() and example_image.exists():
        contour_data = load_medis_contours(example_contour)
        output_dir.mkdir(exist_ok=True)
        viz_path = output_dir / "example_108-0070_MID_RCA_PRE.png"
        visualize_obstruction(example_image, contour_data, viz_path)
        print(f"Example visualization saved: {viz_path}")

    # Process all PRE cases
    print("\nProcessing all PRE-procedure cases...")
    results = process_all_pre_cases(contours_dir, images_dir, output_dir)

    print("\n" + "=" * 60)
    print("Obstruction label generation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
