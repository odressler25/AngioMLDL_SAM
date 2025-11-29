#!/usr/bin/env python3
"""
Decode Medis QAngioXA private tags to extract:
- Frame selection
- Lesion contour tracings (pixel coordinates)
- QCA measurements (MLD, RVD, DS%, lesion length)
- Pixel coordinates for all measurements
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def decode_medis_tags(dicom_path):
    """Decode all Medis private tags"""

    print(f"\n{'='*80}")
    print(f"DECODING MEDIS QANGIOXA PRIVATE TAGS")
    print(f"File: {dicom_path}")
    print(f"{'='*80}\n")

    dcm = pydicom.dcmread(dicom_path)

    # Dictionary to store decoded data
    medis_data = {
        'frame_selection': {},
        'calibration': {},
        'measurements': {},
        'contours': {},
        'other_private_tags': {}
    }

    # Medis private tag groups
    # Tag group 0x7001 - CMSVIEW
    # Tag group 0x7933 - QANGIOXA_CALIBRATION
    # Tag group 0x7935 - QANGIOXA_MEASUREMENT
    # Tag group 0x7901 - QANGIOXA
    # Tag group 0x7931 - QANGIOXA_SUBTRACTION

    print("="*80)
    print("CMSVIEW TAGS (0x7001)")
    print("="*80)
    for elem in dcm:
        if elem.tag.group == 0x7001:
            tag_str = f"{elem.tag}"
            tag_name = elem.keyword if elem.keyword else f"Tag_{elem.tag}"
            try:
                value = elem.value
                # Convert bytes to readable format
                if elem.VR == 'OB' or isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except:
                        value = f"<binary data, {len(value)} bytes>"

                print(f"{tag_str:20s} {elem.VR:4s} {tag_name:40s}: {value}")

                # Store in dictionary
                if 'frame' in str(value).lower():
                    medis_data['frame_selection'][tag_name] = value
                elif 'calib' in str(value).lower() or 'catheter' in str(value).lower():
                    medis_data['calibration'][tag_name] = value
                else:
                    medis_data['other_private_tags'][tag_name] = value
            except Exception as e:
                print(f"{tag_str:20s} {elem.VR:4s} {tag_name:40s}: <error: {e}>")

    print("\n" + "="*80)
    print("QANGIOXA_CALIBRATION TAGS (0x7933)")
    print("="*80)
    for elem in dcm:
        if elem.tag.group == 0x7933:
            tag_str = f"{elem.tag}"
            tag_name = elem.keyword if elem.keyword else f"Tag_{elem.tag}"
            try:
                value = elem.value

                # These are likely coordinate arrays
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    if len(value) <= 5:
                        display_value = value
                    else:
                        display_value = f"Array with {len(value)} values: [{value[0]}, {value[1]}, ..., {value[-1]}]"
                    print(f"{tag_str:20s} {elem.VR:4s} {tag_name:40s}: {display_value}")

                    # Store full array
                    medis_data['calibration'][tag_name] = list(value) if isinstance(value, tuple) else value
                else:
                    print(f"{tag_str:20s} {elem.VR:4s} {tag_name:40s}: {value}")
                    medis_data['calibration'][tag_name] = value

            except Exception as e:
                print(f"{tag_str:20s} {elem.VR:4s} {tag_name:40s}: <error: {e}>")

    print("\n" + "="*80)
    print("QANGIOXA_MEASUREMENT TAGS (0x7935)")
    print("="*80)
    for elem in dcm:
        if elem.tag.group == 0x7935:
            tag_str = f"{elem.tag}"
            tag_name = elem.keyword if elem.keyword else f"Tag_{elem.tag}"
            try:
                value = elem.value

                if isinstance(value, (list, tuple)) and len(value) > 0:
                    if len(value) <= 5:
                        display_value = value
                    else:
                        display_value = f"Array with {len(value)} values: [{value[0]}, {value[1]}, ..., {value[-1]}]"
                    print(f"{tag_str:20s} {elem.VR:4s} {tag_name:40s}: {display_value}")

                    # Store full array - likely contour coordinates!
                    medis_data['measurements'][tag_name] = list(value) if isinstance(value, tuple) else value
                else:
                    print(f"{tag_str:20s} {elem.VR:4s} {tag_name:40s}: {value}")
                    medis_data['measurements'][tag_name] = value

            except Exception as e:
                print(f"{tag_str:20s} {elem.VR:4s} {tag_name:40s}: <error: {e}>")

    print("\n" + "="*80)
    print("QANGIOXA TAGS (0x7901)")
    print("="*80)
    for elem in dcm:
        if elem.tag.group == 0x7901:
            tag_str = f"{elem.tag}"
            tag_name = elem.keyword if elem.keyword else f"Tag_{elem.tag}"
            try:
                value = elem.value
                print(f"{tag_str:20s} {elem.VR:4s} {tag_name:40s}: {value}")
                medis_data['measurements'][tag_name] = value
            except Exception as e:
                print(f"{tag_str:20s} {elem.VR:4s} {tag_name:40s}: <error: {e}>")

    return medis_data, dcm


def analyze_coordinate_arrays(medis_data, dcm):
    """Analyze coordinate arrays to understand structure"""

    print("\n" + "="*80)
    print("ANALYZING COORDINATE ARRAYS")
    print("="*80)

    pixel_array = dcm.pixel_array
    if len(pixel_array.shape) == 3:
        img_height, img_width = pixel_array.shape[1], pixel_array.shape[2]
    else:
        img_height, img_width = pixel_array.shape

    print(f"\nImage dimensions: {img_width} × {img_height}")

    # Analyze calibration arrays
    print("\nCALIBRATION ARRAYS:")

    for key, value in medis_data['calibration'].items():
        if isinstance(value, list) and len(value) > 2:
            print(f"\n{key}:")
            print(f"  Length: {len(value)}")
            print(f"  Min: {min(value):.4f}")
            print(f"  Max: {max(value):.4f}")
            print(f"  Mean: {np.mean(value):.4f}")

            # Check if values are in pixel coordinate range
            if max(value) <= img_width or max(value) <= img_height:
                print(f"  ✓ Values in pixel coordinate range")
            else:
                print(f"  ? Values exceed image dimensions")

            # Check if it's alternating x,y coordinates
            if len(value) % 2 == 0:
                x_coords = value[0::2]
                y_coords = value[1::2]
                print(f"  If (x,y) pairs: {len(x_coords)} points")
                print(f"    X range: {min(x_coords):.2f} - {max(x_coords):.2f}")
                print(f"    Y range: {min(y_coords):.2f} - {max(y_coords):.2f}")

    # Analyze measurement arrays
    print("\n\nMEASUREMENT ARRAYS:")

    for key, value in medis_data['measurements'].items():
        if isinstance(value, list) and len(value) > 2:
            print(f"\n{key}:")
            print(f"  Length: {len(value)}")
            print(f"  Min: {min(value):.4f}")
            print(f"  Max: {max(value):.4f}")
            print(f"  Mean: {np.mean(value):.4f}")

            if max(value) <= img_width or max(value) <= img_height:
                print(f"  ✓ Values in pixel coordinate range")

            if len(value) % 2 == 0:
                x_coords = value[0::2]
                y_coords = value[1::2]
                print(f"  If (x,y) pairs: {len(x_coords)} points")
                print(f"    X range: {min(x_coords):.2f} - {max(x_coords):.2f}")
                print(f"    Y range: {min(y_coords):.2f} - {max(y_coords):.2f}")


def visualize_contours(medis_data, dcm, output_dir="."):
    """Visualize extracted contours on DICOM frames"""

    print("\n" + "="*80)
    print("VISUALIZING CONTOURS")
    print("="*80)

    pixel_array = dcm.pixel_array

    # Get frame 0 for visualization
    if len(pixel_array.shape) == 3:
        frame = pixel_array[0]
    else:
        frame = pixel_array

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(frame, cmap='gray')
    ax.set_title('DICOM with Medis Contours/Calibration Points')

    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    color_idx = 0

    # Plot calibration points
    for key, value in medis_data['calibration'].items():
        if isinstance(value, list) and len(value) >= 2:
            try:
                # Try as (x,y) pairs
                if len(value) == 2:
                    # Single point
                    ax.plot(value[0], value[1], 'o', color=colors[color_idx % len(colors)],
                           markersize=10, label=key)
                    color_idx += 1
                elif len(value) > 2:
                    # Array of points - try different interpretations

                    # Interpretation 1: alternating x,y
                    if len(value) % 2 == 0:
                        x_coords = value[0::2]
                        y_coords = value[1::2]
                        ax.plot(x_coords, y_coords, '-o', color=colors[color_idx % len(colors)],
                               markersize=3, linewidth=1, label=f"{key} (xy pairs)")
                        color_idx += 1

                    # Interpretation 2: all x or all y
                    # (would need another array for the other coordinate)

            except Exception as e:
                print(f"Could not plot {key}: {e}")

    # Plot measurement points/contours
    for key, value in medis_data['measurements'].items():
        if isinstance(value, list) and len(value) >= 2:
            try:
                if len(value) == 2:
                    ax.plot(value[0], value[1], 's', color=colors[color_idx % len(colors)],
                           markersize=10, label=key)
                    color_idx += 1
                elif len(value) > 2 and len(value) % 2 == 0:
                    x_coords = value[0::2]
                    y_coords = value[1::2]
                    ax.plot(x_coords, y_coords, '-s', color=colors[color_idx % len(colors)],
                           markersize=3, linewidth=1, label=f"{key} (xy pairs)")
                    color_idx += 1
            except Exception as e:
                print(f"Could not plot {key}: {e}")

    ax.legend(loc='upper right', fontsize=8)
    ax.axis('on')

    output_path = Path(output_dir) / "medis_contours_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")
    plt.close()


def save_decoded_data(medis_data, output_path):
    """Save decoded Medis data to JSON"""

    # Convert any numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__str__'):
            return str(obj)
        return obj

    def make_serializable(obj):
        """Recursively convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Fallback: convert to string
            return str(obj)

    serializable_data = make_serializable(medis_data)

    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)

    print(f"\nDecoded Medis data saved to: {output_path}")


if __name__ == "__main__":
    dicom_path = Path("samples/MID RCA PRE.dcm")

    if not dicom_path.exists():
        print(f"Error: {dicom_path} not found")
        exit(1)

    # Decode Medis private tags
    medis_data, dcm = decode_medis_tags(dicom_path)

    # Analyze coordinate arrays
    analyze_coordinate_arrays(medis_data, dcm)

    # Visualize contours
    visualize_contours(medis_data, dcm)

    # Save decoded data
    save_decoded_data(medis_data, "medis_decoded_data.json")

    print("\n" + "="*80)
    print("DECODING COMPLETE")
    print("="*80)
    print("\nKey findings:")
    print(f"  Calibration tags: {len(medis_data['calibration'])}")
    print(f"  Measurement tags: {len(medis_data['measurements'])}")
    print(f"  Frame selection tags: {len(medis_data['frame_selection'])}")
    print("\nNext steps:")
    print("  1. Review medis_contours_visualization.png to see if contours overlay correctly")
    print("  2. Check medis_decoded_data.json for detailed coordinate arrays")
    print("  3. Identify which arrays represent vessel contours vs. catheter calibration")
