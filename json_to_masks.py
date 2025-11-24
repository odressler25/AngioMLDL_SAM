"""
Convert Medis CoreLab JSON contours to binary masks for ground truth.

This module provides utilities to:
1. Parse JSON measurement files from CoreLab analysis
2. Convert left_edge + right_edge contours to binary masks
3. Generate ground truth masks for training/validation

The JSON files contain pixel-level vessel contours that serve as expert labels.
"""

import json
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_measurements(json_path: str) -> Dict:
    """
    Load Medis CoreLab JSON measurement file.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary containing measurement data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def json_contours_to_mask(
    json_data: Dict,
    image_shape: Tuple[int, int],
    use_ideal: bool = False
) -> np.ndarray:
    """
    Convert JSON left_edge + right_edge to binary mask.

    This creates ground truth masks by:
    1. Extracting left_edge and right_edge pixel coordinates
    2. Creating a polygon from left + reversed right edges
    3. Filling the polygon to create a binary mask

    Args:
        json_data: Parsed JSON measurement data
        image_shape: (height, width) of the target image
        use_ideal: If True, use ideal_left_edge/ideal_right_edge instead of actual edges

    Returns:
        Binary mask (height, width) with vessel pixels = 1
    """
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Choose which edges to use
    if use_ideal:
        left_key = 'ideal_left_edge'
        right_key = 'ideal_right_edge'
    else:
        left_key = 'left_edge'
        right_key = 'right_edge'

    # Extract edges
    if left_key not in json_data or right_key not in json_data:
        logger.warning(f"JSON missing {left_key} or {right_key}")
        return mask

    left_edge = np.array(json_data[left_key])
    right_edge = np.array(json_data[right_key])

    if len(left_edge) == 0 or len(right_edge) == 0:
        logger.warning("Empty edges in JSON")
        return mask

    # Create closed polygon: left edge + reversed right edge
    polygon = np.vstack([left_edge, right_edge[::-1]])

    # Ensure polygon is in correct format for cv2.fillPoly
    polygon = polygon.astype(np.int32).reshape((-1, 1, 2))

    # Fill polygon on mask
    cv2.fillPoly(mask, [polygon], 1)

    return mask


def extract_vessel_info(json_data: Dict) -> Dict:
    """
    Extract vessel identification and CASS segment information.

    Args:
        json_data: Parsed JSON measurement data

    Returns:
        Dictionary with vessel metadata:
        - segment_name: e.g., "RCA Mid", "LAD Prox"
        - vessel: e.g., "RCA", "LAD", "LCX"
        - cass_segment: CASS segment label
        - view_angles: (primary, secondary) angles
    """
    info = {
        'segment_name': json_data.get('segment_name', 'Unknown'),
        'vessel': 'Unknown',
        'cass_segment': 'Unknown',
        'view_angles': None
    }

    # Parse vessel from segment name
    segment = info['segment_name'].upper()
    if 'RCA' in segment:
        info['vessel'] = 'RCA'
    elif 'LAD' in segment:
        info['vessel'] = 'LAD'
    elif 'LCX' in segment or 'CX' in segment:
        info['vessel'] = 'LCX'
    elif 'LM' in segment:
        info['vessel'] = 'LM'
    elif 'DIAG' in segment or 'D1' in segment or 'D2' in segment:
        info['vessel'] = 'Diagonal'
    elif 'OM' in segment:
        info['vessel'] = 'OM'

    # Extract CASS position (prox, mid, dis)
    if 'PROX' in segment:
        info['cass_segment'] = f"{info['vessel']} Prox"
    elif 'MID' in segment:
        info['cass_segment'] = f"{info['vessel']} Mid"
    elif 'DIS' in segment or 'DIST' in segment:
        info['cass_segment'] = f"{info['vessel']} Dis"

    # Extract view angles if available
    if 'view_angles' in json_data:
        va = json_data['view_angles']
        info['view_angles'] = (va.get('primary_angle'), va.get('secondary_angle'))

    return info


def extract_measurements(json_data: Dict) -> Dict:
    """
    Extract QCA measurements for validation.

    Args:
        json_data: Parsed JSON measurement data

    Returns:
        Dictionary with:
        - mld_mm: Minimum lumen diameter in mm
        - interpolated_rvd_mm: Interpolated reference vessel diameter
        - stenosis_pct: Percent diameter stenosis
        - mld_position: Pixel coordinates of MLD (x, y)
        - lesion_length_mm: Length of stenotic lesion
    """
    measurements = {
        'mld_mm': None,
        'interpolated_rvd_mm': None,
        'stenosis_pct': None,
        'mld_position': None,
        'lesion_length_mm': None
    }

    if 'measurements' in json_data:
        meas = json_data['measurements']
        measurements['mld_mm'] = meas.get('MLD_mm')
        measurements['interpolated_rvd_mm'] = meas.get('interpolated_RVD_mm')
        measurements['stenosis_pct'] = meas.get('diameter_stenosis_pct')
        measurements['lesion_length_mm'] = meas.get('lesion_length_mm')

        # Extract MLD pixel coordinates
        mld_x = meas.get('MLD_x_coord')
        mld_y = meas.get('MLD_y_coord')
        if mld_x is not None and mld_y is not None:
            measurements['mld_position'] = (mld_x, mld_y)

    return measurements


def visualize_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.3
) -> np.ndarray:
    """
    Create visualization of mask overlaid on image.

    Args:
        image: Original image (H, W) or (H, W, 3)
        mask: Binary mask (H, W)
        color: RGB color for mask overlay
        alpha: Transparency (0=transparent, 1=opaque)

    Returns:
        RGB image with mask overlay
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = image.copy()

    # Ensure mask matches image shape
    if mask.shape != vis_image.shape[:2]:
        # Resize mask to match image
        mask = cv2.resize(mask.astype(np.uint8), (vis_image.shape[1], vis_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ensure mask is 2D
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]

    # Create colored mask
    colored_mask = np.zeros_like(vis_image)
    mask_bool = (mask > 0)

    # Apply color to all RGB channels where mask is true
    colored_mask[mask_bool, 0] = color[0]  # R
    colored_mask[mask_bool, 1] = color[1]  # G
    colored_mask[mask_bool, 2] = color[2]  # B

    # Blend
    overlay = cv2.addWeighted(vis_image, 1-alpha, colored_mask, alpha, 0)

    # Draw contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


def get_diverse_test_cases(csv_path: str, n_cases: int = 5) -> List[Dict]:
    """
    Select diverse test cases from dataset.

    Selects cases to cover:
    - Different vessels (RCA, LAD, LCX)
    - Different view angles
    - Different image characteristics

    Args:
        csv_path: Path to corrected_dataset.csv
        n_cases: Number of test cases to select

    Returns:
        List of dicts with cine_path, contours_path, vessel info
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Target diverse vessels - use main_vessel column
    target_vessels = ['RCA', 'LAD', 'LCX']

    selected = []
    for vessel in target_vessels:
        # Find cases with this vessel
        vessel_cases = df[df['main_vessel'] == vessel]

        if len(vessel_cases) > 0:
            # Pick first PRE case (cleaner than FINAL which may have stents)
            pre_cases = vessel_cases[vessel_cases['phase'] == 'PRE']
            if len(pre_cases) > 0:
                selected.append(pre_cases.iloc[0].to_dict())
            else:
                selected.append(vessel_cases.iloc[0].to_dict())

        if len(selected) >= n_cases:
            break

    return selected[:n_cases]


if __name__ == '__main__':
    # Test the utilities
    import sys

    if len(sys.argv) < 3:
        print("Usage: python json_to_masks.py <json_path> <output_mask_path>")
        print("\nTest mode: Creates masks from a sample JSON")

        # Get sample from CSV
        csv_path = r"E:\AngioMLDL_data\corrected_dataset.csv"
        test_cases = get_diverse_test_cases(csv_path, n_cases=1)

        if test_cases:
            case = test_cases[0]
            print(f"\nTesting with: {case['patient_id']} - {case['vessel_pattern']} {case['phase']}")
            print(f"JSON: {case['contours_path']}")

            # Load JSON
            json_data = load_json_measurements(case['contours_path'])

            # Extract info
            vessel_info = extract_vessel_info(json_data)
            measurements = extract_measurements(json_data)

            print(f"\nVessel: {vessel_info['vessel']}")
            print(f"CASS: {vessel_info['cass_segment']}")
            print(f"View angles: {vessel_info['view_angles']}")
            print(f"MLD: {measurements['mld_mm']:.3f} mm")
            print(f"Stenosis: {measurements['stenosis_pct']:.1f}%")

            # Get image resolution from case
            resolution = case['resolution'].split('x')
            image_shape = (int(resolution[0]), int(resolution[1]))

            # Create mask
            mask = json_contours_to_mask(json_data, image_shape=image_shape)
            print(f"\nMask created: {mask.shape}, {mask.sum()} vessel pixels")

        sys.exit(0)

    # Normal usage: convert specific JSON to mask
    json_path = sys.argv[1]
    output_path = sys.argv[2]

    json_data = load_json_measurements(json_path)
    mask = json_contours_to_mask(json_data, image_shape=(512, 512))

    # Save mask
    cv2.imwrite(output_path, mask * 255)
    print(f"Mask saved to {output_path}")
