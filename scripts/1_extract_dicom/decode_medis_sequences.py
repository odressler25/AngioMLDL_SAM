#!/usr/bin/env python3
"""
Decode Medis sequence tags to find lesion contours and measurements
"""

import pydicom
from pathlib import Path
import json


def decode_sequences(dicom_path):
    """Decode all sequence tags including nested structures"""

    print(f"\n{'='*80}")
    print(f"DECODING DICOM SEQUENCES (INCLUDING MEDIS MEASUREMENTS)")
    print(f"File: {dicom_path}")
    print(f"{'='*80}\n")

    dcm = pydicom.dcmread(dicom_path)

    sequences_found = []

    def print_element(elem, indent=0):
        """Recursively print element including sequences"""
        prefix = "  " * indent
        tag_str = f"{elem.tag}"
        tag_name = elem.keyword if elem.keyword else f"Tag_{elem.tag}"

        if elem.VR == 'SQ':
            print(f"{prefix}{tag_str} SQ {tag_name}: Sequence with {len(elem.value)} items")
            sequences_found.append((elem.tag, tag_name, len(elem.value)))

            for i, item in enumerate(elem.value):
                print(f"{prefix}  Item {i}:")
                for sub_elem in item:
                    print_element(sub_elem, indent + 2)
        else:
            try:
                value = elem.value
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except:
                        value = f"<binary, {len(value)} bytes>"
                elif isinstance(value, (list, tuple)) and len(value) > 10:
                    value = f"Array[{len(value)}]: [{value[0]}, {value[1]}, ..., {value[-1]}]"

                # Truncate long strings
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."

                print(f"{prefix}{tag_str} {elem.VR:4s} {tag_name:40s}: {value}")
            except Exception as e:
                print(f"{prefix}{tag_str} {elem.VR:4s} {tag_name:40s}: <error: {e}>")

    # Scan all elements
    print("ALL SEQUENCE TAGS:")
    print("="*80)

    for elem in dcm:
        if elem.VR == 'SQ':
            print_element(elem)
            print()

    if not sequences_found:
        print("No sequence tags found in this DICOM file.\n")

    # Now look for specific Medis measurement tags in more detail
    print("\n" + "="*80)
    print("DETAILED MEDIS TAG SCAN (ALL GROUPS)")
    print("="*80)

    medis_groups = [0x7001, 0x7007, 0x7901, 0x7917, 0x7927, 0x7931, 0x7933, 0x7935, 0x7943]

    for group in medis_groups:
        print(f"\nGroup 0x{group:04X}:")
        found_in_group = False

        for elem in dcm:
            if elem.tag.group == group:
                found_in_group = True
                tag_str = f"{elem.tag}"
                tag_name = elem.keyword if elem.keyword else f"Tag_{elem.tag}"

                try:
                    if elem.VR == 'SQ':
                        print(f"  {tag_str} SQ {tag_name}: Sequence with {len(elem.value)} items")
                        for i, item in enumerate(elem.value):
                            print(f"    Item {i}:")
                            for sub_elem in item:
                                sub_tag_str = f"{sub_elem.tag}"
                                sub_tag_name = sub_elem.keyword if sub_elem.keyword else f"Tag_{sub_elem.tag}"
                                try:
                                    sub_value = sub_elem.value
                                    if isinstance(sub_value, (list, tuple)) and len(sub_value) > 5:
                                        sub_value = f"Array[{len(sub_value)}]"
                                    print(f"      {sub_tag_str} {sub_elem.VR:4s} {sub_tag_name:30s}: {sub_value}")
                                except:
                                    print(f"      {sub_tag_str} {sub_elem.VR:4s} {sub_tag_name:30s}: <error>")
                    else:
                        value = elem.value
                        if isinstance(value, (list, tuple)) and len(value) > 10:
                            value = f"Array[{len(value)}]: first={value[0]}, last={value[-1]}"
                        elif isinstance(value, bytes):
                            value = f"<binary, {len(value)} bytes>"
                        print(f"  {tag_str} {elem.VR:4s} {tag_name:40s}: {value}")
                except Exception as e:
                    print(f"  {tag_str} {elem.VR:4s} {tag_name:40s}: <error: {e}>")

        if not found_in_group:
            print(f"  (no tags in this group)")

    # Check for frame selection
    print("\n" + "="*80)
    print("SEARCHING FOR FRAME SELECTION")
    print("="*80)

    frame_keywords = ['frame', 'Frame', 'FRAME', 'select', 'Select', 'analysis', 'Analysis']

    for elem in dcm:
        tag_name = elem.keyword if elem.keyword else f"Tag_{elem.tag}"
        if any(keyword in str(tag_name) for keyword in frame_keywords):
            tag_str = f"{elem.tag}"
            try:
                value = elem.value
                print(f"{tag_str} {elem.VR:4s} {tag_name:40s}: {value}")
            except:
                pass

    print("\n" + "="*80)
    print(f"Total sequences found: {len(sequences_found)}")
    print("="*80)

    return sequences_found


if __name__ == "__main__":
    dicom_path = Path("samples/MID RCA PRE.dcm")

    if not dicom_path.exists():
        print(f"Error: {dicom_path} not found")
        exit(1)

    sequences_found = decode_sequences(dicom_path)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if sequences_found:
        print(f"\nFound {len(sequences_found)} sequence tags:")
        for tag, name, items in sequences_found:
            print(f"  {tag}: {name} ({items} items)")
    else:
        print("\nNo sequence tags found.")
        print("\nThis DICOM appears to contain only catheter calibration data,")
        print("not lesion measurements or vessel contour tracings.")
        print("\nPossible reasons:")
        print("  1. This is a 'PRE' image - analysis may be in 'POST' images")
        print("  2. Medis analysis stored in separate files")
        print("  3. Need to check other DICOM files in the dataset")
