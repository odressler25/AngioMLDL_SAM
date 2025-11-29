#!/usr/bin/env python3
"""
Extract analyzed frames from ALL patients
"""

import pydicom
from pathlib import Path
import numpy as np
from PIL import Image
import re

class AnalyzedFrameExtractorAll:
    """Extract analyzed frames using frame number from metadata"""

    def __init__(self):
        self.extracted_count = 0
        self.failed_count = 0
        self.patients_processed = 0

    def get_patient_id(self, path):
        """Extract patient ID from path (xxx-xxxx format)"""

        for part in path.parts:
            # Match standalone xxx-xxxx
            if re.match(r'^\d{3}-\d{4}$', part):
                return part
            # Match "ALL RISE xxx-xxxx" format
            match = re.search(r'(\d{3}-\d{4})', part)
            if match:
                return match.group(1)

        return path.parts[-2] if len(path.parts) >= 2 else "Unknown"

    def find_analyzed_frame_number(self, ds):
        """Find analyzed frame number from DICOM tags (0-indexed)"""

        # Check tag (7001,1040) - usually contains the analyzed frame
        if (0x7001, 0x1040) in ds:
            return int(ds[(0x7001, 0x1040)].value)

        # Fallback to (7001,1041)
        if (0x7001, 0x1041) in ds:
            return int(ds[(0x7001, 0x1041)].value)

        return None

    def extract_analyzed_frame(self, dcm_path, output_folder):
        """Extract the analyzed frame"""

        try:
            ds = pydicom.dcmread(dcm_path)

            # Get patient ID
            patient_id = self.get_patient_id(dcm_path)

            # Get analyzed frame number (0-indexed)
            frame_num = self.find_analyzed_frame_number(ds)

            if frame_num is None:
                return None

            # Get pixel data
            pixel_array = ds.pixel_array

            # Handle multi-frame vs single frame
            if len(pixel_array.shape) == 3:
                num_frames = pixel_array.shape[0]
            else:
                num_frames = 1
                pixel_array = np.expand_dims(pixel_array, 0)

            # Check frame is in range
            if frame_num >= num_frames:
                print(f"  [WARN] {dcm_path.name}: Frame {frame_num} out of range (max: {num_frames-1})")
                return None

            # Extract frame
            frame = pixel_array[frame_num]

            # Normalize to 0-255
            frame_norm = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(frame_norm)

            # Generate filename
            dcm_name = dcm_path.stem
            phase = self._get_phase(dcm_name)
            vessel = self._clean_vessel_name(dcm_name)

            # Include both 0-indexed and 1-indexed frame numbers
            output_filename = f"{patient_id}_{phase}_{vessel}_f{frame_num}_f1idx{frame_num+1}.png"
            output_path = output_folder / output_filename

            # Save
            img.save(output_path)

            print(f"  {patient_id}/{dcm_name} -> frame {frame_num} (1-indexed: {frame_num+1})")
            self.extracted_count += 1

            return output_filename

        except Exception as e:
            print(f"  [ERROR] {dcm_path.name}: {e}")
            self.failed_count += 1
            return None

    def _get_phase(self, dcm_name):
        """Determine phase from filename"""

        dcm_upper = dcm_name.upper()

        if "FINAL" in dcm_upper:
            return "FINAL"
        elif "POST" in dcm_upper:
            return "POST"
        elif "PRE" in dcm_upper:
            return "PRE"
        elif "EVENT" in dcm_upper:
            return "EVENT"

        return "ANALYSIS"

    def _clean_vessel_name(self, dcm_name):
        """Clean vessel name"""

        # Remove phase keywords
        vessel = dcm_name
        keywords = ["PRE", "POST", "FINAL", "EVENT", "STENT", "NEW ANALYSIS", "CORRECTED"]

        for keyword in keywords:
            vessel = vessel.replace(keyword, "").replace(keyword.lower(), "")

        # Clean up
        vessel = vessel.strip().strip('-').strip('_').strip()
        vessel = re.sub(r'\s+', '_', vessel)

        return vessel

    def process_patient_folder(self, patient_folder):
        """Process all DICOMs in patient folder"""

        patient_id = patient_folder.name

        # Find all DICOM files
        all_dicoms = list(patient_folder.rglob("*.dcm"))

        # Filter for PRE/POST/FINAL measurement files
        measurement_dcms = []
        for dcm in all_dicoms:
            name_upper = dcm.stem.upper()
            # Include PRE/POST/FINAL, exclude pure complication files
            if any(phase in name_upper for phase in ['PRE', 'POST', 'FINAL', 'EVENT']):
                # Exclude if it's ONLY a complication file
                if not (('DISSECTION' in name_upper or 'SPASM' in name_upper or 'THROMBUS' in name_upper)
                        and not any(p in name_upper for p in ['PRE', 'POST', 'FINAL'])):
                    measurement_dcms.append(dcm)

        if not measurement_dcms:
            return

        # Create output folder
        output_folder = patient_folder / "measurement_frames"
        output_folder.mkdir(exist_ok=True)

        # Extract analyzed frames
        for dcm_file in sorted(measurement_dcms):
            self.extract_analyzed_frame(dcm_file, output_folder)

    def process_all_patients(self, base_dir):
        """Process all patient folders"""

        base_path = Path(base_dir)

        # Find all folders with patient ID (xxx-xxxx or "ALL RISE xxx-xxxx")
        patient_folders = []
        for item in base_path.iterdir():
            if item.is_dir():
                # Match standalone xxx-xxxx OR "ALL RISE xxx-xxxx" format
                if re.match(r'^\d{3}-\d{4}$', item.name) or re.search(r'\d{3}-\d{4}', item.name):
                    patient_folders.append(item)

        patient_folders.sort()

        print(f"Extracting measurement frames from {len(patient_folders)} patients...")
        print("="*80)

        for patient_folder in patient_folders:
            patient_id = patient_folder.name

            # Check if folder has DICOMs
            dicoms = list(patient_folder.rglob("*.dcm"))
            if not dicoms:
                continue

            print(f"\n{patient_id}:")
            print("-"*80)

            self.process_patient_folder(patient_folder)
            self.patients_processed += 1

        print("\n" + "="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)
        print(f"Patients processed: {self.patients_processed}")
        print(f"Measurement frames extracted: {self.extracted_count}")
        print(f"Failed extractions: {self.failed_count}")
        print()
        print("NOTE: Frame numbers shown as both 0-indexed (DICOM) and 1-indexed (viewing software)")


def main():
    """Main execution"""

    print("="*80)
    print("EXTRACT ALL ANALYZED MEASUREMENT FRAMES")
    print("="*80)
    print()

    extractor = AnalyzedFrameExtractorAll()
    extractor.process_all_patients(r'E:\Angios')


if __name__ == "__main__":
    main()
