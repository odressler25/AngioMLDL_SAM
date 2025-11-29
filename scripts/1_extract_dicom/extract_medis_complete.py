#!/usr/bin/env python3
"""
Complete Medis PDF Data Extractor
Extracts both images AND measurements from Medis analysis PDFs
"""

import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io
import re
import pandas as pd
import json

class MedisPDFExtractor:
    """Extract images and measurements from Medis PDFs"""
    
    def __init__(self, pdf_dir, output_dir):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.data_dir = self.output_dir / "measurements"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract all text from PDF"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            return full_text
        except Exception as e:
            print(f"Error extracting text from {pdf_path.name}: {e}")
            return ""
    
    def parse_measurements(self, text, pdf_name):
        """Parse key measurements from Medis PDF text"""
        
        measurements = {
            'pdf_name': pdf_name,
            'vessel': None,
            'phase': None,
            'frame': None,
            'view_angle': None,
            'calibration_method': None,
            'calibration_factor': None,
            'catheter_size': None,
            'mld': None,
            'reference_diameter': None,
            'stenosis_diameter_percent': None,
            'stenosis_area_percent': None,
            'lesion_length': None,
            'mean_diameter': None,
            'segment_length': None,
            'plaque_area': None,
        }
        
        # Extract vessel and phase from filename
        # e.g., "MID RCA PRE.pdf" → vessel="MID RCA", phase="PRE"
        name_parts = pdf_name.replace('.pdf', '').split()
        if len(name_parts) >= 2:
            measurements['phase'] = name_parts[-1]  # PRE, POST, FINAL
            measurements['vessel'] = ' '.join(name_parts[:-1])  # MID RCA
        
        # Frame number: "37 of 66 (37)"
        frame_match = re.search(r'Frame.*?(\d+)\s+of\s+(\d+)', text)
        if frame_match:
            measurements['frame'] = f"{frame_match.group(1)} of {frame_match.group(2)}"
        
        # View angle: "17 LAO, 4 CAU"
        view_match = re.search(r'(\d+)\s+(LAO|RAO)[,\s]+(\d+)\s+(CRA|CAU)', text)
        if view_match:
            measurements['view_angle'] = f"{view_match.group(1)} {view_match.group(2)}, {view_match.group(3)} {view_match.group(4)}"
        
        # Calibration
        cal_method = re.search(r'Calibration Method\s+(\w+)', text)
        if cal_method:
            measurements['calibration_method'] = cal_method.group(1)
        
        # Calibration - search for the value pattern directly
        cal_factor = re.search(r'([\d.]+)\s+mm/pixel', text)
        if cal_factor:
            measurements['calibration_factor'] = float(cal_factor.group(1))
        
        # Catheter size - search for the pattern directly
        catheter = re.search(r'([\d.]+)\s+mm\s+\(([\d.]+)\s+F\)', text)
        if catheter:
            measurements['catheter_size'] = f"{catheter.group(2)}F ({catheter.group(1)}mm)"
        if catheter:
            measurements['catheter_size'] = f"{catheter.group(2)}F ({catheter.group(1)}mm)"
        
        # Key measurements from the table
        # Look for "Obstruction" row in the main table
        
        # Minimal Diameter (MLD)
        mld_match = re.search(r'Obstruction.*?Minimal\s+Diameter.*?([\d.]+)', text, re.DOTALL)
        if mld_match:
            measurements['mld'] = float(mld_match.group(1))
        
        # Reference Diameter
        ref_match = re.search(r'Reference.*?Diameter.*?([\d.]+)', text, re.DOTALL)
        if ref_match:
            measurements['reference_diameter'] = float(ref_match.group(1))
        
        # % Stenosis (both diameter and area)
        stenosis_match = re.search(r'%Stenosis\s+([\d.]+)\s+([\d.]+)', text)
        if stenosis_match:
            measurements['stenosis_diameter_percent'] = float(stenosis_match.group(1))
            measurements['stenosis_area_percent'] = float(stenosis_match.group(2))
        
        # Lesion Length
        length_match = re.search(r'Obstruction.*?Length.*?([\d.]+)', text, re.DOTALL)
        if length_match:
            measurements['lesion_length'] = float(length_match.group(1))
        
        # Mean Diameter (Obstruction segment)
        mean_match = re.search(r'Obstruction.*?Mean\s+Diameter.*?([\d.]+)', text, re.DOTALL)
        if mean_match:
            measurements['mean_diameter'] = float(mean_match.group(1))
        
        # Segment Length
        seg_match = re.search(r'Obstruction.*?Segment\s+Length.*?([\d.]+)', text, re.DOTALL)
        if seg_match:
            measurements['segment_length'] = float(seg_match.group(1))
        
        # Plaque Area
        plaque_match = re.search(r'Plaque.*?([\d.]+)', text)
        if plaque_match:
            measurements['plaque_area'] = float(plaque_match.group(1))
        
        return measurements
    
    def extract_angiogram_image(self, pdf_path):
        """Extract the main angiogram image from PDF"""
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size
                    
                    # Angiogram is typically 1000-1600 pixels
                    if 1000 <= width <= 1700 and 1000 <= height <= 1700:
                        doc.close()
                        return image, page_num
            
            doc.close()
            return None, None
            
        except Exception as e:
            print(f"Error extracting image from {pdf_path.name}: {e}")
            return None, None
    
    def process_pdf(self, pdf_path):
        """Process a single Medis PDF: extract image and measurements"""
        
        print(f"Processing: {pdf_path.name}")
        
        # Extract measurements
        text = self.extract_text_from_pdf(pdf_path)
        measurements = self.parse_measurements(text, pdf_path.name)
        
        # Extract angiogram image
        image, page_num = self.extract_angiogram_image(pdf_path)
        
        if image:
            # Save image
            # Extract patient ID from path (look for pattern like "101-0025")
            patient_id = "UNKNOWN"
            for part in pdf_path.parts:
                if part.startswith("ALL RISE"):
                    patient_id = part.replace("ALL RISE ", "")
                    break
            vessel = measurements['vessel'] or "UNKNOWN"
            phase = measurements['phase'] or "UNKNOWN"
            
            image_name = f"{patient_id}_{vessel}_{phase}.png".replace(" ", "_")
            image_path = self.images_dir / image_name
            image.save(image_path)
            
            measurements['image_file'] = image_name
            measurements['image_size'] = f"{image.size[0]}x{image.size[1]}"
            measurements['patient_id'] = patient_id
            
            print(f"  ✓ Image: {image_name} ({image.size})")
            print(f"  ✓ MLD: {measurements['mld']} mm")
            print(f"  ✓ Stenosis: {measurements['stenosis_diameter_percent']}%")
        else:
            print(f"  ✗ No angiogram image found")
        
        # Save measurements as JSON
        json_name = f"{patient_id}_{vessel}_{phase}.json".replace(" ", "_")
        json_path = self.data_dir / json_name
        with open(json_path, 'w') as f:
            json.dump(measurements, f, indent=2)
        
        self.results.append(measurements)
        
        return measurements
    
    def process_all_pdfs(self):
        """Process all Medis PDFs"""
        
        # Find Analysis PDFs only (not TWS, not CRFs)
        pdf_files = list(self.pdf_dir.rglob("*.pdf"))
        analysis_pdfs = [p for p in pdf_files if "Analysis" in str(p) and "PROCEDURES" in str(p)]
        
        # Filter out TWS forms
        medis_pdfs = [p for p in analysis_pdfs if "Worksheet" not in p.name and "TWS" not in p.name]
        
        print(f"Found {len(medis_pdfs)} Medis analysis PDFs")
        print(f"Processing...\n")
        
        for pdf_file in medis_pdfs:
            self.process_pdf(pdf_file)
            print()
        
        # Save summary CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "medis_dataset.csv"
        df.to_csv(csv_path, index=False)
        
        print("="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)
        print(f"Images extracted: {len([r for r in self.results if r.get('image_file')])}") 
        print(f"Measurements extracted: {len(self.results)}")
        print(f"\nOutput:")
        print(f"  Images: {self.images_dir}")
        print(f"  Measurements (JSON): {self.data_dir}")
        print(f"  Summary CSV: {csv_path}")
        
        return df


def main():
    """Main execution"""
    
    pdf_dir = Path(r"E:\Angios")
    output_dir = Path(r"C:\Users\odressler\AngioMLDL\data\medis_complete")
    
    extractor = MedisPDFExtractor(pdf_dir, output_dir)
    df = extractor.process_all_pdfs()
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total cases: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Vessels: {df['vessel'].value_counts().to_dict()}")
    print(f"Phases: {df['phase'].value_counts().to_dict()}")
    print(f"\nMeasurements available:")
    print(f"  MLD: {df['mld'].notna().sum()} / {len(df)}")
    print(f"  Stenosis: {df['stenosis_diameter_percent'].notna().sum()} / {len(df)}")
    print(f"  Calibration: {df['calibration_factor'].notna().sum()} / {len(df)}")


if __name__ == "__main__":
    main()