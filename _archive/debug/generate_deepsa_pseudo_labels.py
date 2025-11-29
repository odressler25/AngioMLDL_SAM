"""
Generate DeepSA pseudo-labels for Stage 1 training

This script:
1. Loads all cases from corrected_dataset_training.csv
2. For each case, uses pretrained DeepSA to segment the full vessel tree
3. Saves the full vessel masks as pseudo-labels for SAM 3 training
"""

import sys
sys.path.append("DeepSA/")

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from DeepSA.models import UNet
from DeepSA.datasets import tophat


def load_deepsa(ckpt_path="DeepSA/ckpt/fscad_36249.ckpt", device='cuda'):
    """Load pretrained DeepSA model"""
    print(f"Loading DeepSA from {ckpt_path}...")
    model = UNet(1, 1, 32, bilinear=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['netE'].items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print("[OK] DeepSA loaded (frozen)")
    return model


def generate_pseudo_label(deepsa_model, frame, device='cuda'):
    """
    Generate full vessel tree mask using DeepSA

    Args:
        deepsa_model: Pretrained DeepSA
        frame: Numpy array (H, W), grayscale
        device: 'cuda' or 'cpu'

    Returns:
        vessel_mask: Binary mask (H, W), 0=background, 1=vessel
    """
    # Prepare image for DeepSA
    frame_pil = Image.fromarray(frame).convert('L')

    # DeepSA preprocessing
    transform = T.Compose([
        T.Resize(512),
        T.Lambda(lambda img: tophat(img, 50)),
        T.ToTensor(),
        T.Normalize((0.5), (0.5))
    ])

    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

    # DeepSA inference
    with torch.no_grad():
        pred = deepsa_model(frame_tensor)
        vessel_mask = (torch.sign(pred) > 0).cpu().numpy()[0, 0].astype(np.uint8)

    return vessel_mask


def main():
    """
    Generate DeepSA pseudo-labels for all cases
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Paths
    csv_path = "E:/AngioMLDL_data/corrected_dataset_training.csv"
    output_dir = Path("E:/AngioMLDL_data/deepsa_pseudo_labels")
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_dir}\n")

    # Load DeepSA
    deepsa_model = load_deepsa(device=device)

    # Load dataset
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total cases: {len(df)}\n")

    # Generate pseudo-labels
    print("Generating DeepSA pseudo-labels...")
    print("="*70)

    successful = 0
    failed = 0

    # Track processed cines to avoid duplicates
    processed_cines = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        cine_path = row['cine_path']

        # Skip if we already processed this exact cine file
        if cine_path in processed_cines:
            continue

        # Extract unique ID from cine filename
        # E.g., "101-0025_MID_RCA_FINAL_cine.npy" -> "101-0025_MID_RCA_FINAL"
        cine_filename = Path(cine_path).stem  # Remove .npy
        unique_id = cine_filename.replace('_cine', '')  # Remove _cine suffix

        frame_idx = int(row['frame_index'])

        try:
            # Load cine
            cine = np.load(cine_path)

            # Get correct frame
            if frame_idx >= len(cine):
                frame_idx = len(cine) - 1

            frame = cine[frame_idx]

            # Normalize to [0, 255]
            if frame.dtype != np.uint8:
                frame = (frame / frame.max() * 255).astype(np.uint8)

            # Generate pseudo-label with DeepSA
            vessel_mask = generate_pseudo_label(deepsa_model, frame, device)

            # Save with unique ID
            output_path = output_dir / f"{unique_id}_full_vessel_mask.npy"
            np.save(output_path, vessel_mask)

            processed_cines.add(cine_path)
            successful += 1

        except Exception as e:
            print(f"\nError processing {unique_id}: {e}")
            failed += 1
            continue

    print("\n" + "="*70)
    print("PSEUDO-LABEL GENERATION COMPLETE")
    print("="*70)
    print(f"Successful: {successful}/{len(df)}")
    print(f"Failed: {failed}/{len(df)}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Create index file mapping each CSV row to its pseudo-label
    index_path = output_dir / "index.csv"
    df_index = df.copy()

    # Map each row to its corresponding pseudo-label using cine path
    def get_pseudo_label_path(cine_path):
        cine_filename = Path(cine_path).stem
        unique_id = cine_filename.replace('_cine', '')
        return str(output_dir / f"{unique_id}_full_vessel_mask.npy")

    df_index['deepsa_pseudo_label_path'] = df_index['cine_path'].apply(get_pseudo_label_path)
    df_index.to_csv(index_path, index=False)
    print(f"\nIndex file saved: {index_path}")


if __name__ == '__main__':
    main()
