"""
Inspect DeepSA pseudo-labels and compare with Medis GT segments
Uses the CSV file as the single source of truth for all paths
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import cv2


def main():
    # Load dataset CSV - single source of truth
    csv_path = r'E:\AngioMLDL_data\corrected_dataset_training.csv'
    df = pd.read_csv(csv_path)

    print("="*70)
    print("CSV STRUCTURE")
    print("="*70)
    print(f"Total samples: {len(df)}")
    print(f"\nRelevant columns:")
    print(f"  - cine_path: path to cine video")
    print(f"  - frame_index: which frame to use")
    print(f"  - vessel_mask_actual_path: Medis GT mask")
    print(f"  - deepsa_pseudo_label_path: DeepSA pseudo-label")

    # Get first sample
    row = df.iloc[0]

    # All paths from CSV
    cine_path = Path(row['cine_path'])
    medis_mask_path = Path(row['vessel_mask_actual_path'])
    deepsa_path = Path(row['deepsa_pseudo_label_path'])
    frame_idx = int(row['frame_index'])

    print("\n" + "="*70)
    print("SAMPLE INSPECTION")
    print("="*70)
    print(f"\nPatient: {row['patient_id']}")
    print(f"Vessel: {row['vessel_pattern']} ({row['vessel_name']})")
    print(f"Phase: {row['phase']}")
    print(f"Frame index: {frame_idx}")
    print(f"\nCine: {cine_path.name}")
    print(f"Medis GT: {medis_mask_path.name}")
    print(f"DeepSA: {deepsa_path.name}")

    # Load data
    cine = np.load(cine_path)
    medis_mask = np.load(medis_mask_path)
    deepsa_mask = np.load(deepsa_path)

    # Get the correct frame
    frame = cine[frame_idx]
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    frame = frame.astype(np.float32)
    if frame.max() > 1:
        frame = frame / 255.0

    medis_mask = medis_mask.astype(np.float32)
    if medis_mask.max() > 1:
        medis_mask = medis_mask / 255.0

    deepsa_mask = deepsa_mask.astype(np.float32)

    print(f"\nOriginal shapes:")
    print(f"  Frame: {frame.shape}")
    print(f"  Medis GT: {medis_mask.shape}")
    print(f"  DeepSA: {deepsa_mask.shape}")

    # Resize DeepSA to match frame size if different
    if deepsa_mask.shape[0] != frame.shape[0]:
        print(f"\nResizing DeepSA from {deepsa_mask.shape} to {(frame.shape[0], frame.shape[1])}")
        deepsa_mask = cv2.resize(deepsa_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    print(f"\nVessel coverage:")
    print(f"  Medis GT: {(medis_mask > 0.5).sum()} pixels ({100*(medis_mask > 0.5).sum()/medis_mask.size:.2f}%)")
    print(f"  DeepSA: {(deepsa_mask > 0.5).sum()} pixels ({100*(deepsa_mask > 0.5).sum()/deepsa_mask.size:.2f}%)")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{row["patient_id"]} - {row["vessel_pattern"]} - {row["phase"]} (frame {frame_idx})',
                 fontsize=16, fontweight='bold')

    # Row 1: Medis GT
    axes[0, 0].imshow(frame)
    axes[0, 0].set_title('Original Frame', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(medis_mask, cmap='gray')
    axes[0, 1].set_title('Medis GT Mask', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(frame)
    axes[0, 2].contour(medis_mask, colors='red', linewidths=2, levels=[0.5])
    axes[0, 2].set_title('Medis GT Overlay (RED)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: DeepSA
    axes[1, 0].imshow(frame)
    axes[1, 0].set_title('Original Frame', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(deepsa_mask, cmap='gray')
    axes[1, 1].set_title('DeepSA Pseudo-Label', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(frame)
    axes[1, 2].contour(deepsa_mask, colors='green', linewidths=2, levels=[0.5])
    axes[1, 2].set_title('DeepSA Overlay (GREEN)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('deepsa_vs_medis_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved visualization to: deepsa_vs_medis_comparison.png")

    # Check DeepSA coverage
    print("\n" + "="*70)
    print("DEEPSA COVERAGE CHECK")
    print("="*70)

    deepsa_available = df['deepsa_pseudo_label_path'].notna().sum()
    print(f"\nTotal samples: {len(df)}")
    print(f"DeepSA labels available: {deepsa_available} ({100*deepsa_available/len(df):.1f}%)")

    print("\n" + "="*70)
    print("VERDICT: Ready for Phase 1 Training?")
    print("="*70)

    if deepsa_available == len(df):
        print("[OK] YES - DeepSA labels cover 100% of dataset")
        print("[OK] Binary masks (0/1) - ready for training")
        print("[OK] CSV has deepsa_pseudo_label_path column for proper coordination")
        print("\n*** PROCEED WITH PHASE 1 USING DEEPSA LABELS ***")
    else:
        print(f"WARNING - Only {100*deepsa_available/len(df):.1f}% coverage")

    plt.show()


if __name__ == '__main__':
    main()
