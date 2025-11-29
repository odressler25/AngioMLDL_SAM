#!/usr/bin/env python3
"""
Coronary Angiography Pipeline Runner
=====================================

Complete pipeline from DICOM source to COCO dataset ready for training.

Usage:
    # Process new patients
    python scripts/run_pipeline.py --source E:/Angios_new --output E:/AngioMLDL_data/batch2

    # Test on specific patients (inference only)
    python scripts/run_pipeline.py --source E:/Angios_new --output E:/AngioMLDL_data/batch2 --inference-only

    # Dry run (process 5 samples)
    python scripts/run_pipeline.py --source E:/Angios_new --output E:/AngioMLDL_data/batch2 --dry-run

Pipeline Steps:
    1. Extract vessel contours from DICOM (Medis annotations)
    2. Generate vessel masks with DeepSA
    3. Create CASS-labeled COCO dataset
    4. (Optional) Run inference with trained model
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_step(name, cmd, dry_run=False):
    """Run a pipeline step with logging."""
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True

    try:
        result = subprocess.run(cmd, check=True)
        print(f"[OK] {name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {name} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Coronary Angiography Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Source directory containing patient DICOM folders (e.g., E:/Angios_new)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for processed data (e.g., E:/AngioMLDL_data/batch2)"
    )

    # Optional arguments
    parser.add_argument(
        "--deepsa-masks",
        help="Path to existing DeepSA masks (skip mask generation if provided)"
    )
    parser.add_argument(
        "--model-checkpoint",
        default="E:/AngioMLDL_data/experiments/stage2_bifurcation_v7/checkpoints/best.pth",
        help="Model checkpoint for inference"
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Only run inference, don't create training dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip DICOM extraction (use existing contours)"
    )
    parser.add_argument(
        "--skip-masks",
        action="store_true",
        help="Skip DeepSA mask generation"
    )
    parser.add_argument(
        "--skip-coco",
        action="store_true",
        help="Skip COCO dataset creation"
    )

    args = parser.parse_args()

    # Resolve paths
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()
    scripts_dir = Path(__file__).parent.resolve()
    repo_dir = scripts_dir.parent

    # Create output directories
    contours_dir = output_dir / "contours"
    masks_dir = output_dir / "vessel_masks"
    coco_dir = output_dir / "coco_dataset"
    inference_dir = output_dir / "inference_results"

    print("=" * 70)
    print("CORONARY ANGIOGRAPHY PIPELINE")
    print("=" * 70)
    print(f"Source:     {source_dir}")
    print(f"Output:     {output_dir}")
    print(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode:       {'Inference Only' if args.inference_only else 'Full Pipeline'}")

    # Validate source directory
    if not source_dir.exists():
        print(f"\n[ERROR] Source directory does not exist: {source_dir}")
        sys.exit(1)

    # Count patients
    patient_dirs = list(source_dir.glob("ALL RISE *"))
    print(f"\nPatients found: {len(patient_dirs)}")

    if len(patient_dirs) == 0:
        print("[ERROR] No patient directories found (expected 'ALL RISE *' pattern)")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_completed = []
    steps_failed = []

    # =========================================================================
    # STEP 1: Extract vessel contours from DICOM
    # =========================================================================
    if not args.skip_extraction:
        cmd = [
            sys.executable,
            str(scripts_dir / "1_extract_dicom" / "extract_vessel_contours.py"),
            "--source", str(source_dir),
            "--output", str(contours_dir),
        ]
        if args.dry_run:
            cmd.append("--dry-run")

        success = run_step("Extract Vessel Contours from DICOM", cmd, dry_run=args.dry_run)
        (steps_completed if success else steps_failed).append("DICOM Extraction")
    else:
        print("\n[SKIP] DICOM extraction (--skip-extraction)")

    # =========================================================================
    # STEP 2: Generate DeepSA vessel masks
    # =========================================================================
    if not args.skip_masks and not args.deepsa_masks:
        cmd = [
            sys.executable,
            str(repo_dir / "DeepSA" / "inference.py"),
            "--input", str(contours_dir / "images"),
            "--output", str(masks_dir),
        ]

        success = run_step("Generate DeepSA Vessel Masks", cmd, dry_run=args.dry_run)
        (steps_completed if success else steps_failed).append("DeepSA Masks")
    elif args.deepsa_masks:
        masks_dir = Path(args.deepsa_masks)
        print(f"\n[SKIP] Using existing DeepSA masks: {masks_dir}")
    else:
        print("\n[SKIP] DeepSA mask generation (--skip-masks)")

    # =========================================================================
    # STEP 3: Create CASS-labeled COCO dataset
    # =========================================================================
    if not args.inference_only and not args.skip_coco:
        cmd = [
            sys.executable,
            str(scripts_dir / "3_cass_labels" / "create_medis_bifurcation_labels.py"),
            "--contours", str(contours_dir),
            "--masks", str(masks_dir),
            "--output", str(coco_dir),
        ]

        success = run_step("Create CASS-labeled COCO Dataset", cmd, dry_run=args.dry_run)
        (steps_completed if success else steps_failed).append("COCO Dataset")
    else:
        print("\n[SKIP] COCO dataset creation")

    # =========================================================================
    # STEP 4: Run inference (optional)
    # =========================================================================
    if args.inference_only or Path(args.model_checkpoint).exists():
        inference_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(scripts_dir / "4_visualization" / "visualize_predictions.py"),
            "--images", str(contours_dir / "images"),
            "--checkpoint", str(args.model_checkpoint),
            "--output", str(inference_dir),
        ]

        success = run_step("Run Model Inference", cmd, dry_run=args.dry_run)
        (steps_completed if success else steps_failed).append("Inference")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    if steps_completed:
        print(f"\nCompleted steps: {', '.join(steps_completed)}")
    if steps_failed:
        print(f"Failed steps: {', '.join(steps_failed)}")

    print(f"\nOutput directory: {output_dir}")
    print(f"  - Contours:  {contours_dir}")
    print(f"  - Masks:     {masks_dir}")
    if not args.inference_only:
        print(f"  - COCO:      {coco_dir}")
    print(f"  - Inference: {inference_dir}")

    if steps_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
