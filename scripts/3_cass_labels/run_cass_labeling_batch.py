"""
Batch CASS Labeling: Run on all available images and generate report.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
import sys

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cass_anatomical_labeling import (
    load_dinov2_model,
    visualize_cass_labeling,
    CASS_IMAGES,
    DEEPSA_MASKS,
    OUTPUT_DIR
)

# Results directory
RESULTS_DIR = Path("E:/AngioMLDL_data/cass_labeling")
RESULTS_DIR.mkdir(exist_ok=True)


def run_batch_labeling(max_images=None):
    """Run CASS labeling on all available images."""

    print("=" * 70)
    print("CASS Anatomical Labeling - Batch Processing")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load model
    model = load_dinov2_model()

    # Find all mask files
    mask_files = sorted(DEEPSA_MASKS.glob("*_full_vessel_mask.npy"))
    print(f"Found {len(mask_files)} vessel masks")

    # Match with images
    valid_pairs = []
    for mask_path in mask_files:
        base_name = mask_path.stem.replace("_full_vessel_mask", "")
        image_path = CASS_IMAGES / f"{base_name}.png"
        if image_path.exists():
            valid_pairs.append((image_path, mask_path, base_name))

    print(f"Found {len(valid_pairs)} valid image-mask pairs")

    if max_images:
        valid_pairs = valid_pairs[:max_images]
        print(f"Processing first {max_images} images")

    print()

    # Process all images
    results = []
    success_count = 0
    error_count = 0

    for i, (image_path, mask_path, base_name) in enumerate(valid_pairs):
        print(f"[{i+1}/{len(valid_pairs)}] {base_name}...", end=" ")

        output_path = RESULTS_DIR / f"cass_{base_name}.png"

        try:
            result = visualize_cass_labeling(image_path, mask_path, output_path, model)
            result["image_name"] = base_name
            result["success"] = True
            results.append(result)
            success_count += 1
            print(f"OK - {result['vessel_type']}, {result['n_bifurcations']} bifs")

        except Exception as e:
            results.append({
                "image_name": base_name,
                "success": False,
                "error": str(e)
            })
            error_count += 1
            print(f"ERROR: {e}")

    # Generate summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total processed: {len(valid_pairs)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print()

    # Statistics by vessel type
    vessel_stats = {}
    for r in results:
        if r.get("success"):
            vtype = r["vessel_type"]
            if vtype not in vessel_stats:
                vessel_stats[vtype] = {"count": 0, "total_bifs": 0, "segments": []}
            vessel_stats[vtype]["count"] += 1
            vessel_stats[vtype]["total_bifs"] += r["n_bifurcations"]
            vessel_stats[vtype]["segments"].extend([s[2] for s in r["segments"]])

    print("By Vessel Type:")
    for vtype, stats in vessel_stats.items():
        avg_bifs = stats["total_bifs"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {vtype}: {stats['count']} images, avg {avg_bifs:.1f} bifurcations")

    print()

    # Segment distribution
    print("Segment Labels Generated:")
    all_segments = []
    for r in results:
        if r.get("success"):
            all_segments.extend([s[2] for s in r["segments"]])

    from collections import Counter
    segment_counts = Counter(all_segments)
    for seg, count in segment_counts.most_common():
        print(f"  {seg}: {count}")

    # Save results to JSON
    results_file = RESULTS_DIR / "batch_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_processed": len(valid_pairs),
            "success_count": success_count,
            "error_count": error_count,
            "vessel_stats": vessel_stats,
            "segment_counts": dict(segment_counts),
            "results": results
        }, f, indent=2, default=str)

    print()
    print(f"Results saved to: {results_file}")
    print(f"Visualizations saved to: {RESULTS_DIR}")
    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Max images to process")
    args = parser.parse_args()

    run_batch_labeling(max_images=args.max)
