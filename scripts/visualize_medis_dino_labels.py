"""
Visualize what the model sees with the Medis+DINO labeling approach.

Shows:
1. Original angiography image
2. DeepSA vessel mask
3. Medis centerline (ground truth anchor)
4. Final CASS segment masks with labels
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
from pycocotools import mask as mask_utils
import random

# Paths - can switch between datasets
DATASET_DIR = Path("E:/AngioMLDL_data/coco_medis_bifurcation")  # Changed to bifurcation
OUTPUT_DIR = Path("E:/AngioMLDL_data/medis_bifurcation_visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors for each CASS category
CASS_COLORS = {
    "proximal_rca": "#FF6B6B",    # Red
    "mid_rca": "#FF8E8E",          # Light red
    "distal_rca": "#FFB4B4",       # Pale red
    "pda": "#FFD93D",              # Yellow
    "proximal_lad": "#6BCB77",     # Green
    "mid_lad": "#8ED99A",          # Light green
    "distal_lad": "#B4E4BC",       # Pale green
    "diagonal_1": "#4D96FF",       # Blue
    "diagonal_2": "#7DB3FF",       # Light blue
    "proximal_lcx": "#9B59B6",     # Purple
    "distal_lcx": "#BB8FCE",       # Light purple
    "obtuse_marginal_1": "#F39C12", # Orange
    "obtuse_marginal_2": "#F7B84B", # Light orange
    "ramus": "#1ABC9C",            # Teal
    "left_main": "#E74C3C",        # Dark red
}


def load_coco_annotations(split="train"):
    """Load COCO annotations."""
    ann_path = DATASET_DIR / split / "annotations.json"
    with open(ann_path) as f:
        coco = json.load(f)

    # Build lookups
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    ann_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)

    return coco, categories, images, ann_by_image


def decode_rle(rle, height, width):
    """Decode RLE to binary mask."""
    if isinstance(rle, dict):
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')
        rle['size'] = [height, width]
        return mask_utils.decode(rle)
    return np.zeros((height, width), dtype=np.uint8)


def visualize_sample(img_info, annotations, categories, split="train", save_path=None):
    """Create visualization for a single sample."""

    # Load image
    img_path = DATASET_DIR / split / "images" / img_info["file_name"]
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    image = np.array(Image.open(img_path).convert('RGB'))
    height, width = img_info["height"], img_info["width"]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\n{img_info['file_name']}", fontsize=10)
    axes[0].axis('off')

    # 2. Image with segment overlays
    axes[1].imshow(image)

    # Overlay each segment mask
    legend_patches = []
    for ann in annotations:
        cat_name = categories[ann["category_id"]]
        color = CASS_COLORS.get(cat_name, "#FFFFFF")

        # Decode mask
        mask = decode_rle(ann["segmentation"], height, width)

        # Create colored overlay
        overlay = np.zeros((*mask.shape, 4))
        rgb = tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5))
        overlay[mask > 0] = (*rgb, 0.5)  # 50% opacity

        axes[1].imshow(overlay)

        # Add to legend (only if not already added)
        is_anchor = ann.get("is_anchor", False)
        label = f"{cat_name}" + (" (GT)" if is_anchor else " (inferred)")
        if not any(p.get_label() == label for p in legend_patches):
            legend_patches.append(mpatches.Patch(color=color, label=label))

    axes[1].set_title(f"CASS Segments ({len(annotations)} annotations)", fontsize=10)
    axes[1].axis('off')
    axes[1].legend(handles=legend_patches, loc='upper right', fontsize=7)

    # 3. Segment masks only (no image background)
    # Create a composite mask image
    composite = np.zeros((height, width, 3))
    for ann in annotations:
        cat_name = categories[ann["category_id"]]
        color = CASS_COLORS.get(cat_name, "#FFFFFF")
        rgb = tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5))

        mask = decode_rle(ann["segmentation"], height, width)
        for c in range(3):
            composite[:, :, c] = np.where(mask > 0, rgb[c], composite[:, :, c])

    axes[2].imshow(composite)
    axes[2].set_title("Segment Masks Only", fontsize=10)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def create_grid_visualization(split="train", n_samples=12, save_path=None):
    """Create a grid of sample visualizations."""

    coco, categories, images, ann_by_image = load_coco_annotations(split)

    # Select random samples with multiple annotations
    good_samples = [(img_id, anns) for img_id, anns in ann_by_image.items()
                    if len(anns) >= 2]

    if len(good_samples) < n_samples:
        good_samples = list(ann_by_image.items())

    samples = random.sample(good_samples, min(n_samples, len(good_samples)))

    # Calculate grid size
    cols = 4
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, (img_id, anns) in enumerate(samples):
        if idx >= len(axes):
            break

        img_info = images[img_id]
        img_path = DATASET_DIR / split / "images" / img_info["file_name"]

        if not img_path.exists():
            continue

        image = np.array(Image.open(img_path).convert('RGB'))
        height, width = img_info["height"], img_info["width"]

        # Show image with overlays
        axes[idx].imshow(image)

        # Overlay each segment
        for ann in anns:
            cat_name = categories[ann["category_id"]]
            color = CASS_COLORS.get(cat_name, "#FFFFFF")

            mask = decode_rle(ann["segmentation"], height, width)
            overlay = np.zeros((*mask.shape, 4))
            rgb = tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5))
            overlay[mask > 0] = (*rgb, 0.5)

            axes[idx].imshow(overlay)

        # Build title with segment names
        seg_names = [categories[a["category_id"]] for a in anns]
        anchor = [categories[a["category_id"]] for a in anns if a.get("is_anchor", False)]
        title = f"{len(anns)} segs"
        if anchor:
            title += f" | GT: {anchor[0]}"

        axes[idx].set_title(title, fontsize=8)
        axes[idx].axis('off')

    # Hide unused axes
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Medis+DINO CASS Labels - {split} split\n(Colored by segment, GT=ground truth from Medis)",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def create_category_legend(save_path=None):
    """Create a legend showing all CASS categories and colors."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Group by artery
    groups = {
        "RCA": ["proximal_rca", "mid_rca", "distal_rca", "pda"],
        "LAD": ["left_main", "proximal_lad", "mid_lad", "distal_lad", "diagonal_1", "diagonal_2", "ramus"],
        "LCX": ["proximal_lcx", "distal_lcx", "obtuse_marginal_1", "obtuse_marginal_2"],
    }

    y_pos = 0.95
    for group_name, segments in groups.items():
        ax.text(0.1, y_pos, group_name, fontsize=14, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.06

        for seg in segments:
            color = CASS_COLORS.get(seg, "#FFFFFF")
            patch = mpatches.Rectangle((0.12, y_pos - 0.02), 0.05, 0.04,
                                        facecolor=color, edgecolor='black',
                                        transform=ax.transAxes)
            ax.add_patch(patch)
            ax.text(0.19, y_pos, seg.replace("_", " ").title(),
                   fontsize=11, transform=ax.transAxes, verticalalignment='center')
            y_pos -= 0.055

        y_pos -= 0.03  # Extra space between groups

    plt.title("CASS Segment Color Legend", fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def main(n_samples=6):
    """Create visualizations."""

    print("Loading annotations...")

    # Create legend
    create_category_legend(OUTPUT_DIR / "cass_color_legend.png")

    # Create grid visualizations
    print("\nCreating train grid visualization...")
    create_grid_visualization("train", n_samples=12,
                              save_path=OUTPUT_DIR / "train_samples_grid.png")

    print("\nCreating val grid visualization...")
    create_grid_visualization("val", n_samples=12,
                              save_path=OUTPUT_DIR / "val_samples_grid.png")

    # Create individual detailed visualizations
    print("\nCreating detailed sample visualizations...")
    coco, categories, images, ann_by_image = load_coco_annotations("train")

    # Pick samples with good variety
    good_samples = [(img_id, anns) for img_id, anns in ann_by_image.items()
                    if len(anns) >= 2]  # Changed from 3 to 2

    if good_samples:
        samples = random.sample(good_samples, min(n_samples, len(good_samples)))

        for i, (img_id, anns) in enumerate(samples):
            img_info = images[img_id]
            save_path = OUTPUT_DIR / f"angle_constrained_{i+1}.png"  # Updated name
            visualize_sample(img_info, anns, categories, "train", save_path)

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6, help="Number of detailed samples")
    args = parser.parse_args()

    random.seed(42)
    main(args.n)
