"""
Parallel CASS Labeling: GPU-accelerated, memory-optimized.

- Uses both GPUs for DINOv2 inference
- Does NOT expand features to full resolution (samples at skeleton points)
- All cache/temp on E: drive
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
import torch
import torch.multiprocessing as mp
import os
import sys
import gc

# Force all caching to E: drive BEFORE importing torch hub
os.environ['TORCH_HOME'] = 'E:/torch_cache'
os.environ['HF_HOME'] = 'E:/hf_cache'
os.environ['TMPDIR'] = 'E:/tmp'
os.environ['TEMP'] = 'E:/tmp'
os.environ['TMP'] = 'E:/tmp'
os.environ['MPLCONFIGDIR'] = 'E:/matplotlib_cache'

# Create cache dirs
for d in ['E:/torch_cache', 'E:/hf_cache', 'E:/tmp', 'E:/matplotlib_cache']:
    Path(d).mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from cass_anatomical_labeling import (
    clean_vessel_mask,
    get_major_bifurcations,
    find_vessel_origin,
    trace_main_vessel,
    infer_vessel_type,
    label_segments,
    CASS_SEGMENTS
)
from skimage.morphology import skeletonize
from skimage.draw import disk

# Paths
CASS_IMAGES_TRAIN = Path("E:/AngioMLDL_data/coco_cass_segments/train/images")
CASS_IMAGES_VAL = Path("E:/AngioMLDL_data/coco_cass_segments/val/images")
DEEPSA_MASKS = Path("E:/AngioMLDL_data/deepsa_pseudo_labels")
OUTPUT_DIR = Path("E:/AngioMLDL_data/cass_labeling")
OUTPUT_DIR.mkdir(exist_ok=True)

# Batch size - targeting ~20GB VRAM per GPU (you have 24GB each)
BATCH_SIZE = 24


def extract_dinov2_features_lowres(model, images, device, target_size=518):
    """
    Extract DINOv2 features - return LOW RES feature maps (37x37).
    Do NOT resize to full resolution to save RAM.
    """
    tensors = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    for img in images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        t = torch.from_numpy(np.array(img_resized)).float() / 255.0
        t = t.permute(2, 0, 1)
        tensors.append(t)

    batch = torch.stack(tensors).to(device)
    batch = (batch - mean) / std

    with torch.no_grad():
        features = model.forward_features(batch)
        patch_tokens = features['x_norm_patchtokens']

    B, N, D = patch_tokens.shape
    h = w = int(np.sqrt(N))

    # Return LOW RES feature maps (37x37x768) - NOT full resolution
    feature_maps = patch_tokens.reshape(B, h, w, D).cpu().numpy()

    # Clean GPU memory
    del batch, features, patch_tokens
    torch.cuda.empty_cache()

    return feature_maps  # (B, 37, 37, 768)


def sample_features_at_skeleton(feature_map_lowres, skeleton, orig_size):
    """
    Sample features at skeleton points by mapping skeleton coords to low-res feature map.
    Returns PCA-reduced colors for skeleton visualization.
    """
    from sklearn.decomposition import PCA

    h_lr, w_lr = feature_map_lowres.shape[:2]  # 37x37
    h_orig, w_orig = orig_size

    # Get skeleton coordinates
    skel_coords = np.array(np.where(skeleton)).T  # (N, 2) - y, x

    if len(skel_coords) == 0:
        return np.zeros((*skeleton.shape, 3))

    # Map to low-res coordinates
    y_lr = (skel_coords[:, 0] * h_lr / h_orig).astype(int).clip(0, h_lr - 1)
    x_lr = (skel_coords[:, 1] * w_lr / w_orig).astype(int).clip(0, w_lr - 1)

    # Sample features
    skel_features = feature_map_lowres[y_lr, x_lr]  # (N, 768)

    # PCA to 3 components for RGB visualization
    if len(skel_features) > 3:
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(skel_features)
        pca_features = (pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0) + 1e-8)
    else:
        pca_features = np.random.rand(len(skel_features), 3)

    # Create colored skeleton image
    skel_colored = np.zeros((*skeleton.shape, 3))
    skel_colored[skel_coords[:, 0], skel_coords[:, 1]] = pca_features

    return skel_colored


def process_single_image(img_array, mask, feature_map_lowres, image_name, output_path):
    """Process single image with pre-computed low-res features."""

    orig_size = img_array.shape[:2]

    # Process mask and skeleton
    cleaned_mask = clean_vessel_mask(mask)
    skeleton = skeletonize(cleaned_mask)
    major_bifs = get_major_bifurcations(skeleton, cleaned_mask)
    origin = find_vessel_origin(skeleton, cleaned_mask, img_array.shape)

    if origin:
        ordered_bifs, main_path = trace_main_vessel(skeleton, origin, major_bifs)
    else:
        ordered_bifs = major_bifs
        main_path = []

    vessel_type = infer_vessel_type(image_name)
    segments = label_segments(ordered_bifs, vessel_type, len(main_path))

    # Get skeleton colored by DINOv2 features
    skel_colored = sample_features_at_skeleton(feature_map_lowres, skeleton, orig_size)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=100)

    # 1. Original with vessel overlay
    overlay = img_array.astype(np.float32) / 255.0
    vessel_tint = np.zeros_like(overlay)
    vessel_tint[cleaned_mask > 0] = [0, 0.4, 0]
    overlay = overlay * 0.7 + vessel_tint * 0.3
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title(f"Vessel Type: {vessel_type}")
    axes[0, 0].axis("off")

    # 2. Skeleton colored by DINOv2 features + bifurcations
    skel_vis = skel_colored.copy()

    if origin:
        rr, cc = disk(origin, 12, shape=skeleton.shape)
        skel_vis[rr, cc] = [0, 1, 0]

    bif_colors = [[1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0, 1, 1], [0, 0.5, 1], [0.5, 0, 1]]
    for i, bif in enumerate(ordered_bifs):
        rr, cc = disk(bif, 10, shape=skeleton.shape)
        skel_vis[rr, cc] = bif_colors[i % len(bif_colors)]

    axes[0, 1].imshow(skel_vis)
    if origin:
        axes[0, 1].annotate("O", (origin[1], origin[0]), color='white',
                           fontsize=12, fontweight='bold', ha='center', va='center')
    for i, bif in enumerate(ordered_bifs):
        axes[0, 1].annotate(str(i+1), (bif[1], bif[0]), color='white',
                           fontsize=10, fontweight='bold', ha='center', va='center')
    axes[0, 1].set_title(f"DINOv2 Features + Bifurcations (O=Origin)")
    axes[0, 1].axis("off")

    # 3. Segment labels
    segment_overlay = img_array.astype(np.float32) / 255.0
    segment_colors = [[1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1]]

    if main_path and segments:
        path_len = len(main_path)
        for i, (start_pct, end_pct, seg_name, branch) in enumerate(segments):
            start_idx = int(start_pct * path_len)
            end_idx = int(end_pct * path_len)
            color = segment_colors[i % len(segment_colors)]
            for j in range(start_idx, min(end_idx, path_len)):
                py, px = main_path[j]
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                            segment_overlay[ny, nx] = color

    for i, bif in enumerate(ordered_bifs):
        rr, cc = disk(bif, 8, shape=skeleton.shape)
        segment_overlay[rr, cc] = [1, 1, 1]

    axes[1, 0].imshow(segment_overlay)
    segment_text = "\n".join([f"{s[2]}" + (f" (after {s[3]})" if s[3] else "") for s in segments])
    axes[1, 0].text(10, 30, segment_text, color='white', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    axes[1, 0].set_title("CASS Segment Labels")
    axes[1, 0].axis("off")

    # 4. Summary
    axes[1, 1].axis("off")
    summary = f"""CASS SEGMENT ANALYSIS
{'='*40}

Vessel Type: {vessel_type}
Origin Found: {'Yes' if origin else 'No'}
Major Bifurcations: {len(ordered_bifs)}

SEGMENT LABELS:"""
    for i, (start, end, name, branch) in enumerate(segments):
        branch_str = f" (landmark: {branch})" if branch else ""
        summary += f"\n  {i+1}. {name}{branch_str}"

    summary += f"""

CASS DEFINITION ({vessel_type}):
{CASS_SEGMENTS.get(vessel_type, {}).get('description', 'N/A')}"""

    axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 1].set_title("Summary")

    plt.suptitle(f"CASS: {image_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)

    return {
        "vessel_type": vessel_type,
        "n_bifurcations": len(ordered_bifs),
        "segments": [(s[0], s[1], s[2], s[3]) for s in segments],
        "origin_found": origin is not None
    }


def process_gpu_worker(gpu_id, image_pairs, output_dir):
    """Worker for one GPU."""

    print(f"[GPU {gpu_id}] Starting - {len(image_pairs)} images, batch_size={BATCH_SIZE}")

    # Set device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Load model
    print(f"[GPU {gpu_id}] Loading DINOv2...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    model = model.to(device)
    model.eval()
    print(f"[GPU {gpu_id}] Model loaded, VRAM: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

    results = []
    total_batches = (len(image_pairs) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(image_pairs))
        batch_pairs = image_pairs[start_idx:end_idx]

        print(f"[GPU {gpu_id}] Batch {batch_idx+1}/{total_batches}")

        # Load images and masks
        pil_images = []
        img_arrays = []
        masks = []
        names = []
        out_paths = []

        for img_path, mask_path, name, out_path in batch_pairs:
            try:
                img = Image.open(img_path).convert("RGB")
                pil_images.append(img)
                img_arrays.append(np.array(img))

                mask = np.load(mask_path)
                if mask.shape != img_arrays[-1].shape[:2]:
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask_img = mask_img.resize((img_arrays[-1].shape[1], img_arrays[-1].shape[0]), Image.NEAREST)
                    mask = np.array(mask_img) > 127
                masks.append(mask)
                names.append(name)
                out_paths.append(out_path)
            except Exception as e:
                print(f"[GPU {gpu_id}] Load error {name}: {e}")
                results.append({"image_name": name, "success": False, "error": str(e)})

        if not pil_images:
            continue

        # Extract features (LOW RES - 37x37)
        try:
            feature_maps = extract_dinov2_features_lowres(model, pil_images, device)
        except Exception as e:
            print(f"[GPU {gpu_id}] Feature extraction error: {e}")
            for name in names:
                results.append({"image_name": name, "success": False, "error": str(e)})
            continue

        # Process each image
        for i in range(len(pil_images)):
            try:
                print(f"[GPU {gpu_id}]   Processing {names[i]}...", flush=True)
                result = process_single_image(
                    img_arrays[i], masks[i], feature_maps[i],
                    names[i], out_paths[i]
                )
                result["image_name"] = names[i]
                result["success"] = True
                results.append(result)
                print(f"[GPU {gpu_id}]   -> {result['vessel_type']}", flush=True)
            except Exception as e:
                print(f"[GPU {gpu_id}] Process error {names[i]}: {e}", flush=True)
                results.append({"image_name": names[i], "success": False, "error": str(e)})

        # Cleanup
        del pil_images, img_arrays, masks, feature_maps
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}] Done - {sum(1 for r in results if r.get('success'))} successful")
    return results


def main():
    print("=" * 70)
    print("CASS Labeling - GPU Parallel (Memory Optimized)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Cache dir: E:/torch_cache")

    n_gpus = torch.cuda.device_count()
    print(f"\nGPUs: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)")

    n_gpus = min(n_gpus, 2)

    # Collect image pairs
    all_pairs = []

    for img_path in sorted(CASS_IMAGES_TRAIN.glob('*.png')):
        mask_path = DEEPSA_MASKS / f'{img_path.stem}_full_vessel_mask.npy'
        if mask_path.exists():
            all_pairs.append((img_path, mask_path, img_path.stem, OUTPUT_DIR / f'cass_{img_path.stem}.png'))

    val_output = OUTPUT_DIR / 'val'
    val_output.mkdir(exist_ok=True)
    for img_path in sorted(CASS_IMAGES_VAL.glob('*.png')):
        mask_path = DEEPSA_MASKS / f'{img_path.stem}_full_vessel_mask.npy'
        if mask_path.exists():
            all_pairs.append((img_path, mask_path, img_path.stem, val_output / f'cass_{img_path.stem}.png'))

    print(f"\nTotal images: {len(all_pairs)}")

    # Split between GPUs
    chunks = [[] for _ in range(n_gpus)]
    for i, pair in enumerate(all_pairs):
        chunks[i % n_gpus].append(pair)

    for i, chunk in enumerate(chunks):
        print(f"  GPU {i}: {len(chunk)} images ({len(chunk)//BATCH_SIZE} batches)")

    print("\nProcessing...")

    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=n_gpus) as pool:
        results_list = pool.starmap(process_gpu_worker, [(i, chunks[i], OUTPUT_DIR) for i in range(n_gpus)])

    all_results = [r for results in results_list for r in results]

    success_count = sum(1 for r in all_results if r.get('success'))
    print(f"\n{'='*70}")
    print(f"DONE: {success_count}/{len(all_results)} successful")

    vessel_stats = {}
    for r in all_results:
        if r.get('success'):
            vt = r['vessel_type']
            vessel_stats[vt] = vessel_stats.get(vt, 0) + 1
    for vt, c in sorted(vessel_stats.items(), key=lambda x: -x[1]):
        print(f"  {vt}: {c}")

    with open(OUTPUT_DIR / "batch_results.json", "w") as f:
        json.dump({"results": all_results, "vessel_stats": vessel_stats}, f, indent=2, default=str)

    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
