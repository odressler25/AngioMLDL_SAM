"""
Test SAM 3 with downloaded images from the internet.

Usage:
1. Download some images (cat, dog, car, person, etc.) to a folder
2. Run: python test_sam3_with_images.py <image_folder>

This will test if SAM 3 can segment common objects to verify the API works.
"""

import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt

# Import SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_image_with_prompts(sam_processor, image_path, text_prompts):
    """
    Test one image with multiple text prompts.

    Args:
        sam_processor: SAM 3 processor
        image_path: Path to image file
        text_prompts: List of text prompts to try

    Returns:
        Dictionary of results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {image_path.name}")
    logger.info(f"{'='*60}")

    # Load image
    try:
        pil_image = Image.open(image_path).convert("RGB")
        logger.info(f"Image size: {pil_image.size}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None

    # Set image in SAM 3
    state = sam_processor.set_image(pil_image)

    results = {}

    for prompt in text_prompts:
        logger.info(f"\nPrompt: '{prompt}'")

        try:
            # Try text prompt
            output = sam_processor.set_text_prompt(state=state, prompt=prompt)

            # Extract masks and scores
            masks = output.get("masks", [])
            scores = output.get("scores", [])

            # Convert tensors
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            num_masks = len(masks) if hasattr(masks, '__len__') else 0
            avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0

            logger.info(f"  → Found {num_masks} masks, avg score: {avg_score:.3f}")

            # Store result
            results[prompt] = {
                'num_masks': num_masks,
                'avg_score': avg_score,
                'masks': masks if num_masks > 0 else None,
                'success': num_masks > 0
            }

            # If found masks, show pixel counts
            if num_masks > 0:
                for i, mask in enumerate(masks[:3]):  # Show first 3
                    if len(mask.shape) == 3:
                        mask = mask[0]
                    pixels = (mask > 0.5).sum()
                    logger.info(f"    Mask {i}: {pixels:,} pixels")

        except Exception as e:
            logger.error(f"  → Error: {e}")
            results[prompt] = {
                'num_masks': 0,
                'avg_score': 0.0,
                'masks': None,
                'success': False,
                'error': str(e)
            }

    return results


def visualize_results(image_path, results, output_dir):
    """
    Create visualization showing original image and best masks.
    """
    # Load original image
    img = np.array(Image.open(image_path).convert("RGB"))

    # Find best result (most masks or highest score)
    best_prompt = None
    best_masks = None
    best_score = 0

    for prompt, result in results.items():
        if result['success'] and result['avg_score'] > best_score:
            best_prompt = prompt
            best_masks = result['masks']
            best_score = result['avg_score']

    if best_masks is None:
        logger.warning("No masks to visualize")
        return

    # Create figure
    num_masks = min(len(best_masks), 3)
    fig, axes = plt.subplots(1, num_masks + 1, figsize=(5 * (num_masks + 1), 5))

    if num_masks == 0:
        axes = [axes]

    # Show original
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Show masks
    for i in range(num_masks):
        mask = best_masks[i]
        if len(mask.shape) == 3:
            mask = mask[0]

        # Resize mask to match image if needed
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create overlay
        overlay = img.copy()
        mask_bool = mask > 0.5
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5

        axes[i + 1].imshow(overlay.astype(np.uint8))
        axes[i + 1].set_title(f"Mask {i+1}\nPrompt: '{best_prompt}'")
        axes[i + 1].axis('off')

    plt.tight_layout()
    output_path = output_dir / f"{image_path.stem}_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization saved to {output_path}")


def main():
    """
    Main test pipeline.
    """
    # Get image folder from command line
    if len(sys.argv) < 2:
        logger.info("Usage: python test_sam3_with_images.py <image_folder>")
        logger.info("\nNo folder specified, using current directory...")
        image_folder = Path(".")
    else:
        image_folder = Path(sys.argv[1])

    if not image_folder.exists():
        logger.error(f"Folder not found: {image_folder}")
        return

    # Find images (including webp and avif)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_folder.glob(f"*{ext}")))
        image_files.extend(list(image_folder.glob(f"*{ext.upper()}")))

    if not image_files:
        logger.error(f"No images found in {image_folder}")
        logger.info(f"Looking for: {', '.join(image_extensions)}")
        return

    logger.info(f"Found {len(image_files)} images in {image_folder}")
    for img in image_files:
        logger.info(f"  - {img.name}")

    # Initialize SAM 3
    logger.info("\nLoading SAM 3 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    sam_model = build_sam3_image_model(device=device)
    sam_processor = Sam3Processor(sam_model)
    logger.info("SAM 3 loaded successfully")

    # Generic prompts to try on all images
    generic_prompts = [
        "object",
        "main object",
        "foreground",
    ]

    # Specific prompts (you can customize these based on your images)
    specific_prompts = [
        "cat", "dog", "animal", "pet",
        "person", "face", "human",
        "car", "vehicle",
        "building", "house",
        "tree", "plant",
        "food",
    ]

    # Create output directory
    output_dir = Path("sam3_test_results")
    output_dir.mkdir(exist_ok=True)

    # Test each image
    all_results = {}

    for image_path in image_files:
        # Try generic prompts first
        results = test_image_with_prompts(sam_processor, image_path, generic_prompts)

        if results:
            # If generic prompts work, try specific ones
            specific_results = test_image_with_prompts(sam_processor, image_path, specific_prompts)
            if specific_results:
                results.update(specific_results)

            all_results[image_path.name] = results

            # Visualize
            visualize_results(image_path, results, output_dir)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    total_successes = 0
    for image_name, results in all_results.items():
        successful_prompts = [p for p, r in results.items() if r['success']]
        total_successes += len(successful_prompts)

        logger.info(f"\n{image_name}:")
        if successful_prompts:
            logger.info(f"  ✓ Success: {', '.join(successful_prompts)}")
        else:
            logger.info(f"  ✗ No successful prompts")

    if total_successes > 0:
        logger.info("\n✓ SAM 3 API is working!")
        logger.info(f"  Found objects in response to {total_successes} prompts")
        logger.info("  Ready to test on medical images")
    else:
        logger.warning("\n✗ SAM 3 did not find any objects")
        logger.warning("  There may be an API issue")


if __name__ == '__main__':
    main()
