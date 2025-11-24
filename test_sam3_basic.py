"""
Basic SAM 3 test on common objects to verify API usage.

This tests if our SAM 3 calls are working correctly by using:
1. Text prompts for common objects ("cat", "person", "car")
2. Simple test images (we'll create synthetic ones)
3. Verify masks are generated properly

If this works, we know the API is correct.
If this fails, we have a setup/API problem.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Import SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_test_image(size=(512, 512)):
    """
    Create a simple test image with text saying "CAT".

    Returns:
        PIL Image
    """
    # Create white background
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    # Draw a simple cat-like shape (circle for head, triangles for ears)
    center_x, center_y = size[0] // 2, size[1] // 2

    # Head (circle)
    head_radius = 100
    draw.ellipse(
        [center_x - head_radius, center_y - head_radius,
         center_x + head_radius, center_y + head_radius],
        fill='orange', outline='black', width=3
    )

    # Left ear (triangle)
    draw.polygon(
        [(center_x - 80, center_y - 80),
         (center_x - 120, center_y - 140),
         (center_x - 40, center_y - 100)],
        fill='orange', outline='black'
    )

    # Right ear (triangle)
    draw.polygon(
        [(center_x + 80, center_y - 80),
         (center_x + 120, center_y - 140),
         (center_x + 40, center_y - 100)],
        fill='orange', outline='black'
    )

    # Eyes
    draw.ellipse([center_x - 40, center_y - 30, center_x - 10, center_y], fill='black')
    draw.ellipse([center_x + 10, center_y - 30, center_x + 40, center_y], fill='black')

    # Nose
    draw.ellipse([center_x - 10, center_y + 10, center_x + 10, center_y + 30], fill='pink')

    return img


def test_text_prompts(sam_processor, test_image, prompts):
    """
    Test SAM 3 with various text prompts.

    Args:
        sam_processor: SAM 3 processor
        test_image: PIL Image
        prompts: List of text prompts to test

    Returns:
        Dictionary of results per prompt
    """
    results = {}

    # Set image
    state = sam_processor.set_image(test_image)

    for prompt in prompts:
        logger.info(f"\nTesting prompt: '{prompt}'")

        try:
            # Set text prompt
            output = sam_processor.set_text_prompt(state, prompt=prompt)

            # Extract results
            masks = output.get("masks", [])
            scores = output.get("scores", [])

            # Convert tensors to numpy
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            num_masks = len(masks) if hasattr(masks, '__len__') else 0
            avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0

            results[prompt] = {
                'num_masks': num_masks,
                'avg_score': avg_score,
                'success': num_masks > 0
            }

            logger.info(f"  Found {num_masks} masks, avg score: {avg_score:.3f}")

            # Visualize best mask if any
            if num_masks > 0:
                best_mask = masks[0]
                if len(best_mask.shape) == 3:
                    best_mask = best_mask[0]

                mask_pixels = (best_mask > 0.5).sum()
                logger.info(f"  Best mask has {mask_pixels} pixels")

        except Exception as e:
            logger.error(f"  Error: {e}")
            results[prompt] = {
                'num_masks': 0,
                'avg_score': 0.0,
                'success': False,
                'error': str(e)
            }

    return results


def test_geometric_prompts(sam_processor, test_image):
    """
    Test SAM 3 with geometric prompts (bounding boxes).

    Args:
        sam_processor: SAM 3 processor
        test_image: PIL Image

    Returns:
        Dictionary with results
    """
    logger.info("\n=== Testing Geometric Prompts ===")

    # Set image
    state = sam_processor.set_image(test_image)

    # Define a bounding box around the center (where we drew the cat)
    img_width, img_height = test_image.size
    center_bbox = [
        img_width // 2 - 150,  # x1
        img_height // 2 - 150, # y1
        img_width // 2 + 150,  # x2
        img_height // 2 + 150  # y2
    ]

    logger.info(f"Testing bbox: {center_bbox}")

    try:
        # Add geometric prompt
        state = sam_processor.add_geometric_prompt(
            bbox=center_bbox,
            label="object",
            state=state
        )

        # Get predictions
        output = sam_processor._forward_grounding(state)

        # Extract results
        masks = output.get("masks", [])
        scores = output.get("scores", [])

        # Convert tensors
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        num_masks = len(masks) if hasattr(masks, '__len__') else 0
        avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0

        logger.info(f"Found {num_masks} masks, avg score: {avg_score:.3f}")

        return {
            'num_masks': num_masks,
            'avg_score': avg_score,
            'success': num_masks > 0
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            'num_masks': 0,
            'avg_score': 0.0,
            'success': False,
            'error': str(e)
        }


def visualize_results(test_image, results, save_path):
    """
    Visualize the test image with results.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(test_image)
    ax.set_title("Test Image: Simple Cat Drawing")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization saved to {save_path}")


def main():
    """
    Main test pipeline.
    """
    logger.info("=== SAM 3 Basic Functionality Test ===\n")

    # Initialize SAM 3
    logger.info("Loading SAM 3 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    sam_model = build_sam3_image_model(device=device)
    sam_processor = Sam3Processor(sam_model)
    logger.info("SAM 3 loaded successfully\n")

    # Create test image
    logger.info("Creating test image...")
    test_image = create_simple_test_image(size=(512, 512))

    # Save test image
    output_dir = Path("sam3_test_results")
    output_dir.mkdir(exist_ok=True)
    test_image.save(output_dir / "test_image_cat.png")
    logger.info(f"Test image saved to {output_dir / 'test_image_cat.png'}\n")

    # Test 1: Text prompts for common objects
    logger.info("=== Test 1: Text Prompts ===")
    text_prompts = [
        "cat",
        "animal",
        "orange cat",
        "face",
        "circle",
        "person",  # Should NOT find (negative control)
        "car",     # Should NOT find (negative control)
    ]

    text_results = test_text_prompts(sam_processor, test_image, text_prompts)

    # Test 2: Geometric prompts
    geometric_results = test_geometric_prompts(sam_processor, test_image)

    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info("\nText Prompt Results:")
    for prompt, result in text_results.items():
        status = "✓" if result['success'] else "✗"
        logger.info(f"  {status} '{prompt}': {result['num_masks']} masks (score: {result['avg_score']:.3f})")

    logger.info("\nGeometric Prompt Results:")
    status = "✓" if geometric_results['success'] else "✗"
    logger.info(f"  {status} Bounding box: {geometric_results['num_masks']} masks "
                f"(score: {geometric_results['avg_score']:.3f})")

    # Check if API is working
    any_success = any(r['success'] for r in text_results.values()) or geometric_results['success']

    if any_success:
        logger.info("\n✓ SAM 3 API is working correctly!")
        logger.info("  The model can generate masks from prompts.")
        logger.info("  Ready to test on medical images.")
    else:
        logger.warning("\n✗ SAM 3 API may have issues!")
        logger.warning("  No masks were generated for any prompts.")
        logger.warning("  Need to debug the API calls.")

    # Visualize
    visualize_results(test_image, text_results, output_dir / "test_results.png")


if __name__ == '__main__':
    main()
