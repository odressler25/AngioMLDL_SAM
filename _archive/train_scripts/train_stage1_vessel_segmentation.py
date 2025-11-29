"""
Stage 1: Full Vessel Segmentation - Knowledge Transfer from DeepSA to SAM 3

Purpose: Teach SAM 3 what coronary vessels look like
Method: Use DeepSA predictions as pseudo-labels OR use Medis full vessel masks

Key insight: SAM 3 needs to understand vessel topology, boundaries, and anatomy
before it can classify CASS segments or detect stenosis.
"""

import sys
sys.path.append("DeepSA/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image
import json

# DeepSA imports
from DeepSA.models import UNet
from DeepSA.datasets import tophat
import torchvision.transforms as T


class VesselSegmentationDataset(Dataset):
    """
    Dataset for Stage 1: Full vessel segmentation

    Two modes:
    1. Use DeepSA predictions as pseudo-labels (unlabeled data)
    2. Use Medis full vessel masks as ground truth (labeled data)
    """

    def __init__(self, csv_path, use_deepsa_labels=False, deepsa_model=None, max_cases=None):
        """
        Args:
            csv_path: Path to corrected_dataset_training.csv
            use_deepsa_labels: If True, generate labels with DeepSA
            deepsa_model: Pretrained DeepSA model (if use_deepsa_labels=True)
            max_cases: Limit number of cases (for testing)
        """
        self.df = pd.read_csv(csv_path)
        if max_cases:
            self.df = self.df.head(max_cases)

        self.use_deepsa_labels = use_deepsa_labels
        self.deepsa_model = deepsa_model

        if use_deepsa_labels and deepsa_model is None:
            raise ValueError("deepsa_model required when use_deepsa_labels=True")

        # DeepSA preprocessing
        self.deepsa_transform = T.Compose([
            T.Resize(512),
            T.Lambda(lambda img: tophat(img, 50)),
            T.ToTensor(),
            T.Normalize((0.5), (0.5))
        ])

        print(f"Loaded {len(self.df)} cases for Stage 1 training")
        if use_deepsa_labels:
            print("Using DeepSA predictions as pseudo-labels")
        else:
            print("Using Medis vessel masks as ground truth")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load cine
        cine_path = row['cine_path']
        cine = np.load(cine_path)

        # Get correct frame
        frame_idx = int(row['frame_index'])
        frame = cine[frame_idx]

        # Normalize to [0, 255]
        if frame.dtype != np.uint8:
            frame = (frame / frame.max() * 255).astype(np.uint8)

        # Convert to PIL
        frame_pil = Image.fromarray(frame).convert('L')

        # Get vessel mask (ground truth)
        if self.use_deepsa_labels:
            # Generate pseudo-label with DeepSA
            vessel_mask = self.generate_deepsa_label(frame_pil)
        else:
            # Load Medis mask
            mask_path = row['vessel_mask_actual_path']
            vessel_mask = np.load(mask_path)

            # Binarize (vessel vs background)
            vessel_mask = (vessel_mask > 0).astype(np.float32)

        # Resize for SAM 3 (1024x1024 typical SAM input)
        frame_resized = cv2.resize(frame, (1024, 1024))
        vessel_mask_resized = cv2.resize(vessel_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # To tensor
        frame_tensor = torch.from_numpy(frame_resized).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(vessel_mask_resized).float().unsqueeze(0)

        return {
            'image': frame_tensor,
            'mask': mask_tensor,
            'case_id': row['case_id']
        }

    def generate_deepsa_label(self, frame_pil):
        """
        Generate pseudo-label using pretrained DeepSA

        Args:
            frame_pil: PIL Image (grayscale)

        Returns:
            vessel_mask: Binary mask (0/1)
        """
        # Preprocess for DeepSA
        frame_tensor = self.deepsa_transform(frame_pil)

        # DeepSA inference (frozen model)
        with torch.no_grad():
            pred = self.deepsa_model(frame_tensor.unsqueeze(0).to('cuda'))
            vessel_mask = (torch.sign(pred) > 0).cpu().numpy()[0, 0].astype(np.float32)

        return vessel_mask


class Stage1Trainer:
    """
    Train SAM 3 for full vessel segmentation using DeepSA knowledge

    Two training strategies:
    1. Knowledge distillation: DeepSA as teacher
    2. Direct supervision: Medis masks as ground truth
    """

    def __init__(self, sam3_processor, use_deepsa_teacher=True, deepsa_model=None):
        """
        Args:
            sam3_processor: SAM 3 processor (Sam3Processor)
            use_deepsa_teacher: If True, use DeepSA for knowledge distillation
            deepsa_model: Pretrained DeepSA (if using as teacher)
        """
        self.sam3_processor = sam3_processor
        self.use_deepsa_teacher = use_deepsa_teacher
        self.deepsa = deepsa_model

        if use_deepsa_teacher:
            print("Training mode: Knowledge distillation from DeepSA")
            if deepsa_model is None:
                raise ValueError("deepsa_model required for knowledge distillation")
            self.deepsa.eval()
            for param in self.deepsa.parameters():
                param.requires_grad = False
        else:
            print("Training mode: Direct supervision (DeepSA pseudo-labels only)")

    def dice_loss(self, pred, target):
        """Dice loss for segmentation"""
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice

    def combined_loss(self, pred, target):
        """Combined BCE + Dice loss"""
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = self.dice_loss(torch.sigmoid(pred), target)
        return 0.5 * bce + 0.5 * dice

    def train_epoch(self, dataloader, optimizer, device='cuda'):
        """Train one epoch"""
        self.sam3.train()

        total_loss = 0
        total_dice = 0

        for batch in tqdm(dataloader, desc="Training"):
            images = batch['image'].to(device)
            masks_gt = batch['mask'].to(device)

            # TODO: SAM 3 forward pass
            # For now, using placeholder
            # In actual implementation:
            # pred_masks = self.sam3(images, text_prompt="coronary vessels")

            # Placeholder: random predictions
            pred_masks = torch.randn_like(masks_gt)

            # Compute loss
            loss = self.combined_loss(pred_masks, masks_gt)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            with torch.no_grad():
                dice = 1 - self.dice_loss(torch.sigmoid(pred_masks), masks_gt)
                total_dice += dice.item()

        avg_loss = total_loss / len(dataloader)
        avg_dice = total_dice / len(dataloader)

        return avg_loss, avg_dice

    def validate(self, dataloader, device='cuda'):
        """Validate on test set"""
        self.sam3.eval()

        total_dice = 0
        total_iou = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch['image'].to(device)
                masks_gt = batch['mask'].to(device)

                # TODO: SAM 3 inference
                # pred_masks = self.sam3(images, text_prompt="coronary vessels")

                # Placeholder
                pred_masks = torch.sigmoid(torch.randn_like(masks_gt))

                # Metrics
                dice = 1 - self.dice_loss(pred_masks, masks_gt)
                total_dice += dice.item()

                # IoU
                pred_binary = (pred_masks > 0.5).float()
                intersection = (pred_binary * masks_gt).sum()
                union = (pred_binary + masks_gt).clamp(0, 1).sum()
                iou = intersection / (union + 1e-6)
                total_iou += iou.item()

        avg_dice = total_dice / len(dataloader)
        avg_iou = total_iou / len(dataloader)

        return avg_dice, avg_iou

    def train(self, train_loader, val_loader, epochs=20, lr=1e-4, device='cuda'):
        """
        Full training loop for Stage 1

        Args:
            train_loader: Training data
            val_loader: Validation data
            epochs: Number of epochs
            lr: Learning rate
            device: 'cuda' or 'cpu'
        """
        # TODO: Optimizer for SAM 3 (with LoRA if applicable)
        # optimizer = torch.optim.AdamW(self.sam3.parameters(), lr=lr)

        # Placeholder optimizer
        optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=lr)

        best_dice = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("="*70)

            # Train
            train_loss, train_dice = self.train_epoch(train_loader, optimizer, device)
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")

            # Validate
            val_dice, val_iou = self.validate(val_loader, device)
            print(f"Val   - Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                # TODO: Save model
                # torch.save(self.sam3.state_dict(), f'stage1_best_dice{val_dice:.4f}.pth')
                print(f"✓ New best Dice: {best_dice:.4f}")

        print("\n" + "="*70)
        print(f"Stage 1 Training Complete!")
        print(f"Best Validation Dice: {best_dice:.4f}")
        print("="*70)

        return self.sam3


def load_deepsa_pretrained(ckpt_path="DeepSA/ckpt/fscad_36249.ckpt", device='cuda'):
    """Load pretrained DeepSA model (frozen teacher)"""
    print(f"Loading DeepSA from {ckpt_path}...")
    model = UNet(1, 1, 32, bilinear=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['netE'].items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Freeze
    for param in model.parameters():
        param.requires_grad = False

    print("✓ DeepSA loaded (frozen)")
    return model


def main():
    """
    Stage 1 Training: Full Vessel Segmentation

    Transfer DeepSA's vessel knowledge to SAM 3
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Configuration
    csv_path = "E:/AngioMLDL_data/corrected_dataset_training.csv"
    use_deepsa_labels = True  # Use DeepSA pseudo-labels
    batch_size = 4
    epochs = 20
    lr = 1e-4
    max_cases = 100  # For testing; set to None for full dataset

    # Load DeepSA (teacher)
    print("="*70)
    print("STAGE 1: Full Vessel Segmentation")
    print("Training SAM 3 to segment coronary vessels using DeepSA knowledge")
    print("="*70)

    deepsa_model = load_deepsa_pretrained(device=device)

    # Load SAM 3
    print("\nLoading SAM 3...")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_model = build_sam3_image_model(device=device)
    sam3_processor = Sam3Processor(sam3_model)
    print("✓ SAM 3 loaded")

    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = VesselSegmentationDataset(
        csv_path=csv_path,
        use_deepsa_labels=use_deepsa_labels,
        deepsa_model=deepsa_model,
        max_cases=max_cases
    )

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with DeepSA
        pin_memory=True
    )

    print(f"\nTrain batches: {len(train_loader)}")

    # Initialize trainer
    trainer = Stage1Trainer(
        sam3_processor=sam3_processor,
        use_deepsa_teacher=use_deepsa_labels,
        deepsa_model=deepsa_model
    )

    # TODO: Implement actual training loop with SAM 3 API
    print("\n" + "="*70)
    print("NOTE: Training loop needs SAM 3 fine-tuning implementation")
    print("Current status:")
    print("  ✓ SAM 3 loaded")
    print("  ✓ DeepSA loaded (frozen teacher)")
    print("  ✓ Dataset ready (DeepSA pseudo-labels)")
    print("  ⏳ Need: SAM 3 training API (forward pass + LoRA)")
    print("="*70)

    # Test data loading
    print("\nTesting data loading...")
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch mask shape: {batch['mask'].shape}")
    print(f"Case IDs: {batch['case_id']}")


if __name__ == '__main__':
    main()
