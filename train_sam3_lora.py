"""
Train SAM 3 with LoRA for coronary lesion segmentation.

This script shows how to add LoRA adapters to SAM 3 and train them
on your lesion segment data while keeping the base model frozen.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple
from tqdm import tqdm

# SAM 3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh

# LoRA imports (you'll need to install: pip install peft)
from peft import LoraConfig, get_peft_model, TaskType


class CoronaryLesionDataset(Dataset):
    """
    Dataset for coronary lesion segments.
    """

    def __init__(self, base_path: Path, case_ids: List[str]):
        self.base_path = base_path
        self.cases = []

        # Load all cases
        for case_id in case_ids:
            json_path = base_path / "contours" / f"{case_id}_contours.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                self.cases.append({
                    'case_id': case_id,
                    'frame_num': metadata.get('frame_num', 0),
                    'vessel_name': metadata.get('segment_name', ''),
                    'stenosis': metadata.get('Stenosis (%)', 0),
                    'mld': metadata.get('MLD (mm)', 0)
                })

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_id = case['case_id']

        # Load image (correct frame with contrast)
        cine_path = self.base_path / "cines" / f"{case_id}_cine.npy"
        cine = np.load(cine_path)
        frame = cine[min(case['frame_num'], len(cine)-1)]

        # Normalize to [0, 1]
        frame = frame.astype(np.float32)
        if frame.max() > 1:
            frame = frame / 255.0

        # Load mask
        mask_path = self.base_path / "vessel_masks" / f"{case_id}_mask.npy"
        mask = np.load(mask_path).astype(np.float32)

        # Get bbox from mask
        bbox = self.get_bbox_from_mask(mask)

        # Parse vessel info
        vessel_type = self.parse_vessel_type(case['vessel_name'])

        return {
            'image': frame,
            'mask': mask,
            'bbox': bbox,
            'vessel_type': vessel_type,
            'vessel_name': case['vessel_name'],
            'stenosis': case['stenosis'],
            'mld': case['mld']
        }

    def get_bbox_from_mask(self, mask: np.ndarray, padding: int = 20):
        """Get normalized bbox from mask."""
        vessel_pixels = np.argwhere(mask > 0)

        if len(vessel_pixels) == 0:
            # Return full image bbox if no vessel
            return [0.5, 0.5, 1.0, 1.0]  # cx, cy, w, h normalized

        y_coords = vessel_pixels[:, 0]
        x_coords = vessel_pixels[:, 1]

        h, w = mask.shape

        y_min = max(0, y_coords.min() - padding) / h
        y_max = min(h, y_coords.max() + padding) / h
        x_min = max(0, x_coords.min() - padding) / w
        x_max = min(w, x_coords.max() + padding) / w

        # Return in (cx, cy, w, h) normalized format
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return [cx, cy, width, height]

    def parse_vessel_type(self, vessel_name: str) -> str:
        """Extract vessel type from name."""
        if "RCA" in vessel_name:
            return "RCA"
        elif "LAD" in vessel_name:
            return "LAD"
        elif "LCX" in vessel_name or "LCx" in vessel_name:
            return "LCX"
        elif "OM" in vessel_name:
            return "OM"
        elif "DIAG" in vessel_name:
            return "DIAG"
        else:
            return "Unknown"


class SAM3WithLoRA(nn.Module):
    """
    SAM 3 model with LoRA adapters and additional task heads.
    """

    def __init__(self, sam3_model, lora_config: LoraConfig):
        super().__init__()

        # Freeze base model
        for param in sam3_model.parameters():
            param.requires_grad = False

        # Add LoRA adapters
        self.sam3 = get_peft_model(sam3_model, lora_config)

        # Print trainable parameters
        self.sam3.print_trainable_parameters()

        # Add task-specific heads (optional)
        # These could predict stenosis %, CASS segment, etc.
        hidden_dim = 256
        self.stenosis_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output 0-1 for stenosis percentage
        )

    def forward(self, image, bbox, text_prompt):
        """
        Forward pass through SAM 3 with LoRA.

        Returns:
            mask: Segmentation mask
            stenosis: Predicted stenosis percentage (optional)
        """
        # This is simplified - actual implementation depends on SAM 3 API
        output = self.sam3(image, bbox, text_prompt)
        mask = output['masks']

        # Optional: extract features for stenosis prediction
        # features = output.get('features')
        # stenosis = self.stenosis_head(features)

        return mask


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        bboxes = batch['bbox'].to(device)
        vessel_names = batch['vessel_name']

        # Create text prompts
        prompts = [f"{name} stenosis" for name in vessel_names]

        # Forward pass
        pred_masks = model(images, bboxes, prompts)

        # Calculate loss (IoU + Dice)
        loss = calculate_segmentation_loss(pred_masks, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def calculate_segmentation_loss(pred, target):
    """
    Combined IoU and Dice loss for segmentation.
    """
    # Binary cross entropy
    bce_loss = nn.BCEWithLogitsLoss()(pred, target)

    # Dice loss
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target).sum(dim=(1, 2))
    union = pred_sigmoid.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice_loss = 1 - (2 * intersection / (union + 1e-8)).mean()

    return bce_loss + dice_loss


def main():
    """
    Main training loop.
    """
    print("="*60)
    print("SAM 3 LoRA Training for Coronary Lesions")
    print("="*60)

    # Configuration
    base_path = Path(r"E:\AngioMLDL_data\corrected_vessel_dataset")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 10
    lora_rank = 16

    # Get all case IDs
    contours_path = base_path / "contours"
    case_ids = [f.stem.replace("_contours", "")
                for f in contours_path.glob("*.json")]

    print(f"Found {len(case_ids)} cases")

    # Split train/val (80/20)
    split_idx = int(0.8 * len(case_ids))
    train_ids = case_ids[:split_idx]
    val_ids = case_ids[split_idx:]

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Create datasets
    train_dataset = CoronaryLesionDataset(base_path, train_ids)
    val_dataset = CoronaryLesionDataset(base_path, val_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Load SAM 3 base model
    print("\nLoading SAM 3 base model...")
    sam3_base = build_sam3_image_model(device=device)

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,  # Low rank
        lora_alpha=32,  # LoRA scaling
        target_modules=["q_proj", "v_proj", "k_proj"],  # Target attention
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Create model with LoRA
    print("\nAdding LoRA adapters...")
    model = SAM3WithLoRA(sam3_base, lora_config)
    model = model.to(device)

    # Optimizer (only LoRA parameters will be updated)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate (implement validation loop similarly)
        # val_loss = validate(model, val_loader, device)
        # print(f"Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"sam3_coronary_lora_epoch_{epoch+1}.pth"
            model.sam3.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    print("\nSaving final LoRA weights...")
    model.sam3.save_pretrained("sam3_coronary_lora_final")

    print("\nTraining complete!")
    print("LoRA weights saved to: sam3_coronary_lora_final/")
    print("\nTo use the trained model:")
    print("```python")
    print("from peft import PeftModel")
    print("sam3_base = build_sam3_image_model()")
    print("sam3_with_lora = PeftModel.from_pretrained(sam3_base, 'sam3_coronary_lora_final')")
    print("```")


if __name__ == '__main__':
    # Note: This is a template. The actual SAM 3 API integration
    # needs to be adapted based on the specific SAM 3 interface.
    # Key parts that need adaptation:
    # 1. How to pass bbox/text to SAM 3
    # 2. How to extract features for additional tasks
    # 3. Loss calculation with SAM 3 outputs

    print("\n⚠️ This is a training template!")
    print("Actual implementation needs:")
    print("1. pip install peft transformers")
    print("2. Adaptation to exact SAM 3 API")
    print("3. Proper loss functions for SAM 3 outputs")
    print("\nRefer to SAM3_TRAINING_STRATEGY.md for details")