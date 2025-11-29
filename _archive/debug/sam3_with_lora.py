"""
SAM 3 with LoRA Adapters for Efficient Fine-tuning

Supports multi-GPU training with DataParallel for 2x RTX 3090
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sam3.model_builder import build_sam3_image_model


class SAM3WithLoRA(nn.Module):
    """
    SAM 3 model with LoRA adapters for efficient fine-tuning

    Wraps SAM 3 with LoRA and DataParallel for multi-GPU training
    """

    def __init__(
        self,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=None,
        device='cuda',
        use_multi_gpu=True
    ):
        """
        Args:
            lora_r: LoRA rank (higher = more capacity, but slower)
            lora_alpha: LoRA alpha (scaling factor)
            lora_dropout: Dropout for LoRA layers
            target_modules: Which modules to apply LoRA to (default: attention)
            device: 'cuda' or 'cpu'
            use_multi_gpu: If True, use DataParallel for multi-GPU
        """
        super().__init__()

        print(f"Building SAM 3 with LoRA...")
        print(f"  LoRA rank (r): {lora_r}")
        print(f"  LoRA alpha: {lora_alpha}")
        print(f"  LoRA dropout: {lora_dropout}")

        # Build base SAM 3 model
        self.sam3_base = build_sam3_image_model(device=device)

        # Default target modules (attention layers in transformer)
        if target_modules is None:
            # Target the attention projection layers
            target_modules = [
                "q_proj",  # Query projection
                "v_proj",  # Value projection
                "k_proj",  # Key projection (optional, can comment out)
                "out_proj"  # Output projection (optional)
            ]

        print(f"  Target modules: {target_modules}")

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none"
            # task_type not needed for direct model wrapping
        )

        # Apply LoRA to SAM 3
        self.model = get_peft_model(self.sam3_base, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n  Trainable params: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"  Total params: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  LoRA efficiency: {trainable_params/total_params*100:.2f}% trainable")

        # Multi-GPU support
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1

        if self.use_multi_gpu:
            print(f"\n  🚀 Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
            print(f"  GPU 1: {torch.cuda.get_device_name(1)}")
        else:
            print(f"\n  Using single GPU: {torch.cuda.get_device_name(0)}")

        self.device = device

    def forward(self, images, bboxes=None):
        """
        Forward pass through SAM 3 with LoRA

        Args:
            images: Batch of images (B, C, H, W)
            bboxes: Optional bounding boxes (B, 4) in normalized [cx, cy, w, h]

        Returns:
            Segmentation masks (B, 1, H, W)
        """
        # TODO: Implement actual SAM 3 forward pass with prompts
        # This depends on SAM 3's exact API

        # Placeholder for now
        batch_size = images.shape[0]
        h, w = images.shape[2], images.shape[3]

        # Return dummy masks (will be replaced with actual SAM 3 forward)
        return torch.zeros(batch_size, 1, h, w, device=images.device)

    def save_lora_weights(self, path):
        """
        Save only the LoRA adapter weights (not the full model)

        Args:
            path: Where to save (e.g., 'lora_stage1.pth')
        """
        if self.use_multi_gpu:
            # Access the wrapped model
            self.model.module.save_pretrained(path)
        else:
            self.model.save_pretrained(path)

        print(f"✓ LoRA weights saved to {path}")

    def load_lora_weights(self, path):
        """
        Load LoRA adapter weights

        Args:
            path: Path to LoRA weights
        """
        from peft import PeftModel

        if self.use_multi_gpu:
            # Load into the base model, then re-wrap with DataParallel
            base_model = self.model.module.base_model
            self.model = PeftModel.from_pretrained(base_model, path)
            self.model = nn.DataParallel(self.model)
        else:
            base_model = self.model.base_model
            self.model = PeftModel.from_pretrained(base_model, path)

        print(f"✓ LoRA weights loaded from {path}")

    def print_trainable_parameters(self):
        """Print detailed breakdown of trainable parameters"""
        trainable_params = 0
        all_param = 0

        for name, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'lora' in name.lower():
                    print(f"  ✓ {name}: {param.numel():,} params")

        print(f"\nTotal trainable: {trainable_params:,} / {all_param:,}")
        print(f"Percentage: {100 * trainable_params / all_param:.2f}%")


def test_lora_sam3():
    """
    Test LoRA-enabled SAM 3 with multi-GPU
    """
    print("="*70)
    print("Testing SAM 3 with LoRA (Multi-GPU)")
    print("="*70)

    # Build model
    model = SAM3WithLoRA(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        device='cuda',
        use_multi_gpu=True  # Use both GPUs!
    )

    # Test forward pass with batch
    print("\nTesting forward pass...")

    # Simulate batch of 8 images (will be split across 2 GPUs: 4 per GPU)
    batch_size = 8
    images = torch.randn(batch_size, 3, 1024, 1024).cuda()

    print(f"Input batch size: {batch_size}")
    print(f"Split across GPUs: {batch_size // 2} per GPU")

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    print(f"Output shape: {outputs.shape}")

    # Print trainable parameters
    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # Test save/load
    print("\nTesting save/load...")
    save_path = "test_lora_weights"
    model.save_lora_weights(save_path)

    print("\n" + "="*70)
    print("✓ SAM 3 with LoRA test complete!")
    print("="*70)


if __name__ == '__main__':
    test_lora_sam3()
