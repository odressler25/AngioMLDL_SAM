"""
SAM 3 with manual LoRA implementation + Multi-GPU support

Uses custom LoRA layers to avoid peft dependency conflicts
Supports DataParallel for 2x RTX 3090 GPUs
"""

import torch
import torch.nn as nn
from sam3.model_builder import build_sam3_image_model
from lora_layers import add_lora_to_model


class SAM3WithLoRA(nn.Module):
    """
    SAM 3 with LoRA adapters for efficient fine-tuning

    Features:
    - Manual LoRA implementation (no peft dependency)
    - Multi-GPU support via DataParallel
    - Optimized for 2x RTX 3090 (48GB total VRAM)
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
            lora_r: LoRA rank (16 recommended for balance)
            lora_alpha: LoRA alpha scaling (32 recommended)
            lora_dropout: Dropout for LoRA layers
            target_modules: Which modules to add LoRA to
            device: 'cuda' or 'cpu'
            use_multi_gpu: Use DataParallel for multi-GPU
        """
        super().__init__()

        print("="*70)
        print("Building SAM 3 with LoRA")
        print("="*70)
        print(f"LoRA rank (r): {lora_r}")
        print(f"LoRA alpha: {lora_alpha}")
        print(f"LoRA dropout: {lora_dropout}")

        # Build base SAM 3
        print("\nLoading base SAM 3...")
        self.sam3_base = build_sam3_image_model(device=device)
        print("Base SAM 3 loaded")

        # Default target modules (SAM 3 attention layers)
        if target_modules is None:
            target_modules = [
                "attn.qkv",   # Combined Q,K,V projection in SAM 3
                "attn.proj",  # Output projection in SAM 3
            ]

        print(f"\nTarget modules for LoRA: {target_modules}")

        # Add LoRA adapters
        print("\nAdding LoRA adapters...")
        self.model, num_lora_params = add_lora_to_model(
            model=self.sam3_base,
            target_modules=target_modules,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # Count parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"\nParameter Summary:")
        print(f"  Total params: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  Trainable (LoRA): {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"  Frozen (base SAM 3): {total_params - trainable_params:,}")
        print(f"  LoRA efficiency: {trainable_params/total_params*100:.2f}% trainable")

        # Multi-GPU info (DataParallel applied by training script, not here)
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        self.device = device

        if self.use_multi_gpu:
            num_gpus = torch.cuda.device_count()
            print(f"\nMulti-GPU Available:")
            print(f"  {num_gpus} GPUs detected")
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Note: DataParallel will be applied by training script")
        else:
            print(f"\nSingle GPU: {torch.cuda.get_device_name(0)}")

        print("="*70)

    def forward(self, images, bboxes=None):
        """
        Forward pass with RoPE-compatible preprocessing

        Args:
            images: (B, C, H, W)
            bboxes: (B, 4) normalized bboxes (optional, not used currently)

        Returns:
            image_embeddings: Feature embeddings from SAM 3 backbone
        """
        # 1. FORCE RESOLUTION
        # SAM 3's RoPE cache is pre-calculated for 1008x1008 (not 1024).
        # We interpolate to 1008x1008 to guarantee the sequence length matches RoPE.
        target_size = 1008
        if images.shape[-1] != target_size or images.shape[-2] != target_size:
            images = torch.nn.functional.interpolate(
                images,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )

        # 2. FORWARD THROUGH BACKBONE
        # Use the LoRA-adapted backbone's forward_image method
        # self.model is the LoRA-wrapped SAM3 model
        features = self.model.backbone.forward_image(images)

        # 3. EXTRACT FEATURES
        # Handle dict output if SAM 3 returns {vision_features, ...}
        if isinstance(features, dict):
            image_embeddings = features.get('vision_features', features.get('image_embeddings', features))
        else:
            image_embeddings = features

        # 4. (Optional) MASK DECODER
        # If you have a mask decoder/head, call it here using image_embeddings
        # For now, we return the embeddings as the primary output of this wrapper
        return image_embeddings

    def save_lora_weights(self, path):
        """
        Save only LoRA weights (not full model)

        Args:
            path: Save path (e.g., 'lora_stage1.pth')
        """
        # Extract LoRA parameters only
        lora_state_dict = {}

        if self.use_multi_gpu:
            model = self.model.module
        else:
            model = self.model

        for name, param in model.named_parameters():
            if param.requires_grad:  # Only trainable (LoRA) params
                lora_state_dict[name] = param.cpu()

        torch.save(lora_state_dict, path)
        print(f"Saved LoRA weights to {path}")
        print(f"  Size: {sum(p.numel() for p in lora_state_dict.values()):,} params")

    def load_lora_weights(self, path):
        """
        Load LoRA weights

        Args:
            path: Path to LoRA checkpoint
        """
        lora_state_dict = torch.load(path)

        if self.use_multi_gpu:
            model = self.model.module
        else:
            model = self.model

        # Load only LoRA parameters
        model.load_state_dict(lora_state_dict, strict=False)
        print(f"Loaded LoRA weights from {path}")

    def get_trainable_parameters(self):
        """Get list of trainable parameters for optimizer"""
        return [p for p in self.model.parameters() if p.requires_grad]


def test_sam3_lora():
    """Test SAM 3 with LoRA and multi-GPU"""
    print("\n" + "="*70)
    print("Testing SAM 3 + LoRA + Multi-GPU")
    print("="*70 + "\n")

    # Build model
    model = SAM3WithLoRA(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        device='cuda',
        use_multi_gpu=True
    )

    # Test forward pass with batch
    print("\nTesting forward pass...")
    batch_size = 8  # Will split 4 per GPU with DataParallel
    images = torch.randn(batch_size, 3, 1024, 1024).cuda()

    print(f"  Input batch: {batch_size} images")
    print(f"  Image size: {images.shape[2]}x{images.shape[3]}")

    if model.use_multi_gpu:
        print(f"  Split: {batch_size//2} images per GPU")

    # Forward
    with torch.no_grad():
        outputs = model(images)

    print(f"  Output shape: {outputs.shape}")

    # Test save/load
    print("\nTesting save/load...")
    save_path = "test_lora.pth"
    model.save_lora_weights(save_path)

    # Test optimizer setup
    print("\nSetting up optimizer...")
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    print(f"  Optimizer params: {len(list(optimizer.param_groups[0]['params']))}")

    print("\n" + "="*70)
    print("SAM 3 + LoRA test complete!")
    print("="*70)


if __name__ == '__main__':
    test_sam3_lora()
