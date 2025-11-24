"""
Manual LoRA implementation for SAM 3

Implements LoRA (Low-Rank Adaptation) without relying on peft library
to avoid dependency conflicts.
"""

import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer

    Adds trainable low-rank matrices A and B to a frozen linear layer:
    h = W_0 x + (B A) x

    where:
    - W_0 is frozen (pretrained weights)
    - A, B are trainable (low-rank adaptation)
    - rank of A and B is much smaller than W_0
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            r: LoRA rank (bottleneck dimension)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability
        """
        super().__init__()

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Apply LoRA adaptation

        Args:
            x: Input tensor (..., in_features)

        Returns:
            LoRA output: (B @ A) @ x, scaled
        """
        # x: (..., in_features)
        # A: (r, in_features)
        # B: (out_features, r)

        # Compute (B @ A) @ x = B @ (A @ x)
        result = self.lora_dropout(x) @ self.lora_A.T  # (..., r)
        result = result @ self.lora_B.T  # (..., out_features)

        return self.scaling * result


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adapter

    Combines frozen pretrained weights with trainable LoRA adaptation
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05
    ):
        """
        Args:
            linear: Pretrained linear layer to adapt
            r: LoRA rank
            lora_alpha: LoRA scaling
            lora_dropout: LoRA dropout
        """
        super().__init__()

        # Store frozen pretrained layer
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False

        # Add LoRA adapter
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

    def forward(self, x):
        """
        Forward pass: pretrained output + LoRA adaptation

        Args:
            x: Input tensor

        Returns:
            h = W_0 x + (B A) x
        """
        # Frozen pretrained forward
        pretrained_output = self.linear(x)

        # LoRA adaptation
        lora_output = self.lora(x)

        # Combined
        # FIX: Add .contiguous() to ensure memory layout is safe for
        # DataParallel scattering and SAM 3's internal reshaping/RoPE
        return (pretrained_output + lora_output).contiguous()


def add_lora_to_model(model, target_modules=None, r=16, lora_alpha=32.0, lora_dropout=0.05):
    """
    Add LoRA adapters to specified modules in a model

    Args:
        model: PyTorch model
        target_modules: List of module name patterns to add LoRA to
                       (e.g., ['q_proj', 'v_proj'])
        r: LoRA rank
        lora_alpha: LoRA scaling
        lora_dropout: LoRA dropout

    Returns:
        model: Modified model with LoRA adapters
        num_trainable: Number of trainable parameters added
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']  # Default: attention

    # First, freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False

    num_replaced = 0
    num_trainable = 0

    # Recursively replace linear layers
    for name, module in model.named_modules():
        # Check if this module should get LoRA
        should_add_lora = any(target in name for target in target_modules)

        if should_add_lora and isinstance(module, nn.Linear):
            # Get parent module and attribute name
            *parent_names, attr_name = name.split('.')

            parent = model
            for parent_name in parent_names:
                parent = getattr(parent, parent_name)

            # Replace with LinearWithLoRA
            lora_layer = LinearWithLoRA(
                linear=module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

            # CRITICAL: Move LoRA layer to same device as original linear layer
            lora_layer = lora_layer.to(module.weight.device)

            setattr(parent, attr_name, lora_layer)

            num_replaced += 1
            num_trainable += r * (module.in_features + module.out_features)

            print(f"  [+] Added LoRA to: {name} ({module.in_features} -> {module.out_features})")

    print(f"\n  Total LoRA layers added: {num_replaced}")
    print(f"  Total trainable params: {num_trainable:,}")

    return model, num_trainable


def test_lora():
    """Test LoRA layer"""
    print("Testing LoRA implementation...")

    # Create a simple linear layer
    linear = nn.Linear(512, 256)

    # Freeze it
    for param in linear.parameters():
        param.requires_grad = False

    # Add LoRA
    lora_linear = LinearWithLoRA(linear, r=16)

    # Test forward pass
    x = torch.randn(4, 512)
    output = lora_linear(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count trainable params
    trainable = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_linear.parameters())

    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    print("[OK] LoRA test passed!")


if __name__ == '__main__':
    test_lora()
