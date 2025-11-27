"""
Clean SAM3 training launcher for Windows.

Handles:
1. Loading Phase 1 weights (model only, no optimizer)
2. Gloo backend float32 conversion
3. Standard resolution (no RoPE hacks)
"""

import os
import sys

# Add sam3 to path
sys.path.insert(0, r"C:\Users\odressler\sam3")

import torch
import torch.multiprocessing as mp
from argparse import ArgumentParser
from omegaconf import OmegaConf

from sam3.train.utils.train_utils import register_omegaconf_resolvers
from iopath.common.file_io import g_pathmgr


def get_1d_freqs(dim: int, end: int, theta: float = 10000.0):
    """Generate 1D frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return freqs


def precompute_freqs_cis(dim: int, grid_size: int, theta: float = 10000.0):
    """
    Precompute RoPE frequencies for 2D grid (ViTDet formula).

    This is the CORRECT way to generate RoPE - can't just interpolate!
    """
    freqs_1d = get_1d_freqs(dim, grid_size, theta)
    freqs_x = freqs_1d.repeat(grid_size, 1)
    freqs_y = freqs_1d.repeat_interleave(grid_size, dim=0)
    freqs = torch.cat([freqs_x, freqs_y], dim=1)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis



def resize_rope_embeddings(model, target_resolution, patch_size=14):
    """Resize RoPE embeddings by regenerating them for target resolution."""
    import math

    target_grid = target_resolution // patch_size
    target_seq_len = target_grid * target_grid

    print(f"[RoPE] Target: {target_resolution}px -> {target_grid}x{target_grid} grid ({target_seq_len} tokens)")

    def resize_module(module, window_context=None):
        """Recursively find and regenerate freqs_cis buffers."""
        found = False
        current_window = window_context

        module_window_attr = getattr(module, "window_size", None)
        if isinstance(module_window_attr, int):
            current_window = module_window_attr

        if (
            hasattr(module, "freqs_cis")
            and isinstance(module.freqs_cis, torch.Tensor)
            and getattr(module, "input_size", None) is not None
        ):
            uses_window = current_window is not None and current_window > 0
            if not uses_window:
                cls_offset = 1 if getattr(module, "cls_token", False) else 0
                seq_len = module.freqs_cis.shape[0] - cls_offset
                if seq_len > 0:
                    grid_candidate = int(math.sqrt(seq_len))
                    if grid_candidate * grid_candidate == seq_len and grid_candidate != target_grid:
                        print(
                            f"[RoPE] Regenerating {type(module).__name__}: "
                            f"{grid_candidate}x{grid_candidate} -> {target_grid}x{target_grid}"
                        )
                        head_dim_complex = module.freqs_cis.shape[1]
                        device = module.freqs_cis.device
                        dtype = module.freqs_cis.dtype
                        new_freqs = precompute_freqs_cis(
                            head_dim_complex, target_grid, theta=10000.0
                        )
                        new_freqs = new_freqs.to(device=device, dtype=dtype)
                        if cls_offset:
                            zeros = torch.zeros(
                                head_dim_complex, dtype=torch.float32, device=device
                            )
                            cls_freqs_cis = torch.polar(torch.ones_like(zeros), zeros).to(dtype=dtype)[
                                None, :
                            ]
                            new_freqs = torch.cat([cls_freqs_cis, new_freqs], dim=0)
                        module.register_buffer("freqs_cis", new_freqs, persistent=False)
                        found = True

        for child in module.children():
            if resize_module(child, current_window):
                found = True

        return found

    found = resize_module(model)
    if found:
        print(f"[RoPE] Regeneration complete!")
    else:
        print(f"[RoPE] No freqs_cis found - using default")

    return model


def worker_process(rank, world_size, main_port, cfg_dict, weights_path):
    """Worker process - runs in isolated subprocess via spawn."""

    # Convert dict back to OmegaConf
    cfg = OmegaConf.create(cfg_dict)

    # Setup distributed env
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Import and patch AFTER spawn (fresh process)
    from sam3.train import trainer as trainer_module

    # Patch 1: Load weights after model creation
    _original_load_checkpoint = trainer_module.Trainer.load_checkpoint

    def _patched_load_checkpoint(self):
        """Load weights from Phase 1, ignore missing optimizer/scheduler."""
        # Check for checkpoint.pt in save_dir (from previous runs)
        save_dir = self.checkpoint_conf.save_dir if hasattr(self.checkpoint_conf, 'save_dir') else None
        ckpt_in_savedir = os.path.join(save_dir, "checkpoint.pt") if save_dir and os.path.exists(save_dir) else None

        # If checkpoint.pt exists, use original loader (resuming)
        if ckpt_in_savedir and os.path.exists(ckpt_in_savedir):
            print(f"[Worker {rank}] Resuming from {ckpt_in_savedir}")
            result = _original_load_checkpoint(self)
            _maybe_resize_rope(self)
            return result

        # Otherwise, load Phase 1 weights if provided
        if weights_path and os.path.exists(weights_path):
            print(f"[Worker {rank}] Loading Phase 1 weights: {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')

            if 'model' in checkpoint:
                missing, unexpected = self.model.load_state_dict(checkpoint['model'], strict=False)
                print(f"[Worker {rank}] Loaded {len(checkpoint['model'])} keys")
                if len(missing) > 0:
                    print(f"[Worker {rank}] Missing {len(missing)} keys (random init)")
                if len(unexpected) > 0:
                    print(f"[Worker {rank}] Unexpected {len(unexpected)} keys (ignored)")
            else:
                print(f"[Worker {rank}] WARNING: No 'model' key in checkpoint")
        else:
            print(f"[Worker {rank}] No weights to load, starting from scratch")

        _maybe_resize_rope(self)

    # Patch 2: Gloo backend float32 conversion
    _original_setup_ddp = trainer_module.Trainer._setup_ddp_distributed_training

    def _patched_setup_ddp(self, distributed_conf, accelerator):
        """Convert bfloat16 to float32 for gloo backend."""
        if hasattr(self, 'model') and self.model is not None:
            bf16_params = sum(1 for p in self.model.parameters() if p.dtype == torch.bfloat16)
            if bf16_params > 0:
                print(f"[Worker {rank}] Converting {bf16_params} bfloat16 params to float32 (gloo)")
                self.model = self.model.float()

        # Call original DDP setup
        return _original_setup_ddp(self, distributed_conf, accelerator)

    def _maybe_resize_rope(self):
        target_res = cfg.scratch.resolution if 'scratch' in cfg and 'resolution' in cfg.scratch else 1008
        if target_res == 1008 or not weights_path:
            return
        if not hasattr(self, 'model') or self.model is None:
            return
        model_to_resize = self.model.module if hasattr(self.model, 'module') else self.model
        print(f"[Worker {rank}] Regenerating RoPE for resolution {target_res}")
        resize_rope_embeddings(model_to_resize, target_res)

    # Apply patches

    trainer_module.Trainer.load_checkpoint = _patched_load_checkpoint

    trainer_module.Trainer._setup_ddp_distributed_training = _patched_setup_ddp


    # Register resolvers and instantiate trainer
    register_omegaconf_resolvers()

    from hydra.utils import instantiate
    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


if __name__ == "__main__":
    # Force spawn (required for Windows)
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Config file path")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs")
    args = parser.parse_args()

    # Load config
    register_omegaconf_resolvers()
    config_path = os.path.abspath(args.config)
    if not config_path.endswith('.yaml'):
        config_path += '.yaml'

    print(f"Loading config: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Remove defaults key
    if 'defaults' in cfg:
        OmegaConf.set_struct(cfg, False)
        del cfg['defaults']

    # Apply overrides
    cfg.launcher.gpus_per_node = args.num_gpus
    cfg.launcher.num_nodes = 1

    # Extract init_weights_from before passing to workers
    weights_path = None
    if 'checkpoint' in cfg.trainer and 'init_weights_from' in cfg.trainer.checkpoint:
        weights_path = cfg.trainer.checkpoint.init_weights_from
        print(f"Will load Phase 1 weights from: {weights_path}")
        # Remove from config so trainer doesn't see it
        OmegaConf.set_struct(cfg, False)
        del cfg.trainer.checkpoint['init_weights_from']

    # Create experiment dir and save config
    exp_dir = cfg.paths.experiment_log_dir
    os.makedirs(exp_dir, exist_ok=True)

    with g_pathmgr.open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print(f"Experiment dir: {exp_dir}")
    print(f"Resolution: {cfg.scratch.resolution}")
    print(f"Batch size: {cfg.scratch.train_batch_size}")
    print(f"GPUs: {args.num_gpus}")

    # Clean up old checkpoints
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f == "checkpoint.pt":
                old_ckpt = os.path.join(ckpt_dir, f)
                print(f"Removing old checkpoint: {old_ckpt}")
                os.remove(old_ckpt)

    # Spawn workers
    import random
    main_port = random.randint(29500, 29600)
    world_size = args.num_gpus

    # Convert config to dict for pickling (spawn requirement)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if world_size == 1:
        worker_process(0, 1, main_port, cfg_dict, weights_path)
    else:
        mp.spawn(
            worker_process,
            args=(world_size, main_port, cfg_dict, weights_path),
            nprocs=world_size,
            join=True
        )

    print("Training complete!")
