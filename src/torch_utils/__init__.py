"""Shared PyTorch helpers with no project-specific logic -- a leaf module
every training loop (contrastive_pretrain, ppo) can depend on without either
depending on the other."""

from __future__ import annotations

import torch


def autocast_dtype(device: torch.device) -> torch.dtype:
    """bf16 everywhere except MPS. The target production hardware is a CUDA
    A100 (bf16-native, Ampere+), and CPU's own recommended autocast dtype is
    also bf16 -- but MPS has weak bf16 kernel coverage and torch's own
    per-device guidance there is fp16 (see scripts/check_env.py's autocast
    table in the pytorch skill). No GradScaler is used anywhere in this loop
    (torch reports none available on MPS either), so this only changes local
    MPS dev-loop numerics -- never the CUDA A100 production path.
    """
    return torch.float16 if device.type == "mps" else torch.bfloat16
