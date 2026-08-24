"""Full training-state checkpointing to the local RunPod network volume.
Separate from contrastive_pretrain.encoder_io, which handles the
weights-only frozen artifact pushed to the HF Hub -- these are the two
tiers described in the design spec, split by cost/purpose, not just by
file.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_checkpoint_state(
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    dataloader_state: dict,
    best_val_loss: float,
) -> dict:
    """`model` must be the raw, uncompiled module -- callers must never
    pass a torch.compile-wrapped model here, since its state_dict keys
    may carry an `_orig_mod.` prefix that a freshly-constructed raw
    module on resume won't have."""
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "dataloader": dataloader_state,
        "best_val_loss": best_val_loss,
    }


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)  # atomic on POSIX -- no half-written checkpoint on a mid-save crash


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(Path(path), weights_only=True)


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
    return candidates[-1] if candidates else None


def restore_optimizer_and_scheduler(optimizer: Optimizer, scheduler: LRScheduler, state: dict) -> None:
    """Caller must construct `scheduler` (attached to `optimizer`) BEFORE
    calling this function. This follows PyTorch's documented recommended order
    for restoring optimizer+scheduler state and is defensive/forward-compatible
    best practice, even though it may be empirically a no-op on the current
    torch version for some scheduler types."""
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
