"""Full training-state checkpointing to the local RunPod network volume.
Separate from contrastive_pretrain.encoder_io, which handles the
weights-only frozen artifact pushed to the HF Hub -- these are the two
tiers described in the design spec, split by cost/purpose, not just by
file.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_checkpoint_state(
    epoch: int,
    global_step: int,
    model: nn.Module,
    projector: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    dataloader_state: dict | None,
    best_val_loss: float,
    local_cache_dir: str | None = None,
) -> dict:
    """`model` must be the raw, uncompiled module -- callers must never
    pass a torch.compile-wrapped model here, since its state_dict keys
    may carry an `_orig_mod.` prefix that a freshly-constructed raw
    module on resume won't have. The same rule applies to `projector`,
    though in practice the projector is never compiled, so its
    state_dict keys are already prefix-free.

    `projector` MUST be checkpointed alongside `model`: the optimizer's
    parameter groups span both modules, so restoring optimizer moments
    onto a freshly-random projector would silently reset the projection
    head on every resume while pretending its Adam state was still valid.

    `dataloader_state` may be None, meaning "nothing to restore" -- used
    at epoch boundaries, where the just-finished iterator's state would
    otherwise resume as immediately-exhausted and train zero steps.

    `local_cache_dir` is recorded so a resume can detect the data source
    changed since this checkpoint was saved -- restoring dataloader state
    built under a different pipeline structure (e.g. streaming vs.
    local-cache, which drops a pre-shuffle resize-map stage) corrupts
    `StatefulDataLoader.load_state_dict()` silently, then raises
    `KeyError: 'examples_iterable'` on the first batch, not at load
    time."""
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "dataloader": dataloader_state,
        "best_val_loss": best_val_loss,
        "local_cache_dir": local_cache_dir,
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


def prune_checkpoints(checkpoint_dir: str | Path, keep_last_n: int) -> list[Path]:
    """Deletes all but the `keep_last_n` newest checkpoints, returning the
    paths removed (oldest first). Newest is by the same zero-padded-step
    filename sort `find_latest_checkpoint` uses, so the resume point is
    always among the survivors.

    Retention is safe here specifically because the *best* encoder is not
    kept only on this volume: run_training pushes it to the Hub via
    push_frozen_encoder whenever val loss improves. These files are resume
    state, not the artifact -- so keeping a short tail is enough.

    Sized against a real constraint, not taste: a checkpoint is ~336MB
    (ResNet-50 encoder + projector + AdamW moments, measured), and an
    unpruned 100-epoch run writes ~138 of them -- ~46GB on a 50GB network
    volume that must also hold the resize cache."""
    if keep_last_n < 1:
        raise ValueError(f"keep_last_n must be at least 1, got {keep_last_n}")
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    candidates = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
    stale = candidates[:-keep_last_n]
    for path in stale:
        path.unlink(missing_ok=True)
    return stale


def restore_optimizer_and_scheduler(optimizer: Optimizer, scheduler: LRScheduler, state: dict) -> None:
    """Caller must construct `scheduler` (attached to `optimizer`) BEFORE
    calling this function. This follows PyTorch's documented recommended order
    for restoring optimizer+scheduler state and is defensive/forward-compatible
    best practice, even though it may be empirically a no-op on the current
    torch version for some scheduler types."""
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
