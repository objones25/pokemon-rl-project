"""Full training-state checkpointing to the local RunPod network volume.
Separate from contrastive_pretrain.encoder_io, which handles the
weights-only frozen artifact pushed to the HF Hub -- these are the two
tiers described in the design spec, split by cost/purpose, not just by
file.

This module owns the *schema* -- what a contrastive-pretraining checkpoint
contains. The file I/O underneath it (atomic write, discovery, retention)
is shared with the other sub-projects in checkpointing.io and re-exported
here so existing call sites keep one import.

Pruning is safe for this run specifically because the *best* encoder is not
kept only on the network volume: run_training pushes it to the Hub via
push_frozen_encoder whenever val loss improves. These files are resume
state, not the artifact -- so keeping a short tail is enough. See
ContrastivePretrainConfig.checkpoint_keep_last_n for the volume arithmetic.
"""

from __future__ import annotations

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from checkpointing.io import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)

__all__ = [
    "build_checkpoint_state",
    "find_latest_checkpoint",
    "load_checkpoint",
    "prune_checkpoints",
    "restore_optimizer_and_scheduler",
    "save_checkpoint",
]


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


def restore_optimizer_and_scheduler(optimizer: Optimizer, scheduler: LRScheduler, state: dict) -> None:
    """Caller must construct `scheduler` (attached to `optimizer`) BEFORE
    calling this function. This follows PyTorch's documented recommended order
    for restoring optimizer+scheduler state and is defensive/forward-compatible
    best practice, even though it may be empirically a no-op on the current
    torch version for some scheduler types."""
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
