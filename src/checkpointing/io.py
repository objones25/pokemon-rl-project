"""Atomic checkpoint writes, newest-first discovery, and retention.

Shared by every sub-project that drives a long unattended paid run. These
four functions were originally written for contrastive_pretrain; they are
here because a second copy guarding a second paid run would drift from the
first, and the failure mode of drift is a run that cannot resume.

`pattern` is a parameter rather than a constant so two runs can share one
network volume without one pruning the other's resume point.
"""

from __future__ import annotations

from pathlib import Path

import torch

DEFAULT_PATTERN = "checkpoint_step*.pt"


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)  # atomic on POSIX -- no half-written checkpoint on a mid-save crash


def save_text_atomic(path: str | Path, text: str) -> None:
    """Same temp-file-then-replace pattern as `save_checkpoint`, for a
    caller's own manifest/commit-point file rather than a torch state dict --
    a plain `path.write_text(text)` can leave a truncated file on a mid-write
    crash, which is exactly the failure a "commit point" file exists to not
    have."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(Path(path), weights_only=True)


def find_latest_checkpoint(
    checkpoint_dir: str | Path, pattern: str = DEFAULT_PATTERN
) -> Path | None:
    """Newest by zero-padded-step filename sort, which is why callers must
    zero-pad: `step900` sorts after `step1300` as a string."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob(pattern))
    return candidates[-1] if candidates else None


def prune_checkpoints(
    checkpoint_dir: str | Path, keep_last_n: int, pattern: str = DEFAULT_PATTERN
) -> list[Path]:
    """Deletes all but the `keep_last_n` newest checkpoints matching
    `pattern`, returning the paths removed (oldest first). Newest is by the
    same filename sort `find_latest_checkpoint` uses, so the resume point is
    always among the survivors."""
    if keep_last_n < 1:
        raise ValueError(f"keep_last_n must be at least 1, got {keep_last_n}")
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    candidates = sorted(checkpoint_dir.glob(pattern))
    stale = candidates[:-keep_last_n]
    for path in stale:
        path.unlink(missing_ok=True)
    return stale
