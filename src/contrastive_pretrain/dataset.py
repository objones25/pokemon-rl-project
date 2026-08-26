"""Streaming dataset construction. Augmentation randomness is a per-row
deterministic seed derived from (base_seed, video_id, timestamp_s), NOT
a shared torch.Generator -- a shared Generator would be copied identically
into every StatefulDataLoader worker process, producing correlated (not
independent) augmentation sequences across workers. This also means
resuming needs zero augmentation-RNG checkpoint state: the seed for a
given row is always re-derivable from data the row already carries.
"""

from __future__ import annotations

import functools
import hashlib
import logging

import datasets
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision.transforms.v2 import functional as TF

from contrastive_pretrain.augmentation import AugmentationConfig, make_pair
from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.model import INPUT_HEIGHT, INPUT_WIDTH

logger = logging.getLogger(__name__)


def row_seed(base_seed: int, video_id: str, timestamp_s: float) -> int:
    digest = hashlib.sha256(f"{base_seed}:{video_id}:{timestamp_s}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _resize_to_canonical(example: dict) -> dict:
    """Must run BEFORE .shuffle() in build_train_dataset -- source videos are
    stored at their native crop resolution (up to 2400x2160, per
    configs/video_sources.yaml), not pre-resized to the model's 144x160
    input. .shuffle()'s buffer holds buffer_size examples PER WORKER
    (independent buffers, not shared), so buffering native-resolution
    frames there is what actually OOM-killed a real training pod at
    shuffle_buffer_size=10_000 x num_workers=8: up to 80,000 buffered
    frames at ~5MB each (2400x2160 uint8) is tens of GB, not the
    design spec's ~460MB estimate, which assumed already-small frames.
    Resizing first means the buffer only ever holds ~23KB (144x160) frames."""
    frame = TF.to_image(example["image"])  # (1, H, W) uint8
    frame = TF.resize(frame, [INPUT_HEIGHT, INPUT_WIDTH], antialias=True)
    return {"image": frame}


class _ResizeToCanonicalWithProgress:
    """Wraps _resize_to_canonical with periodic progress logging, kept as a
    separate class rather than baked into _resize_to_canonical itself so
    that function stays a plain, stateless, directly-unit-testable
    transform. This is the step that runs while .shuffle()'s buffer is
    filling (up to shuffle_buffer_size rows per DataLoader worker, before
    ANY training batch can be produced) -- previously the only wall-clock
    evidence of that multi-minute wait was tqdm progress bars from
    unrelated Hub calls, with nothing from this project's own pipeline in
    between. Runs inside each DataLoader worker subprocess when
    num_workers > 0 (forked on Linux, inheriting the parent's already-
    configured root logger), so `rows_processed` counts per-worker, not
    globally across all workers."""

    def __init__(self, log_every_n: int = 1000) -> None:
        self._log_every_n = log_every_n
        self._count = 0

    def __call__(self, example: dict) -> dict:
        result = _resize_to_canonical(example)
        self._count += 1
        if self._count % self._log_every_n == 0:
            logger.info(
                "resize_to_canonical_progress",
                extra={"rows_processed": self._count, "video_id": example.get("video_id")},
            )
        return result


def to_pair_transform(example: dict, augmentation_config: AugmentationConfig, base_seed: int) -> dict:
    frame = TF.to_image(example["image"])  # (1, H, W) uint8
    frame = TF.resize(frame, [INPUT_HEIGHT, INPUT_WIDTH], antialias=True)
    seed = row_seed(base_seed, example["video_id"], example["timestamp_s"])
    rng = torch.Generator().manual_seed(seed)
    view_a, view_b = make_pair(frame, augmentation_config, rng)
    return {"original": frame, "view_a": view_a, "view_b": view_b}


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    snapshot_every_n_steps: int,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> StatefulDataLoader:
    """drop_last defaults to True for the training loader for two reasons:
    it keeps batch shape fixed (cudnn.benchmark and torch.compile both
    depend on that), and a variable-size final batch would itself break
    that fixed-shape assumption. The validation loader has no such
    requirement (compute_val_loss runs eval-mode, batch-shape-agnostic) and
    must pass drop_last=False -- the held-out val_video_ids can easily
    total fewer rows than one training batch_size, and drop_last=True on a
    dataset smaller than one batch yields zero batches, which
    compute_val_loss treats as a hard failure. num_workers must stay the
    same between the run that saved a dataloader checkpoint and the run
    that resumes it -- StatefulDataLoader.load_state_dict requires it."""
    return StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        snapshot_every_n_steps=snapshot_every_n_steps,
    )


def _load_base_stream(config: TrainingConfig):
    if config.local_cache_dir:
        return datasets.load_dataset(
            "parquet",
            data_files=f"{config.local_cache_dir}/shards/**/*.parquet",
            split="train",
            streaming=True,
        )
    return datasets.load_dataset(config.dataset_repo_id, streaming=True, split="train")


def build_train_dataset(config: TrainingConfig):
    # Local import: resize_cache.py imports _resize_to_canonical from this
    # module at module level, so importing resize_cache back at this
    # module's top level would be circular.
    from contrastive_pretrain.resize_cache import ensure_local_cache

    ensure_local_cache(config)
    logger.info(
        "build_train_dataset",
        extra={
            "dataset_repo_id": config.dataset_repo_id,
            "shuffle_buffer_size": config.shuffle_buffer_size,
            "val_video_ids": config.val_video_ids,
        },
    )
    ds = _load_base_stream(config)
    ds = ds.filter(lambda ex: ex["video_id"] not in config.val_video_ids)
    if not config.local_cache_dir:
        ds = ds.map(_ResizeToCanonicalWithProgress())  # BEFORE shuffle -- see its
        # docstring; skipped for the local-cache path, whose frames are
        # already canonical-sized (see resize_cache.py's design rationale).
    ds = ds.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.seed)
    ds = ds.map(
        functools.partial(to_pair_transform, augmentation_config=AugmentationConfig(), base_seed=config.seed),
        remove_columns=["image"],
    )
    return ds


def build_val_dataset(config: TrainingConfig):
    # Local import: resize_cache.py imports _resize_to_canonical from this
    # module at module level, so importing resize_cache back at this
    # module's top level would be circular. Same reason as build_train_dataset.
    from contrastive_pretrain.resize_cache import ensure_local_cache

    ensure_local_cache(config)
    logger.info(
        "build_val_dataset",
        extra={"dataset_repo_id": config.dataset_repo_id, "val_video_ids": config.val_video_ids},
    )
    ds = _load_base_stream(config)
    ds = ds.filter(lambda ex: ex["video_id"] in config.val_video_ids)
    if not config.local_cache_dir:
        ds = ds.map(_ResizeToCanonicalWithProgress())
    ds = ds.map(
        functools.partial(to_pair_transform, augmentation_config=AugmentationConfig(), base_seed=config.seed),
        remove_columns=["image"],
    )
    return ds
