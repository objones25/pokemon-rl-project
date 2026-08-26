"""One-time local Parquet cache of resized frames, built from
objones25/pokemon-frames' native-resolution shards. See
docs/superpowers/specs/2026-08-25-contrastive-pretrain-resize-cache-design.md
for the full design rationale.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import datasets
from PIL import Image

from contrastive_pretrain.dataset import _resize_to_canonical

logger = logging.getLogger(__name__)


def _resize_row_for_cache(example: dict) -> dict:
    """_resize_to_canonical returns a channel-first (1, H, W) uint8 tensor --
    correct for the streaming pipeline (which never re-serializes it), but
    writing that tensor (or a raw (H, W) numpy array) straight into a
    datasets.Image()-typed column silently corrupts it on parquet
    round-trip: verified empirically that it either collapses the column
    to a nested-list type (losing the Image feature entirely) or, with the
    schema preserved explicitly, downcasts pixels to 32-bit int mode.
    Returning an actual PIL.Image sidesteps this -- datasets recognizes it
    directly and PNG-encodes it under the Image() feature, no explicit
    features= needed on .map()."""
    frame = _resize_to_canonical(example)["image"]  # (1, H, W) uint8
    return {"image": Image.fromarray(frame.squeeze(0).numpy(), mode="L")}


def build_local_resize_cache(
    list_shard_paths: Callable[[], list[str]],
    download_shard: Callable[[str], bytes],
    local_cache_dir: Path,
) -> None:
    """Downloads and resizes every shard `list_shard_paths()` returns that
    isn't already present under `local_cache_dir`, writing each as a local
    Parquet shard at the same relative path. Safe to interrupt and rerun:
    already-completed shards are skipped, and a shard is never considered
    complete until its output file has been atomically renamed into
    place."""
    shard_paths = list_shard_paths()
    completed = 0
    skipped = 0
    total_rows = 0
    for shard_path in shard_paths:
        output_path = local_cache_dir / shard_path
        if output_path.exists():
            skipped += 1
            logger.info("resize_cache_shard_skipped", extra={"shard_path": shard_path})
            continue

        start = time.monotonic()
        raw_bytes = download_shard(shard_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_tmp_path = output_path.parent / f"{output_path.name}.raw.tmp"
        output_tmp_path = output_path.parent / f"{output_path.name}.tmp"
        try:
            raw_tmp_path.write_bytes(raw_bytes)
            raw_dataset = datasets.Dataset.from_parquet(str(raw_tmp_path))
            resized_dataset = raw_dataset.map(_resize_row_for_cache)
            resized_dataset.to_parquet(str(output_tmp_path))
            os.replace(output_tmp_path, output_path)
        finally:
            raw_tmp_path.unlink(missing_ok=True)

        elapsed = time.monotonic() - start
        completed += 1
        total_rows += resized_dataset.num_rows
        logger.info(
            "resize_cache_shard_done",
            extra={"shard_path": shard_path, "rows": resized_dataset.num_rows, "elapsed_s": elapsed},
        )

    logger.info(
        "resize_cache_complete",
        extra={"shard_count": completed, "skipped_count": skipped, "total_rows": total_rows},
    )
