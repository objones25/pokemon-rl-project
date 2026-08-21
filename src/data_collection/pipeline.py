"""Orchestrate Phase B: per approved video, extract -> validate -> dedup ->
batch -> upload, with resume-by-manifest and bounded retry on failure."""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from data_collection.batcher import FrameBatcher, FrameRecord, batch_to_parquet
from data_collection.dedup import PerceptualHashDeduper
from data_collection.frame_validator import FrameValidator
from data_collection.hf_uploader import HfUploader
from data_collection.matching import load_template_gray
from data_collection.observability import build_contact_sheet
from data_collection.registry import VideoSource


class TrackioRunLike(Protocol):
    def log(self, metrics: dict) -> None: ...
    def finish(self) -> None: ...


def retry_with_backoff(
    func: Callable[[], None],
    max_retries: int,
    base_delay: float,
    sleep_func: Callable[[float], None],
) -> None:
    attempt = 0
    while True:
        try:
            func()
            return
        except Exception:
            attempt += 1
            if attempt >= max_retries:
                raise
            sleep_func(base_delay * (2 ** (attempt - 1)))


@dataclass
class PipelineDeps:
    frame_source: Callable[[VideoSource], Iterator[np.ndarray]]
    uploader: HfUploader
    logger: logging.Logger
    trackio_run: TrackioRunLike | None = None
    batch_size: int = 500
    max_retries: int = 3
    sleep_func: Callable[[float], None] = field(default=time.sleep)


def _process_video(video: VideoSource, deps: PipelineDeps) -> None:
    reference_patch = load_template_gray(video.reference_patch_path)
    validator = FrameValidator(reference_patch, baseline_score=video.match_confidence_baseline)
    deduper = PerceptualHashDeduper()
    batcher = FrameBatcher(batch_size=deps.batch_size)

    sampled = kept = dropped_dedup = dropped_anomaly = shard_index = 0

    for frame in deps.frame_source(video):
        sampled += 1
        result = validator.validate(frame)
        if not result.keep:
            dropped_anomaly += 1
            continue
        if result.halted:
            deps.logger.warning(
                "video_halted_on_anomaly_rate", extra={"video_id": video.video_id}
            )
            break
        if deduper.is_duplicate(frame):
            dropped_dedup += 1
            continue

        kept += 1
        record = FrameRecord(
            image=frame, video_id=video.video_id, timestamp_s=sampled / 2.0, game=video.game
        )
        full_batch = batcher.add(record)
        if full_batch is not None:
            shard_index = _flush_batch(full_batch, video.video_id, shard_index, deps)

    trailing = batcher.flush()
    if trailing is not None:
        _flush_batch(trailing, video.video_id, shard_index, deps)

    dedup_rejection_rate = dropped_dedup / sampled if sampled else 0.0
    anomaly_drop_rate = dropped_anomaly / sampled if sampled else 0.0
    metrics = {
        "video_id": video.video_id,
        "sampled": sampled,
        "kept": kept,
        "dropped_dedup": dropped_dedup,
        "dropped_anomaly": dropped_anomaly,
        "dedup_rejection_rate": dedup_rejection_rate,
        "anomaly_drop_rate": anomaly_drop_rate,
    }
    deps.logger.info("video_complete", extra=metrics)
    if deps.trackio_run is not None:
        deps.trackio_run.log(metrics)


def _flush_batch(
    batch: list[FrameRecord], video_id: str, shard_index: int, deps: PipelineDeps
) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        shard_path = Path(tmp_dir) / "shard.parquet"
        batch_to_parquet(batch, shard_path)
        deps.uploader.upload_shard(shard_path, video_id, shard_index)

        contact_sheet = build_contact_sheet([record.image for record in batch])
        preview_path = Path(tmp_dir) / "contact_sheet.png"
        cv2.imwrite(str(preview_path), contact_sheet)
        deps.uploader.upload_preview(preview_path, video_id, shard_index)
    return shard_index + 1


def run_pipeline(registry: list[VideoSource], deps: PipelineDeps) -> None:
    manifest = deps.uploader.load_manifest()

    for video in registry:
        if manifest.is_complete(video.video_id):
            continue

        try:
            retry_with_backoff(
                lambda v=video: _process_video(v, deps),
                max_retries=deps.max_retries,
                base_delay=1.0,
                sleep_func=deps.sleep_func,
            )
        except Exception as exc:
            deps.logger.error(
                "video_failed", extra={"video_id": video.video_id, "reason": str(exc)}
            )
            manifest.mark_failed(video.video_id, str(exc))
            deps.uploader.save_manifest(manifest)
            continue

        manifest.mark_complete(video.video_id)
        deps.uploader.save_manifest(manifest)

    if deps.trackio_run is not None:
        deps.trackio_run.finish()
