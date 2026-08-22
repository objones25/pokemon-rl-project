"""Orchestrate Phase B: per approved video, extract -> dedup -> batch ->
upload, with resume-by-manifest and bounded retry on failure.

Videos are processed concurrently (PipelineDeps.max_concurrent_videos, default
1 = sequential) via a thread pool -- the heavy per-video work (ffmpeg decode)
already happens in a separate OS process outside the GIL, so threads give
real concurrency here without ProcessPoolExecutor's pickling constraints on
the frame_source closure. The Manifest is shared, mutable state read/written
from every worker thread, so every access to it is guarded by manifest_lock.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from data_collection.batcher import FrameBatcher, FrameRecord, batch_to_parquet
from data_collection.dedup import PerceptualHashDeduper
from data_collection.hf_uploader import HfUploader, Manifest
from data_collection.registry import VideoSource
from observability.visualization import build_contact_sheet

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class PipelineResult:
    completed: int
    failed: int


@dataclass
class PipelineDeps:
    # (video, resume_seconds) -> frames. resume_seconds is 0.0 for a video
    # with no saved progress; a real offset seeks past already-processed
    # content on a resumed video, via ffmpeg's -ss (see extract.stream_frames).
    frame_source: Callable[[VideoSource, float], Iterator[np.ndarray]]
    uploader: HfUploader
    trackio_run: TrackioRunLike | None = None
    batch_size: int = 500
    max_retries: int = 3
    sleep_func: Callable[[float], None] = field(default=time.sleep)
    sample_fps: float = 2.0
    max_concurrent_videos: int = 1


_PROGRESS_CHECKPOINT_INTERVAL_SAMPLES = 200


def _checkpoint_progress(
    video: VideoSource,
    deps: PipelineDeps,
    manifest: Manifest,
    manifest_lock: threading.Lock,
    started_at: float,
    sampled: int,
    kept: int,
    dropped_dedup: int,
    shard_index: int,
) -> None:
    """Periodic checkpoint during a single video's (possibly hours-long)
    frame loop: logs progress (without this, nothing is logged between the
    video starting and finishing -- indistinguishable from "hung" on an
    8+ hour video) AND persists a resume point to the manifest, so a killed
    run picks up from here instead of restarting the video from frame 0.
    """
    resume_seconds = sampled / deps.sample_fps
    metrics = {
        "video_id": video.video_id,
        "sampled": sampled,
        "kept": kept,
        "dropped_dedup": dropped_dedup,
        "video_time_s": resume_seconds,
        "elapsed_wall_s": time.monotonic() - started_at,
    }
    logger.info("video_progress", extra=metrics)
    if deps.trackio_run is not None:
        deps.trackio_run.log(metrics)

    with manifest_lock:
        manifest.save_progress(video.video_id, resume_seconds, shard_index)
        deps.uploader.save_manifest(manifest)


def _process_video(
    video: VideoSource, deps: PipelineDeps, manifest: Manifest, manifest_lock: threading.Lock
) -> None:
    deduper = PerceptualHashDeduper()
    batcher = FrameBatcher(batch_size=deps.batch_size)

    with manifest_lock:
        saved_progress = manifest.get_progress(video.video_id)
    resume_seconds = saved_progress["resume_seconds"] if saved_progress else 0.0
    shard_index = saved_progress["next_shard_index"] if saved_progress else 0
    if saved_progress:
        logger.info(
            "video_resuming",
            extra={
                "video_id": video.video_id,
                "resume_seconds": resume_seconds,
                "next_shard_index": shard_index,
            },
        )

    sampled = round(resume_seconds * deps.sample_fps)
    kept = dropped_dedup = 0
    started_at = time.monotonic()

    for frame in deps.frame_source(video, resume_seconds):
        sampled += 1
        if sampled % _PROGRESS_CHECKPOINT_INTERVAL_SAMPLES == 0:
            _checkpoint_progress(
                video,
                deps,
                manifest,
                manifest_lock,
                started_at,
                sampled,
                kept,
                dropped_dedup,
                shard_index,
            )
        if deduper.is_duplicate(frame):
            dropped_dedup += 1
            continue

        kept += 1
        record = FrameRecord(
            image=frame,
            video_id=video.video_id,
            timestamp_s=(sampled - 1) / deps.sample_fps,
            game=video.game,
        )
        full_batch = batcher.add(record)
        if full_batch is not None:
            shard_index = _flush_batch(full_batch, video.video_id, shard_index, deps)

    trailing = batcher.flush()
    if trailing is not None:
        shard_index = _flush_batch(trailing, video.video_id, shard_index, deps)

    dedup_rejection_rate = dropped_dedup / sampled if sampled else 0.0
    metrics = {
        "video_id": video.video_id,
        "sampled": sampled,
        "kept": kept,
        "dropped_dedup": dropped_dedup,
        "dedup_rejection_rate": dedup_rejection_rate,
    }
    # "video_processed" (not "video_complete"): this event fires whenever the
    # frame loop finishes for any reason, including a zero-frame extraction,
    # which is about to be raised as a failure below. Logging/reporting
    # happens first so accurate counts reach Trackio even though the video
    # will be marked failed, not complete.
    logger.info("video_processed", extra=metrics)
    if deps.trackio_run is not None:
        deps.trackio_run.log(metrics)

    if sampled == 0:
        raise RuntimeError(f"no frames extracted for video {video.video_id}")

    # Save the final position too: if this attempt got cut off by an
    # exception on a *later* frame than the last periodic checkpoint (or
    # never hit one at all, for a short video), the next retry should still
    # resume from here rather than from a stale or absent checkpoint.
    with manifest_lock:
        manifest.save_progress(video.video_id, sampled / deps.sample_fps, shard_index)
        deps.uploader.save_manifest(manifest)


_CONTACT_SHEET_SAMPLE_SIZE = 64


def _sample_for_contact_sheet(
    images: list[np.ndarray], sample_size: int = _CONTACT_SHEET_SAMPLE_SIZE
) -> list[np.ndarray]:
    if len(images) <= sample_size:
        return images
    step = len(images) / sample_size
    return [images[int(i * step)] for i in range(sample_size)]


def _flush_batch(
    batch: list[FrameRecord], video_id: str, shard_index: int, deps: PipelineDeps
) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        shard_path = Path(tmp_dir) / "shard.parquet"
        batch_to_parquet(batch, shard_path)
        deps.uploader.upload_shard(shard_path, video_id, shard_index)

        sampled_images = _sample_for_contact_sheet([record.image for record in batch])
        contact_sheet = build_contact_sheet(sampled_images)
        preview_path = Path(tmp_dir) / "contact_sheet.png"
        cv2.imwrite(str(preview_path), contact_sheet)
        deps.uploader.upload_preview(preview_path, video_id, shard_index)
    return shard_index + 1


def _run_one_video(
    video: VideoSource, deps: PipelineDeps, manifest: Manifest, manifest_lock: threading.Lock
) -> bool:
    """Runs one video through retry_with_backoff and updates the manifest
    with the outcome. Returns True on success, False on exhausted-retries
    failure -- never raises, so it's safe to submit directly to a thread
    pool without a wrapper."""
    try:
        retry_with_backoff(
            lambda: _process_video(video, deps, manifest, manifest_lock),
            max_retries=deps.max_retries,
            base_delay=1.0,
            sleep_func=deps.sleep_func,
        )
    except Exception as exc:
        logger.error("video_failed", extra={"video_id": video.video_id, "reason": str(exc)})
        with manifest_lock:
            manifest.mark_failed(video.video_id, str(exc))
            deps.uploader.save_manifest(manifest)
        return False

    with manifest_lock:
        manifest.mark_complete(video.video_id)
        deps.uploader.save_manifest(manifest)
    return True


def run_pipeline(registry: list[VideoSource], deps: PipelineDeps) -> PipelineResult:
    manifest = deps.uploader.load_manifest()
    manifest_lock = threading.Lock()

    pending = [video for video in registry if not manifest.is_complete(video.video_id)]
    completed = len(registry) - len(pending)
    failed = 0

    if pending:
        max_workers = max(1, deps.max_concurrent_videos)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_one_video, video, deps, manifest, manifest_lock): video
                for video in pending
            }
            for future in as_completed(futures):
                if future.result():
                    completed += 1
                else:
                    failed += 1

    if deps.trackio_run is not None:
        deps.trackio_run.finish()

    return PipelineResult(completed=completed, failed=failed)
