import logging

import numpy as np
import pytest

from data_collection.hf_uploader import HfUploader, Manifest
from data_collection.pipeline import (
    PipelineDeps,
    PipelineResult,
    retry_with_backoff,
    run_pipeline,
)
from data_collection.registry import VideoSource


class FakeHfClient:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.uploads[path_in_repo] = data

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.uploads.get(path_in_repo)


def _source(video_id: str, reference_patch_path: str) -> VideoSource:
    return VideoSource(
        video_id=video_id,
        url=f"https://youtube.com/watch?v={video_id}",
        game="red",
        crop_x=0,
        crop_y=0,
        crop_w=4,
        crop_h=3,
        reference_patch_path=reference_patch_path,
        match_confidence_baseline=1.0,
    )


def _reference_frame() -> np.ndarray:
    return np.full((3, 4), 100, dtype=np.uint8)


def test_run_pipeline_uploads_shards_and_marks_manifest_complete(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        for _ in range(3):
            yield _reference_frame()

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        batch_size=10,
    )

    run_pipeline([source], deps)

    manifest = uploader.load_manifest()
    assert manifest.is_complete("abc123") is True
    assert any(path.startswith("shards/abc123/") for path in client.uploads)


def test_process_video_logs_periodic_progress_for_long_videos(tmp_path, caplog) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        for _ in range(250):  # > the 200-sample progress-log interval
            yield _reference_frame()

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(frame_source=frame_source, uploader=uploader, batch_size=1000)

    with caplog.at_level(logging.INFO, logger="data_collection.pipeline"):
        run_pipeline([source], deps)

    progress_records = [r for r in caplog.records if r.message == "video_progress"]
    assert len(progress_records) >= 1
    assert progress_records[0].sampled == 200
    # Every yielded frame is pixel-identical, so the deduper correctly
    # collapses frames 2+ as duplicates of frame 1 -- only 1 is ever kept.
    assert progress_records[0].kept == 1
    assert progress_records[0].video_time_s == 100.0  # 200 samples / 2.0 default sample_fps


def test_run_pipeline_uploads_a_contact_sheet_preview_per_batch(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        for _ in range(3):
            yield _reference_frame()

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        batch_size=10,
    )

    run_pipeline([source], deps)

    assert any(path.startswith("previews/abc123/") for path in client.uploads)


def test_run_pipeline_skips_already_completed_videos(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))

    calls = []

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        calls.append(video_source.video_id)
        return iter([])

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    manifest = Manifest()
    manifest.mark_complete("abc123")
    uploader.save_manifest(manifest)

    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
    )

    run_pipeline([source], deps)

    assert calls == []


def test_run_pipeline_marks_failed_after_exhausting_retries(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))

    def failing_frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        raise RuntimeError("network blip")
        yield  # pragma: no cover - unreachable, makes this a generator

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    sleeps: list[float] = []
    deps = PipelineDeps(
        frame_source=failing_frame_source,
        uploader=uploader,
        max_retries=2,
        sleep_func=sleeps.append,
    )

    run_pipeline([source], deps)

    manifest = uploader.load_manifest()
    assert manifest.is_complete("abc123") is False
    assert len(sleeps) == 1  # max_retries=2 total attempts -> 1 sleep between them


def test_run_pipeline_zero_frames_leaves_video_incomplete(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        return iter([])

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        max_retries=1,
        sleep_func=lambda _: None,
    )

    run_pipeline([source], deps)

    manifest = uploader.load_manifest()
    assert manifest.is_complete("abc123") is False


def test_run_pipeline_halts_video_on_sustained_anomaly_and_stops_early(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    # NOTE: a constant-valued reference (like the shared `_reference_frame()`
    # helper) is degenerate for cv2's TM_CCOEFF_NORMED: with zero variance in
    # the template, OpenCV's normalized cross-correlation always reports a
    # perfect score of 1.0 regardless of the frame content, so no frame could
    # ever fail to match and the halt could never trip. Use a reference patch
    # with real variance here so mismatches actually score low.
    reference = np.array(
        [[10, 50, 90, 130], [170, 210, 250, 30], [60, 100, 140, 180]], dtype=np.uint8
    )
    cv2.imwrite(str(reference_path), reference)
    source = _source("abc123", str(reference_path))
    unrelated = np.random.default_rng(seed=99).integers(0, 255, size=(3, 4), dtype=np.uint8)

    frames_consumed = {"count": 0}

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        # 300 frames total: never matches the reference, so the validator
        # should halt well before all 300 are consumed.
        for _ in range(300):
            frames_consumed["count"] += 1
            yield unrelated

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        batch_size=1000,
    )

    run_pipeline([source], deps)

    # FrameValidator's default window_size=50 means it can halt as early as
    # frame 50; well under the full 300-frame source either way.
    assert frames_consumed["count"] < 300


def test_run_pipeline_halted_video_is_not_marked_complete(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    # See NOTE above: a constant-valued reference is degenerate for
    # cv2's TM_CCOEFF_NORMED, so use a reference patch with real variance.
    reference = np.array(
        [[10, 50, 90, 130], [170, 210, 250, 30], [60, 100, 140, 180]], dtype=np.uint8
    )
    cv2.imwrite(str(reference_path), reference)
    source = _source("abc123", str(reference_path))
    unrelated = np.random.default_rng(seed=99).integers(0, 255, size=(3, 4), dtype=np.uint8)

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        for _ in range(300):
            yield unrelated

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        batch_size=1000,
        max_retries=1,
        sleep_func=lambda _: None,
    )

    run_pipeline([source], deps)

    manifest = uploader.load_manifest()
    assert manifest.is_complete("abc123") is False


def test_run_pipeline_processes_next_video_after_one_fails(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    bad_source = _source("bad_video", str(reference_path))
    good_source = _source("good_video", str(reference_path))

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        if video_source.video_id == "bad_video":
            raise RuntimeError("network blip")
            yield  # pragma: no cover - unreachable, makes this a generator
        for _ in range(3):
            yield _reference_frame()

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        max_retries=1,
        sleep_func=lambda _: None,
        batch_size=10,
    )

    result = run_pipeline([bad_source, good_source], deps)

    manifest = uploader.load_manifest()
    assert manifest.is_complete("bad_video") is False
    assert manifest.is_complete("good_video") is True
    assert result == PipelineResult(completed=1, failed=1)


def test_run_pipeline_resumes_from_last_checkpoint_after_failure(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))
    call_resume_seconds: list[float] = []

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        call_resume_seconds.append(resume_seconds)
        rng = np.random.default_rng(seed=len(call_resume_seconds))
        # Distinct frames each call so nothing gets deduped and the sample
        # count during the checkpoint is exact. 250 > the 200-sample
        # checkpoint interval, so a checkpoint fires mid-stream.
        for _ in range(250):
            yield rng.integers(0, 255, size=(3, 4), dtype=np.uint8)
        if len(call_resume_seconds) == 1:
            raise RuntimeError("network blip after first pass")

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        batch_size=1000,
        max_retries=2,
        sleep_func=lambda _: None,
    )

    result = run_pipeline([source], deps)

    assert result == PipelineResult(completed=1, failed=0)
    # First attempt starts from scratch; the retry resumes from the
    # checkpoint the first attempt saved at sample 200 (200 / 2.0 fps),
    # not from 0 again.
    assert call_resume_seconds == [0.0, 100.0]


def test_run_pipeline_processes_multiple_videos_concurrently(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    sources = [_source(f"video{i}", str(reference_path)) for i in range(5)]

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        for _ in range(3):
            yield _reference_frame()

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        batch_size=10,
        max_concurrent_videos=4,
    )

    result = run_pipeline(sources, deps)

    assert result == PipelineResult(completed=5, failed=0)
    manifest = uploader.load_manifest()
    for source in sources:
        # Every video's completion must actually land in the manifest --
        # this is the thing a manifest-save race between threads would lose.
        assert manifest.is_complete(source.video_id) is True
        assert any(path.startswith(f"shards/{source.video_id}/") for path in client.uploads)


def test_retry_with_backoff_succeeds_after_transient_failures() -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def flaky() -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")

    retry_with_backoff(flaky, max_retries=3, base_delay=1.0, sleep_func=sleeps.append)

    assert attempts["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_retry_with_backoff_raises_after_exhausting_retries() -> None:
    def always_fails() -> None:
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        retry_with_backoff(always_fails, max_retries=2, base_delay=1.0, sleep_func=lambda _: None)
