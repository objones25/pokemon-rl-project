from pathlib import Path

import pytest

from data_collection.hf_uploader import HfUploader, Manifest
from tests.conftest import FakeHfClient


def test_manifest_starts_empty() -> None:
    manifest = Manifest()
    assert manifest.is_complete("abc123") is False


def test_manifest_mark_complete() -> None:
    manifest = Manifest()
    manifest.mark_complete("abc123")
    assert manifest.is_complete("abc123") is True


def test_manifest_json_round_trip() -> None:
    manifest = Manifest()
    manifest.mark_complete("abc123")
    manifest.mark_failed("def456", "ffmpeg crashed")

    restored = Manifest.from_json(manifest.to_json())

    assert restored.is_complete("abc123") is True
    assert restored.is_complete("def456") is False


def test_manifest_progress_starts_absent() -> None:
    manifest = Manifest()
    assert manifest.get_progress("abc123") is None


def test_manifest_save_and_get_progress() -> None:
    manifest = Manifest()
    manifest.save_progress("abc123", resume_seconds=120.5, next_shard_index=3)

    progress = manifest.get_progress("abc123")

    assert progress == {"resume_seconds": 120.5, "next_shard_index": 3}


def test_manifest_mark_complete_clears_progress() -> None:
    manifest = Manifest()
    manifest.save_progress("abc123", resume_seconds=120.5, next_shard_index=3)
    manifest.mark_complete("abc123")

    assert manifest.get_progress("abc123") is None


def test_manifest_progress_round_trips_through_json() -> None:
    manifest = Manifest()
    manifest.save_progress("abc123", resume_seconds=120.5, next_shard_index=3)

    restored = Manifest.from_json(manifest.to_json())

    assert restored.get_progress("abc123") == {"resume_seconds": 120.5, "next_shard_index": 3}


def test_upload_shard_writes_to_expected_path(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.parquet"
    shard_path.write_bytes(b"fake-parquet-bytes")
    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")

    path_in_repo = uploader.upload_shard(shard_path, video_id="abc123", shard_index=3)

    assert path_in_repo == "shards/abc123/00003.parquet"
    assert client.files["shards/abc123/00003.parquet"] == b"fake-parquet-bytes"


def test_upload_preview_writes_to_expected_path(tmp_path: Path) -> None:
    preview_path = tmp_path / "contact_sheet.png"
    preview_path.write_bytes(b"fake-png-bytes")
    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")

    path_in_repo = uploader.upload_preview(preview_path, video_id="abc123", shard_index=3)

    assert path_in_repo == "previews/abc123/00003.png"
    assert client.files["previews/abc123/00003.png"] == b"fake-png-bytes"


def test_load_manifest_returns_empty_when_not_yet_uploaded() -> None:
    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")

    manifest = uploader.load_manifest()

    assert manifest.is_complete("anything") is False


def test_save_then_load_manifest_round_trips() -> None:
    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    manifest = Manifest()
    manifest.mark_complete("abc123")

    uploader.save_manifest(manifest)
    reloaded = uploader.load_manifest()

    assert reloaded.is_complete("abc123") is True


class _FlakyThenSucceedsClient(FakeHfClient):
    """Fails upload_bytes a fixed number of times (any error, not rate-limit
    specific) before succeeding -- simulates an ordinary transient failure."""

    def __init__(self, fail_times: int) -> None:
        super().__init__()
        self._remaining_failures = fail_times

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("connection reset")
        super().upload_bytes(data, path_in_repo)


class _AlwaysRateLimitedClient(FakeHfClient):
    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        raise RuntimeError(
            "429 Too Many Requests for url: ...\n"
            "You have exceeded the rate limit for repository commits (256 per hour)."
        )


def test_upload_shard_retries_transient_upload_failures(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.parquet"
    shard_path.write_bytes(b"fake-parquet-bytes")
    client = _FlakyThenSucceedsClient(fail_times=2)
    sleeps: list[float] = []
    uploader = HfUploader(
        client, repo_id="me/pokemon-frames", max_retries=5, base_delay=1.0, sleep_func=sleeps.append
    )

    path_in_repo = uploader.upload_shard(shard_path, video_id="abc123", shard_index=3)

    assert path_in_repo == "shards/abc123/00003.parquet"
    assert client.files["shards/abc123/00003.parquet"] == b"fake-parquet-bytes"
    # Ordinary transient failures use the normal short exponential backoff.
    assert sleeps == [1.0, 2.0]


def test_save_manifest_gives_up_after_exhausting_retries(tmp_path: Path) -> None:
    client = _FlakyThenSucceedsClient(fail_times=99)
    uploader = HfUploader(
        client, repo_id="me/pokemon-frames", max_retries=3, base_delay=0, sleep_func=lambda _: None
    )

    with pytest.raises(RuntimeError, match="connection reset"):
        uploader.save_manifest(Manifest())


def test_save_manifest_uses_rate_limit_delay_on_a_429(tmp_path: Path) -> None:
    client = _AlwaysRateLimitedClient()
    sleeps: list[float] = []
    uploader = HfUploader(
        client,
        repo_id="me/pokemon-frames",
        max_retries=3,
        base_delay=1.0,
        rate_limit_delay=120.0,
        sleep_func=sleeps.append,
    )

    with pytest.raises(RuntimeError, match="429"):
        uploader.save_manifest(Manifest())

    # A rate-limit error waits the dedicated long delay, not the short
    # exponential schedule used for ordinary transient failures.
    assert sleeps == [120.0, 120.0]
