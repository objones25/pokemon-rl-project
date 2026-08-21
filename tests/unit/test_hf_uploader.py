from pathlib import Path

import pytest

from data_collection.hf_uploader import HfUploader, Manifest


class FakeHfClient:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.uploads[path_in_repo] = data

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.uploads.get(path_in_repo)


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


def test_upload_shard_writes_to_expected_path(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.parquet"
    shard_path.write_bytes(b"fake-parquet-bytes")
    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")

    path_in_repo = uploader.upload_shard(shard_path, video_id="abc123", shard_index=3)

    assert path_in_repo == "shards/abc123/00003.parquet"
    assert client.uploads["shards/abc123/00003.parquet"] == b"fake-parquet-bytes"


def test_upload_preview_writes_to_expected_path(tmp_path: Path) -> None:
    preview_path = tmp_path / "contact_sheet.png"
    preview_path.write_bytes(b"fake-png-bytes")
    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")

    path_in_repo = uploader.upload_preview(preview_path, video_id="abc123", shard_index=3)

    assert path_in_repo == "previews/abc123/00003.png"
    assert client.uploads["previews/abc123/00003.png"] == b"fake-png-bytes"


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
