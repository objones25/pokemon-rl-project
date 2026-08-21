"""Push frame shards to the HF dataset repo and track per-video progress
in a manifest.json stored in that same repo, for crash resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class HfClient(Protocol):
    def upload_bytes(self, data: bytes, path_in_repo: str) -> None: ...
    def download_bytes(self, path_in_repo: str) -> bytes | None: ...


_MANIFEST_PATH = "manifest.json"


class Manifest:
    def __init__(
        self, completed: set[str] | None = None, failed: dict[str, str] | None = None
    ) -> None:
        self._completed = set(completed or set())
        self._failed = dict(failed or {})

    def is_complete(self, video_id: str) -> bool:
        return video_id in self._completed

    def mark_complete(self, video_id: str) -> None:
        self._completed.add(video_id)
        self._failed.pop(video_id, None)

    def mark_failed(self, video_id: str, reason: str) -> None:
        self._failed[video_id] = reason

    def to_json(self) -> str:
        return json.dumps({"completed": sorted(self._completed), "failed": self._failed})

    @classmethod
    def from_json(cls, data: str) -> "Manifest":
        parsed = json.loads(data)
        return cls(completed=set(parsed.get("completed", [])), failed=parsed.get("failed", {}))


class HfUploader:
    def __init__(self, client: HfClient, repo_id: str) -> None:
        self._client = client
        self._repo_id = repo_id

    def upload_shard(self, local_path: str | Path, video_id: str, shard_index: int) -> str:
        path_in_repo = f"shards/{video_id}/{shard_index:05d}.parquet"
        data = Path(local_path).read_bytes()
        self._client.upload_bytes(data, path_in_repo)
        return path_in_repo

    def upload_preview(self, local_path: str | Path, video_id: str, shard_index: int) -> str:
        path_in_repo = f"previews/{video_id}/{shard_index:05d}.png"
        data = Path(local_path).read_bytes()
        self._client.upload_bytes(data, path_in_repo)
        return path_in_repo

    def load_manifest(self) -> Manifest:
        data = self._client.download_bytes(_MANIFEST_PATH)
        if data is None:
            return Manifest()
        return Manifest.from_json(data.decode("utf-8"))

    def save_manifest(self, manifest: Manifest) -> None:
        self._client.upload_bytes(manifest.to_json().encode("utf-8"), _MANIFEST_PATH)
