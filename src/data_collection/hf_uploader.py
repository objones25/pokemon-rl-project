"""Push frame shards to the HF dataset repo and track per-video progress
in a manifest.json stored in that same repo, for crash resume."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from hf_storage.client import HfClient
from hf_storage.retry import rate_limit_aware_backoff, retry_with_backoff

_MANIFEST_PATH = "manifest.json"


class Manifest:
    def __init__(
        self,
        completed: set[str] | None = None,
        failed: dict[str, str] | None = None,
        progress: dict[str, dict] | None = None,
    ) -> None:
        self._completed = set(completed or set())
        self._failed = dict(failed or {})
        self._progress: dict[str, dict] = dict(progress or {})

    def is_complete(self, video_id: str) -> bool:
        return video_id in self._completed

    def mark_complete(self, video_id: str) -> None:
        self._completed.add(video_id)
        self._failed.pop(video_id, None)
        self._progress.pop(video_id, None)

    def mark_failed(self, video_id: str, reason: str) -> None:
        self._failed[video_id] = reason

    def get_progress(self, video_id: str) -> dict | None:
        return self._progress.get(video_id)

    def save_progress(self, video_id: str, resume_seconds: float, next_shard_index: int) -> None:
        self._progress[video_id] = {
            "resume_seconds": resume_seconds,
            "next_shard_index": next_shard_index,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "completed": sorted(self._completed),
                "failed": self._failed,
                "progress": self._progress,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> Manifest:
        parsed = json.loads(data)
        return cls(
            completed=set(parsed.get("completed", [])),
            failed=parsed.get("failed", {}),
            progress=parsed.get("progress", {}),
        )


class HfUploader:
    """Retries each write call individually (rather than leaving retry to
    the caller) so a single flaky/rate-limited upload doesn't force
    pipeline.py to redo an entire video's frame extraction just to retry
    one HTTP call -- see retry_with_backoff's docstring and
    data_collection.retry.rate_limit_aware_backoff."""

    def __init__(
        self,
        client: HfClient,
        repo_id: str,
        max_retries: int = 5,
        base_delay: float = 2.0,
        rate_limit_delay: float = 120.0,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._repo_id = repo_id
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._sleep_func = sleep_func
        self._backoff = rate_limit_aware_backoff(base_delay, rate_limit_delay)

    def _upload_with_retry(self, data: bytes, path_in_repo: str) -> None:
        retry_with_backoff(
            lambda: self._client.upload_bytes(data, path_in_repo),
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            sleep_func=self._sleep_func,
            backoff_seconds=self._backoff,
        )

    def upload_shard(self, local_path: str | Path, video_id: str, shard_index: int) -> str:
        path_in_repo = f"shards/{video_id}/{shard_index:05d}.parquet"
        data = Path(local_path).read_bytes()
        self._upload_with_retry(data, path_in_repo)
        return path_in_repo

    def upload_preview(self, local_path: str | Path, video_id: str, shard_index: int) -> str:
        path_in_repo = f"previews/{video_id}/{shard_index:05d}.png"
        data = Path(local_path).read_bytes()
        self._upload_with_retry(data, path_in_repo)
        return path_in_repo

    def load_manifest(self) -> Manifest:
        data = self._client.download_bytes(_MANIFEST_PATH)
        if data is None:
            return Manifest()
        return Manifest.from_json(data.decode("utf-8"))

    def save_manifest(self, manifest: Manifest) -> None:
        self._upload_with_retry(manifest.to_json().encode("utf-8"), _MANIFEST_PATH)
