"""HfClient Protocol + a real huggingface_hub.HfApi-backed implementation,
shared by every package in this project that persists something to the HF
Hub (data_collection's shards/manifest, contrastive_pretrain's checkpoints
and frozen encoder artifact). repo_type defaults to "dataset" to preserve
data_collection's existing behavior unchanged; contrastive_pretrain passes
repo_type="model" for the frozen encoder repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError


class HfClient(Protocol):
    def upload_bytes(self, data: bytes, path_in_repo: str) -> None: ...
    def download_bytes(self, path_in_repo: str) -> bytes | None: ...


class RealHfClient:
    """Adapts huggingface_hub.HfApi to the HfClient protocol."""

    def __init__(self, api: HfApi, repo_id: str, repo_type: str = "dataset") -> None:
        self._api = api
        self._repo_id = repo_id
        self._repo_type = repo_type

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self._api.upload_file(
            path_or_fileobj=data,
            path_in_repo=path_in_repo,
            repo_id=self._repo_id,
            repo_type=self._repo_type,
        )

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        try:
            local_path = self._api.hf_hub_download(
                repo_id=self._repo_id, filename=path_in_repo, repo_type=self._repo_type
            )
        except EntryNotFoundError:
            return None
        return Path(local_path).read_bytes()
