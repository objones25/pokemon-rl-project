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

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import EntryNotFoundError


class HfClient(Protocol):
    def upload_bytes(self, data: bytes, path_in_repo: str) -> None: ...
    def download_bytes(self, path_in_repo: str) -> bytes | None: ...


class AtomicHfClient(HfClient, Protocol):
    """An HfClient that can also publish several files as one atomic
    commit. Kept as its own (narrower) Protocol rather than added to
    HfClient directly: data_collection's per-shard uploads are correctly
    modeled as independent single-file commits and never need this, so
    adding it to the base Protocol would force every HfClient fake in this
    project to implement a method most of them never call (interface
    segregation). Only consumers that publish a multi-file artifact where
    partial-upload is not an acceptable state -- e.g.
    contrastive_pretrain's frozen encoder (weights + config + latent
    stats) -- should require this Protocol."""

    def upload_many_bytes(self, files: dict[str, bytes], commit_message: str) -> None: ...


class RealHfClient:
    """Adapts huggingface_hub.HfApi to the HfClient protocol.

    `revision` pins `download_bytes` to a resolved commit (or any git ref
    `HfApi.hf_hub_download` accepts) -- verified by introspection against the
    installed `huggingface_hub`: `hf_hub_download`'s `revision` keyword
    defaults to `None`, so passing it through unconditionally is a no-op for
    an unpinned client. Uploads are deliberately NOT pinned: they always
    target the branch head, and a `revision` there would be meaningless (a
    non-branch ref) or actively harmful (silently committing to a stale
    branch instead of the current one)."""

    def __init__(
        self, api: HfApi, repo_id: str, repo_type: str = "dataset", revision: str | None = None
    ) -> None:
        self._api = api
        self._repo_id = repo_id
        self._repo_type = repo_type
        self._revision = revision

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self._api.upload_file(
            path_or_fileobj=data,
            path_in_repo=path_in_repo,
            repo_id=self._repo_id,
            repo_type=self._repo_type,
        )

    def upload_many_bytes(self, files: dict[str, bytes], commit_message: str) -> None:
        operations = [
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=data)
            for path_in_repo, data in files.items()
        ]
        self._api.create_commit(
            repo_id=self._repo_id,
            repo_type=self._repo_type,
            operations=operations,
            commit_message=commit_message,
        )

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        try:
            local_path = self._api.hf_hub_download(
                repo_id=self._repo_id,
                filename=path_in_repo,
                repo_type=self._repo_type,
                revision=self._revision,
            )
        except EntryNotFoundError:
            return None
        return Path(local_path).read_bytes()
