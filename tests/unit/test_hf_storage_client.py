from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError

from hf_storage.client import RealHfClient


class _FakeHfApi:
    """Stands in for huggingface_hub.HfApi: upload_file/hf_hub_download/
    create_commit are the only methods RealHfClient calls, so this fake
    only implements those, backed by a tmp_path directory instead of the
    real Hub."""

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self.uploaded_calls: list[tuple[str, str, str]] = []
        self.commits: list[dict] = []
        self.download_calls: list[tuple[str, str, str, str | None]] = []

    def upload_file(self, path_or_fileobj: bytes, path_in_repo: str, repo_id: str, repo_type: str) -> None:
        self.uploaded_calls.append((path_in_repo, repo_id, repo_type))
        dest = self._tmp_path / path_in_repo.replace("/", "_")
        dest.write_bytes(path_or_fileobj)

    def create_commit(self, repo_id, repo_type, operations, commit_message) -> None:
        self.commits.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "commit_message": commit_message,
                "paths": [op.path_in_repo for op in operations],
            }
        )
        for op in operations:
            dest = self._tmp_path / op.path_in_repo.replace("/", "_")
            data = op.path_or_fileobj
            assert isinstance(data, bytes)  # every caller in this codebase passes bytes, never a path/file object
            dest.write_bytes(data)

    def hf_hub_download(
        self, repo_id: str, filename: str, repo_type: str, revision: str | None = None
    ) -> str:
        self.download_calls.append((repo_id, filename, repo_type, revision))
        dest = self._tmp_path / filename.replace("/", "_")
        if not dest.exists():
            raise EntryNotFoundError("not found")
        return str(dest)


def test_real_hf_client_round_trips_bytes(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)
    # _FakeHfApi structurally satisfies what RealHfClient actually calls
    # (upload_file/hf_hub_download) but isn't nominally an HfApi.
    client = RealHfClient(api, "me/repo", repo_type="model")  # type: ignore[arg-type]

    client.upload_bytes(b"hello", "config.json")
    result = client.download_bytes("config.json")

    assert result == b"hello"
    assert api.uploaded_calls == [("config.json", "me/repo", "model")]


def test_real_hf_client_download_bytes_returns_none_when_missing(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo")  # type: ignore[arg-type]

    assert client.download_bytes("missing.json") is None


def test_real_hf_client_defaults_to_dataset_repo_type(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo")  # type: ignore[arg-type]

    client.upload_bytes(b"x", "manifest.json")

    assert api.uploaded_calls == [("manifest.json", "me/repo", "dataset")]


def test_real_hf_client_upload_many_bytes_makes_one_commit_for_all_files(tmp_path) -> None:
    """The whole point of upload_many_bytes: several files land in a
    single Hub commit, not one commit per file -- so a caller publishing a
    multi-file artifact can never observe a partial (some-files-landed)
    state, which the old upload_bytes-per-file approach could leave behind
    on a mid-publish failure."""
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo", repo_type="model")  # type: ignore[arg-type]

    client.upload_many_bytes(
        {"model.safetensors": b"weights", "config.json": b"{}"},
        commit_message="Publish frozen encoder artifact",
    )

    assert len(api.commits) == 1
    commit = api.commits[0]
    assert commit["repo_id"] == "me/repo"
    assert commit["repo_type"] == "model"
    assert commit["commit_message"] == "Publish frozen encoder artifact"
    assert set(commit["paths"]) == {"model.safetensors", "config.json"}
    assert client.download_bytes("model.safetensors") == b"weights"
    assert client.download_bytes("config.json") == b"{}"


def test_download_bytes_forwards_repo_id_and_repo_type(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo", repo_type="model")  # type: ignore[arg-type]
    client.upload_bytes(b"hi", "file.txt")

    client.download_bytes("file.txt")

    assert api.download_calls == [("me/repo", "file.txt", "model", None)]


def test_download_bytes_forwards_a_pinned_revision_to_hf_hub_download(tmp_path) -> None:
    """PPOConfig.frozen_encoder_revision is required and pinned precisely so
    a later push to the encoder repo cannot change features underneath a
    running agent -- this is the plumbing that pin actually depends on."""
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo", repo_type="model", revision="abc123")  # type: ignore[arg-type]
    client.upload_bytes(b"hi", "file.txt")

    client.download_bytes("file.txt")

    assert api.download_calls == [("me/repo", "file.txt", "model", "abc123")]


def test_download_bytes_passes_none_revision_when_the_client_is_unpinned(tmp_path) -> None:
    """hf_hub_download's own `revision` keyword defaults to None (verified
    by introspection against the installed huggingface_hub), so an unpinned
    RealHfClient must pass None through explicitly rather than omit the
    keyword -- either behaves the same against the real API, but a caller
    stubbing hf_hub_download without a default for `revision` would only
    catch the omitted-keyword case, not a wrong non-None value."""
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo")  # type: ignore[arg-type]
    client.upload_bytes(b"hi", "file.txt")

    client.download_bytes("file.txt")

    assert api.download_calls == [("me/repo", "file.txt", "dataset", None)]


def test_upload_bytes_ignores_the_configured_revision(tmp_path) -> None:
    """Uploads always target the branch head -- pinning them to the same
    revision as downloads would be meaningless (a non-branch ref) or
    actively harmful (silently committing to a stale branch)."""
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo", repo_type="model", revision="abc123")  # type: ignore[arg-type]

    client.upload_bytes(b"hi", "file.txt")

    assert api.uploaded_calls == [("file.txt", "me/repo", "model")]


def test_download_bytes_does_not_swallow_unrelated_errors() -> None:
    class _BrokenApi:
        def hf_hub_download(
            self, repo_id: str, filename: str, repo_type: str, revision: str | None = None
        ) -> str:
            raise RuntimeError("connection reset")

    client = RealHfClient(_BrokenApi(), "me/repo")  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="connection reset"):
        client.download_bytes("file.txt")
