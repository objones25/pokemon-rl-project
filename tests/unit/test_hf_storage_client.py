from pathlib import Path

from huggingface_hub.errors import EntryNotFoundError

from hf_storage.client import RealHfClient


class _FakeHfApi:
    """Stands in for huggingface_hub.HfApi: upload_file/hf_hub_download are
    the only two methods RealHfClient calls, so this fake only implements
    those, backed by a tmp_path directory instead of the real Hub."""

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self.uploaded_calls: list[tuple[str, str, str]] = []

    def upload_file(self, path_or_fileobj: bytes, path_in_repo: str, repo_id: str, repo_type: str) -> None:
        self.uploaded_calls.append((path_in_repo, repo_id, repo_type))
        dest = self._tmp_path / path_in_repo.replace("/", "_")
        dest.write_bytes(path_or_fileobj)

    def hf_hub_download(self, repo_id: str, filename: str, repo_type: str) -> str:
        dest = self._tmp_path / filename.replace("/", "_")
        if not dest.exists():
            raise EntryNotFoundError("not found")
        return str(dest)


def test_real_hf_client_round_trips_bytes(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo", repo_type="model")

    client.upload_bytes(b"hello", "config.json")
    result = client.download_bytes("config.json")

    assert result == b"hello"
    assert api.uploaded_calls == [("config.json", "me/repo", "model")]


def test_real_hf_client_download_bytes_returns_none_when_missing(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo")

    assert client.download_bytes("missing.json") is None


def test_real_hf_client_defaults_to_dataset_repo_type(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo")

    client.upload_bytes(b"x", "manifest.json")

    assert api.uploaded_calls == [("manifest.json", "me/repo", "dataset")]
