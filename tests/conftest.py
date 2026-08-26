"""Pytest configuration, loaded before any test module.

Disables torch.compile's on-disk FX graph cache (TORCHINDUCTOR_FX_GRAPH_CACHE
-- a real, torch-documented env var, see torch._inductor.config.fx_graph_cache)
for the test suite only. Compilation itself is unaffected; only the pickling
step that hashes/caches the compiled graph to disk is skipped.

This exists because that pickling step walks global interpreter state and, in
this test suite specifically, ends up probing yt_dlp's lazy `Cryptodome`
compat-shim module (imported transitively by data_collection's tests, which
run in the same pytest session as contrastive_pretrain's torch.compile-using
tests) with an attribute name it doesn't handle, raising
`ModuleNotFoundError: No module named 'no_Cryptodome'`. Set as an environment
variable (not a torch._inductor.config.fx_graph_cache = False assignment) so
it takes effect before torch._inductor.config is first imported by any test
module, regardless of import order.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import pytest


class FakeHfClient:
    """In-memory stand-in for hf_storage.client.AtomicHfClient's Protocol
    (upload_bytes/upload_many_bytes/download_bytes) -- shared by every
    contrastive_pretrain test that needs an HfClient without a real Hub
    call. upload_calls records every path written, in order (a
    upload_many_bytes call appends all its paths at once, matching it
    landing as one commit), so tests can assert not just final file
    contents but how many upload attempts/publishes happened."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.upload_calls: list[str] = []
        self.commits: list[dict] = []

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.upload_calls.append(path_in_repo)
        self.files[path_in_repo] = data

    def upload_many_bytes(self, files: dict[str, bytes], commit_message: str) -> None:
        self.commits.append({"paths": sorted(files), "commit_message": commit_message})
        self.upload_calls.extend(files.keys())
        self.files.update(files)

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.files.get(path_in_repo)


@pytest.fixture
def fake_hf_client() -> FakeHfClient:
    return FakeHfClient()
