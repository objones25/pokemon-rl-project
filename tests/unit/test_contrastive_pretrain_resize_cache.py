import io
import os

import datasets
import numpy as np
import pytest
from PIL import Image

import contrastive_pretrain.resize_cache
from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.dataset import _resize_to_canonical
from contrastive_pretrain.resize_cache import build_local_resize_cache, ensure_local_cache


_SOURCE_PIXELS = np.random.default_rng(0).integers(0, 256, (2160, 2400), dtype=np.uint8)


def _native_res_shard_bytes(video_id: str) -> bytes:
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "video_id": datasets.Value("string"),
            "timestamp_s": datasets.Value("float64"),
        }
    )
    pixels = _SOURCE_PIXELS
    rows = [{"image": Image.fromarray(pixels, mode="L"), "video_id": video_id, "timestamp_s": 0.0}]
    ds = datasets.Dataset.from_list(rows, features=features)
    buf = io.BytesIO()
    ds.to_parquet(buf)
    return buf.getvalue()


def test_build_local_resize_cache_writes_resized_shard_for_each_listed_path(tmp_path) -> None:
    shard_bytes = {
        "shards/vidA/00000.parquet": _native_res_shard_bytes("vidA"),
        "shards/vidB/00000.parquet": _native_res_shard_bytes("vidB"),
    }

    build_local_resize_cache(
        list_shard_paths=lambda: list(shard_bytes),
        download_shard=lambda path: shard_bytes[path],
        local_cache_dir=tmp_path,
    )

    # The whole reason _resize_row_for_cache exists is that the obvious
    # implementations corrupt pixels on the parquet round-trip while
    # PRESERVING shape (verified during design: a raw tensor/array write comes
    # back in 32-bit int mode, still 144x160). A size/mode-only assertion would
    # not have caught the bug this module was written to avoid, so compare
    # actual pixels against _resize_to_canonical applied to the same source.
    expected = _resize_to_canonical({"image": Image.fromarray(_SOURCE_PIXELS, mode="L")})[
        "image"
    ]  # (1, H, W) uint8
    expected_pixels = expected.squeeze(0).numpy()

    for shard_path in shard_bytes:
        output_path = tmp_path / shard_path
        assert output_path.exists()
        reloaded = datasets.Dataset.from_parquet(str(output_path))
        assert reloaded.num_rows == 1
        assert reloaded[0]["image"].size == (160, 144)  # PIL size is (width, height)
        assert reloaded[0]["image"].mode == "L"
        assert np.array_equal(np.array(reloaded[0]["image"]), expected_pixels)


def test_build_local_resize_cache_leaves_no_temp_files_behind(tmp_path) -> None:
    """Every intermediate -- the raw bytes, the in-progress output, and the
    parquet builder's Arrow copy -- is scratch on the same volume as the cache
    and must not survive the shard that produced it. Left alone, the Arrow
    copies alone would be roughly the size of the entire source dataset."""
    shard_path = "shards/vidA/00000.parquet"

    build_local_resize_cache(
        list_shard_paths=lambda: [shard_path],
        download_shard=lambda path: _native_res_shard_bytes("vidA"),
        local_cache_dir=tmp_path,
    )

    survivors = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert survivors == ["shards", "shards/vidA", shard_path]


def test_build_local_resize_cache_skips_shards_whose_output_already_exists(tmp_path) -> None:
    existing_shard = "shards/vidA/00000.parquet"
    new_shard = "shards/vidB/00000.parquet"
    output_path = tmp_path / existing_shard
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(_native_res_shard_bytes("vidA"))  # any prior content marks it done

    download_calls: list[str] = []

    def download_shard(path: str) -> bytes:
        download_calls.append(path)
        return _native_res_shard_bytes("vidB")

    build_local_resize_cache(
        list_shard_paths=lambda: [existing_shard, new_shard],
        download_shard=download_shard,
        local_cache_dir=tmp_path,
    )

    assert download_calls == [new_shard]
    assert (tmp_path / new_shard).exists()


def test_build_local_resize_cache_leaves_no_partial_file_on_failure_and_resumes_correctly(tmp_path) -> None:
    ok_shard = "shards/vidA/00000.parquet"
    failing_shard = "shards/vidB/00000.parquet"

    def failing_download(path: str) -> bytes:
        if path == failing_shard:
            raise RuntimeError("simulated network failure")
        return _native_res_shard_bytes("vidA")

    with pytest.raises(RuntimeError, match="simulated network failure"):
        build_local_resize_cache(
            list_shard_paths=lambda: [ok_shard, failing_shard],
            download_shard=failing_download,
            local_cache_dir=tmp_path,
        )

    assert (tmp_path / ok_shard).exists()
    assert not (tmp_path / failing_shard).exists()

    download_calls: list[str] = []

    def succeeding_download(path: str) -> bytes:
        download_calls.append(path)
        return _native_res_shard_bytes("vidB")

    build_local_resize_cache(
        list_shard_paths=lambda: [ok_shard, failing_shard],
        download_shard=succeeding_download,
        local_cache_dir=tmp_path,
    )

    assert download_calls == [failing_shard]  # ok_shard skipped, not re-downloaded
    assert (tmp_path / failing_shard).exists()


def test_ensure_local_cache_is_a_noop_when_local_cache_dir_is_unset(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.build_local_resize_cache",
        lambda **kwargs: calls.append(kwargs),
    )
    config = TrainingConfig(local_cache_dir=None)

    ensure_local_cache(config)

    assert calls == []


@pytest.fixture
def isolated_hf_hub_cache(monkeypatch):
    """ensure_local_cache redirects huggingface_hub's download cache onto the
    local_cache_dir volume, which means mutating process-global state (the
    HF_HUB_CACHE env var and huggingface_hub.constants.HF_HUB_CACHE, since the
    library only reads the env var at import time). Register both with
    monkeypatch so a test's tmp_path doesn't leak into the rest of the
    session -- notably the @slow tests that hit the real Hub.

    setenv-then-delenv, not delenv(raising=False): monkeypatch records an
    undo entry only for env keys that already exist, so on a machine with no
    HF_HUB_CACHE set, delenv alone records nothing and the tmp_path that
    ensure_local_cache goes on to setdefault survives teardown -- exactly the
    leak this fixture exists to prevent (verified). Setting a placeholder
    first forces the undo entry to exist; both entries are then reversed in
    order by monkeypatch's single teardown, restoring "absent" or the
    operator's real value as appropriate."""
    from huggingface_hub import constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", hf_constants.HF_HUB_CACHE)
    monkeypatch.setenv("HF_HUB_CACHE", "placeholder-forcing-an-undo-entry")
    monkeypatch.delenv("HF_HUB_CACHE")
    return hf_constants


def test_ensure_local_cache_redirects_hf_hub_download_cache_onto_the_volume(
    monkeypatch, tmp_path, isolated_hf_hub_cache
) -> None:
    """Without this, hf_hub_download persists every downloaded shard under
    ~/.cache/huggingface/hub -- the container's small overlay disk on a RunPod
    pod, not the /workspace volume -- and a full build ENOSPCs. Asserts the
    CONSTANT, not just the env var: huggingface_hub snapshots HF_HUB_CACHE at
    import time, so setting the env var alone would be a silent no-op here."""
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.build_local_resize_cache",
        lambda **kwargs: None,
    )
    monkeypatch.setattr("contrastive_pretrain.resize_cache.HfApi", lambda: object())
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.RealHfClient",
        lambda api, repo_id, repo_type: object(),
    )
    cache_dir = tmp_path / "cache"

    ensure_local_cache(TrainingConfig(local_cache_dir=str(cache_dir)))

    assert isolated_hf_hub_cache.HF_HUB_CACHE == str(cache_dir / ".hf_hub_cache")
    assert os.environ["HF_HUB_CACHE"] == str(cache_dir / ".hf_hub_cache")


def test_ensure_local_cache_leaves_an_operator_set_hf_hub_cache_alone(
    monkeypatch, tmp_path, isolated_hf_hub_cache
) -> None:
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.build_local_resize_cache",
        lambda **kwargs: None,
    )
    monkeypatch.setattr("contrastive_pretrain.resize_cache.HfApi", lambda: object())
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.RealHfClient",
        lambda api, repo_id, repo_type: object(),
    )
    monkeypatch.setenv("HF_HUB_CACHE", "/operator/choice")

    ensure_local_cache(TrainingConfig(local_cache_dir=str(tmp_path / "cache")))

    assert os.environ["HF_HUB_CACHE"] == "/operator/choice"
    assert isolated_hf_hub_cache.HF_HUB_CACHE == "/operator/choice"


def test_ensure_local_cache_raises_file_not_found_for_a_shard_missing_on_hub(
    monkeypatch, tmp_path, isolated_hf_hub_cache
) -> None:
    """Not an assert: asserts are stripped under `python -O`, where the
    failure would resurface as a confusing TypeError from write_bytes(None)."""
    captured = {}
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.build_local_resize_cache",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        contrastive_pretrain.resize_cache.time, "sleep", lambda seconds: None
    )

    class _FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["shards/vidA/00000.parquet"]

    monkeypatch.setattr("contrastive_pretrain.resize_cache.HfApi", _FakeApi)

    class _MissingClient:
        def __init__(self, api, repo_id, repo_type):
            pass

        def download_bytes(self, path):
            return None

    monkeypatch.setattr("contrastive_pretrain.resize_cache.RealHfClient", _MissingClient)

    ensure_local_cache(TrainingConfig(local_cache_dir=str(tmp_path / "cache")))

    with pytest.raises(FileNotFoundError, match="shard listed but missing on Hub"):
        captured["download_shard"]("shards/vidA/00000.parquet")


def test_ensure_local_cache_retries_a_failing_list_repo_files(
    monkeypatch, tmp_path, isolated_hf_hub_cache
) -> None:
    """list_repo_files runs on every build_train_dataset/build_val_dataset
    call, cache fully built or not, so an un-retried transient Hub failure
    here kills the training start before a single batch."""
    captured = {}

    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.build_local_resize_cache",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.RealHfClient",
        lambda api, repo_id, repo_type: object(),
    )
    monkeypatch.setattr(
        contrastive_pretrain.resize_cache.time, "sleep", lambda seconds: None
    )

    attempts = []

    class _FlakyApi:
        def list_repo_files(self, repo_id, repo_type):
            attempts.append(repo_id)
            if len(attempts) < 3:
                raise RuntimeError("transient hub failure")
            return ["shards/vidA/00000.parquet", "README.md"]

    monkeypatch.setattr("contrastive_pretrain.resize_cache.HfApi", _FlakyApi)

    ensure_local_cache(TrainingConfig(local_cache_dir=str(tmp_path / "cache")))

    assert captured["list_shard_paths"]() == ["shards/vidA/00000.parquet"]
    assert len(attempts) == 3  # two transient failures, retried, then success


def test_ensure_local_cache_wires_build_local_resize_cache_when_set(
    monkeypatch, tmp_path, isolated_hf_hub_cache
) -> None:
    captured = {}

    def fake_build_local_resize_cache(*, list_shard_paths, download_shard, local_cache_dir):
        captured["list_shard_paths"] = list_shard_paths
        captured["download_shard"] = download_shard
        captured["local_cache_dir"] = local_cache_dir

    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.build_local_resize_cache",
        fake_build_local_resize_cache,
    )

    class _FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            assert repo_id == "objones25/pokemon-frames"
            assert repo_type == "dataset"
            return ["shards/vidA/00000.parquet", "README.md", "shards/vidB/00000.parquet"]

    monkeypatch.setattr("contrastive_pretrain.resize_cache.HfApi", _FakeApi)

    class _FakeClient:
        def __init__(self, api, repo_id, repo_type):
            pass

        def download_bytes(self, path):
            return b"raw-bytes-for-" + path.encode()

    monkeypatch.setattr("contrastive_pretrain.resize_cache.RealHfClient", _FakeClient)

    config = TrainingConfig(local_cache_dir=str(tmp_path / "cache"))

    ensure_local_cache(config)

    assert captured["local_cache_dir"] == tmp_path / "cache"
    assert captured["list_shard_paths"]() == ["shards/vidA/00000.parquet", "shards/vidB/00000.parquet"]
    assert captured["download_shard"]("shards/vidA/00000.parquet") == b"raw-bytes-for-shards/vidA/00000.parquet"


@pytest.mark.slow
def test_build_local_resize_cache_against_real_hub_shard(tmp_path) -> None:
    """Confirms the real Hub schema round-trips through the real resize +
    local Parquet write correctly -- every other test in this file uses
    synthetic fixtures; this is the one check against the real
    objones25/pokemon-frames dataset."""
    from huggingface_hub import HfApi

    from hf_storage.client import RealHfClient

    api = HfApi()
    client = RealHfClient(api, "objones25/pokemon-frames", repo_type="dataset")
    shard_paths = [
        p
        for p in api.list_repo_files("objones25/pokemon-frames", repo_type="dataset")
        if p.startswith("shards/")
    ][:1]
    assert shard_paths, "expected at least one shard under shards/ in objones25/pokemon-frames"

    build_local_resize_cache(
        list_shard_paths=lambda: shard_paths,
        download_shard=lambda path: client.download_bytes(path),
        local_cache_dir=tmp_path,
    )

    output_path = tmp_path / shard_paths[0]
    assert output_path.exists()
    reloaded = datasets.Dataset.from_parquet(str(output_path))
    assert reloaded.num_rows > 0
    assert reloaded[0]["image"].size == (160, 144)
