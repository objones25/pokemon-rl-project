import io

import datasets
import numpy as np
import pytest
from PIL import Image

from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.resize_cache import build_local_resize_cache, ensure_local_cache


def _native_res_shard_bytes(video_id: str) -> bytes:
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "video_id": datasets.Value("string"),
            "timestamp_s": datasets.Value("float64"),
        }
    )
    pixels = np.random.default_rng(0).integers(0, 256, (2160, 2400), dtype=np.uint8)
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

    for shard_path in shard_bytes:
        output_path = tmp_path / shard_path
        assert output_path.exists()
        reloaded = datasets.Dataset.from_parquet(str(output_path))
        assert reloaded.num_rows == 1
        assert reloaded[0]["image"].size == (160, 144)  # PIL size is (width, height)
        assert reloaded[0]["image"].mode == "L"


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


def test_ensure_local_cache_wires_build_local_resize_cache_when_set(monkeypatch, tmp_path) -> None:
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
