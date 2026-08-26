import datasets
import numpy as np
import pytest
import torch
from huggingface_hub import HfApi
from PIL import Image

from contrastive_pretrain.augmentation import AugmentationConfig
from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.dataset import (
    _load_base_stream,
    _resize_to_canonical,
    _ResizeToCanonicalWithProgress,
    build_dataloader,
    build_train_dataset,
    build_val_dataset,
    row_seed,
    to_pair_transform,
)
from tests.conftest import requires_hf_credentials

_ROW_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "video_id": datasets.Value("string"),
        "timestamp_s": datasets.Value("float64"),
        "game": datasets.Value("string"),
    }
)


class _CountingResize:
    """Stub for _ResizeToCanonicalWithProgress that records how many times
    it was instantiated, letting a test assert whether build_train_dataset/
    build_val_dataset added the resize map to the pipeline without needing
    a working resize implementation. Tests using this must reset
    `instantiation_count` to 0 (e.g. via monkeypatch.setattr) before
    running, so counts don't leak across tests."""

    instantiation_count = 0

    def __init__(self) -> None:
        type(self).instantiation_count += 1

    def __call__(self, example: dict) -> dict:
        return example


def _grayscale_example(video_id: str = "abc123", timestamp_s: float = 5.0) -> dict:
    pixels = np.random.default_rng(0).integers(0, 256, (144, 160), dtype=np.uint8)
    return {
        "image": Image.fromarray(pixels, mode="L"),
        "video_id": video_id,
        "timestamp_s": timestamp_s,
        "game": "red",
    }


def test_row_seed_is_deterministic() -> None:
    assert row_seed(0, "abc123", 12.5) == row_seed(0, "abc123", 12.5)


def test_row_seed_differs_for_different_rows() -> None:
    assert row_seed(0, "abc123", 12.5) != row_seed(0, "abc123", 12.6)
    assert row_seed(0, "abc123", 12.5) != row_seed(0, "xyz789", 12.5)
    assert row_seed(0, "abc123", 12.5) != row_seed(1, "abc123", 12.5)


def test_to_pair_transform_produces_original_and_two_views() -> None:
    result = to_pair_transform(_grayscale_example(), AugmentationConfig(), base_seed=0)

    assert result["original"].shape == (1, 144, 160)
    assert result["view_a"].shape == (1, 144, 160)
    assert result["view_b"].shape == (1, 144, 160)
    assert result["original"].dtype == torch.uint8


def test_to_pair_transform_resizes_non_canonical_resolution_frames() -> None:
    """Real dataset frames are stored at 2400x2160 (15x the Game Boy's
    native 160x144), not pre-resized to canonical resolution. Regression
    test for that mismatch: any input resolution must come out at
    (1, 144, 160), since both augmentation.py's pixel-tuned parameters and
    model.py's encoder hard-require that exact shape."""
    pixels = np.random.default_rng(0).integers(0, 256, (2160, 2400), dtype=np.uint8)
    example = {
        "image": Image.fromarray(pixels, mode="L"),
        "video_id": "abc123",
        "timestamp_s": 5.0,
        "game": "red",
    }

    result = to_pair_transform(example, AugmentationConfig(), base_seed=0)

    assert result["original"].shape == (1, 144, 160)
    assert result["view_a"].shape == (1, 144, 160)
    assert result["view_b"].shape == (1, 144, 160)
    assert result["original"].dtype == torch.uint8


def test_resize_to_canonical_resizes_native_resolution_frames() -> None:
    pixels = np.random.default_rng(0).integers(0, 256, (2160, 2400), dtype=np.uint8)
    example = {"image": Image.fromarray(pixels, mode="L")}

    result = _resize_to_canonical(example)

    assert result["image"].shape == (1, 144, 160)
    assert result["image"].dtype == torch.uint8


def test_resize_to_canonical_with_progress_delegates_to_resize_to_canonical() -> None:
    example = _grayscale_example()

    result = _ResizeToCanonicalWithProgress()(example)

    assert result["image"].shape == (1, 144, 160)
    assert result["image"].dtype == torch.uint8


def test_resize_to_canonical_with_progress_logs_every_n_rows(caplog) -> None:
    import logging

    transform = _ResizeToCanonicalWithProgress(log_every_n=3)

    with caplog.at_level(logging.INFO, logger="contrastive_pretrain.dataset"):
        for i in range(7):
            transform(_grayscale_example(timestamp_s=float(i)))

    progress_logs = [
        r for r in caplog.records if r.message == "resize_to_canonical_progress"
    ]
    # Rows 3 and 6 (1-indexed count % 3 == 0), not every row and not row 7.
    assert [r.rows_processed for r in progress_logs] == [3, 6]


def test_to_pair_transform_is_deterministic_for_the_same_row() -> None:
    result1 = to_pair_transform(_grayscale_example(), AugmentationConfig(), base_seed=0)
    result2 = to_pair_transform(_grayscale_example(), AugmentationConfig(), base_seed=0)

    assert torch.equal(result1["view_a"], result2["view_a"])
    assert torch.equal(result1["view_b"], result2["view_b"])


def test_to_pair_transform_differs_across_rows() -> None:
    result1 = to_pair_transform(
        _grayscale_example(timestamp_s=5.0), AugmentationConfig(), base_seed=0
    )
    result2 = to_pair_transform(
        _grayscale_example(timestamp_s=6.0), AugmentationConfig(), base_seed=0
    )

    assert not torch.equal(result1["view_a"], result2["view_a"])


def _synthetic_iterable_dataset():
    base = datasets.Dataset.from_dict({"value": list(range(20))})
    return base.to_iterable_dataset(num_shards=4)


def test_build_dataloader_yields_batches_of_configured_size() -> None:
    loader = build_dataloader(
        _synthetic_iterable_dataset(),
        batch_size=4,
        num_workers=0,
        snapshot_every_n_steps=1,
        pin_memory=False,
    )

    batch = next(iter(loader))

    assert batch["value"].shape == (4,)


def test_build_dataloader_drop_last_false_yields_a_final_partial_batch() -> None:
    """The validation loader must pass drop_last=False: held-out
    val_video_ids can easily total fewer rows than one training
    batch_size, and drop_last=True on a stream smaller than one batch
    yields zero batches -- which compute_val_loss treats as a hard
    failure that would kill an unattended training run."""
    loader = build_dataloader(
        _synthetic_iterable_dataset(),
        batch_size=8,
        num_workers=0,
        snapshot_every_n_steps=1,
        pin_memory=False,
        drop_last=False,
    )

    batches = list(loader)

    assert [b["value"].shape[0] for b in batches] == [8, 8, 4]


def test_build_dataloader_resumes_from_exact_position() -> None:
    """Verifies the StatefulDataLoader checkpoint/resume mechanic the
    design spec depends on: a fresh dataloader over the same underlying
    (unconsumed) dataset, given a saved state_dict, continues from
    exactly where the original left off -- no re-served, no skipped
    examples."""
    loader = build_dataloader(
        _synthetic_iterable_dataset(),
        batch_size=2,
        num_workers=0,
        snapshot_every_n_steps=1,
        pin_memory=False,
    )

    it = iter(loader)
    for _ in range(3):
        next(it)
    state = loader.state_dict()
    expected_next_two_batches = [next(it)["value"].tolist() for _ in range(2)]

    resumed_loader = build_dataloader(
        _synthetic_iterable_dataset(),
        batch_size=2,
        num_workers=0,
        snapshot_every_n_steps=1,
        pin_memory=False,
    )
    resumed_loader.load_state_dict(state)
    actual_next_two_batches = [
        batch["value"].tolist() for _, batch in zip(range(2), resumed_loader)
    ]

    assert actual_next_two_batches == expected_next_two_batches


def _synthetic_frame_stream(video_ids: list[str]):
    rows = [
        _grayscale_example(video_id=vid, timestamp_s=float(i))
        for i, vid in enumerate(video_ids)
    ]
    return datasets.Dataset.from_list(rows, features=_ROW_FEATURES).to_iterable_dataset()


@pytest.fixture
def patch_load_base_stream(monkeypatch):
    def _apply(video_ids: list[str]):
        stream = _synthetic_frame_stream(video_ids)
        monkeypatch.setattr("contrastive_pretrain.dataset._load_base_stream", lambda config: stream)
        return stream

    return _apply


def test_build_train_dataset_excludes_val_videos_fast(patch_load_base_stream) -> None:
    """Fast, network-free regression test for the train/val split filter
    itself -- the only prior coverage of this exact filter logic was
    @pytest.mark.slow and needs real Hub access, so a swapped in/not-in
    would previously ship undetected by the fast suite."""
    patch_load_base_stream(["train_a", "val_a", "val_b", "train_b"])
    config = TrainingConfig(val_video_ids=("val_a", "val_b"))

    ds = build_train_dataset(config)

    assert [row["video_id"] for row in ds] == ["train_a", "train_b"]


def test_build_val_dataset_only_yields_held_out_videos_fast(patch_load_base_stream) -> None:
    patch_load_base_stream(["train_a", "val_a", "val_b", "train_b"])
    config = TrainingConfig(val_video_ids=("val_a", "val_b"))

    ds = build_val_dataset(config)

    assert [row["video_id"] for row in ds] == ["val_a", "val_b"]


def test_build_train_dataset_resizes_native_resolution_frames_before_shuffling(
    monkeypatch,
) -> None:
    """Regression test for a real production OOM: .shuffle()'s buffer holds
    buffer_size examples PER WORKER (independent, unshared buffers), so it
    must run on already-small frames, not raw native-resolution ones --
    real source videos are stored up to 2400x2160 (see
    configs/video_sources.yaml), and buffering those at
    shuffle_buffer_size=10_000 x num_workers=8 OOM-killed a real training
    pod. This asserts the full build_train_dataset pipeline (filter ->
    resize -> shuffle -> to_pair_transform) still produces correctly-shaped
    views when fed an oversized frame -- i.e. the resize-before-shuffle
    reordering didn't break the pipeline."""

    def _large_frame_stream(config):
        pixels = np.random.default_rng(0).integers(0, 256, (2160, 2400), dtype=np.uint8)
        rows = [
            {
                "image": Image.fromarray(pixels, mode="L"),
                "video_id": "train_a",
                "timestamp_s": 0.0,
                "game": "red",
            }
        ]
        return datasets.Dataset.from_list(rows, features=_ROW_FEATURES).to_iterable_dataset()

    monkeypatch.setattr(
        "contrastive_pretrain.dataset._load_base_stream", _large_frame_stream
    )
    config = TrainingConfig(val_video_ids=("val_a",))

    ds = build_train_dataset(config)
    row = next(iter(ds))

    assert row["view_a"].shape == (1, 144, 160)
    assert row["view_b"].shape == (1, 144, 160)


@pytest.fixture(scope="module")
def real_hub_first_row():
    """One streamed row from the real private objones25/pokemon-frames, shared
    by the live schema-contract tests below so the whole live tier costs a
    single Hub interaction. Module-scoped rather than per-test purely to avoid
    repeating that fetch; nothing here mutates the row."""
    return next(iter(_load_base_stream(TrainingConfig())))


@pytest.mark.slow
@requires_hf_credentials
def test_real_hub_rows_carry_exactly_the_columns_the_streaming_pipeline_reads(
    real_hub_first_row,
) -> None:
    """Live contract check, and the only thing in this file a fake cannot
    cover: every other dataset test builds its own synthetic stream, so all of
    them keep passing if the *real* repo's schema drifts. to_pair_transform
    reads image/video_id/timestamp_s by name, so a rename there surfaces as a
    KeyError hours into a paid pod run rather than here."""
    assert sorted(real_hub_first_row) == ["game", "image", "timestamp_s", "video_id"]


@pytest.mark.slow
@requires_hf_credentials
def test_real_hub_frames_are_single_channel_grayscale(real_hub_first_row) -> None:
    """The `L` mode is load-bearing, not incidental: TF.to_image on an RGB
    frame yields (3, H, W), which GrayscaleResNetEncoder.forward rejects with
    "expected 1-channel grayscale input" -- again only at training time."""
    assert real_hub_first_row["image"].mode == "L"


@pytest.mark.slow
@requires_hf_credentials
def test_configured_val_video_ids_all_exist_as_shards_in_the_real_dataset_repo() -> None:
    """A val_video_id with no shards behind it makes build_val_dataset yield
    zero rows, which compute_val_loss turns into a hard "no batches" failure at
    the first epoch boundary. Cheap metadata call -- no shard bytes are read."""
    config = TrainingConfig()
    shard_video_ids = {
        path.split("/")[1]
        for path in HfApi().list_repo_files(config.dataset_repo_id, repo_type="dataset")
        if path.startswith("shards/")
    }

    missing = sorted(set(config.val_video_ids) - shard_video_ids)

    assert missing == []


def test_load_base_stream_reads_from_local_cache_when_configured(tmp_path) -> None:
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "video_id": datasets.Value("string"),
            "timestamp_s": datasets.Value("float64"),
        }
    )
    pixels = np.random.default_rng(0).integers(0, 256, (144, 160), dtype=np.uint8)
    rows = [{"image": Image.fromarray(pixels, mode="L"), "video_id": "cachedA", "timestamp_s": 0.0}]
    ds = datasets.Dataset.from_list(rows, features=features)
    shard_path = tmp_path / "shards" / "cachedA" / "00000.parquet"
    shard_path.parent.mkdir(parents=True)
    ds.to_parquet(str(shard_path))

    config = TrainingConfig(local_cache_dir=str(tmp_path))

    stream = _load_base_stream(config)

    assert [row["video_id"] for row in stream] == ["cachedA"]


def test_build_train_dataset_calls_ensure_local_cache_before_loading(monkeypatch) -> None:
    call_order: list[str] = []
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.ensure_local_cache",
        lambda config: call_order.append("ensure_local_cache"),
    )

    def fake_load_base_stream(config):
        call_order.append("load_base_stream")
        return _synthetic_frame_stream(["train_a"])

    monkeypatch.setattr("contrastive_pretrain.dataset._load_base_stream", fake_load_base_stream)
    config = TrainingConfig(val_video_ids=("val_a",))

    list(build_train_dataset(config))

    assert call_order == ["ensure_local_cache", "load_base_stream"]


def test_build_train_dataset_skips_resize_map_when_local_cache_dir_is_set(
    monkeypatch, tmp_path, patch_load_base_stream
) -> None:
    monkeypatch.setattr("contrastive_pretrain.resize_cache.ensure_local_cache", lambda config: None)
    monkeypatch.setattr(_CountingResize, "instantiation_count", 0)
    monkeypatch.setattr("contrastive_pretrain.dataset._ResizeToCanonicalWithProgress", _CountingResize)
    patch_load_base_stream(["train_a"])
    config = TrainingConfig(val_video_ids=("val_a",), local_cache_dir=str(tmp_path))

    list(build_train_dataset(config))

    assert _CountingResize.instantiation_count == 0


def test_build_train_dataset_still_uses_resize_map_when_local_cache_dir_is_unset(
    monkeypatch, patch_load_base_stream
) -> None:
    monkeypatch.setattr(_CountingResize, "instantiation_count", 0)
    monkeypatch.setattr("contrastive_pretrain.dataset._ResizeToCanonicalWithProgress", _CountingResize)
    patch_load_base_stream(["train_a"])
    config = TrainingConfig(val_video_ids=("val_a",))

    list(build_train_dataset(config))

    assert _CountingResize.instantiation_count == 1


def test_build_val_dataset_calls_ensure_local_cache_before_loading(monkeypatch) -> None:
    call_order: list[str] = []
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.ensure_local_cache",
        lambda config: call_order.append("ensure_local_cache"),
    )

    def fake_load_base_stream(config):
        call_order.append("load_base_stream")
        return _synthetic_frame_stream(["val_a"])

    monkeypatch.setattr("contrastive_pretrain.dataset._load_base_stream", fake_load_base_stream)
    config = TrainingConfig(val_video_ids=("val_a",))

    list(build_val_dataset(config))

    assert call_order == ["ensure_local_cache", "load_base_stream"]


def test_build_val_dataset_skips_resize_map_when_local_cache_dir_is_set(
    monkeypatch, tmp_path, patch_load_base_stream
) -> None:
    monkeypatch.setattr("contrastive_pretrain.resize_cache.ensure_local_cache", lambda config: None)
    monkeypatch.setattr(_CountingResize, "instantiation_count", 0)
    monkeypatch.setattr("contrastive_pretrain.dataset._ResizeToCanonicalWithProgress", _CountingResize)
    patch_load_base_stream(["val_a"])
    config = TrainingConfig(val_video_ids=("val_a",), local_cache_dir=str(tmp_path))

    list(build_val_dataset(config))

    assert _CountingResize.instantiation_count == 0


def test_build_val_dataset_still_uses_resize_map_when_local_cache_dir_is_unset(
    monkeypatch, patch_load_base_stream
) -> None:
    monkeypatch.setattr(_CountingResize, "instantiation_count", 0)
    monkeypatch.setattr("contrastive_pretrain.dataset._ResizeToCanonicalWithProgress", _CountingResize)
    patch_load_base_stream(["val_a"])
    config = TrainingConfig(val_video_ids=("val_a",))

    list(build_val_dataset(config))

    assert _CountingResize.instantiation_count == 1


def _write_parquet_shards(root, video_ids: list[str], rows_per_shard: int) -> None:
    """Writes one shards/<video_id>/00000.parquet per id, in the same layout
    and with the same Features as the real dataset repo."""
    for video_id in video_ids:
        rows = [
            _grayscale_example(video_id=video_id, timestamp_s=float(i))
            for i in range(rows_per_shard)
        ]
        shard_path = root / "shards" / video_id / "00000.parquet"
        shard_path.parent.mkdir(parents=True)
        datasets.Dataset.from_list(rows, features=_ROW_FEATURES).to_parquet(str(shard_path))


@pytest.fixture
def patch_load_base_stream_with_parquet_shards(monkeypatch, tmp_path):
    """Points the dataset builders at real on-disk Parquet shards streamed by
    datasets.load_dataset(..., streaming=True) -- the same reader, shard layout
    and Features the Hub path uses, minus the Hub. A fresh stream per call, so
    two pipelines built from it share no iteration state."""

    def _apply(video_ids: list[str], rows_per_shard: int):
        _write_parquet_shards(tmp_path, video_ids, rows_per_shard)

        def _load(config):
            return datasets.load_dataset(
                "parquet",
                data_files=f"{tmp_path}/shards/**/*.parquet",
                split="train",
                streaming=True,
            )

        monkeypatch.setattr("contrastive_pretrain.dataset._load_base_stream", _load)

    return _apply


def test_build_val_dataloader_resumes_from_exact_position_over_streamed_parquet_shards(
    patch_load_base_stream_with_parquet_shards,
) -> None:
    """StatefulDataLoader's shard-skipping over the real Parquet streaming
    reader and a real build_* pipeline, rather than an in-memory Dataset.
    Replaces a @slow test that ran the same assertion against
    objones25/pokemon-frames: the Hub only changed where the bytes came from,
    and the shards written here have the same layout and Features.

    Deliberately the VAL pipeline. build_train_dataset ends with .shuffle(),
    whose buffer contents are not part of the checkpointed state, so the train
    loader does NOT resume to the exact next batch -- measured, and the reason
    the old @slow version asserted a guarantee that does not exist. The val
    pipeline has no shuffle, so exact resume is a real, assertable property
    there."""
    # One row per shard, every row's video_id distinct: with only a couple of
    # ids the post-resume batch could match the expected one by coincidence, so
    # a loader that silently restarted at position zero would still pass.
    video_ids = [f"t{i:02d}" for i in range(16)]
    patch_load_base_stream_with_parquet_shards(video_ids, 1)
    config = TrainingConfig(val_video_ids=tuple(video_ids))

    loader = build_dataloader(
        build_val_dataset(config),
        batch_size=2,
        num_workers=0,
        snapshot_every_n_steps=1,
        pin_memory=False,
        drop_last=False,
    )
    it = iter(loader)
    for _ in range(3):
        next(it)
    state = loader.state_dict()
    expected_video_ids = next(it)["video_id"]

    resumed_loader = build_dataloader(
        build_val_dataset(config),
        batch_size=2,
        num_workers=0,
        snapshot_every_n_steps=1,
        pin_memory=False,
        drop_last=False,
    )
    resumed_loader.load_state_dict(state)
    actual_video_ids = next(iter(resumed_loader))["video_id"]

    assert actual_video_ids == expected_video_ids
