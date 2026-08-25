import numpy as np
import torch
from PIL import Image

from contrastive_pretrain.augmentation import AugmentationConfig
from contrastive_pretrain.dataset import (
    _resize_to_canonical,
    row_seed,
    to_pair_transform,
)


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


import datasets

from contrastive_pretrain.dataset import build_dataloader


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


import pytest

from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.dataset import build_train_dataset, build_val_dataset


def _synthetic_frame_stream(video_ids: list[str]):
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "video_id": datasets.Value("string"),
            "timestamp_s": datasets.Value("float64"),
            "game": datasets.Value("string"),
        }
    )
    rows = [
        _grayscale_example(video_id=vid, timestamp_s=float(i))
        for i, vid in enumerate(video_ids)
    ]
    return datasets.Dataset.from_list(rows, features=features).to_iterable_dataset()


def test_build_train_dataset_excludes_val_videos_fast(monkeypatch) -> None:
    """Fast, network-free regression test for the train/val split filter
    itself -- the only prior coverage of this exact filter logic was
    @pytest.mark.slow and needs real Hub access, so a swapped in/not-in
    would previously ship undetected by the fast suite."""
    monkeypatch.setattr(
        "contrastive_pretrain.dataset._load_base_stream",
        lambda config: _synthetic_frame_stream(
            ["train_a", "val_a", "val_b", "train_b"]
        ),
    )
    config = TrainingConfig(val_video_ids=("val_a", "val_b"))

    ds = build_train_dataset(config)

    assert [row["video_id"] for row in ds] == ["train_a", "train_b"]


def test_build_val_dataset_only_yields_held_out_videos_fast(monkeypatch) -> None:
    monkeypatch.setattr(
        "contrastive_pretrain.dataset._load_base_stream",
        lambda config: _synthetic_frame_stream(
            ["train_a", "val_a", "val_b", "train_b"]
        ),
    )
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
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "video_id": datasets.Value("string"),
            "timestamp_s": datasets.Value("float64"),
            "game": datasets.Value("string"),
        }
    )

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
        return datasets.Dataset.from_list(rows, features=features).to_iterable_dataset()

    monkeypatch.setattr(
        "contrastive_pretrain.dataset._load_base_stream", _large_frame_stream
    )
    config = TrainingConfig(val_video_ids=("val_a",))

    ds = build_train_dataset(config)
    row = next(iter(ds))

    assert row["view_a"].shape == (1, 144, 160)
    assert row["view_b"].shape == (1, 144, 160)


@pytest.mark.slow
def test_build_train_dataset_excludes_val_videos() -> None:
    config = TrainingConfig()
    ds = build_train_dataset(config)

    row = next(iter(ds))

    assert row["video_id"] not in config.val_video_ids
    assert row["view_a"].shape == (1, 144, 160)


@pytest.mark.slow
def test_build_val_dataset_only_yields_held_out_videos() -> None:
    config = TrainingConfig()
    ds = build_val_dataset(config)

    for _, row in zip(range(5), ds):
        assert row["video_id"] in config.val_video_ids


@pytest.mark.slow
def test_build_dataloader_resumes_against_real_streaming_data() -> None:
    """Same resume guarantee as Task 9's synthetic test, but against the
    real Hub-backed streaming dataset -- confirms StatefulDataLoader's
    shard-skipping behavior holds for real parquet shards, not just an
    in-memory Dataset."""
    config = TrainingConfig()

    loader = build_dataloader(
        build_train_dataset(config),
        batch_size=2,
        num_workers=0,
        snapshot_every_n_steps=1,
    )
    it = iter(loader)
    for _ in range(3):
        next(it)
    state = loader.state_dict()
    expected_video_ids = next(it)["video_id"]

    resumed_loader = build_dataloader(
        build_train_dataset(config),
        batch_size=2,
        num_workers=0,
        snapshot_every_n_steps=1,
    )
    resumed_loader.load_state_dict(state)
    actual_video_ids = next(iter(resumed_loader))["video_id"]

    assert actual_video_ids == expected_video_ids
