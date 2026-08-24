import numpy as np
import torch
from PIL import Image

from contrastive_pretrain.augmentation import AugmentationConfig
from contrastive_pretrain.dataset import row_seed, to_pair_transform


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


def test_to_pair_transform_is_deterministic_for_the_same_row() -> None:
    result1 = to_pair_transform(_grayscale_example(), AugmentationConfig(), base_seed=0)
    result2 = to_pair_transform(_grayscale_example(), AugmentationConfig(), base_seed=0)

    assert torch.equal(result1["view_a"], result2["view_a"])
    assert torch.equal(result1["view_b"], result2["view_b"])


def test_to_pair_transform_differs_across_rows() -> None:
    result1 = to_pair_transform(_grayscale_example(timestamp_s=5.0), AugmentationConfig(), base_seed=0)
    result2 = to_pair_transform(_grayscale_example(timestamp_s=6.0), AugmentationConfig(), base_seed=0)

    assert not torch.equal(result1["view_a"], result2["view_a"])


import datasets

from contrastive_pretrain.dataset import build_dataloader


def _synthetic_iterable_dataset():
    base = datasets.Dataset.from_dict({"value": list(range(20))})
    return base.to_iterable_dataset(num_shards=4)


def test_build_dataloader_yields_batches_of_configured_size() -> None:
    loader = build_dataloader(
        _synthetic_iterable_dataset(), batch_size=4, num_workers=0,
        snapshot_every_n_steps=1, pin_memory=False,
    )

    batch = next(iter(loader))

    assert batch["value"].shape == (4,)


def test_build_dataloader_resumes_from_exact_position() -> None:
    """Verifies the StatefulDataLoader checkpoint/resume mechanic the
    design spec depends on: a fresh dataloader over the same underlying
    (unconsumed) dataset, given a saved state_dict, continues from
    exactly where the original left off -- no re-served, no skipped
    examples."""
    loader = build_dataloader(
        _synthetic_iterable_dataset(), batch_size=2, num_workers=0,
        snapshot_every_n_steps=1, pin_memory=False,
    )

    it = iter(loader)
    for _ in range(3):
        next(it)
    state = loader.state_dict()
    expected_next_two_batches = [next(it)["value"].tolist() for _ in range(2)]

    resumed_loader = build_dataloader(
        _synthetic_iterable_dataset(), batch_size=2, num_workers=0,
        snapshot_every_n_steps=1, pin_memory=False,
    )
    resumed_loader.load_state_dict(state)
    actual_next_two_batches = [batch["value"].tolist() for _, batch in zip(range(2), resumed_loader)]

    assert actual_next_two_batches == expected_next_two_batches
