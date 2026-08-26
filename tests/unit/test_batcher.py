from pathlib import Path

import datasets
import numpy as np
import pytest

from data_collection.batcher import FrameBatcher, FrameRecord, batch_to_parquet


def _record(i: int) -> FrameRecord:
    return FrameRecord(
        image=np.full((144, 160), i % 256, dtype=np.uint8),
        video_id="abc123",
        timestamp_s=float(i),
        game="red",
    )


def test_add_returns_none_until_batch_size_reached() -> None:
    batcher = FrameBatcher(batch_size=3)

    assert batcher.add(_record(0)) is None
    assert batcher.add(_record(1)) is None
    full_batch = batcher.add(_record(2))

    assert full_batch is not None
    assert len(full_batch) == 3


def test_batcher_resets_after_emitting_a_full_batch() -> None:
    batcher = FrameBatcher(batch_size=2)
    batcher.add(_record(0))
    batcher.add(_record(1))  # emits and resets

    assert batcher.add(_record(2)) is None


def test_flush_returns_partial_batch() -> None:
    batcher = FrameBatcher(batch_size=10)
    batcher.add(_record(0))
    batcher.add(_record(1))

    partial = batcher.flush()

    assert partial is not None
    assert len(partial) == 2


def test_flush_returns_none_when_empty() -> None:
    batcher = FrameBatcher(batch_size=10)

    assert batcher.flush() is None


def test_batch_to_parquet_round_trips(tmp_path: Path) -> None:
    records = [_record(i) for i in range(5)]
    path = tmp_path / "shard.parquet"

    batch_to_parquet(records, path)
    reloaded = datasets.Dataset.from_parquet(str(path))

    assert len(reloaded) == 5
    assert reloaded.column_names == ["image", "video_id", "timestamp_s", "game"]
    assert reloaded[0]["video_id"] == "abc123"
    assert reloaded[2]["timestamp_s"] == pytest.approx(2.0)
    assert reloaded[0]["image"].size == (160, 144)  # PIL Image (width, height)
