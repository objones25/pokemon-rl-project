"""Accumulate accepted frames and write them as Parquet shards with the
`datasets` Image feature."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FrameRecord:
    image: np.ndarray
    video_id: str
    timestamp_s: float
    game: str


class FrameBatcher:
    def __init__(self, batch_size: int = 500) -> None:
        self._batch_size = batch_size
        self._buffer: list[FrameRecord] = []

    def add(self, record: FrameRecord) -> list[FrameRecord] | None:
        self._buffer.append(record)
        if len(self._buffer) >= self._batch_size:
            return self.flush()
        return None

    def flush(self) -> list[FrameRecord] | None:
        if not self._buffer:
            return None
        batch, self._buffer = self._buffer, []
        return batch


def batch_to_parquet(batch: list[FrameRecord], path: str | Path) -> None:
    rows = {
        "image": [Image.fromarray(r.image, mode="L") for r in batch],
        "video_id": [r.video_id for r in batch],
        "timestamp_s": [r.timestamp_s for r in batch],
        "game": [r.game for r in batch],
    }
    dataset = datasets.Dataset.from_dict(rows)
    dataset = dataset.cast_column("image", datasets.Image())
    dataset.to_parquet(str(path))
