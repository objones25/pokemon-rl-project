"""Grid-image previews for spot-checking sampled frames."""

from __future__ import annotations

import numpy as np


def build_contact_sheet(frames: list[np.ndarray], cols: int = 8) -> np.ndarray:
    if not frames:
        return np.empty((0, 0), dtype=np.uint8)

    height, width = frames[0].shape
    rows = -(-len(frames) // cols)  # ceil division
    sheet = np.zeros((height * rows, width * cols), dtype=np.uint8)

    for i, frame in enumerate(frames):
        row, col = divmod(i, cols)
        sheet[row * height : (row + 1) * height, col * width : (col + 1) * width] = frame

    return sheet


def build_pair_preview(original: np.ndarray, view_a: np.ndarray, view_b: np.ndarray) -> np.ndarray:
    """Concatenate three images horizontally: original | view_a | view_b."""
    return np.concatenate([original, view_a, view_b], axis=1)


def build_augmentation_contact_sheet(triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Stack augmentation pair previews vertically."""
    if not triples:
        return np.empty((0, 0), dtype=np.uint8)

    rows = [build_pair_preview(original, view_a, view_b) for original, view_a, view_b in triples]
    return np.concatenate(rows, axis=0)
