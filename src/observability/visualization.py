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
