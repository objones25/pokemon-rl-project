"""Perceptual-hash near-duplicate filter, compared against the last kept frame."""

from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image


class PerceptualHashDeduper:
    def __init__(self, hamming_threshold: int = 5) -> None:
        self._hamming_threshold = hamming_threshold
        self._last_kept_hash: imagehash.ImageHash | None = None

    def is_duplicate(self, frame_gray: np.ndarray) -> bool:
        current_hash = imagehash.phash(Image.fromarray(frame_gray, mode="L"))

        if self._last_kept_hash is not None:
            distance = current_hash - self._last_kept_hash
            if distance <= self._hamming_threshold:
                return True

        self._last_kept_hash = current_hash
        return False
