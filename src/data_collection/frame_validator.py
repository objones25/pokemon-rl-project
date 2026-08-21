"""Re-check extracted frames against the curation-time reference patch."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from data_collection.matching import match_crop


@dataclass(frozen=True)
class ValidationResult:
    keep: bool
    halted: bool


class FrameValidator:
    def __init__(
        self,
        reference_patch_gray: np.ndarray,
        baseline_score: float,
        score_ratio_threshold: float = 0.6,
        window_size: int = 50,
        drop_ratio_threshold: float = 0.2,
    ) -> None:
        self._reference = reference_patch_gray
        self._min_score = baseline_score * score_ratio_threshold
        self._drop_ratio_threshold = drop_ratio_threshold
        self._window: deque[int] = deque(maxlen=window_size)
        self._halted = False

    def validate(self, frame_crop_gray: np.ndarray) -> ValidationResult:
        score = match_crop(frame_crop_gray, self._reference).score
        keep = score >= self._min_score

        self._window.append(0 if keep else 1)
        if len(self._window) == self._window.maxlen:
            drop_rate = sum(self._window) / len(self._window)
            if drop_rate > self._drop_ratio_threshold:
                self._halted = True

        return ValidationResult(keep=keep, halted=self._halted)
