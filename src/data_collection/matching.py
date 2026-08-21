"""Grayscale template matching, shared by curation (Phase A) and the
runtime frame validator (Phase B) so both use exactly the same logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchResult:
    x: int
    y: int
    score: float


def load_template_gray(path: str | Path) -> np.ndarray:
    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"could not read template image: {path}")
    return template


def match_crop(frame_gray: np.ndarray, template_gray: np.ndarray) -> MatchResult:
    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    x, y = max_loc
    return MatchResult(x=x, y=y, score=float(max_val))
