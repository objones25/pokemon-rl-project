import numpy as np

from data_collection.matching import match_crop


def _embed(frame: np.ndarray, template: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = template.shape
    out = frame.copy()
    out[y : y + h, x : x + w] = template
    return out


def test_match_crop_finds_known_offset() -> None:
    rng = np.random.default_rng(seed=0)
    frame = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    template = rng.integers(0, 255, size=(40, 50), dtype=np.uint8)
    frame = _embed(frame, template, x=75, y=60)

    result = match_crop(frame, template)

    assert result.x == 75
    assert result.y == 60
    assert result.score > 0.99


def test_match_crop_low_score_when_template_absent() -> None:
    rng = np.random.default_rng(seed=1)
    frame = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    template = rng.integers(0, 255, size=(40, 50), dtype=np.uint8)

    result = match_crop(frame, template)

    assert result.score < 0.5
