import numpy as np

from data_collection.dedup import PerceptualHashDeduper


def _solid(value: int) -> np.ndarray:
    return np.full((144, 160), value, dtype=np.uint8)


def _checkerboard() -> np.ndarray:
    frame = np.zeros((144, 160), dtype=np.uint8)
    frame[::2, ::2] = 255
    frame[1::2, 1::2] = 255
    return frame


def test_first_frame_is_never_a_duplicate() -> None:
    deduper = PerceptualHashDeduper()
    assert deduper.is_duplicate(_solid(100)) is False


def test_identical_frame_is_a_duplicate() -> None:
    deduper = PerceptualHashDeduper()
    frame = _solid(100)
    deduper.is_duplicate(frame)

    assert deduper.is_duplicate(frame.copy()) is True


def test_very_different_frame_is_not_a_duplicate() -> None:
    deduper = PerceptualHashDeduper()
    deduper.is_duplicate(_solid(0))

    assert deduper.is_duplicate(_checkerboard()) is False


def test_is_duplicate_treats_distance_equal_to_threshold_as_duplicate() -> None:
    deduper = PerceptualHashDeduper(hamming_threshold=0)
    frame = _solid(100)

    assert deduper.is_duplicate(frame) is False  # first frame: nothing to compare against yet
    assert deduper.is_duplicate(frame.copy()) is True  # identical frame -> distance 0 == threshold 0


def test_duplicate_frames_do_not_reset_the_reference() -> None:
    deduper = PerceptualHashDeduper()
    kept = _solid(100)
    deduper.is_duplicate(kept)

    # A near-duplicate is dropped and must not become the new reference.
    near_duplicate = kept.copy()
    near_duplicate[0, 0] = 101
    assert deduper.is_duplicate(near_duplicate) is True

    # Still compared against the original `kept` frame, not the dropped one.
    assert deduper.is_duplicate(kept.copy()) is True
