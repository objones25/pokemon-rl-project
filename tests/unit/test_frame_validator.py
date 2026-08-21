import numpy as np

from data_collection.frame_validator import FrameValidator


def _reference() -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, size=(144, 160), dtype=np.uint8)


def test_matching_frame_is_kept_and_not_halted() -> None:
    reference = _reference()
    validator = FrameValidator(reference, baseline_score=1.0)

    result = validator.validate(reference.copy())

    assert result.keep is True
    assert result.halted is False


def test_wildly_different_frame_is_dropped() -> None:
    reference = _reference()
    validator = FrameValidator(reference, baseline_score=1.0)
    unrelated = np.random.default_rng(seed=99).integers(0, 255, size=(144, 160), dtype=np.uint8)

    result = validator.validate(unrelated)

    assert result.keep is False


def test_sustained_anomalies_trigger_halt() -> None:
    reference = _reference()
    validator = FrameValidator(
        reference, baseline_score=1.0, window_size=10, drop_ratio_threshold=0.2
    )
    unrelated = np.random.default_rng(seed=99).integers(0, 255, size=(144, 160), dtype=np.uint8)

    results = [validator.validate(unrelated) for _ in range(10)]

    assert results[-1].halted is True


def test_occasional_anomalies_do_not_trigger_halt() -> None:
    reference = _reference()
    validator = FrameValidator(
        reference, baseline_score=1.0, window_size=10, drop_ratio_threshold=0.2
    )
    unrelated = np.random.default_rng(seed=99).integers(0, 255, size=(144, 160), dtype=np.uint8)

    # 1 anomaly out of 10 frames = 10% drop rate, below the 20% threshold.
    results = [validator.validate(reference.copy()) for _ in range(9)]
    results.append(validator.validate(unrelated))

    assert all(r.halted is False for r in results)
