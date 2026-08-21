import numpy as np

from data_collection.curation import propose_crop_box, render_preview


def test_render_preview_draws_a_visible_box() -> None:
    frame = np.zeros((144, 160), dtype=np.uint8)

    preview = render_preview(frame, x=10, y=10, w=50, h=40)

    assert preview.shape == frame.shape
    # The rectangle outline should have introduced non-zero pixels
    # that weren't in the all-black source frame.
    assert preview.max() > 0
    # The original frame must not be mutated in place.
    assert frame.max() == 0


def test_propose_crop_box_finds_best_matching_template() -> None:
    rng = np.random.default_rng(seed=0)
    frame = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    good_template = rng.integers(0, 255, size=(144, 160), dtype=np.uint8)
    frame[30:174, 60:220] = good_template
    bad_template = np.random.default_rng(seed=7).integers(0, 255, size=(144, 160), dtype=np.uint8)

    box = propose_crop_box(
        frame,
        bank_templates={"bad": bad_template, "good": good_template},
        min_confidence=0.7,
    )

    assert box == (60, 30, 160, 144)


def test_propose_crop_box_returns_none_when_no_template_confident() -> None:
    rng = np.random.default_rng(seed=0)
    frame = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    unrelated_template = np.random.default_rng(seed=9).integers(
        0, 255, size=(144, 160), dtype=np.uint8
    )

    box = propose_crop_box(
        frame, bank_templates={"unrelated": unrelated_template}, min_confidence=0.7
    )

    assert box is None
