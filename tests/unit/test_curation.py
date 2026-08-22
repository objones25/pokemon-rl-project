import numpy as np

from data_collection.curation import render_preview


def test_render_preview_draws_a_visible_box() -> None:
    frame = np.zeros((144, 160), dtype=np.uint8)

    preview = render_preview(frame, x=10, y=10, w=50, h=40)

    assert preview.shape == frame.shape
    # The rectangle outline should have introduced non-zero pixels
    # that weren't in the all-black source frame.
    assert preview.max() > 0
    # The original frame must not be mutated in place.
    assert frame.max() == 0
