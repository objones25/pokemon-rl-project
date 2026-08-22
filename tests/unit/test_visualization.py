import numpy as np

from observability.visualization import build_contact_sheet


def test_build_contact_sheet_grid_dimensions() -> None:
    frames = [np.full((144, 160), i, dtype=np.uint8) for i in range(10)]

    sheet = build_contact_sheet(frames, cols=4)

    # 10 frames at 4 cols -> 3 rows (ceil(10/4)), each cell 144x160.
    assert sheet.shape == (144 * 3, 160 * 4)


def test_build_contact_sheet_empty_input_returns_empty_array() -> None:
    sheet = build_contact_sheet([], cols=4)
    assert sheet.size == 0
