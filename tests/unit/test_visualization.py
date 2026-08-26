import numpy as np

from observability.visualization import (
    build_augmentation_contact_sheet,
    build_contact_sheet,
    build_pair_preview,
)


def test_build_contact_sheet_grid_dimensions() -> None:
    frames = [np.full((144, 160), i, dtype=np.uint8) for i in range(10)]

    sheet = build_contact_sheet(frames, cols=4)

    # 10 frames at 4 cols -> 3 rows (ceil(10/4)), each cell 144x160.
    assert sheet.shape == (144 * 3, 160 * 4)


def test_build_contact_sheet_places_each_frame_in_its_own_cell_and_zero_pads_the_rest() -> None:
    # Values start at 1, not 0: 0 is reserved as the zero-fill padding
    # default, so a value of 0 in a frame cell would be indistinguishable
    # from that cell never having been written (e.g. an off-by-one bug that
    # skips the first frame). Starting at 1 makes every written cell
    # unambiguously non-zero.
    frames = [np.full((144, 160), value, dtype=np.uint8) for value in range(1, 11)]

    sheet = build_contact_sheet(frames, cols=4)

    # 10 frames at 4 cols -> 3 rows (ceil(10/4)), each cell 144x160.
    assert sheet.shape == (144 * 3, 160 * 4)
    # Spot-check placement: frame 0 (value 1) in the top-left cell, frame 9
    # (value 10, the last frame) in row 2, col 1 -- divmod(9, 4) == (2, 1).
    assert np.all(sheet[:144, :160] == 1)
    assert np.all(sheet[288:, 160:320] == 10)
    # The last row's remaining two cells (cols 2 and 3) were never written --
    # must stay at the zero-fill default, not leak stale/garbage data.
    assert np.all(sheet[288:, 320:480] == 0)
    assert np.all(sheet[288:, 480:] == 0)


def test_build_contact_sheet_empty_input_returns_empty_array() -> None:
    sheet = build_contact_sheet([], cols=4)
    assert sheet.size == 0


def test_build_pair_preview_concatenates_horizontally() -> None:
    original = np.full((144, 160), 10, dtype=np.uint8)
    view_a = np.full((144, 160), 20, dtype=np.uint8)
    view_b = np.full((144, 160), 30, dtype=np.uint8)

    preview = build_pair_preview(original, view_a, view_b)

    assert preview.shape == (144, 480)
    assert np.all(preview[:, :160] == 10)
    assert np.all(preview[:, 160:320] == 20)
    assert np.all(preview[:, 320:] == 30)


def test_build_augmentation_contact_sheet_stacks_rows_vertically() -> None:
    triples = [
        (
            np.full((144, 160), i, dtype=np.uint8),
            np.full((144, 160), i + 1, dtype=np.uint8),
            np.full((144, 160), i + 2, dtype=np.uint8),
        )
        for i in range(3)
    ]

    sheet = build_augmentation_contact_sheet(triples)

    assert sheet.shape == (144 * 3, 480)


def test_build_augmentation_contact_sheet_empty_input_returns_empty_array() -> None:
    sheet = build_augmentation_contact_sheet([])
    assert sheet.size == 0
