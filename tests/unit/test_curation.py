import numpy as np
import pytest

from data_collection.curation import extract_video_id, render_preview


def test_render_preview_draws_a_visible_box() -> None:
    frame = np.zeros((144, 160), dtype=np.uint8)

    preview = render_preview(frame, x=10, y=10, w=50, h=40)

    assert preview.shape == frame.shape
    # The rectangle outline should have introduced non-zero pixels
    # that weren't in the all-black source frame.
    assert preview.max() > 0
    # The original frame must not be mutated in place.
    assert frame.max() == 0


@pytest.mark.parametrize(
    ("video_url", "expected_id"),
    [
        ("https://www.youtube.com/watch?v=abc123XYZ_-", "abc123XYZ_-"),
        ("https://youtu.be/abc123XYZ_-", "abc123XYZ_-"),
        # A timestamped share link: a second "=" (in "t=30s") after "v=" is
        # exactly what makes video_url.split("=")[-1] pick the timestamp
        # fragment instead of the video id.
        ("https://www.youtube.com/watch?v=abc123XYZ_-&t=30s", "abc123XYZ_-"),
        # A timestamped youtu.be short link: no "v=" query param at all, so
        # the id must come from the path, not the (absent) query value.
        ("https://youtu.be/abc123XYZ_-?t=30", "abc123XYZ_-"),
        ("https://youtu.be/abc123XYZ_-/", "abc123XYZ_-"),
    ],
    ids=["plain_watch_url", "plain_short_url", "watch_url_with_timestamp", "short_url_with_timestamp", "trailing_slash"],
)
def test_extract_video_id_returns_the_video_id_not_a_query_fragment(
    video_url: str, expected_id: str
) -> None:
    assert extract_video_id(video_url) == expected_id
