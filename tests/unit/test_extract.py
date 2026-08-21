import io

import numpy as np

from data_collection.extract import build_ffmpeg_command, parse_frame_stream


def test_build_ffmpeg_command_applies_crop_gray_and_fps() -> None:
    cmd = build_ffmpeg_command(
        "https://example.com/stream.m3u8", crop_x=10, crop_y=20, crop_w=160, crop_h=144, fps=2
    )

    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd and cmd[cmd.index("-i") + 1] == "https://example.com/stream.m3u8"
    vf_index = cmd.index("-vf")
    assert cmd[vf_index + 1] == "crop=160:144:10:20,format=gray,fps=2"
    assert cmd[-4:] == ["-f", "image2pipe", "-pix_fmt", "gray"] or "-vcodec" in cmd


def test_parse_frame_stream_yields_correct_shapes_and_values() -> None:
    width, height = 4, 3
    frame_a = np.arange(width * height, dtype=np.uint8).reshape(height, width)
    frame_b = np.full((height, width), 42, dtype=np.uint8)
    raw = frame_a.tobytes() + frame_b.tobytes()
    stdout = io.BytesIO(raw)

    frames = list(parse_frame_stream(stdout, width=width, height=height))

    assert len(frames) == 2
    assert frames[0].shape == (height, width)
    np.testing.assert_array_equal(frames[0], frame_a)
    np.testing.assert_array_equal(frames[1], frame_b)


def test_parse_frame_stream_drops_trailing_partial_frame() -> None:
    width, height = 4, 3
    frame_a = np.arange(width * height, dtype=np.uint8).reshape(height, width)
    raw = frame_a.tobytes() + b"\x00\x01"  # incomplete trailing frame
    stdout = io.BytesIO(raw)

    frames = list(parse_frame_stream(stdout, width=width, height=height))

    assert len(frames) == 1
