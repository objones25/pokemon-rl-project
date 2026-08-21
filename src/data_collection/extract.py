"""Stream Pokemon longplay frames without ever writing video to disk.

`get_stream_url` and `stream_frames` are thin glue around `yt_dlp` and a real
`ffmpeg` subprocess -- covered by the Task 13 integration test, not unit
tests. `build_ffmpeg_command` and `parse_frame_stream` are pure and unit
tested directly.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from typing import BinaryIO

import numpy as np
import yt_dlp


def build_ffmpeg_command(
    stream_url: str, crop_x: int, crop_y: int, crop_w: int, crop_h: int, fps: int = 2
) -> list[str]:
    return [
        "ffmpeg",
        "-i",
        stream_url,
        "-vf",
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},format=gray,fps={fps}",
        "-f",
        "image2pipe",
        "-pix_fmt",
        "gray",
        "-vcodec",
        "rawvideo",
        "-",
    ]


def parse_frame_stream(stdout: BinaryIO, width: int, height: int) -> Iterator[np.ndarray]:
    frame_size = width * height
    while True:
        chunk = stdout.read(frame_size)
        if len(chunk) < frame_size:
            return
        yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width)


def get_stream_url(video_url: str) -> str:
    opts = {"format": "bestvideo", "quiet": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    if info.get("url"):
        return info["url"]
    for fmt in reversed(info.get("formats", [])):
        if fmt.get("vcodec") not in (None, "none") and fmt.get("url"):
            return fmt["url"]
    raise RuntimeError(f"no playable video stream found for {video_url}")


def stream_frames(
    stream_url: str, crop_x: int, crop_y: int, crop_w: int, crop_h: int, fps: int = 2
) -> Iterator[np.ndarray]:
    cmd = build_ffmpeg_command(stream_url, crop_x, crop_y, crop_w, crop_h, fps)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert proc.stdout is not None
        yield from parse_frame_stream(proc.stdout, width=crop_w, height=crop_h)
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
