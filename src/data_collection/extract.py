"""Stream Pokemon longplay frames without ever writing video to disk.

`get_stream_url` and `stream_frames` are thin glue around `yt_dlp` and a real
`ffmpeg` subprocess -- covered by the Task 13 integration test, not unit
tests. `build_ffmpeg_command` and `parse_frame_stream` are pure and unit
tested directly.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from typing import IO

import numpy as np
import yt_dlp

_CROPDETECT_PATTERN = re.compile(rb"crop=(\d+):(\d+):(\d+):(\d+)")


def _format_headers(headers: dict[str, str] | None) -> str | None:
    if not headers:
        return None
    return "".join(f"{key}: {value}\r\n" for key, value in headers.items())


def build_ffmpeg_command(
    stream_url: str,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    fps: int = 2,
    start_seconds: float = 0,
    headers: dict[str, str] | None = None,
) -> list[str]:
    cmd = ["ffmpeg", "-loglevel", "error"]
    if start_seconds:
        cmd += ["-ss", str(start_seconds)]
    header_str = _format_headers(headers)
    if header_str:
        cmd += ["-headers", header_str]
    cmd += [
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
    return cmd


def parse_frame_stream(stdout: IO[bytes], width: int, height: int) -> Iterator[np.ndarray]:
    frame_size = width * height
    while True:
        chunk = stdout.read(frame_size)
        if len(chunk) < frame_size:
            return
        yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width)


def get_stream_info(video_url: str) -> tuple[str, int, int, dict[str, str]]:
    """Returns (stream_url, width, height, http_headers) for the best
    available video-only stream. Longplay uploads range from ~360p to 4K, so
    callers that need to grab a full, uncropped frame (curation's smoke
    test) must use the real dimensions here rather than guessing a fixed
    resolution.

    `http_headers` (User-Agent etc.) must be replayed on every subsequent
    request to this URL. YouTube's CDN validates them alongside the URL's
    own signature -- fetching the bare URL with ffmpeg's default headers
    works inconsistently from a residential IP and reliably 403s from a
    datacenter IP (e.g. a RunPod pod), since the mismatched headers read as
    bot traffic.
    """
    opts = {"format": "bestvideo", "quiet": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore[arg-type]
        info = ydl.extract_info(video_url, download=False)

    url = info.get("url")
    width = info.get("width")
    height = info.get("height")
    if url and width and height:
        return url, width, height, info.get("http_headers") or {}
    for fmt in reversed(info.get("formats") or []):
        fmt_url = fmt.get("url")
        if fmt.get("vcodec") not in (None, "none") and fmt_url:
            headers = fmt.get("http_headers") or info.get("http_headers") or {}
            return fmt_url, fmt.get("width") or width or 0, fmt.get("height") or height or 0, headers
    raise RuntimeError(f"no playable video stream found for {video_url}")


def get_stream_url(video_url: str) -> str:
    url, _, _, _ = get_stream_info(video_url)
    return url


def detect_crop_box(
    stream_url: str,
    start_seconds: float,
    duration_seconds: float = 20,
    headers: dict[str, str] | None = None,
) -> tuple[int, int, int, int] | None:
    """Auto-detects the real content region via ffmpeg's `cropdetect` filter
    over a short segment, so curation proposes a starting crop box derived
    from the actual video instead of a guess. Returns (x, y, w, h), or None
    if `cropdetect` produced no usable output at all (ffmpeg failure, an
    unreadable stream, etc. -- NOT the same as "no letterbox found", which
    `cropdetect` reports as a crop matching the full frame).
    """
    cmd = ["ffmpeg", "-loglevel", "info", "-ss", str(start_seconds)]
    header_str = _format_headers(headers)
    if header_str:
        cmd += ["-headers", header_str]
    cmd += [
        "-i",
        stream_url,
        "-t",
        str(duration_seconds),
        "-vf",
        "cropdetect=24:2:0",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    matches = _CROPDETECT_PATTERN.findall(proc.stderr)
    if not matches:
        return None
    w, h, x, y = (int(v) for v in matches[-1])
    return (x, y, w, h)


def stream_frames(
    stream_url: str,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    fps: int = 2,
    start_seconds: float = 0,
    headers: dict[str, str] | None = None,
) -> Iterator[np.ndarray]:
    cmd = build_ffmpeg_command(
        stream_url, crop_x, crop_y, crop_w, crop_h, fps, start_seconds, headers
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    exhausted_naturally = False
    try:
        assert proc.stdout is not None
        yield from parse_frame_stream(proc.stdout, width=crop_w, height=crop_h)
        exhausted_naturally = True
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if exhausted_naturally and proc.returncode != 0:
            stderr_text = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise RuntimeError(
                f"ffmpeg exited with code {proc.returncode} for {stream_url}: "
                f"{stderr_text[-2000:]}"
            )
