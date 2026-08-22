"""Phase A: interactive, human-gated crop-box curation.

`render_preview` is pure and unit tested. `run_curation` drives the actual
interactive terminal flow (fetch smoke-test frame, propose a box via ffmpeg
cropdetect, let the human confirm/adjust, append to the registry) and is
exercised manually, per the spec's testing strategy.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from data_collection import extract
from data_collection.registry import VideoSource, append_to_registry


def render_preview(frame_gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    preview = frame_gray.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), color=255, thickness=2)
    return preview


_SMOKE_TEST_SEEK_SECONDS = 120


def _grab_smoke_test_frame(
    stream_url: str, width: int, height: int, headers: dict[str, str] | None = None
) -> np.ndarray:
    """Grab one full, uncropped frame -- the crop box doesn't exist yet.

    Seeks past the first couple of minutes rather than grabbing frame 0:
    longplay uploads reliably open on a channel intro or title card, not
    gameplay, so frame 0 is useless for proposing/confirming a crop box.
    """
    frames = extract.stream_frames(
        stream_url,
        crop_x=0,
        crop_y=0,
        crop_w=width,
        crop_h=height,
        fps=1,
        start_seconds=_SMOKE_TEST_SEEK_SECONDS,
        headers=headers,
    )
    return next(frames)


def run_curation(
    video_url: str,
    approved_dir: Path,
    registry_path: Path,
    game: str,
    input_func=input,
) -> None:
    stream_url, width, height, headers = extract.get_stream_info(video_url)
    frame = _grab_smoke_test_frame(stream_url, width, height, headers)

    detected = extract.detect_crop_box(
        stream_url, start_seconds=_SMOKE_TEST_SEEK_SECONDS, headers=headers
    )

    if detected is not None:
        x, y, w, h = detected
        print(f"Auto-detected crop box (ffmpeg cropdetect): x={x} y={y} w={w} h={h}")
    else:
        print("No boundary detected -- starting from the full frame.")
        x = y = 0
        w, h = width, height

    while True:
        preview = render_preview(frame, x, y, w, h)
        preview_path = approved_dir / "_preview.png"
        cv2.imwrite(str(preview_path), preview)
        print(f"Preview written to {preview_path}")
        answer = input_func(
            f"Crop box x={x} y={y} w={w} h={h} -- [a]pprove / [m]anual entry / [r]eject? "
        ).strip().lower()

        if answer == "a":
            break
        if answer == "r":
            print("Video rejected -- nothing written to the registry.")
            return
        if answer == "m":
            x = int(input_func(f"x [{x}]: ") or x)
            y = int(input_func(f"y [{y}]: ") or y)
            w = int(input_func(f"w [{w}]: ") or w)
            h = int(input_func(f"h [{h}]: ") or h)

    video_id = video_url.rstrip("/").split("=")[-1].split("/")[-1]

    source = VideoSource(
        video_id=video_id,
        url=video_url,
        game=game,
        crop_x=x,
        crop_y=y,
        crop_w=w,
        crop_h=h,
    )
    append_to_registry(registry_path, source)
    print(f"Approved and added {video_id} to {registry_path}")
