"""Phase A: interactive, human-gated crop-box curation.

`render_preview` and `propose_crop_box` are pure and unit tested. `run_curation`
drives the actual interactive terminal flow (fetch smoke-test frame, propose
a box, let the human confirm/adjust, capture the reference patch, append to
the registry) and is exercised manually, per the spec's testing strategy.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from data_collection import extract
from data_collection.matching import load_template_gray, match_crop
from data_collection.registry import VideoSource, append_to_registry


def render_preview(frame_gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    preview = frame_gray.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), color=255, thickness=2)
    return preview


def propose_crop_box(
    frame_gray: np.ndarray,
    bank_templates: dict[str, np.ndarray],
    min_confidence: float = 0.7,
) -> tuple[int, int, int, int] | None:
    best_name: str | None = None
    best_score = -1.0
    best_x = best_y = 0

    for name, template in bank_templates.items():
        result = match_crop(frame_gray, template)
        if result.score > best_score:
            best_name, best_score, best_x, best_y = name, result.score, result.x, result.y

    if best_name is None or best_score < min_confidence:
        return None

    h, w = bank_templates[best_name].shape
    return (best_x, best_y, w, h)


def _grab_smoke_test_frame(stream_url: str, width: int, height: int) -> np.ndarray:
    """Grab one full, uncropped frame -- the crop box doesn't exist yet."""
    frames = extract.stream_frames(
        stream_url, crop_x=0, crop_y=0, crop_w=width, crop_h=height, fps=1
    )
    return next(frames)


def run_curation(
    video_url: str,
    bank_dir: Path,
    approved_dir: Path,
    registry_path: Path,
    game: str,
    input_func=input,
) -> None:
    stream_url, width, height = extract.get_stream_info(video_url)
    frame = _grab_smoke_test_frame(stream_url, width, height)

    bank_templates = {
        p.stem: load_template_gray(p) for p in sorted(bank_dir.glob("*.png"))
    }
    proposed = propose_crop_box(frame, bank_templates) if bank_templates else None

    if proposed is not None:
        x, y, w, h = proposed
        print(f"Proposed crop box from template match: x={x} y={y} w={w} h={h}")
    else:
        print("No confident template match found -- enter a crop box manually.")
        x = y = 0
        w, h = 160, 144

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
    reference_patch = frame[y : y + h, x : x + w]
    reference_patch_path = approved_dir / f"{video_id}.png"
    cv2.imwrite(str(reference_patch_path), reference_patch)

    source = VideoSource(
        video_id=video_id,
        url=video_url,
        game=game,
        crop_x=x,
        crop_y=y,
        crop_w=w,
        crop_h=h,
        reference_patch_path=str(reference_patch_path),
        match_confidence_baseline=1.0,
    )
    append_to_registry(registry_path, source)
    print(f"Approved and added {video_id} to {registry_path}")
