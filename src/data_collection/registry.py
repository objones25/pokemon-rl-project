"""Load and update the approved-video registry (configs/video_sources.yaml)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

_VALID_GAMES = {"red", "blue"}
_REQUIRED_FIELDS = (
    "video_id",
    "url",
    "game",
    "crop_x",
    "crop_y",
    "crop_w",
    "crop_h",
)


@dataclass(frozen=True)
class VideoSource:
    video_id: str
    url: str
    game: str
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int


def _parse_entry(entry: dict) -> VideoSource:
    missing = [f for f in _REQUIRED_FIELDS if f not in entry]
    if missing:
        raise ValueError(f"registry entry missing fields: {missing}")
    if entry["game"] not in _VALID_GAMES:
        raise ValueError(
            f"invalid game {entry['game']!r}, expected one of {_VALID_GAMES}"
        )
    return VideoSource(
        video_id=entry["video_id"],
        url=entry["url"],
        game=entry["game"],
        crop_x=entry["crop_x"],
        crop_y=entry["crop_y"],
        crop_w=entry["crop_w"],
        crop_h=entry["crop_h"],
    )


def load_registry(path: str | Path) -> list[VideoSource]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    entries = data.get("videos") or []
    return [_parse_entry(entry) for entry in entries]


def append_to_registry(path: str | Path, source: VideoSource) -> None:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    videos = data.get("videos") or []
    videos.append(asdict(source))
    data["videos"] = videos
    path.write_text(yaml.safe_dump(data, sort_keys=False))
