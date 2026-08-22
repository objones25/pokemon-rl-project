from pathlib import Path

import pytest
import yaml

from data_collection.registry import VideoSource, append_to_registry, load_registry


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data))


def test_load_registry_parses_valid_entries(tmp_path: Path) -> None:
    path = tmp_path / "video_sources.yaml"
    _write_yaml(
        path,
        {
            "videos": [
                {
                    "video_id": "abc123",
                    "url": "https://youtube.com/watch?v=abc123",
                    "game": "red",
                    "crop_x": 10,
                    "crop_y": 20,
                    "crop_w": 160,
                    "crop_h": 144,
                }
            ]
        },
    )

    sources = load_registry(path)

    assert sources == [
        VideoSource(
            video_id="abc123",
            url="https://youtube.com/watch?v=abc123",
            game="red",
            crop_x=10,
            crop_y=20,
            crop_w=160,
            crop_h=144,
        )
    ]


def test_load_registry_empty_videos_list(tmp_path: Path) -> None:
    path = tmp_path / "video_sources.yaml"
    _write_yaml(path, {"videos": []})

    assert load_registry(path) == []


def test_load_registry_rejects_invalid_game(tmp_path: Path) -> None:
    path = tmp_path / "video_sources.yaml"
    _write_yaml(
        path,
        {
            "videos": [
                {
                    "video_id": "abc123",
                    "url": "https://youtube.com/watch?v=abc123",
                    "game": "yellow",
                    "crop_x": 0,
                    "crop_y": 0,
                    "crop_w": 160,
                    "crop_h": 144,
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="game"):
        load_registry(path)


def test_load_registry_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "video_sources.yaml"
    _write_yaml(
        path,
        {
            "videos": [
                {
                    "video_id": "abc123",
                    "url": "https://youtube.com/watch?v=abc123",
                    "game": "red",
                    "crop_x": 0,
                    "crop_y": 0,
                    # missing crop_w and crop_h
                }
            ]
        },
    )

    with pytest.raises(ValueError):
        load_registry(path)


def test_append_to_registry_adds_entry(tmp_path: Path) -> None:
    path = tmp_path / "video_sources.yaml"
    _write_yaml(path, {"videos": []})
    new_source = VideoSource(
        video_id="def456",
        url="https://youtube.com/watch?v=def456",
        game="blue",
        crop_x=5,
        crop_y=5,
        crop_w=160,
        crop_h=144,
    )

    append_to_registry(path, new_source)

    assert load_registry(path) == [new_source]


def test_append_to_registry_preserves_existing_entries(tmp_path: Path) -> None:
    path = tmp_path / "video_sources.yaml"
    first = VideoSource(
        video_id="abc123",
        url="https://youtube.com/watch?v=abc123",
        game="red",
        crop_x=0,
        crop_y=0,
        crop_w=160,
        crop_h=144,
    )
    _write_yaml(path, {"videos": []})
    append_to_registry(path, first)

    second = VideoSource(
        video_id="def456",
        url="https://youtube.com/watch?v=def456",
        game="blue",
        crop_x=5,
        crop_y=5,
        crop_w=160,
        crop_h=144,
    )
    append_to_registry(path, second)

    assert load_registry(path) == [first, second]
