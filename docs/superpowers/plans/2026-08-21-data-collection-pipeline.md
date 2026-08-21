# Data Collection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-phase pipeline that turns human-curated YouTube Pokemon Red/Blue longplay videos into a grayscale, deduplicated frame dataset on Hugging Face Hub, ready for contrastive (SimCLR/BYOL) CNN pretraining.

**Architecture:** Phase A is an interactive CLI tool that proposes a crop box against a candidate video's smoke-test frame, lets a human confirm/adjust it, and saves an approved `VideoSource` (crop box + a self-captured reference patch image) to a YAML registry. Phase B is an unattended pipeline (intended to run on a RunPod CPU pod) that, per approved video, streams frames via `yt-dlp` + `ffmpeg`, validates each frame against its video's own reference patch (dropping anomalies, halting on a sustained bad streak), deduplicates near-identical frames via perceptual hashing, batches survivors into Parquet shards with the `datasets` `Image` feature, and pushes them to a private HF dataset repo — with a `manifest.json` (also in the repo) enabling resume-by-video after a crash.

**Tech Stack:** Python 3.12, `uv`, `click` (CLI), `opencv-python-headless` (template matching), `imagehash` + `Pillow` (dedup), `datasets` + `huggingface_hub` (storage), `yt-dlp` (stream extraction), `ffmpeg` (system binary, crop/grayscale/fps filtering), `trackio` (live metrics), stdlib `logging` (structured JSON logs), `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-21-data-collection-pipeline-design.md`

## Global Constraints

- Package management is `uv` only (`uv add`, `uv run`) — never bare `pip`/`venv`. (spec: Tooling)
- Frame sample rate is 2 FPS. (spec: Purpose)
- Every frame is converted to grayscale (`format=gray` in the `ffmpeg` filter chain) before any storage or comparison. (spec: Color)
- Storage is a private Hugging Face Hub dataset repo; shards are Parquet using the `datasets` library's `Image` feature. (spec: Storage, Shard format)
- Dedup compares each frame's perceptual hash only against the last _kept_ frame, not full history. (spec: Architecture, Phase B step 3)
- A video's rolling anomaly drop-rate halting threshold defaults to >20% over a window (exact window size fixed in Task 6). (spec: Architecture, Phase B step 3)
- Crop-box approval is always human-gated in Phase A; Phase B never auto-detects or auto-approves a crop. (spec: Crop-box detection)
- Phase B is designed to run unattended on a RunPod CPU pod; `ffmpeg` and `yt-dlp` must be available there — do not assume they exist on the dev machine used to write/review this code. (spec: Execution environment)
- No frame is ever written to local disk as video — only extracted frame arrays and Parquet shards touch disk/memory. (spec: Execution environment)
- Every module takes its external dependencies (HTTP/Hub client, subprocess runner, filesystem) as constructor/function arguments, so core logic is unit-testable without network, `ffmpeg`, or a GPU. (spec: Components)

---

## Design note: per-video reference patch, not a shared runtime-baseline template

The spec says Phase A finds a crop box by matching against "a small bank of reference Game Boy UI crops" and stores "the winning template" and its match-confidence as the runtime baseline (spec: Architecture, Phase A step 2 & 4). This plan implements that with one refinement not pinned down by the spec: the shared bank (`configs/templates/bank/`) is used only to _propose_ a starting crop box for a brand-new video — it plays no role at runtime. Once a human confirms a crop box, the tool cuts that exact region out of the smoke-test frame and saves it as `configs/templates/approved/{video_id}.png` — a per-video reference patch, captured at that video's own resolution/scale. Phase B's frame validator matches each frame against _that video's own patch_, not the shared bank. This sidesteps template/video scale mismatches entirely (a shared low-res bank template would otherwise need multi-scale matching to work across videos of different upload resolutions) and keeps the "runtime baseline" score meaningful: it starts at (very close to) 1.0 by construction, since the reference patch is cut directly from real footage of that same video.

---

## Task 1: Project scaffolding

**Files:**

- Modify: `pyproject.toml`
- Create: `src/data_collection/__init__.py`
- Create: `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
- Create: `configs/templates/bank/README.md`
- Create: `configs/templates/approved/.gitkeep`
- Create: `configs/video_sources.yaml`

**Interfaces:**

- Produces: an importable `data_collection` package under `src/`, a working `uv run pytest` command, a `data-collection` console script entry point (wired to `data_collection.cli:main`, implemented in Task 12), and a `slow` pytest marker deselected by default.

- [ ] **Step 1: Add runtime and dev dependencies**

```bash
uv add opencv-python-headless numpy pillow imagehash pyyaml datasets huggingface_hub click trackio yt-dlp
uv add --dev pytest
```

- [ ] **Step 2: Configure `pyproject.toml` for a `src/` package layout, console script, and pytest**

Edit `pyproject.toml` so it reads:

```toml
[project]
name = "pokemon-rl-project"
version = "0.1.0"
description = "Data collection pipeline for Pokemon Red/Blue contrastive CNN pretraining"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "opencv-python-headless",
    "numpy",
    "pillow",
    "imagehash",
    "pyyaml",
    "datasets",
    "huggingface_hub",
    "click",
    "trackio",
    "yt-dlp",
]

[project.scripts]
data-collection = "data_collection.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/data_collection"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: requires real network/ffmpeg/HF credentials (deselect with -m 'not slow')",
]
addopts = "-m \"not slow\""

[dependency-groups]
dev = ["pytest"]
```

(`uv add` will have already inserted the `dependencies` and `[dependency-groups]` entries — reconcile rather than duplicate; the important additions to make by hand are `[project.scripts]`, `[build-system]`, `[tool.hatch.build.targets.wheel]`, and `[tool.pytest.ini_options]`.)

- [ ] **Step 3: Create package and test package skeletons**

```bash
mkdir -p src/data_collection tests/unit tests/integration
touch src/data_collection/__init__.py tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
```

- [ ] **Step 4: Create the templates and registry directories with placeholders**

Create `configs/templates/bank/README.md`:

```markdown
# Reference template bank

Used only by `data_collection.curation` to _propose_ a starting crop box for
a brand-new video. Add a few PNG screenshots here of distinctive, mostly-static
Game Boy Pokemon Red/Blue screens (e.g. the start menu fully open). Capture
them yourself from a video you have rights to view, at roughly the resolution
you expect most candidate videos to be uploaded at (curation still lets you
manually adjust the proposed box, so this doesn't need to be exact).

This directory ships empty — there is nothing here until you add templates.
```

```bash
mkdir -p configs/templates/bank configs/templates/approved
touch configs/templates/approved/.gitkeep
```

Create `configs/video_sources.yaml`:

```yaml
# Approved video sources. Populated by `data-collection curate <url>`.
# Do not hand-edit crop coordinates without re-running curation --
# they must match the reference patch image captured at approval time.
videos: []
```

- [ ] **Step 5: Verify the scaffold**

```bash
uv sync
uv run pytest
```

Expected: `uv sync` succeeds; `pytest` reports "no tests ran" (0 collected) with exit code 0 (or 5, pytest's "no tests collected" code — either is acceptable here).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/ configs/
git commit -m "chore: scaffold data collection package, deps, pytest config"
```

---

## Task 2: Video registry (`registry.py`)

**Files:**

- Create: `src/data_collection/registry.py`
- Test: `tests/unit/test_registry.py`

**Interfaces:**

- Produces:

  ```python
  @dataclass(frozen=True)
  class VideoSource:
      video_id: str
      url: str
      game: str  # "red" or "blue"
      crop_x: int
      crop_y: int
      crop_w: int
      crop_h: int
      reference_patch_path: str
      match_confidence_baseline: float

  def load_registry(path: str | Path) -> list[VideoSource]: ...
  def append_to_registry(path: str | Path, source: VideoSource) -> None: ...
  ```

- Consumes: nothing (pure I/O + validation over YAML).

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_registry.py`:

```python
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
                    "reference_patch_path": "configs/templates/approved/abc123.png",
                    "match_confidence_baseline": 0.999,
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
            reference_patch_path="configs/templates/approved/abc123.png",
            match_confidence_baseline=0.999,
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
                    "reference_patch_path": "x.png",
                    "match_confidence_baseline": 1.0,
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
                    "crop_w": 160,
                    "crop_h": 144,
                    # missing reference_patch_path and match_confidence_baseline
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
        reference_patch_path="configs/templates/approved/def456.png",
        match_confidence_baseline=1.0,
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
        reference_patch_path="configs/templates/approved/abc123.png",
        match_confidence_baseline=1.0,
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
        reference_patch_path="configs/templates/approved/def456.png",
        match_confidence_baseline=1.0,
    )
    append_to_registry(path, second)

    assert load_registry(path) == [first, second]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_registry.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.registry'`.

- [ ] **Step 3: Implement `registry.py`**

```python
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
    "reference_patch_path",
    "match_confidence_baseline",
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
    reference_patch_path: str
    match_confidence_baseline: float


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
        reference_patch_path=entry["reference_patch_path"],
        match_confidence_baseline=entry["match_confidence_baseline"],
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_registry.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_collection/registry.py tests/unit/test_registry.py
git commit -m "feat: add video source registry load/append"
```

---

## Task 3: Template matching (`matching.py`)

**Files:**

- Create: `src/data_collection/matching.py`
- Test: `tests/unit/test_matching.py`

**Interfaces:**

- Produces:

  ```python
  @dataclass(frozen=True)
  class MatchResult:
      x: int
      y: int
      score: float

  def load_template_gray(path: str | Path) -> np.ndarray: ...
  def match_crop(frame_gray: np.ndarray, template_gray: np.ndarray) -> MatchResult: ...
  ```

- Consumes: nothing beyond numpy/cv2 arrays — pure, no I/O in `match_crop`.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_matching.py`:

```python
import numpy as np

from data_collection.matching import match_crop


def _embed(frame: np.ndarray, template: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = template.shape
    out = frame.copy()
    out[y : y + h, x : x + w] = template
    return out


def test_match_crop_finds_known_offset() -> None:
    rng = np.random.default_rng(seed=0)
    frame = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    template = rng.integers(0, 255, size=(40, 50), dtype=np.uint8)
    frame = _embed(frame, template, x=75, y=60)

    result = match_crop(frame, template)

    assert result.x == 75
    assert result.y == 60
    assert result.score > 0.99


def test_match_crop_low_score_when_template_absent() -> None:
    rng = np.random.default_rng(seed=1)
    frame = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    template = rng.integers(0, 255, size=(40, 50), dtype=np.uint8)

    result = match_crop(frame, template)

    assert result.score < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_matching.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.matching'`.

- [ ] **Step 3: Implement `matching.py`**

```python
"""Grayscale template matching, shared by curation (Phase A) and the
runtime frame validator (Phase B) so both use exactly the same logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchResult:
    x: int
    y: int
    score: float


def load_template_gray(path: str | Path) -> np.ndarray:
    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"could not read template image: {path}")
    return template


def match_crop(frame_gray: np.ndarray, template_gray: np.ndarray) -> MatchResult:
    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    x, y = max_loc
    return MatchResult(x=x, y=y, score=float(max_val))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_matching.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_collection/matching.py tests/unit/test_matching.py
git commit -m "feat: add grayscale template matching"
```

---

## Task 4: Perceptual-hash dedup (`dedup.py`)

**Files:**

- Create: `src/data_collection/dedup.py`
- Test: `tests/unit/test_dedup.py`

**Interfaces:**

- Produces:
  ```python
  class PerceptualHashDeduper:
      def __init__(self, hamming_threshold: int = 5) -> None: ...
      def is_duplicate(self, frame_gray: np.ndarray) -> bool: ...
  ```
  Calling `is_duplicate` updates the "last kept frame" hash whenever it returns `False`; a `True` return leaves state unchanged (the frame is dropped, not kept).
- Consumes: nothing beyond numpy arrays.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_dedup.py`:

```python
import numpy as np

from data_collection.dedup import PerceptualHashDeduper


def _solid(value: int) -> np.ndarray:
    return np.full((144, 160), value, dtype=np.uint8)


def _checkerboard() -> np.ndarray:
    frame = np.zeros((144, 160), dtype=np.uint8)
    frame[::2, ::2] = 255
    frame[1::2, 1::2] = 255
    return frame


def test_first_frame_is_never_a_duplicate() -> None:
    deduper = PerceptualHashDeduper()
    assert deduper.is_duplicate(_solid(100)) is False


def test_identical_frame_is_a_duplicate() -> None:
    deduper = PerceptualHashDeduper()
    frame = _solid(100)
    deduper.is_duplicate(frame)

    assert deduper.is_duplicate(frame.copy()) is True


def test_very_different_frame_is_not_a_duplicate() -> None:
    deduper = PerceptualHashDeduper()
    deduper.is_duplicate(_solid(0))

    assert deduper.is_duplicate(_checkerboard()) is False


def test_duplicate_frames_do_not_reset_the_reference() -> None:
    deduper = PerceptualHashDeduper()
    kept = _solid(100)
    deduper.is_duplicate(kept)

    # A near-duplicate is dropped and must not become the new reference.
    near_duplicate = kept.copy()
    near_duplicate[0, 0] = 101
    assert deduper.is_duplicate(near_duplicate) is True

    # Still compared against the original `kept` frame, not the dropped one.
    assert deduper.is_duplicate(kept.copy()) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_dedup.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.dedup'`.

- [ ] **Step 3: Implement `dedup.py`**

```python
"""Perceptual-hash near-duplicate filter, compared against the last kept frame."""

from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image


class PerceptualHashDeduper:
    def __init__(self, hamming_threshold: int = 5) -> None:
        self._hamming_threshold = hamming_threshold
        self._last_kept_hash: imagehash.ImageHash | None = None

    def is_duplicate(self, frame_gray: np.ndarray) -> bool:
        current_hash = imagehash.phash(Image.fromarray(frame_gray, mode="L"))

        if self._last_kept_hash is not None:
            distance = current_hash - self._last_kept_hash
            if distance <= self._hamming_threshold:
                return True

        self._last_kept_hash = current_hash
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_dedup.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_collection/dedup.py tests/unit/test_dedup.py
git commit -m "feat: add perceptual-hash dedup filter"
```

---

## Task 5: Frame stream extraction (`extract.py`)

**Files:**

- Create: `src/data_collection/extract.py`
- Test: `tests/unit/test_extract.py`

**Interfaces:**

- Produces:
  ```python
  def build_ffmpeg_command(stream_url: str, crop_x: int, crop_y: int, crop_w: int, crop_h: int, fps: int = 2) -> list[str]: ...
  def parse_frame_stream(stdout: BinaryIO, width: int, height: int) -> Iterator[np.ndarray]: ...
  def get_stream_url(video_url: str) -> str: ...  # thin yt-dlp wrapper, not unit tested
  def stream_frames(stream_url: str, crop_x: int, crop_y: int, crop_w: int, crop_h: int, fps: int = 2) -> Iterator[np.ndarray]: ...  # thin subprocess wrapper, not unit tested
  ```
- Consumes: `data_collection.matching` not required here — this module has no dependency on Task 2-4.

Only `build_ffmpeg_command` and `parse_frame_stream` are pure and unit-tested; `get_stream_url` and `stream_frames` are thin I/O glue exercised by the Task 13 integration test.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_extract.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_extract.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.extract'`.

- [ ] **Step 3: Implement `extract.py`**

```python
"""Stream Pokemon longplay frames without ever writing video to disk.

`get_stream_url` and `stream_frames` are thin glue around `yt_dlp` and a real
`ffmpeg` subprocess -- covered by the Task 13 integration test, not unit
tests. `build_ffmpeg_command` and `parse_frame_stream` are pure and unit
tested directly.
"""

from __future__ import annotations

import subprocess
from typing import BinaryIO, Iterator

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
        proc.stdout.close() if proc.stdout else None
        proc.wait()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_extract.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_collection/extract.py tests/unit/test_extract.py
git commit -m "feat: add ffmpeg/yt-dlp frame streaming"
```

---

## Task 6: Frame validator (`frame_validator.py`)

**Files:**

- Create: `src/data_collection/frame_validator.py`
- Test: `tests/unit/test_frame_validator.py`

**Interfaces:**

- Consumes: `data_collection.matching.match_crop`, `data_collection.matching.MatchResult` (Task 3).
- Produces:

  ```python
  @dataclass(frozen=True)
  class ValidationResult:
      keep: bool
      halted: bool

  class FrameValidator:
      def __init__(
          self,
          reference_patch_gray: np.ndarray,
          baseline_score: float,
          score_ratio_threshold: float = 0.6,
          window_size: int = 50,
          drop_ratio_threshold: float = 0.2,
      ) -> None: ...
      def validate(self, frame_crop_gray: np.ndarray) -> ValidationResult: ...
  ```

  `window_size=50` at 2 FPS is a 25-second rolling window — long enough that a few noisy frames don't trip a halt, short enough to catch a sustained anomaly (an ad break, a bad re-encode) within about half a minute.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_frame_validator.py`:

```python
import numpy as np

from data_collection.frame_validator import FrameValidator


def _reference() -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, size=(144, 160), dtype=np.uint8)


def test_matching_frame_is_kept_and_not_halted() -> None:
    reference = _reference()
    validator = FrameValidator(reference, baseline_score=1.0)

    result = validator.validate(reference.copy())

    assert result.keep is True
    assert result.halted is False


def test_wildly_different_frame_is_dropped() -> None:
    reference = _reference()
    validator = FrameValidator(reference, baseline_score=1.0)
    unrelated = np.random.default_rng(seed=99).integers(0, 255, size=(144, 160), dtype=np.uint8)

    result = validator.validate(unrelated)

    assert result.keep is False


def test_sustained_anomalies_trigger_halt() -> None:
    reference = _reference()
    validator = FrameValidator(
        reference, baseline_score=1.0, window_size=10, drop_ratio_threshold=0.2
    )
    unrelated = np.random.default_rng(seed=99).integers(0, 255, size=(144, 160), dtype=np.uint8)

    results = [validator.validate(unrelated) for _ in range(10)]

    assert results[-1].halted is True


def test_occasional_anomalies_do_not_trigger_halt() -> None:
    reference = _reference()
    validator = FrameValidator(
        reference, baseline_score=1.0, window_size=10, drop_ratio_threshold=0.2
    )
    unrelated = np.random.default_rng(seed=99).integers(0, 255, size=(144, 160), dtype=np.uint8)

    # 1 anomaly out of 10 frames = 10% drop rate, below the 20% threshold.
    results = [validator.validate(reference.copy()) for _ in range(9)]
    results.append(validator.validate(unrelated))

    assert all(r.halted is False for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_frame_validator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.frame_validator'`.

- [ ] **Step 3: Implement `frame_validator.py`**

```python
"""Re-check extracted frames against the curation-time reference patch."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from data_collection.matching import match_crop


@dataclass(frozen=True)
class ValidationResult:
    keep: bool
    halted: bool


class FrameValidator:
    def __init__(
        self,
        reference_patch_gray: np.ndarray,
        baseline_score: float,
        score_ratio_threshold: float = 0.6,
        window_size: int = 50,
        drop_ratio_threshold: float = 0.2,
    ) -> None:
        self._reference = reference_patch_gray
        self._min_score = baseline_score * score_ratio_threshold
        self._drop_ratio_threshold = drop_ratio_threshold
        self._window: deque[int] = deque(maxlen=window_size)
        self._halted = False

    def validate(self, frame_crop_gray: np.ndarray) -> ValidationResult:
        score = match_crop(frame_crop_gray, self._reference).score
        keep = score >= self._min_score

        self._window.append(0 if keep else 1)
        if len(self._window) == self._window.maxlen:
            drop_rate = sum(self._window) / len(self._window)
            if drop_rate > self._drop_ratio_threshold:
                self._halted = True

        return ValidationResult(keep=keep, halted=self._halted)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_frame_validator.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_collection/frame_validator.py tests/unit/test_frame_validator.py
git commit -m "feat: add runtime frame validator with anomaly-rate halting"
```

---

## Task 7: Frame batching and Parquet shards (`batcher.py`)

**Files:**

- Create: `src/data_collection/batcher.py`
- Test: `tests/unit/test_batcher.py`

**Interfaces:**

- Produces:

  ```python
  @dataclass(frozen=True)
  class FrameRecord:
      image: np.ndarray  # grayscale HxW uint8
      video_id: str
      timestamp_s: float
      game: str

  class FrameBatcher:
      def __init__(self, batch_size: int = 500) -> None: ...
      def add(self, record: FrameRecord) -> list[FrameRecord] | None: ...
      def flush(self) -> list[FrameRecord] | None: ...

  def batch_to_parquet(batch: list[FrameRecord], path: str | Path) -> None: ...
  ```

- Consumes: nothing from earlier tasks — `FrameRecord.image` is a plain grayscale numpy array, produced by `extract.stream_frames`/`frame_validator`/`dedup` in Task 11's orchestration.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_batcher.py`:

```python
from pathlib import Path

import datasets
import numpy as np

from data_collection.batcher import FrameBatcher, FrameRecord, batch_to_parquet


def _record(i: int) -> FrameRecord:
    return FrameRecord(
        image=np.full((144, 160), i % 256, dtype=np.uint8),
        video_id="abc123",
        timestamp_s=float(i),
        game="red",
    )


def test_add_returns_none_until_batch_size_reached() -> None:
    batcher = FrameBatcher(batch_size=3)

    assert batcher.add(_record(0)) is None
    assert batcher.add(_record(1)) is None
    full_batch = batcher.add(_record(2))

    assert full_batch is not None
    assert len(full_batch) == 3


def test_batcher_resets_after_emitting_a_full_batch() -> None:
    batcher = FrameBatcher(batch_size=2)
    batcher.add(_record(0))
    batcher.add(_record(1))  # emits and resets

    assert batcher.add(_record(2)) is None


def test_flush_returns_partial_batch() -> None:
    batcher = FrameBatcher(batch_size=10)
    batcher.add(_record(0))
    batcher.add(_record(1))

    partial = batcher.flush()

    assert partial is not None
    assert len(partial) == 2


def test_flush_returns_none_when_empty() -> None:
    batcher = FrameBatcher(batch_size=10)

    assert batcher.flush() is None


def test_batch_to_parquet_round_trips(tmp_path: Path) -> None:
    records = [_record(i) for i in range(5)]
    path = tmp_path / "shard.parquet"

    batch_to_parquet(records, path)
    reloaded = datasets.Dataset.from_parquet(str(path))

    assert len(reloaded) == 5
    assert reloaded.column_names == ["image", "video_id", "timestamp_s", "game"]
    assert reloaded[0]["video_id"] == "abc123"
    assert reloaded[2]["timestamp_s"] == 2.0
    assert reloaded[0]["image"].size == (160, 144)  # PIL Image (width, height)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_batcher.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.batcher'`.

- [ ] **Step 3: Implement `batcher.py`**

```python
"""Accumulate accepted frames and write them as Parquet shards with the
`datasets` Image feature."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FrameRecord:
    image: np.ndarray
    video_id: str
    timestamp_s: float
    game: str


class FrameBatcher:
    def __init__(self, batch_size: int = 500) -> None:
        self._batch_size = batch_size
        self._buffer: list[FrameRecord] = []

    def add(self, record: FrameRecord) -> list[FrameRecord] | None:
        self._buffer.append(record)
        if len(self._buffer) >= self._batch_size:
            return self.flush()
        return None

    def flush(self) -> list[FrameRecord] | None:
        if not self._buffer:
            return None
        batch, self._buffer = self._buffer, []
        return batch


def batch_to_parquet(batch: list[FrameRecord], path: str | Path) -> None:
    rows = {
        "image": [Image.fromarray(r.image, mode="L") for r in batch],
        "video_id": [r.video_id for r in batch],
        "timestamp_s": [r.timestamp_s for r in batch],
        "game": [r.game for r in batch],
    }
    dataset = datasets.Dataset.from_dict(rows)
    dataset = dataset.cast_column("image", datasets.Image())
    dataset.to_parquet(str(path))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_batcher.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_collection/batcher.py tests/unit/test_batcher.py
git commit -m "feat: add frame batching and Parquet shard writer"
```

---

## Task 8: HF Hub upload and manifest (`hf_uploader.py`)

**Files:**

- Create: `src/data_collection/hf_uploader.py`
- Test: `tests/unit/test_hf_uploader.py`

**Interfaces:**

- Produces:

  ```python
  class Manifest:
      def __init__(self, completed: set[str] | None = None, failed: dict[str, str] | None = None) -> None: ...
      def is_complete(self, video_id: str) -> bool: ...
      def mark_complete(self, video_id: str) -> None: ...
      def mark_failed(self, video_id: str, reason: str) -> None: ...
      def to_json(self) -> str: ...
      @classmethod
      def from_json(cls, data: str) -> "Manifest": ...

  class HfUploader:
      def __init__(self, client: HfClient, repo_id: str) -> None: ...
      def upload_shard(self, local_path: str | Path, video_id: str, shard_index: int) -> str: ...
      def upload_preview(self, local_path: str | Path, video_id: str, shard_index: int) -> str: ...
      def load_manifest(self) -> Manifest: ...
      def save_manifest(self, manifest: Manifest) -> None: ...
  ```

  `HfClient` is a duck-typed protocol with two methods: `upload_bytes(data: bytes, path_in_repo: str) -> None` and `download_bytes(path_in_repo: str) -> bytes | None` (`None` means "not found yet", used for a manifest that doesn't exist on first run). Task 12 provides the real adapter over `huggingface_hub.HfApi`.

- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_hf_uploader.py`:

```python
from pathlib import Path

import pytest

from data_collection.hf_uploader import HfUploader, Manifest


class FakeHfClient:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.uploads[path_in_repo] = data

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.uploads.get(path_in_repo)


def test_manifest_starts_empty() -> None:
    manifest = Manifest()
    assert manifest.is_complete("abc123") is False


def test_manifest_mark_complete() -> None:
    manifest = Manifest()
    manifest.mark_complete("abc123")
    assert manifest.is_complete("abc123") is True


def test_manifest_json_round_trip() -> None:
    manifest = Manifest()
    manifest.mark_complete("abc123")
    manifest.mark_failed("def456", "ffmpeg crashed")

    restored = Manifest.from_json(manifest.to_json())

    assert restored.is_complete("abc123") is True
    assert restored.is_complete("def456") is False


def test_upload_shard_writes_to_expected_path(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.parquet"
    shard_path.write_bytes(b"fake-parquet-bytes")
    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")

    path_in_repo = uploader.upload_shard(shard_path, video_id="abc123", shard_index=3)

    assert path_in_repo == "shards/abc123/00003.parquet"
    assert client.uploads["shards/abc123/00003.parquet"] == b"fake-parquet-bytes"


def test_upload_preview_writes_to_expected_path(tmp_path: Path) -> None:
    preview_path = tmp_path / "contact_sheet.png"
    preview_path.write_bytes(b"fake-png-bytes")
    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")

    path_in_repo = uploader.upload_preview(preview_path, video_id="abc123", shard_index=3)

    assert path_in_repo == "previews/abc123/00003.png"
    assert client.uploads["previews/abc123/00003.png"] == b"fake-png-bytes"


def test_load_manifest_returns_empty_when_not_yet_uploaded() -> None:
    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")

    manifest = uploader.load_manifest()

    assert manifest.is_complete("anything") is False


def test_save_then_load_manifest_round_trips() -> None:
    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    manifest = Manifest()
    manifest.mark_complete("abc123")

    uploader.save_manifest(manifest)
    reloaded = uploader.load_manifest()

    assert reloaded.is_complete("abc123") is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_hf_uploader.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.hf_uploader'`.

- [ ] **Step 3: Implement `hf_uploader.py`**

```python
"""Push frame shards to the HF dataset repo and track per-video progress
in a manifest.json stored in that same repo, for crash resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class HfClient(Protocol):
    def upload_bytes(self, data: bytes, path_in_repo: str) -> None: ...
    def download_bytes(self, path_in_repo: str) -> bytes | None: ...


_MANIFEST_PATH = "manifest.json"


class Manifest:
    def __init__(
        self, completed: set[str] | None = None, failed: dict[str, str] | None = None
    ) -> None:
        self._completed = set(completed or set())
        self._failed = dict(failed or {})

    def is_complete(self, video_id: str) -> bool:
        return video_id in self._completed

    def mark_complete(self, video_id: str) -> None:
        self._completed.add(video_id)
        self._failed.pop(video_id, None)

    def mark_failed(self, video_id: str, reason: str) -> None:
        self._failed[video_id] = reason

    def to_json(self) -> str:
        return json.dumps({"completed": sorted(self._completed), "failed": self._failed})

    @classmethod
    def from_json(cls, data: str) -> "Manifest":
        parsed = json.loads(data)
        return cls(completed=set(parsed.get("completed", [])), failed=parsed.get("failed", {}))


class HfUploader:
    def __init__(self, client: HfClient, repo_id: str) -> None:
        self._client = client
        self._repo_id = repo_id

    def upload_shard(self, local_path: str | Path, video_id: str, shard_index: int) -> str:
        path_in_repo = f"shards/{video_id}/{shard_index:05d}.parquet"
        data = Path(local_path).read_bytes()
        self._client.upload_bytes(data, path_in_repo)
        return path_in_repo

    def upload_preview(self, local_path: str | Path, video_id: str, shard_index: int) -> str:
        path_in_repo = f"previews/{video_id}/{shard_index:05d}.png"
        data = Path(local_path).read_bytes()
        self._client.upload_bytes(data, path_in_repo)
        return path_in_repo

    def load_manifest(self) -> Manifest:
        data = self._client.download_bytes(_MANIFEST_PATH)
        if data is None:
            return Manifest()
        return Manifest.from_json(data.decode("utf-8"))

    def save_manifest(self, manifest: Manifest) -> None:
        self._client.upload_bytes(manifest.to_json().encode("utf-8"), _MANIFEST_PATH)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_hf_uploader.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_collection/hf_uploader.py tests/unit/test_hf_uploader.py
git commit -m "feat: add HF shard upload and resumable manifest"
```

---

## Task 9: Observability — structured logging, Trackio, contact sheets (`observability.py`)

**Files:**

- Create: `src/data_collection/observability.py`
- Test: `tests/unit/test_observability.py`

**Interfaces:**

- Produces:

  ```python
  def configure_logging(stream: TextIO | None = None) -> logging.Logger: ...  # logger name "data_collection", one JSON object per line

  class TrackioRun:
      def __init__(self, trackio_module, project: str, name: str) -> None: ...
      def log(self, metrics: dict) -> None: ...
      def finish(self) -> None: ...

  def build_contact_sheet(frames: list[np.ndarray], cols: int = 8) -> np.ndarray: ...
  ```

- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_observability.py`:

```python
import io
import json
import logging

import numpy as np

from data_collection.observability import TrackioRun, build_contact_sheet, configure_logging


def test_configure_logging_emits_one_json_object_per_line() -> None:
    stream = io.StringIO()
    logger = configure_logging(stream=stream)

    logger.info("frames_kept", extra={"video_id": "abc123", "count": 42})

    stream.seek(0)
    lines = [line for line in stream.read().splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["message"] == "frames_kept"
    assert payload["video_id"] == "abc123"
    assert payload["count"] == 42
    assert payload["level"] == "INFO"


def test_configure_logging_returns_same_named_logger_on_repeat_calls() -> None:
    logger_a = configure_logging(stream=io.StringIO())
    logger_b = configure_logging(stream=io.StringIO())
    assert logger_a.name == logger_b.name == "data_collection"


class FakeTrackioModule:
    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.log_calls: list[dict] = []
        self.finished = False

    def init(self, project: str, name: str) -> None:
        self.init_calls.append({"project": project, "name": name})

    def log(self, metrics: dict) -> None:
        self.log_calls.append(metrics)

    def finish(self) -> None:
        self.finished = True


def test_trackio_run_forwards_calls() -> None:
    fake = FakeTrackioModule()
    run = TrackioRun(fake, project="pokemon-data-collection", name="run-1")

    run.log({"frames_per_sec": 12.5})
    run.finish()

    assert fake.init_calls == [{"project": "pokemon-data-collection", "name": "run-1"}]
    assert fake.log_calls == [{"frames_per_sec": 12.5}]
    assert fake.finished is True


def test_build_contact_sheet_grid_dimensions() -> None:
    frames = [np.full((144, 160), i, dtype=np.uint8) for i in range(10)]

    sheet = build_contact_sheet(frames, cols=4)

    # 10 frames at 4 cols -> 3 rows (ceil(10/4)), each cell 144x160.
    assert sheet.shape == (144 * 3, 160 * 4)


def test_build_contact_sheet_empty_input_returns_empty_array() -> None:
    sheet = build_contact_sheet([], cols=4)
    assert sheet.size == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_observability.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.observability'`.

- [ ] **Step 3: Implement `observability.py`**

```python
"""Structured logging, a thin Trackio wrapper, and contact-sheet previews."""

from __future__ import annotations

import json
import logging
from typing import TextIO

import numpy as np


class _JsonFormatter(logging.Formatter):
    _RESERVED = set(logging.LogRecord(
        "", 0, "", 0, "", (), None
    ).__dict__.keys()) | {"message"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value
        return json.dumps(payload)


def configure_logging(stream: TextIO | None = None) -> logging.Logger:
    logger = logging.getLogger("data_collection")
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class TrackioRun:
    def __init__(self, trackio_module, project: str, name: str) -> None:
        self._trackio = trackio_module
        self._trackio.init(project=project, name=name)

    def log(self, metrics: dict) -> None:
        self._trackio.log(metrics)

    def finish(self) -> None:
        self._trackio.finish()


def build_contact_sheet(frames: list[np.ndarray], cols: int = 8) -> np.ndarray:
    if not frames:
        return np.empty((0, 0), dtype=np.uint8)

    height, width = frames[0].shape
    rows = -(-len(frames) // cols)  # ceil division
    sheet = np.zeros((height * rows, width * cols), dtype=np.uint8)

    for i, frame in enumerate(frames):
        row, col = divmod(i, cols)
        sheet[row * height : (row + 1) * height, col * width : (col + 1) * width] = frame

    return sheet
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_observability.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_collection/observability.py tests/unit/test_observability.py
git commit -m "feat: add structured logging, Trackio wrapper, contact sheets"
```

---

## Task 10: Curation tool (`curation.py`)

**Files:**

- Create: `src/data_collection/curation.py`
- Test: `tests/unit/test_curation.py`

**Interfaces:**

- Consumes: `data_collection.matching.match_crop`, `MatchResult` (Task 3).
- Produces:

  ```python
  def render_preview(frame_gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray: ...
  def propose_crop_box(frame_gray: np.ndarray, bank_templates: dict[str, np.ndarray], min_confidence: float = 0.7) -> tuple[int, int, int, int] | None: ...
  def run_curation(video_url: str, bank_dir: Path, approved_dir: Path, registry_path: Path, game: str, smoke_test_width: int = 1920, smoke_test_height: int = 1080, input_func=input) -> None: ...  # interactive orchestrator, not unit tested here
  ```

  `run_curation` is exercised manually per the spec's testing strategy ("exercised interactively per video; it is Phase A's product, not a separate test suite") and wired into the CLI in Task 12.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_curation.py`:

```python
import numpy as np

from data_collection.curation import propose_crop_box, render_preview


def test_render_preview_draws_a_visible_box() -> None:
    frame = np.zeros((144, 160), dtype=np.uint8)

    preview = render_preview(frame, x=10, y=10, w=50, h=40)

    assert preview.shape == frame.shape
    # The rectangle outline should have introduced non-zero pixels
    # that weren't in the all-black source frame.
    assert preview.max() > 0
    # The original frame must not be mutated in place.
    assert frame.max() == 0


def test_propose_crop_box_finds_best_matching_template() -> None:
    rng = np.random.default_rng(seed=0)
    frame = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    good_template = rng.integers(0, 255, size=(144, 160), dtype=np.uint8)
    frame[30:174, 60:220] = good_template
    bad_template = np.random.default_rng(seed=7).integers(0, 255, size=(144, 160), dtype=np.uint8)

    box = propose_crop_box(
        frame,
        bank_templates={"bad": bad_template, "good": good_template},
        min_confidence=0.7,
    )

    assert box == (60, 30, 160, 144)


def test_propose_crop_box_returns_none_when_no_template_confident() -> None:
    rng = np.random.default_rng(seed=0)
    frame = rng.integers(0, 255, size=(200, 300), dtype=np.uint8)
    unrelated_template = np.random.default_rng(seed=9).integers(
        0, 255, size=(144, 160), dtype=np.uint8
    )

    box = propose_crop_box(
        frame, bank_templates={"unrelated": unrelated_template}, min_confidence=0.7
    )

    assert box is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_curation.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.curation'`.

- [ ] **Step 3: Implement `curation.py`**

```python
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
    smoke_test_width: int = 1920,
    smoke_test_height: int = 1080,
    input_func=input,
) -> None:
    stream_url = extract.get_stream_url(video_url)
    frame = _grab_smoke_test_frame(stream_url, smoke_test_width, smoke_test_height)

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_curation.py -v
```

Expected: both tests PASS (`run_curation` is not exercised by these tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_collection/curation.py tests/unit/test_curation.py
git commit -m "feat: add curation preview rendering and crop-box proposal"
```

---

## Task 11: Pipeline orchestration (`pipeline.py`)

**Files:**

- Create: `src/data_collection/pipeline.py`
- Test: `tests/unit/test_pipeline.py`

**Interfaces:**

- Consumes:
  - `data_collection.registry.VideoSource` (Task 2)
  - `data_collection.matching.load_template_gray` (Task 3)
  - `data_collection.dedup.PerceptualHashDeduper` (Task 4)
  - `data_collection.frame_validator.FrameValidator`, `ValidationResult` (Task 6)
  - `data_collection.batcher.FrameBatcher`, `FrameRecord`, `batch_to_parquet` (Task 7)
  - `data_collection.hf_uploader.HfUploader`, `Manifest` (Task 8)
  - `data_collection.observability.build_contact_sheet` (Task 9) — called once per
    flushed batch so a spot-checkable preview PNG is pushed to the repo
    alongside every shard, per spec: Observability
- Produces:

  ```python
  @dataclass
  class PipelineDeps:
      frame_source: Callable[[VideoSource], Iterator[np.ndarray]]
      uploader: HfUploader
      logger: logging.Logger
      trackio_run: TrackioRunLike | None = None
      batch_size: int = 500
      max_retries: int = 3
      sleep_func: Callable[[float], None] = time.sleep

  def retry_with_backoff(func: Callable[[], None], max_retries: int, base_delay: float, sleep_func: Callable[[float], None]) -> None: ...
  def run_pipeline(registry: list[VideoSource], deps: PipelineDeps) -> None: ...
  ```

  `frame_source` abstracts "extract + apply crop for this video" so tests never touch real `ffmpeg`/`yt-dlp`; the real one built from `extract.stream_frames` is wired in Task 12.

Scope note on the spec's Observability metrics list (frames/sec, dedup-rejection rate, anomaly-drop rate, cumulative frames uploaded): this task logs `dedup_rejection_rate` and `anomaly_drop_rate` exactly, plus `sampled`/`kept` counts, once per completed video (not a live per-second stream). A true live frames/sec gauge and a running cross-video cumulative counter add real complexity (timers, shared mutable state across videos) for a benefit Trackio's own dashboard already provides — it can plot `kept` over time per video and sum it visually. If live/cumulative counters turn out to matter in practice once this runs for real, add them as a follow-up rather than speculatively now.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_pipeline.py`:

```python
import io
import logging

import numpy as np
import pytest

from data_collection.hf_uploader import HfUploader, Manifest
from data_collection.observability import configure_logging
from data_collection.pipeline import PipelineDeps, retry_with_backoff, run_pipeline
from data_collection.registry import VideoSource


class FakeHfClient:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.uploads[path_in_repo] = data

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.uploads.get(path_in_repo)


def _source(video_id: str, reference_patch_path: str) -> VideoSource:
    return VideoSource(
        video_id=video_id,
        url=f"https://youtube.com/watch?v={video_id}",
        game="red",
        crop_x=0,
        crop_y=0,
        crop_w=4,
        crop_h=3,
        reference_patch_path=reference_patch_path,
        match_confidence_baseline=1.0,
    )


def _reference_frame() -> np.ndarray:
    return np.full((3, 4), 100, dtype=np.uint8)


def test_run_pipeline_uploads_shards_and_marks_manifest_complete(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))

    def frame_source(video_source: VideoSource):
        for _ in range(3):
            yield _reference_frame()

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        logger=configure_logging(stream=io.StringIO()),
        batch_size=10,
    )

    run_pipeline([source], deps)

    manifest = uploader.load_manifest()
    assert manifest.is_complete("abc123") is True
    assert any(path.startswith("shards/abc123/") for path in client.uploads)


def test_run_pipeline_uploads_a_contact_sheet_preview_per_batch(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))

    def frame_source(video_source: VideoSource):
        for _ in range(3):
            yield _reference_frame()

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        logger=configure_logging(stream=io.StringIO()),
        batch_size=10,
    )

    run_pipeline([source], deps)

    assert any(path.startswith("previews/abc123/") for path in client.uploads)


def test_run_pipeline_skips_already_completed_videos(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))

    calls = []

    def frame_source(video_source: VideoSource):
        calls.append(video_source.video_id)
        return iter([])

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    manifest = Manifest()
    manifest.mark_complete("abc123")
    uploader.save_manifest(manifest)

    deps = PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        logger=configure_logging(stream=io.StringIO()),
    )

    run_pipeline([source], deps)

    assert calls == []


def test_run_pipeline_marks_failed_after_exhausting_retries(tmp_path) -> None:
    reference_path = tmp_path / "ref.png"
    import cv2

    cv2.imwrite(str(reference_path), _reference_frame())
    source = _source("abc123", str(reference_path))

    def failing_frame_source(video_source: VideoSource):
        raise RuntimeError("network blip")
        yield  # pragma: no cover - unreachable, makes this a generator

    client = FakeHfClient()
    uploader = HfUploader(client, repo_id="me/pokemon-frames")
    sleeps: list[float] = []
    deps = PipelineDeps(
        frame_source=failing_frame_source,
        uploader=uploader,
        logger=configure_logging(stream=io.StringIO()),
        max_retries=2,
        sleep_func=sleeps.append,
    )

    run_pipeline([source], deps)

    manifest = uploader.load_manifest()
    assert manifest.is_complete("abc123") is False
    assert len(sleeps) == 1  # max_retries=2 total attempts -> 1 sleep between them


def test_retry_with_backoff_succeeds_after_transient_failures() -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def flaky() -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")

    retry_with_backoff(flaky, max_retries=3, base_delay=1.0, sleep_func=sleeps.append)

    assert attempts["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_retry_with_backoff_raises_after_exhausting_retries() -> None:
    def always_fails() -> None:
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        retry_with_backoff(always_fails, max_retries=2, base_delay=1.0, sleep_func=lambda _: None)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.pipeline'`.

- [ ] **Step 3: Implement `pipeline.py`**

```python
"""Orchestrate Phase B: per approved video, extract -> validate -> dedup ->
batch -> upload, with resume-by-manifest and bounded retry on failure."""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

from data_collection.batcher import FrameBatcher, FrameRecord, batch_to_parquet
from data_collection.dedup import PerceptualHashDeduper
from data_collection.frame_validator import FrameValidator
from data_collection.hf_uploader import HfUploader
from data_collection.matching import load_template_gray
from data_collection.observability import build_contact_sheet
from data_collection.registry import VideoSource
import cv2


class TrackioRunLike(Protocol):
    def log(self, metrics: dict) -> None: ...
    def finish(self) -> None: ...


def retry_with_backoff(
    func: Callable[[], None],
    max_retries: int,
    base_delay: float,
    sleep_func: Callable[[float], None],
) -> None:
    attempt = 0
    while True:
        try:
            func()
            return
        except Exception:
            attempt += 1
            if attempt >= max_retries:
                raise
            sleep_func(base_delay * (2 ** (attempt - 1)))


@dataclass
class PipelineDeps:
    frame_source: Callable[[VideoSource], Iterator[np.ndarray]]
    uploader: HfUploader
    logger: logging.Logger
    trackio_run: TrackioRunLike | None = None
    batch_size: int = 500
    max_retries: int = 3
    sleep_func: Callable[[float], None] = field(default=time.sleep)


def _process_video(video: VideoSource, deps: PipelineDeps) -> None:
    reference_patch = load_template_gray(video.reference_patch_path)
    validator = FrameValidator(reference_patch, baseline_score=video.match_confidence_baseline)
    deduper = PerceptualHashDeduper()
    batcher = FrameBatcher(batch_size=deps.batch_size)

    sampled = kept = dropped_dedup = dropped_anomaly = shard_index = 0

    for frame in deps.frame_source(video):
        sampled += 1
        result = validator.validate(frame)
        if not result.keep:
            dropped_anomaly += 1
            continue
        if result.halted:
            deps.logger.warning(
                "video_halted_on_anomaly_rate", extra={"video_id": video.video_id}
            )
            break
        if deduper.is_duplicate(frame):
            dropped_dedup += 1
            continue

        kept += 1
        record = FrameRecord(
            image=frame, video_id=video.video_id, timestamp_s=sampled / 2.0, game=video.game
        )
        full_batch = batcher.add(record)
        if full_batch is not None:
            shard_index = _flush_batch(full_batch, video.video_id, shard_index, deps)

    trailing = batcher.flush()
    if trailing is not None:
        _flush_batch(trailing, video.video_id, shard_index, deps)

    dedup_rejection_rate = dropped_dedup / sampled if sampled else 0.0
    anomaly_drop_rate = dropped_anomaly / sampled if sampled else 0.0
    metrics = {
        "video_id": video.video_id,
        "sampled": sampled,
        "kept": kept,
        "dropped_dedup": dropped_dedup,
        "dropped_anomaly": dropped_anomaly,
        "dedup_rejection_rate": dedup_rejection_rate,
        "anomaly_drop_rate": anomaly_drop_rate,
    }
    deps.logger.info("video_complete", extra=metrics)
    if deps.trackio_run is not None:
        deps.trackio_run.log(metrics)


def _flush_batch(
    batch: list[FrameRecord], video_id: str, shard_index: int, deps: PipelineDeps
) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        shard_path = Path(tmp_dir) / "shard.parquet"
        batch_to_parquet(batch, shard_path)
        deps.uploader.upload_shard(shard_path, video_id, shard_index)

        contact_sheet = build_contact_sheet([record.image for record in batch])
        preview_path = Path(tmp_dir) / "contact_sheet.png"
        cv2.imwrite(str(preview_path), contact_sheet)
        deps.uploader.upload_preview(preview_path, video_id, shard_index)
    return shard_index + 1


def run_pipeline(registry: list[VideoSource], deps: PipelineDeps) -> None:
    manifest = deps.uploader.load_manifest()

    for video in registry:
        if manifest.is_complete(video.video_id):
            continue

        try:
            retry_with_backoff(
                lambda v=video: _process_video(v, deps),
                max_retries=deps.max_retries,
                base_delay=1.0,
                sleep_func=deps.sleep_func,
            )
        except Exception as exc:
            deps.logger.error(
                "video_failed", extra={"video_id": video.video_id, "reason": str(exc)}
            )
            manifest.mark_failed(video.video_id, str(exc))
            deps.uploader.save_manifest(manifest)
            continue

        manifest.mark_complete(video.video_id)
        deps.uploader.save_manifest(manifest)

    if deps.trackio_run is not None:
        deps.trackio_run.finish()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_pipeline.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_collection/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: add Phase B pipeline orchestration with resume and retry"
```

---

## Task 12: CLI entry points (`cli.py`)

**Files:**

- Create: `src/data_collection/cli.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**

- Consumes: everything from Tasks 2–11, including `data_collection.curation.run_curation` (Task 10).
- Produces: `main()` — the Click group registered as the `data-collection` console script (`pyproject.toml`, Task 1).

- [ ] **Step 1: Write failing CLI tests**

Create `tests/unit/test_cli.py`:

```python
from click.testing import CliRunner

from data_collection.cli import main


def test_cli_help_lists_curate_and_run_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "curate" in result.output
    assert "run" in result.output


def test_curate_command_requires_url_argument() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["curate"])

    assert result.exit_code != 0


def test_run_command_requires_repo_id_option() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["run"])

    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.cli'`.

- [ ] **Step 3: Implement `cli.py`**

```python
"""Console entry points: `data-collection curate <url>` (Phase A) and
`data-collection run --repo-id <repo>` (Phase B)."""

from __future__ import annotations

from pathlib import Path

import click
import trackio
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError

from data_collection import curation, extract, pipeline
from data_collection.hf_uploader import HfClient, HfUploader
from data_collection.observability import TrackioRun, configure_logging
from data_collection.registry import load_registry

_DEFAULT_REGISTRY = Path("configs/video_sources.yaml")
_DEFAULT_BANK_DIR = Path("configs/templates/bank")
_DEFAULT_APPROVED_DIR = Path("configs/templates/approved")


class RealHfClient:
    """Adapts huggingface_hub.HfApi to the HfClient protocol hf_uploader expects."""

    def __init__(self, api: HfApi, repo_id: str) -> None:
        self._api = api
        self._repo_id = repo_id

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self._api.upload_file(
            path_or_fileobj=data,
            path_in_repo=path_in_repo,
            repo_id=self._repo_id,
            repo_type="dataset",
        )

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        try:
            local_path = self._api.hf_hub_download(
                repo_id=self._repo_id, filename=path_in_repo, repo_type="dataset"
            )
        except EntryNotFoundError:
            return None
        return Path(local_path).read_bytes()


@click.group()
def main() -> None:
    """Pokemon Red/Blue data collection pipeline."""


@main.command()
@click.argument("url")
@click.option("--game", type=click.Choice(["red", "blue"]), required=True)
@click.option("--registry", type=click.Path(path_type=Path), default=_DEFAULT_REGISTRY)
@click.option("--bank-dir", type=click.Path(path_type=Path), default=_DEFAULT_BANK_DIR)
@click.option("--approved-dir", type=click.Path(path_type=Path), default=_DEFAULT_APPROVED_DIR)
def curate(url: str, game: str, registry: Path, bank_dir: Path, approved_dir: Path) -> None:
    """Phase A: interactively review and approve a candidate video."""
    curation.run_curation(
        video_url=url,
        bank_dir=bank_dir,
        approved_dir=approved_dir,
        registry_path=registry,
        game=game,
    )


@main.command()
@click.option("--repo-id", required=True, help="HF dataset repo, e.g. me/pokemon-frames")
@click.option("--registry", type=click.Path(path_type=Path), default=_DEFAULT_REGISTRY)
@click.option("--batch-size", type=int, default=500)
def run(repo_id: str, registry: Path, batch_size: int) -> None:
    """Phase B: unattended extraction across all approved, incomplete videos."""
    sources = load_registry(registry)
    logger = configure_logging()

    client: HfClient = RealHfClient(HfApi(), repo_id)
    uploader = HfUploader(client, repo_id)
    trackio_run = TrackioRun(trackio, project="pokemon-data-collection", name=repo_id)

    def frame_source(video_source):
        return extract.stream_frames(
            extract.get_stream_url(video_source.url),
            crop_x=video_source.crop_x,
            crop_y=video_source.crop_y,
            crop_w=video_source.crop_w,
            crop_h=video_source.crop_h,
        )

    deps = pipeline.PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        logger=logger,
        trackio_run=trackio_run,
        batch_size=batch_size,
    )
    pipeline.run_pipeline(sources, deps)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_cli.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run the full unit test suite**

```bash
uv run pytest -v
```

Expected: every unit test across Tasks 2-12 PASSes (the `slow`-marked integration test from Task 13 does not exist yet).

- [ ] **Step 6: Commit**

```bash
git add src/data_collection/cli.py tests/unit/test_cli.py
git commit -m "feat: add CLI entry points for curation and pipeline"
```

---

## Task 13: Integration smoke test (opt-in, slow)

**Files:**

- Create: `tests/integration/test_extraction_smoke.py`

**Interfaces:**

- Consumes: `data_collection.extract.stream_frames`, `data_collection.frame_validator.FrameValidator`, `data_collection.dedup.PerceptualHashDeduper`, `data_collection.batcher.FrameBatcher`, `batch_to_parquet` — the real Phase B chain, minus the HF upload step (writes to a local scratch Parquet file instead, per spec: Testing strategy).

This test requires a real short video file and real `ffmpeg`/`yt-dlp` and is skipped unless both the `slow` marker is explicitly selected and a test clip path is provided — it will not run in CI or on a machine without `ffmpeg` installed.

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_extraction_smoke.py`:

```python
"""Opt-in integration test for the real Phase B chain.

Run explicitly with a short (~30s), locally-available test clip you have the
rights to use:

    POKEMON_RL_TEST_CLIP=/path/to/clip.mp4 uv run pytest -m slow tests/integration/test_extraction_smoke.py -v

Skipped by default (see the `addopts = -m "not slow"` in pyproject.toml) and
skipped automatically if POKEMON_RL_TEST_CLIP is unset, so this never fails
CI or a fresh checkout.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import pytest

from data_collection import extract
from data_collection.batcher import FrameBatcher, FrameRecord, batch_to_parquet
from data_collection.dedup import PerceptualHashDeduper
from data_collection.frame_validator import FrameValidator
from data_collection.matching import load_template_gray

pytestmark = pytest.mark.slow

_CLIP_ENV_VAR = "POKEMON_RL_TEST_CLIP"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
@pytest.mark.skipif(_CLIP_ENV_VAR not in os.environ, reason=f"set {_CLIP_ENV_VAR} to run this")
def test_real_extraction_chain_produces_a_local_parquet_shard(tmp_path: Path) -> None:
    clip_path = os.environ[_CLIP_ENV_VAR]

    # Grab one full frame first to let a human-equivalent crop box be chosen;
    # here we just crop the top-left 160x144 region as a deterministic smoke test.
    crop_x, crop_y, crop_w, crop_h = 0, 0, 160, 144

    frames = list(
        extract.stream_frames(clip_path, crop_x=crop_x, crop_y=crop_y, crop_w=crop_w, crop_h=crop_h, fps=2)
    )
    assert len(frames) > 0

    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), frames[0])
    reference = load_template_gray(reference_path)

    validator = FrameValidator(reference, baseline_score=1.0)
    deduper = PerceptualHashDeduper()
    batcher = FrameBatcher(batch_size=1000)

    kept_count = 0
    for i, frame in enumerate(frames):
        result = validator.validate(frame)
        if not result.keep or deduper.is_duplicate(frame):
            continue
        kept_count += 1
        batcher.add(
            FrameRecord(image=frame, video_id="smoke-test", timestamp_s=i / 2.0, game="red")
        )

    batch = batcher.flush()
    assert batch is not None
    assert kept_count > 0

    shard_path = tmp_path / "shard.parquet"
    batch_to_parquet(batch, shard_path)

    assert shard_path.exists()
    assert shard_path.stat().st_size > 0
```

- [ ] **Step 2: Verify it's skipped by default**

```bash
uv run pytest tests/integration/ -v
```

Expected: the test is deselected/skipped (not run), because of the default `-m "not slow"` in `pyproject.toml`.

- [ ] **Step 3: Verify the full suite still passes with the new file present**

```bash
uv run pytest -v
```

Expected: all unit tests still PASS; the integration test is skipped, not failed.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_extraction_smoke.py
git commit -m "test: add opt-in integration smoke test for the real Phase B chain"
```

---

## Final verification

- [ ] Run `uv run pytest -v` — every unit test (Tasks 2–12) passes, the Task 13 integration test is skipped.
- [ ] Run `uv run data-collection --help` — shows `curate` and `run` subcommands.
- [ ] Confirm `configs/templates/bank/` is empty except its README, and `configs/video_sources.yaml` still has `videos: []` — nothing in this plan fabricates fake curation approvals; real curation happens by running `data-collection curate <url> --game <red|blue>` against the candidate list in the spec's appendix once reference templates are added to the bank.
