# Contrastive Augmentation Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the domain-adapted augmentation transforms, positive-pair construction, and a no-GPU visual validation tool for the contrastive-pretraining sub-project, per the approved design spec.

**Architecture:** A new `contrastive_pretrain` package (sibling to `data_collection` and `observability`) holds a single `augmentation.py` module: a frozen `AugmentationConfig` dataclass of tunable parameter ranges, and one `_resolve_*` (pure, bounds-testable) + `_apply_*` (pure, transform) pair per augmentation family, composed into `augment_view` and `make_pair`. Transforms operate on `torch.Tensor` frames (shape `(1, H, W)`, dtype `torch.uint8`) built on TorchVision's `transforms.v2.functional` and `torchvision.io` primitives — not hand-rolled numpy/cv2 — so this module is directly reusable by the deferred training-loop's `Dataset`/`DataLoader` without a rewrite, and so blur/resize/JPEG-artifact logic reuses TorchVision's tested implementations rather than reimplementing them. `observability/visualization.py` stays numpy-based (its existing, already-tested convention, shared with `data_collection`'s contact sheets) and gains a small extension to build (original, view A, view B) contact sheets; the CLI is the boundary that converts torch tensors to numpy right before building the sheet. A `contrastive-pretrain preview` CLI command ties it together for human spot-checking against sample frames, with no HF/GPU dependency (CPU-only torch is sufficient for this validation tool).

**Tech Stack:** Python 3.12, PyTorch (`torch`), TorchVision (`torchvision`), Pillow, click. `torch`/`torchvision` are new project dependencies — justified because the deferred encoder/training-loop spec will need them regardless (frozen CNN, projection head, InfoNCE/BYOL loss all require an autograd framework), so building augmentation on tested TorchVision primitives now avoids a rewrite later and matches how SimCLR/BYOL reference implementations actually do this.

**Spec:** `docs/superpowers/specs/2026-08-23-contrastive-augmentation-policy-design.md`

## Global Constraints

- Package management: `uv` only (`uv add`, `uv run`) — no bare `pip`/`venv`.
- TDD: write the failing test first for every function below, run it, confirm the failure, then implement.
- No comments in implementation code unless a line encodes a non-obvious constraint; none are expected in this plan's code.
- Every stochastic parameter is resolved via an injected `torch.Generator` (never global `torch.manual_seed` state), so tests are deterministic and reproducible via seeding.
- Follow existing repo conventions: `from __future__ import annotations` at the top of every module, one-line module docstrings, `@dataclass(frozen=True)` for config objects (see `data_collection/batcher.py`'s `FrameRecord`).
- All frames are single-channel grayscale `torch.Tensor`, shape `(1, height, width)`, dtype `torch.uint8` — the standard TorchVision channel-first convention. `observability/visualization.py` continues to operate on numpy arrays (its existing convention); conversion between the two happens only at the CLI boundary (Task 9).

---

### Task 1: Package scaffold, dependencies, and `AugmentationConfig`

**Files:**

- Modify: `pyproject.toml`
- Create: `src/contrastive_pretrain/__init__.py`
- Create: `src/contrastive_pretrain/augmentation.py`
- Test: `tests/unit/test_augmentation.py`

**Interfaces:**

- Produces: `AugmentationConfig` — a frozen dataclass with fields `max_translate_px: int`, `crop_min_area_fraction: float`, `brightness_range: float`, `contrast_range: float`, `noise_sigma_max: float`, `blur_sigma_max: float`, `blur_kernel_size: int`, `jpeg_quality_min: int`, `jpeg_quality_max: int`, each with a default matching the design spec's suggested magnitude. `brightness_range`/`contrast_range` are multiplicative factor ranges (a value of `0.15` means a factor sampled in `[0.85, 1.15]`), matching TorchVision's `adjust_brightness`/`adjust_contrast` factor convention used in Task 3.

- [ ] **Step 1: Add the new dependencies**

Run: `uv add torch torchvision`
Expected: `pyproject.toml`'s `[project] dependencies` list gains `torch` and `torchvision` entries, and `uv.lock` updates.

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_augmentation.py
from contrastive_pretrain.augmentation import AugmentationConfig


def test_augmentation_config_has_spec_defaults() -> None:
    config = AugmentationConfig()

    assert config.max_translate_px == 4
    assert config.crop_min_area_fraction == 0.90
    assert config.brightness_range == 0.15
    assert config.contrast_range == 0.15
    assert config.noise_sigma_max == 8.0
    assert config.blur_sigma_max == 0.8
    assert config.blur_kernel_size == 3
    assert config.jpeg_quality_min == 60
    assert config.jpeg_quality_max == 95
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contrastive_pretrain'`

- [ ] **Step 4: Create the package and config**

Create `src/contrastive_pretrain/__init__.py` as an empty file (matching
`data_collection/__init__.py` — no content, just marks the directory as a
package).

```python
# src/contrastive_pretrain/augmentation.py
"""Domain-adapted contrastive-pretraining augmentation transforms, built on
PyTorch/TorchVision primitives so this module is directly reusable by the
(deferred) training-loop's Dataset/DataLoader without a rewrite.

See docs/superpowers/specs/2026-08-23-contrastive-augmentation-policy-design.md
for the rationale behind each transform's inclusion and parameter range.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentationConfig:
    max_translate_px: int = 4
    crop_min_area_fraction: float = 0.90
    brightness_range: float = 0.15
    contrast_range: float = 0.15
    noise_sigma_max: float = 8.0
    blur_sigma_max: float = 0.8
    blur_kernel_size: int = 3
    jpeg_quality_min: int = 60
    jpeg_quality_max: int = 95
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/contrastive_pretrain/__init__.py src/contrastive_pretrain/augmentation.py tests/unit/test_augmentation.py
git commit -m "feat: scaffold contrastive_pretrain package with AugmentationConfig, add torch/torchvision"
```

---

### Task 2: Geometric transforms — translate and crop-resize

**Files:**

- Modify: `src/contrastive_pretrain/augmentation.py`
- Test: `tests/unit/test_augmentation.py`

**Interfaces:**

- Consumes: `AugmentationConfig` (Task 1).
- Produces: `random_translate(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor`, `random_crop_resize(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor`, plus internal `_resolve_translate_offset`, `_apply_translate`, `_resolve_crop_box`, `_apply_crop_resize` used directly by their own bounds tests.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_augmentation.py
import torch

from contrastive_pretrain.augmentation import (
    AugmentationConfig,
    _apply_crop_resize,
    _apply_translate,
    _resolve_crop_box,
    _resolve_translate_offset,
    random_crop_resize,
    random_translate,
)


def _marker_frame(size: tuple[int, int] = (144, 160), marker_at: tuple[int, int] = (72, 80)) -> torch.Tensor:
    frame = torch.zeros((1, *size), dtype=torch.uint8)
    frame[0, marker_at[0], marker_at[1]] = 255
    return frame


def test_resolve_translate_offset_stays_within_configured_bounds() -> None:
    config = AugmentationConfig(max_translate_px=4)
    rng = torch.Generator().manual_seed(0)

    for _ in range(1000):
        dy, dx = _resolve_translate_offset(config, rng)
        assert -4 <= dy <= 4
        assert -4 <= dx <= 4


def test_apply_translate_shifts_marker_by_exact_offset() -> None:
    frame = _marker_frame(marker_at=(72, 80))

    shifted = _apply_translate(frame, dy=3, dx=-2, max_px=4)

    ys, xs = torch.where(shifted[0] == 255)
    assert (int(ys[0]), int(xs[0])) == (75, 78)


def test_random_translate_preserves_shape_and_dtype() -> None:
    frame = _marker_frame()
    config = AugmentationConfig(max_translate_px=4)
    rng = torch.Generator().manual_seed(1)

    result = random_translate(frame, config, rng)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_resolve_crop_box_never_exceeds_frame_bounds() -> None:
    config = AugmentationConfig(crop_min_area_fraction=0.90)
    rng = torch.Generator().manual_seed(2)
    shape = (144, 160)

    for _ in range(1000):
        y, x, crop_h, crop_w = _resolve_crop_box(shape, config, rng)
        assert y >= 0
        assert y + crop_h <= shape[0]
        assert x >= 0
        assert x + crop_w <= shape[1]
        area_fraction = (crop_h * crop_w) / (shape[0] * shape[1])
        assert area_fraction >= config.crop_min_area_fraction - 0.02


def test_apply_crop_resize_returns_original_shape() -> None:
    frame = _marker_frame()

    result = _apply_crop_resize(frame, y=2, x=2, crop_h=140, crop_w=156)

    assert result.shape == frame.shape


def test_random_crop_resize_preserves_shape_and_dtype() -> None:
    frame = _marker_frame()
    config = AugmentationConfig()
    rng = torch.Generator().manual_seed(3)

    result = random_crop_resize(frame, config, rng)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: FAIL with `ImportError` (names not yet defined)

- [ ] **Step 3: Implement the transforms**

Add these imports at the top of `src/contrastive_pretrain/augmentation.py`,
directly below `from dataclasses import dataclass` (this task is the first
to need `torch`/`torchvision` — only import what a task actually uses,
never front-load imports a later task will consume, to avoid dead-import
lint failures on intermediate commits):

```python
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF
```

Then append the transforms:

```python
# append to src/contrastive_pretrain/augmentation.py
def _resolve_translate_offset(config: AugmentationConfig, rng: torch.Generator) -> tuple[int, int]:
    bound = config.max_translate_px
    dy = int(torch.randint(-bound, bound + 1, (1,), generator=rng).item())
    dx = int(torch.randint(-bound, bound + 1, (1,), generator=rng).item())
    return dy, dx


def _apply_translate(frame: torch.Tensor, dy: int, dx: int, max_px: int) -> torch.Tensor:
    padded = F.pad(frame, (max_px, max_px, max_px, max_px), mode="replicate")
    h, w = frame.shape[-2:]
    y0 = max_px - dy
    x0 = max_px - dx
    return padded[..., y0 : y0 + h, x0 : x0 + w]


def random_translate(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor:
    dy, dx = _resolve_translate_offset(config, rng)
    return _apply_translate(frame, dy, dx, config.max_translate_px)


def _resolve_crop_box(
    frame_shape: tuple[int, int], config: AugmentationConfig, rng: torch.Generator
) -> tuple[int, int, int, int]:
    h, w = frame_shape
    area_fraction = torch.empty(1).uniform_(config.crop_min_area_fraction, 1.0, generator=rng).item()
    scale = area_fraction**0.5
    crop_h = max(1, round(h * scale))
    crop_w = max(1, round(w * scale))
    max_y = h - crop_h
    max_x = w - crop_w
    y = int(torch.randint(0, max_y + 1, (1,), generator=rng).item()) if max_y > 0 else 0
    x = int(torch.randint(0, max_x + 1, (1,), generator=rng).item()) if max_x > 0 else 0
    return y, x, crop_h, crop_w


def _apply_crop_resize(frame: torch.Tensor, y: int, x: int, crop_h: int, crop_w: int) -> torch.Tensor:
    h, w = frame.shape[-2:]
    cropped = frame[..., y : y + crop_h, x : x + crop_w]
    return TF.resize(cropped, [h, w], antialias=True)


def random_crop_resize(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor:
    y, x, crop_h, crop_w = _resolve_crop_box(frame.shape[-2:], config, rng)
    return _apply_crop_resize(frame, y, x, crop_h, crop_w)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/augmentation.py tests/unit/test_augmentation.py
git commit -m "feat: add translate and crop-resize augmentation transforms"
```

---

### Task 3: Color transform — brightness/contrast jitter

**Files:**

- Modify: `src/contrastive_pretrain/augmentation.py`
- Test: `tests/unit/test_augmentation.py`

**Interfaces:**

- Consumes: `AugmentationConfig` (Task 1).
- Produces: `random_brightness_contrast(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor`, plus `_resolve_brightness_contrast`, `_apply_brightness_contrast`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_augmentation.py
from contrastive_pretrain.augmentation import (
    _apply_brightness_contrast,
    _resolve_brightness_contrast,
    random_brightness_contrast,
)


def test_resolve_brightness_contrast_stays_within_configured_range() -> None:
    config = AugmentationConfig(brightness_range=0.15, contrast_range=0.15)
    rng = torch.Generator().manual_seed(4)

    for _ in range(1000):
        brightness_factor, contrast_factor = _resolve_brightness_contrast(config, rng)
        assert 0.85 <= brightness_factor <= 1.15
        assert 0.85 <= contrast_factor <= 1.15


def test_apply_brightness_contrast_on_solid_frame_scales_by_brightness_factor() -> None:
    frame = torch.full((1, 144, 160), 100, dtype=torch.uint8)

    result = _apply_brightness_contrast(frame, brightness_factor=1.2, contrast_factor=1.0)

    # A uniform frame's own mean equals its value, so contrast (which blends
    # toward that mean) is a no-op here; only brightness scaling shows up.
    assert torch.all(result == 120)


def test_apply_brightness_contrast_clips_to_valid_pixel_range() -> None:
    frame = torch.full((1, 144, 160), 250, dtype=torch.uint8)

    result = _apply_brightness_contrast(frame, brightness_factor=1.15, contrast_factor=1.0)

    assert torch.all(result == 255)


def test_random_brightness_contrast_preserves_shape_and_dtype() -> None:
    frame = torch.full((1, 144, 160), 100, dtype=torch.uint8)
    config = AugmentationConfig()
    rng = torch.Generator().manual_seed(5)

    result = random_brightness_contrast(frame, config, rng)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the transform**

```python
# append to src/contrastive_pretrain/augmentation.py
def _resolve_brightness_contrast(config: AugmentationConfig, rng: torch.Generator) -> tuple[float, float]:
    brightness_factor = (
        torch.empty(1).uniform_(1 - config.brightness_range, 1 + config.brightness_range, generator=rng).item()
    )
    contrast_factor = (
        torch.empty(1).uniform_(1 - config.contrast_range, 1 + config.contrast_range, generator=rng).item()
    )
    return brightness_factor, contrast_factor


def _apply_brightness_contrast(frame: torch.Tensor, brightness_factor: float, contrast_factor: float) -> torch.Tensor:
    adjusted = TF.adjust_brightness(frame, brightness_factor)
    return TF.adjust_contrast(adjusted, contrast_factor)


def random_brightness_contrast(
    frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator
) -> torch.Tensor:
    brightness_factor, contrast_factor = _resolve_brightness_contrast(config, rng)
    return _apply_brightness_contrast(frame, brightness_factor, contrast_factor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/augmentation.py tests/unit/test_augmentation.py
git commit -m "feat: add brightness/contrast augmentation transform"
```

---

### Task 4: Noise transform — Gaussian noise

**Files:**

- Modify: `src/contrastive_pretrain/augmentation.py`
- Test: `tests/unit/test_augmentation.py`

**Interfaces:**

- Consumes: `AugmentationConfig` (Task 1).
- Produces: `random_gaussian_noise(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor`, plus `_resolve_noise_sigma`, `_apply_gaussian_noise`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_augmentation.py
from contrastive_pretrain.augmentation import (
    _apply_gaussian_noise,
    _resolve_noise_sigma,
    random_gaussian_noise,
)


def test_resolve_noise_sigma_stays_within_configured_bound() -> None:
    config = AugmentationConfig(noise_sigma_max=8.0)
    rng = torch.Generator().manual_seed(6)

    for _ in range(1000):
        sigma = _resolve_noise_sigma(config, rng)
        assert 0.0 <= sigma <= 8.0


def test_apply_gaussian_noise_zero_sigma_is_identity() -> None:
    frame = torch.full((1, 144, 160), 100, dtype=torch.uint8)
    rng = torch.Generator().manual_seed(7)

    result = _apply_gaussian_noise(frame, sigma=0.0, rng=rng)

    assert torch.equal(result, frame)


def test_apply_gaussian_noise_std_matches_requested_sigma() -> None:
    frame = torch.full((1, 144, 160), 128, dtype=torch.uint8)
    rng = torch.Generator().manual_seed(8)

    result = _apply_gaussian_noise(frame, sigma=8.0, rng=rng)

    diff_std = (result.to(torch.float32) - frame.to(torch.float32)).std().item()
    assert 6.0 <= diff_std <= 10.0


def test_random_gaussian_noise_preserves_shape_and_dtype() -> None:
    frame = torch.full((1, 144, 160), 100, dtype=torch.uint8)
    config = AugmentationConfig()
    rng = torch.Generator().manual_seed(9)

    result = random_gaussian_noise(frame, config, rng)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the transform**

```python
# append to src/contrastive_pretrain/augmentation.py
def _resolve_noise_sigma(config: AugmentationConfig, rng: torch.Generator) -> float:
    return torch.empty(1).uniform_(0.0, config.noise_sigma_max, generator=rng).item()


def _apply_gaussian_noise(frame: torch.Tensor, sigma: float, rng: torch.Generator) -> torch.Tensor:
    if sigma <= 0:
        return frame.clone()
    noise = torch.randn(frame.shape, generator=rng) * sigma
    return (frame.to(torch.float32) + noise).clamp(0, 255).to(torch.uint8)


def random_gaussian_noise(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor:
    sigma = _resolve_noise_sigma(config, rng)
    return _apply_gaussian_noise(frame, sigma, rng)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/augmentation.py tests/unit/test_augmentation.py
git commit -m "feat: add Gaussian noise augmentation transform"
```

---

### Task 5: Blur transform — Gaussian blur

**Files:**

- Modify: `src/contrastive_pretrain/augmentation.py`
- Test: `tests/unit/test_augmentation.py`

**Interfaces:**

- Consumes: `AugmentationConfig` (Task 1).
- Produces: `random_gaussian_blur(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor`, plus `_resolve_blur_sigma`, `_resolve_blur_kernel`, `_apply_gaussian_blur`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_augmentation.py
from contrastive_pretrain.augmentation import (
    _apply_gaussian_blur,
    _resolve_blur_kernel,
    _resolve_blur_sigma,
    random_gaussian_blur,
)


def test_resolve_blur_sigma_stays_within_configured_bound() -> None:
    config = AugmentationConfig(blur_sigma_max=0.8)
    rng = torch.Generator().manual_seed(10)

    for _ in range(1000):
        sigma = _resolve_blur_sigma(config, rng)
        assert 0.0 <= sigma <= 0.8


def test_resolve_blur_kernel_is_always_odd() -> None:
    assert _resolve_blur_kernel(AugmentationConfig(blur_kernel_size=3)) == 3
    assert _resolve_blur_kernel(AugmentationConfig(blur_kernel_size=4)) == 5


def test_apply_gaussian_blur_zero_sigma_is_identity() -> None:
    frame = torch.full((1, 144, 160), 100, dtype=torch.uint8)

    result = _apply_gaussian_blur(frame, sigma=0.0, kernel_size=3)

    assert torch.equal(result, frame)


def test_apply_gaussian_blur_preserves_shape_and_dtype() -> None:
    frame = torch.zeros((1, 144, 160), dtype=torch.uint8)
    frame[0, 70:74, 78:82] = 255

    result = _apply_gaussian_blur(frame, sigma=0.8, kernel_size=3)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_random_gaussian_blur_preserves_shape_and_dtype() -> None:
    frame = torch.zeros((1, 144, 160), dtype=torch.uint8)
    frame[0, 70:74, 78:82] = 255
    config = AugmentationConfig()
    rng = torch.Generator().manual_seed(11)

    result = random_gaussian_blur(frame, config, rng)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the transform**

```python
# append to src/contrastive_pretrain/augmentation.py
def _resolve_blur_sigma(config: AugmentationConfig, rng: torch.Generator) -> float:
    return torch.empty(1).uniform_(0.0, config.blur_sigma_max, generator=rng).item()


def _resolve_blur_kernel(config: AugmentationConfig) -> int:
    kernel = config.blur_kernel_size
    return kernel if kernel % 2 == 1 else kernel + 1


def _apply_gaussian_blur(frame: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    if sigma <= 0:
        return frame.clone()
    return TF.gaussian_blur(frame, [kernel_size, kernel_size], [sigma, sigma])


def random_gaussian_blur(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor:
    sigma = _resolve_blur_sigma(config, rng)
    kernel_size = _resolve_blur_kernel(config)
    return _apply_gaussian_blur(frame, sigma, kernel_size)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/augmentation.py tests/unit/test_augmentation.py
git commit -m "feat: add Gaussian blur augmentation transform"
```

---

### Task 6: Noise transform — JPEG-artifact simulation

**Files:**

- Modify: `src/contrastive_pretrain/augmentation.py`
- Test: `tests/unit/test_augmentation.py`

**Interfaces:**

- Consumes: `AugmentationConfig` (Task 1).
- Produces: `random_jpeg_artifact(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor`, plus `_resolve_jpeg_quality`, `_apply_jpeg_artifact`. Uses `torchvision.io.encode_jpeg`/`decode_jpeg` (imported in Task 1), which operate directly on `uint8` tensors of shape `(1 or 3, H, W)` — no PIL/cv2 round trip needed.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_augmentation.py
from contrastive_pretrain.augmentation import (
    _apply_jpeg_artifact,
    _resolve_jpeg_quality,
    random_jpeg_artifact,
)


def test_resolve_jpeg_quality_stays_within_configured_bounds() -> None:
    config = AugmentationConfig(jpeg_quality_min=60, jpeg_quality_max=95)
    rng = torch.Generator().manual_seed(12)

    for _ in range(1000):
        quality = _resolve_jpeg_quality(config, rng)
        assert 60 <= quality <= 95


def test_apply_jpeg_artifact_preserves_shape_and_dtype() -> None:
    frame = torch.zeros((1, 144, 160), dtype=torch.uint8)
    frame[0, 70:74, 78:82] = 255

    result = _apply_jpeg_artifact(frame, quality=80)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_apply_jpeg_artifact_at_high_quality_stays_close_to_original() -> None:
    frame = torch.zeros((1, 144, 160), dtype=torch.uint8)
    frame[0, 70:74, 78:82] = 255

    result = _apply_jpeg_artifact(frame, quality=95)

    diff = (result.to(torch.int16) - frame.to(torch.int16)).abs()
    assert diff.to(torch.float32).mean().item() < 5.0


def test_random_jpeg_artifact_preserves_shape_and_dtype() -> None:
    frame = torch.zeros((1, 144, 160), dtype=torch.uint8)
    frame[0, 70:74, 78:82] = 255
    config = AugmentationConfig()
    rng = torch.Generator().manual_seed(13)

    result = random_jpeg_artifact(frame, config, rng)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the transform**

Add this import at the top of `src/contrastive_pretrain/augmentation.py`,
below the existing `torch`/`torchvision` imports (this task is the first
to need `torchvision.io`):

```python
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg
```

Then append the transform:

```python
# append to src/contrastive_pretrain/augmentation.py
def _resolve_jpeg_quality(config: AugmentationConfig, rng: torch.Generator) -> int:
    return int(torch.randint(config.jpeg_quality_min, config.jpeg_quality_max + 1, (1,), generator=rng).item())


def _apply_jpeg_artifact(frame: torch.Tensor, quality: int) -> torch.Tensor:
    encoded = encode_jpeg(frame, quality=quality)
    return decode_jpeg(encoded, mode=ImageReadMode.GRAY)


def random_jpeg_artifact(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor:
    quality = _resolve_jpeg_quality(config, rng)
    return _apply_jpeg_artifact(frame, quality)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/augmentation.py tests/unit/test_augmentation.py
git commit -m "feat: add JPEG-artifact simulation augmentation transform"
```

---

### Task 7: Compose — `augment_view` and `make_pair`

**Files:**

- Modify: `src/contrastive_pretrain/augmentation.py`
- Test: `tests/unit/test_augmentation.py`

**Interfaces:**

- Consumes: `random_translate`, `random_crop_resize` (Task 2), `random_brightness_contrast` (Task 3), `random_gaussian_noise` (Task 4), `random_gaussian_blur` (Task 5), `random_jpeg_artifact` (Task 6).
- Produces: `augment_view(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor` and `make_pair(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]` — the two names Task 9's CLI imports.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_augmentation.py
from contrastive_pretrain.augmentation import augment_view, make_pair


def test_augment_view_preserves_shape_and_dtype() -> None:
    frame = torch.zeros((1, 144, 160), dtype=torch.uint8)
    frame[0, 70:74, 78:82] = 255
    config = AugmentationConfig()
    rng = torch.Generator().manual_seed(14)

    result = augment_view(frame, config, rng)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_make_pair_produces_two_independently_sampled_views() -> None:
    frame = torch.zeros((1, 144, 160), dtype=torch.uint8)
    frame[0, 70:74, 78:82] = 255
    config = AugmentationConfig()
    rng = torch.Generator().manual_seed(15)

    view_a, view_b = make_pair(frame, config, rng)

    assert view_a.shape == frame.shape
    assert view_b.shape == frame.shape
    assert not torch.equal(view_a, view_b)


def test_make_pair_is_reproducible_given_the_same_seed() -> None:
    frame = torch.zeros((1, 144, 160), dtype=torch.uint8)
    frame[0, 70:74, 78:82] = 255
    config = AugmentationConfig()

    view_a1, view_b1 = make_pair(frame, config, torch.Generator().manual_seed(42))
    view_a2, view_b2 = make_pair(frame, config, torch.Generator().manual_seed(42))

    assert torch.equal(view_a1, view_a2)
    assert torch.equal(view_b1, view_b2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the composition**

```python
# append to src/contrastive_pretrain/augmentation.py
def augment_view(frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator) -> torch.Tensor:
    view = random_translate(frame, config, rng)
    view = random_crop_resize(view, config, rng)
    view = random_brightness_contrast(view, config, rng)
    view = random_gaussian_noise(view, config, rng)
    view = random_gaussian_blur(view, config, rng)
    view = random_jpeg_artifact(view, config, rng)
    return view


def make_pair(
    frame: torch.Tensor, config: AugmentationConfig, rng: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    view_a = augment_view(frame, config, rng)
    view_b = augment_view(frame, config, rng)
    return view_a, view_b
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_augmentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/augmentation.py tests/unit/test_augmentation.py
git commit -m "feat: compose augmentation transforms into augment_view and make_pair"
```

---

### Task 8: Visualization — augmentation contact sheet

**Files:**

- Modify: `src/observability/visualization.py`
- Modify: `src/observability/__init__.py`
- Test: `tests/unit/test_visualization.py`

**Interfaces:**

- Produces: `build_pair_preview(original: np.ndarray, view_a: np.ndarray, view_b: np.ndarray) -> np.ndarray` and `build_augmentation_contact_sheet(triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray` — the name Task 9's CLI imports. This module stays numpy-based (unchanged convention, shared with `data_collection`'s existing contact sheets); Task 9's CLI is the boundary that converts `augmentation.py`'s torch tensors to numpy right before calling these.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_visualization.py
from observability.visualization import build_augmentation_contact_sheet, build_pair_preview


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_visualization.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the visualization functions**

```python
# append to src/observability/visualization.py
def build_pair_preview(original: np.ndarray, view_a: np.ndarray, view_b: np.ndarray) -> np.ndarray:
    return np.concatenate([original, view_a, view_b], axis=1)


def build_augmentation_contact_sheet(triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    if not triples:
        return np.empty((0, 0), dtype=np.uint8)

    rows = [build_pair_preview(original, view_a, view_b) for original, view_a, view_b in triples]
    return np.concatenate(rows, axis=0)
```

```python
# src/observability/__init__.py -- replace entire file contents with:
from observability.logging_config import JSONFormatter, configure_logging
from observability.tracking import TrackioRun
from observability.visualization import (
    build_augmentation_contact_sheet,
    build_contact_sheet,
    build_pair_preview,
)

__all__ = [
    "JSONFormatter",
    "TrackioRun",
    "build_augmentation_contact_sheet",
    "build_contact_sheet",
    "build_pair_preview",
    "configure_logging",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_visualization.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/observability/visualization.py src/observability/__init__.py tests/unit/test_visualization.py
git commit -m "feat: add augmentation contact-sheet visualization"
```

---

### Task 9: CLI — `contrastive-pretrain preview` command

**Files:**

- Create: `src/contrastive_pretrain/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_contrastive_pretrain_cli.py`

**Interfaces:**

- Consumes: `AugmentationConfig`, `make_pair` (Task 7); `build_augmentation_contact_sheet` (Task 8).
- Produces: `main` — a `click.Group` console entry point registered as `contrastive-pretrain` in `pyproject.toml`, with one subcommand `preview`. Converts torch tensors to numpy (`.squeeze(0).numpy()`) at the boundary before calling `build_augmentation_contact_sheet`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_contrastive_pretrain_cli.py
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image

from contrastive_pretrain.cli import main


def test_preview_command_writes_contact_sheet(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(2):
        frame = np.full((144, 160), 50 + i, dtype=np.uint8)
        Image.fromarray(frame).save(frames_dir / f"frame_{i}.png")

    out_path = tmp_path / "preview.png"
    runner = CliRunner()
    result = runner.invoke(
        main, ["preview", "--frames-dir", str(frames_dir), "--out", str(out_path)]
    )

    assert result.exit_code == 0
    assert out_path.exists()
    saved = np.array(Image.open(out_path))
    assert saved.shape == (144 * 2, 480)


def test_preview_command_errors_on_empty_directory(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(main, ["preview", "--frames-dir", str(frames_dir)])

    assert result.exit_code != 0
    assert "No .png frames found" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contrastive_pretrain.cli'`

- [ ] **Step 3: Implement the CLI**

```python
# src/contrastive_pretrain/cli.py
"""Console entry point: `contrastive-pretrain preview --frames-dir <dir> --out <path>`
generates a visual (original, view A, view B) contact sheet for spot-checking
the augmentation policy before spending GPU time on a training run."""

from __future__ import annotations

from pathlib import Path

import click
import torch
from PIL import Image
from torchvision.io import ImageReadMode, read_image

from contrastive_pretrain.augmentation import AugmentationConfig, make_pair
from observability.visualization import build_augmentation_contact_sheet


def _load_grayscale_frames(frames_dir: Path) -> list[torch.Tensor]:
    paths = sorted(frames_dir.glob("*.png"))
    return [read_image(str(p), mode=ImageReadMode.GRAY) for p in paths]


@click.group()
def main() -> None:
    """Pokemon Red/Blue contrastive-pretraining tools."""


@main.command()
@click.option("--frames-dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("augmentation_preview.png"))
@click.option("--seed", type=int, default=0)
def preview(frames_dir: Path, out: Path, seed: int) -> None:
    """Build an (original, view A, view B) contact sheet from sample frames."""
    frames = _load_grayscale_frames(frames_dir)
    if not frames:
        raise click.ClickException(f"No .png frames found in {frames_dir}")

    config = AugmentationConfig()
    rng = torch.Generator().manual_seed(seed)
    triples = []
    for frame in frames:
        view_a, view_b = make_pair(frame, config, rng)
        triples.append(
            (frame.squeeze(0).numpy(), view_a.squeeze(0).numpy(), view_b.squeeze(0).numpy())
        )

    sheet = build_augmentation_contact_sheet(triples)
    Image.fromarray(sheet).save(out)
    click.echo(f"Wrote {len(frames)}-frame augmentation preview to {out}")
```

```toml
# pyproject.toml -- in [project.scripts], add:
contrastive-pretrain = "contrastive_pretrain.cli:main"
```

```toml
# pyproject.toml -- in [tool.hatch.build.targets.wheel], change to:
packages = ["src/data_collection", "src/observability", "src/contrastive_pretrain"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run the full unit test suite**

Run: `uv run pytest tests/unit -v`
Expected: PASS (all tests, including Tasks 1-8)

- [ ] **Step 6: Commit**

```bash
git add src/contrastive_pretrain/cli.py pyproject.toml tests/unit/test_contrastive_pretrain_cli.py
git commit -m "feat: add contrastive-pretrain preview CLI command"
```

---

## Manual validation (not automated, do after Task 9)

Run the preview command against the existing local sample frames to eyeball the augmentation policy end to end. Note: `scratch/smoke_frames/*_t120s.png` are pre-crop smoke-test captures (full video frame, not the final 160x144 pipeline crop), so this is a code-path smoke test, not a validation of final-pipeline-format frames — re-run this against real extracted frames once Phase B output exists.

```bash
uv run contrastive-pretrain preview --frames-dir scratch/smoke_frames --out /tmp/augmentation_preview.png
```

Open `/tmp/augmentation_preview.png` and confirm: text stays legible, no view has cropped out the bottom text-box area, brightness/contrast shifts don't wash out detail.
