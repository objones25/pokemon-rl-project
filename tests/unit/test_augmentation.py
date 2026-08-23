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
