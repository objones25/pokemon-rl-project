import torch

from contrastive_pretrain.augmentation import (
    AugmentationConfig,
    _apply_brightness_contrast,
    _apply_crop_resize,
    _apply_translate,
    _resolve_brightness_contrast,
    _resolve_crop_box,
    _resolve_translate_offset,
    random_brightness_contrast,
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
