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
