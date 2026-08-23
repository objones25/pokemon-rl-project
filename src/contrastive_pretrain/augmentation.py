"""Domain-adapted contrastive-pretraining augmentation transforms, built on
PyTorch/TorchVision primitives so this module is directly reusable by the
(deferred) training-loop's Dataset/DataLoader without a rewrite.

See docs/superpowers/specs/2026-08-23-contrastive-augmentation-policy-design.md
for the rationale behind each transform's inclusion and parameter range.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF


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
