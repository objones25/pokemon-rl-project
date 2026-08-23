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
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg
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
