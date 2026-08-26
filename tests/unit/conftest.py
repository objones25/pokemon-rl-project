"""Fixtures shared across contrastive_pretrain's test files under tests/unit/.
Scoped here (not the root tests/conftest.py) because build_encoder is
specific to contrastive_pretrain -- no other package's tests need it.
"""

import pytest
import torch
from torch import nn

from contrastive_pretrain.model import build_encoder


@pytest.fixture
def encoder_and_dim() -> tuple[nn.Module, int]:
    """Fresh, untrained GrayscaleResNetEncoder + its embedding dim -- shared
    by every contrastive_pretrain test that needs a real encoder instance
    without downloading pretrained ImageNet weights."""
    return build_encoder(pretrained=False)


@pytest.fixture
def encoder(encoder_and_dim: tuple[nn.Module, int]) -> nn.Module:
    return encoder_and_dim[0]
