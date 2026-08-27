"""Fixtures shared across test files under tests/unit/. Scoped here (not the
root tests/conftest.py) because these fixtures are specific to individual
sub-packages -- other sub-packages' tests don't need them.
"""

import pytest
from torch import nn

from contrastive_pretrain.model import build_encoder

from .fakes import FakeEmulator


@pytest.fixture
def encoder_and_dim() -> tuple[nn.Module, int]:
    """Fresh, untrained GrayscaleResNetEncoder + its embedding dim -- shared
    by every contrastive_pretrain test that needs a real encoder instance
    without downloading pretrained ImageNet weights."""
    return build_encoder(pretrained=False)


@pytest.fixture
def encoder(encoder_and_dim: tuple[nn.Module, int]) -> nn.Module:
    return encoder_and_dim[0]


@pytest.fixture
def fake_emulator() -> FakeEmulator:
    return FakeEmulator()
