"""SimCLR ResNet-50-style encoder + projector. The encoder drops the
standard ImageNet stem's initial maxpool (kept: pretrained weights,
which are fully compatible since maxpool owns zero learnable params —
see the design spec) to preserve spatial detail this domain's small
UI elements (HP bars, text glyphs) depend on. Grayscale-to-3-channel
replication lives inside the module so training and load_frozen_encoder
share one external contract: 1-channel grayscale in, 2048-d feature out.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

EMBEDDING_DIM = 2048


class GrayscaleResNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 1:
            raise ValueError(f"expected 1-channel grayscale input, got shape {tuple(x.shape)}")
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def build_encoder(pretrained: bool = True) -> tuple[nn.Module, int]:
    return GrayscaleResNetEncoder(pretrained=pretrained), EMBEDDING_DIM


class SimCLRProjector(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_projector(in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128) -> nn.Module:
    return SimCLRProjector(in_dim, hidden_dim, out_dim)
