"""RMSNorm and SwiGLU -- the two non-attention pieces of the block.

RMSNorm computes the mean square in float32 regardless of the autocast
dtype, and adds eps INSIDE the square root. eps outside is a real bug
(it changes the normalizer's asymptote); the bf16 squaring question is a
0.18% norm shift per site, not the accumulation-precision story usually
told, but float32 here costs nothing.

SwiGLU is three matrices, not two. d_ff is sized round_to_128(8/3 *
d_model) precisely so a gated MLP holds parameters level against a plain
4x one -- writing d_ff = 4 * d_model here would silently add 50% more MLP
parameters than the budget accounts for."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
