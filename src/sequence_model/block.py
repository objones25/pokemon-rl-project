"""Pre-norm transformer block: RMSNorm -> GQA -> residual, RMSNorm ->
SwiGLU -> residual.

Pre-norm, not post-norm: it trains without a warmup schedule, and this
model's optimizer never gets a stable loss surface to warm up against --
PPO's objective moves under it every update.

The residual stream itself is never normalized, only what feeds each
sublayer. The zeroed-sublayer identity test is the cheap check that this
stayed true through refactors."""

from __future__ import annotations

import torch
from torch import nn

from sequence_model.attention import GroupedQueryAttention
from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig
from sequence_model.layers import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config.d_model, config.d_ff)

    def forward_chunk(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attention.forward_chunk(self.attn_norm(x), cos, sin, mask)
        return x + self.mlp(self.mlp_norm(x))

    def forward_step(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: RolloutCache,
        layer: int,
    ) -> torch.Tensor:
        x = x + self.attention.forward_step(self.attn_norm(x), cos, sin, cache, layer)
        return x + self.mlp(self.mlp_norm(x))
