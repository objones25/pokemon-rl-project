"""Grouped-query attention with RoPE, one module serving both the chunked
training forward and the KV-cached rollout step.

Three ordering/flag decisions here each produce correctly-shaped tensors
and a broken model:

1. Heads are split BEFORE RoPE. Rotating the flat (B, T, d_model) tensor
   makes the rotation straddle head boundaries.
2. `enable_gqa=True` rather than a hand-written repeat_kv. The tempting
   `x.repeat()` implementation is exactly a query-head permutation: it
   trains fine from scratch and breaks checkpoint interop and fused
   kernels, which is a different bug class than a quality regression.
3. `is_causal=False` on the step path. The query is length 1 against
   `context_len` cached keys, so is_causal would mask everything but slot
   0 -- prefill perfect, rollout garbage.

SDPA with an explicit boolean mask is used rather than FlexAttention
because FlexAttention has no CPU backward in torch 2.13, which would make
the chunked-forward tests unrunnable under the CPU-only test rule."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig
from sequence_model.layers import RMSNorm
from sequence_model.rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        if config.qk_norm:
            self.q_norm: nn.Module = RMSNorm(config.head_dim, config.rms_norm_eps)
            self.k_norm: nn.Module = RMSNorm(config.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _project(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        cfg = self.config
        q = self.q_proj(x).view(batch, seq_len, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        return apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin), v

    def _merge_heads(self, attended: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = attended.shape
        merged = attended.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(merged)

    def forward_chunk(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """`x` is (B, L, d_model); `mask` is the (B, 1, L, L) bool mask
        from masks.build_chunk_mask."""
        q, k, v = self._project(x, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return self._merge_heads(attended)

    def forward_step(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: RolloutCache,
        layer: int,
    ) -> torch.Tensor:
        """`x` is (N, 1, d_model). Writes this step's rotated K/V into the
        cache and attends over every valid slot."""
        q, k, v = self._project(x, cos, sin)
        k_all, v_all = cache.write(layer, k, v)
        attended = F.scaled_dot_product_attention(
            q, k_all, v_all, attn_mask=cache.attention_mask(), enable_gqa=True, is_causal=False
        )
        return self._merge_heads(attended)
