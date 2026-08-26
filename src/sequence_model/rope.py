"""Rotary position embeddings, halves pairing.

Two decisions here are load-bearing and invisible in a shape check.

First, HALVES pairing: element i is rotated against element i + head_dim/2.
The interleaved convention (i against i+1) trains just as well and is
silently incompatible with any checkpoint written by the other one, so the
choice is pinned by a test against an exact tensor rather than left to a
future refactor.

Second, the angle is computed in float64 and reduced mod 2*pi before
casting. Episodes run to ~163,840 steps; at that magnitude a float32
product t * inv_freq carries max error 3.0e-03 across channels (2.9e-08
via float64). The damage is in the MID-frequency channels, not the
highest -- 163840 is exactly representable in float32 -- which is why the
test asserts on the max over all channels."""

from __future__ import annotations

import math

import torch


def rope_tables(
    positions: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`positions` is an integer tensor of absolute step indices, shape
    (B, T). Returns (cos, sin), each (B, T, head_dim // 2), float32."""
    half = head_dim // 2
    inv_freq = theta ** (
        -torch.arange(0, half, dtype=torch.float64, device=positions.device) / half
    )
    angle = (positions.to(torch.float64).unsqueeze(-1) * inv_freq) % (2 * math.pi)
    return angle.cos().to(torch.float32), angle.sin().to(torch.float32)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x` is (B, H, T, head_dim) -- ALREADY split into heads. Applying
    RoPE before the head split makes the rotation straddle head
    boundaries, which costs a few percent of loss and gets blamed on the
    data. `cos`/`sin` are (B, T, head_dim // 2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
