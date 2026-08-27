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
    (B, T). Returns (cos, sin), each (B, T, head_dim // 2), float32, on
    `positions.device`.

    The float64 reduction stays ON-DEVICE everywhere it can, which is the
    case that matters: training runs on RunPod CUDA, where float64 is
    supported natively and this is a pure device-local computation. Routing
    it through the host instead would put a copy and a synchronisation point
    on the rollout hot path, thousands of times per second, for no gain.

    MPS is the sole exception -- it has no float64 at all and raises
    "Cannot convert a MPS Tensor to float64" -- so on Apple Silicon, which
    is only ever the local test machine, the reduction falls back to CPU.
    The `.to(device)` calls below are no-ops when the compute device already
    is the target, so CUDA pays nothing for this branch.

    The move and the cast are two separate calls on purpose. The fused
    `positions.to(device="cpu", dtype=torch.float64)` REINTERPRETS an MPS
    int64 tensor's bit patterns as float64 instead of converting them --
    position 1 comes back as 5e-324, 163840 as 8.09e-319 -- so it silently
    yields an all-zero angle table rather than raising. Moving first and
    casting second is correct on every device."""
    half = head_dim // 2
    device = positions.device
    compute_device = torch.device("cpu") if device.type == "mps" else device
    inv_freq = theta ** (
        -torch.arange(0, half, dtype=torch.float64, device=compute_device) / half
    )
    angle = (
        positions.to(compute_device).to(torch.float64).unsqueeze(-1) * inv_freq
    ) % (2 * math.pi)
    return (
        angle.cos().to(torch.float32).to(device),
        angle.sin().to(torch.float32).to(device),
    )


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x` is (B, H, T, head_dim) -- ALREADY split into heads. Applying
    RoPE before the head split makes the rotation straddle head
    boundaries, which costs a few percent of loss and gets blamed on the
    data. `cos`/`sin` are (B, T, head_dim // 2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
