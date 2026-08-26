import math

import pytest
import torch

from sequence_model.rope import apply_rope, rope_tables


def test_rope_tables_returns_cos_and_sin_of_half_head_dim() -> None:
    positions = torch.arange(5).unsqueeze(0)

    cos, sin = rope_tables(positions, head_dim=8, theta=10000.0)

    assert (cos.shape, sin.shape) == ((1, 5, 4), (1, 5, 4))


def test_rope_tables_at_position_zero_is_all_cos_one_sin_zero() -> None:
    cos, sin = rope_tables(torch.zeros(1, 1, dtype=torch.long), head_dim=8, theta=10000.0)

    assert cos.tolist() == [[[1.0, 1.0, 1.0, 1.0]]]
    assert sin.tolist() == [[[0.0, 0.0, 0.0, 0.0]]]


def test_rope_tables_are_exact_at_large_absolute_positions() -> None:
    """Episodes run to 163,840 steps. Computing t * inv_freq in float32
    loses precision in the mid-frequency channels (max error 3.0e-03);
    computing in float64 with a mod-2pi reduction keeps it at 2.9e-08.
    Asserted across ALL channels: channel 0 alone passes either way,
    because 163840 is exactly representable in float32."""
    head_dim, theta, t = 64, 10000.0, 163840
    half = head_dim // 2
    inv_freq = theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
    expected = ((torch.tensor([[t]], dtype=torch.float64).unsqueeze(-1) * inv_freq) % (2 * math.pi)).cos()

    cos, _ = rope_tables(torch.tensor([[t]]), head_dim=head_dim, theta=theta)

    assert (cos.double() - expected).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_apply_rope_uses_halves_pairing_not_interleaved() -> None:
    """head_dim=4 at position 1 with theta such that inv_freq = [1, 1].
    Halves pairing rotates (x0, x2) and (x1, x3) together. Interleaved
    would rotate (x0, x1) and (x2, x3) -- same shape, different model,
    and silently incompatible with any external checkpoint."""
    x = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    cos = torch.full((1, 1, 2), math.cos(1.0))
    sin = torch.full((1, 1, 2), math.sin(1.0))

    out = apply_rope(x, cos, sin)

    assert out.flatten().tolist() == pytest.approx(
        [math.cos(1.0), 0.0, math.sin(1.0), 0.0], abs=1e-6
    )


def test_rope_attention_score_depends_only_on_relative_distance() -> None:
    """The whole point of RoPE. A query at 100 against a key at 105 must
    score identically to a query at 0 against a key at 5."""
    torch.manual_seed(0)
    head_dim = 8
    q_vec = torch.randn(1, 1, 1, head_dim)
    k_vec = torch.randn(1, 1, 1, head_dim)

    near_cos, near_sin = rope_tables(torch.tensor([[0, 5]]), head_dim, 10000.0)
    far_cos, far_sin = rope_tables(torch.tensor([[100, 105]]), head_dim, 10000.0)
    near = (
        apply_rope(q_vec, near_cos[:, :1], near_sin[:, :1])
        * apply_rope(k_vec, near_cos[:, 1:], near_sin[:, 1:])
    ).sum()
    far = (
        apply_rope(q_vec, far_cos[:, :1], far_sin[:, :1])
        * apply_rope(k_vec, far_cos[:, 1:], far_sin[:, 1:])
    ).sum()

    assert near.item() == pytest.approx(far.item(), abs=1e-5)
