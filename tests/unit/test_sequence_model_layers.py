import pytest
import torch

from sequence_model.layers import RMSNorm, SwiGLU


def test_rmsnorm_scales_input_to_unit_root_mean_square() -> None:
    norm = RMSNorm(4, eps=0.0)
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]])

    out = norm(x)

    assert out.pow(2).mean().item() == pytest.approx(1.0, abs=1e-5)


def test_rmsnorm_matches_hand_derivation() -> None:
    """x = [2, 2, 2, 2] has RMS 2, so the normalized value is 1.0 in
    every channel and the default unit weight leaves it there."""
    norm = RMSNorm(4, eps=0.0)

    out = norm(torch.tensor([[2.0, 2.0, 2.0, 2.0]]))

    assert torch.allclose(out, torch.tensor([[1.0, 1.0, 1.0, 1.0]]), atol=1e-6)


def test_rmsnorm_applies_eps_inside_the_square_root() -> None:
    """eps outside the sqrt is a real bug. With x all zeros and eps=4,
    inside gives 0/sqrt(0+4)=0 and no NaN; the value that distinguishes
    the two is the scale on a nonzero input: RMS^2 = 1, eps = 3, so
    inside-sqrt gives 1/sqrt(1+3) = 0.5."""
    norm = RMSNorm(4, eps=3.0)

    out = norm(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))

    assert torch.allclose(out, torch.tensor([[0.5, 0.5, 0.5, 0.5]]), atol=1e-6)


def test_rmsnorm_weight_is_a_learnable_parameter() -> None:
    norm = RMSNorm(4)

    assert [name for name, _ in norm.named_parameters()] == ["weight"]


def test_swiglu_output_has_d_model_width() -> None:
    torch.manual_seed(0)
    mlp = SwiGLU(d_model=8, d_ff=16)

    out = mlp(torch.randn(2, 3, 8))

    assert tuple(out.shape) == (2, 3, 8)


def test_swiglu_has_three_projections_and_no_biases() -> None:
    """d_ff = 4 * d_model with a GATED mlp is silently 50% more MLP
    parameters than intended, so the three-matrix shape is asserted."""
    mlp = SwiGLU(d_model=8, d_ff=16)

    names = sorted(name for name, _ in mlp.named_parameters())

    assert names == ["down_proj.weight", "gate_proj.weight", "up_proj.weight"]


def test_swiglu_matches_hand_derivation_for_unit_weights() -> None:
    """With gate=up=identity-like all-ones projections on x=[1], the
    gate is silu(1) = 1/(1+e^-1) and up is 1, so the product is silu(1)
    and the summing down_proj returns d_ff * silu(1)."""
    mlp = SwiGLU(d_model=1, d_ff=2)
    torch.nn.init.ones_(mlp.gate_proj.weight)
    torch.nn.init.ones_(mlp.up_proj.weight)
    torch.nn.init.ones_(mlp.down_proj.weight)

    out = mlp(torch.tensor([[1.0]]))

    expected = 2.0 * (1.0 / (1.0 + torch.tensor(-1.0).exp().item()))
    assert out.item() == pytest.approx(expected, abs=1e-6)
