import math

import pytest
import torch
import torch.nn.functional as F

from contrastive_pretrain.losses import nt_xent_loss


def test_nt_xent_loss_matches_hand_derivation_for_orthogonal_negatives() -> None:
    """2 examples, embedding dim 2. a0=b0=e1, a1=b1=e2 (e1 orthogonal to
    e2), temperature=1.0. For anchor a0, the positive is b0 (sim=1) and
    the two negatives (a1, b1) both have sim=0 (self a0-a0 is masked
    out). NT-Xent for this anchor is
    -log(exp(1) / (exp(1) + exp(0) + exp(0))) = log(e + 2) - 1.
    All 4 anchors are symmetric here, so the batch-mean loss equals this
    same value."""
    e1 = torch.tensor([1.0, 0.0])
    e2 = torch.tensor([0.0, 1.0])
    z_a = torch.stack([e1, e2])
    z_b = torch.stack([e1, e2])

    loss = nt_xent_loss(z_a, z_b, temperature=1.0)

    expected = math.log(math.e + 2) - 1
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_nt_xent_loss_lower_for_better_aligned_positives() -> None:
    torch.manual_seed(0)
    n, d = 8, 16
    base = F.normalize(torch.randn(n, d), dim=1)
    noise = torch.randn(n, d) * 0.01
    z_a = base
    z_b_close = F.normalize(base + noise, dim=1)
    z_b_far = F.normalize(torch.randn(n, d), dim=1)

    loss_close = nt_xent_loss(z_a, z_b_close, temperature=0.5)
    loss_far = nt_xent_loss(z_a, z_b_far, temperature=0.5)

    assert loss_close.item() < loss_far.item()


def test_nt_xent_loss_is_differentiable() -> None:
    z_a = torch.randn(4, 8, requires_grad=True)
    z_b = torch.randn(4, 8, requires_grad=True)

    loss = nt_xent_loss(z_a, z_b, temperature=0.1)
    loss.backward()

    assert z_a.grad is not None
    assert torch.isfinite(z_a.grad).all()
