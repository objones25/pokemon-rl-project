"""ReturnScaler: divides value targets by a running std, never shifts the mean."""

from __future__ import annotations

import pytest
import torch

from ppo.normalizer import ReturnScaler


def test_scale_starts_at_one_so_the_first_update_is_unscaled() -> None:
    assert ReturnScaler(gamma=0.99).scale == pytest.approx(1.0)


def test_scale_approaches_the_standard_deviation_of_the_returns_it_has_seen() -> None:
    scaler = ReturnScaler(gamma=0.99)
    returns = torch.tensor([[-10.0, 10.0, -10.0, 10.0]])

    scaler.update(returns)

    assert scaler.scale == pytest.approx(10.0, rel=0.05)


def test_the_scaler_never_shifts_the_mean_so_advantage_signs_survive() -> None:
    """The brief's own value here (-5, asserted negative) does not
    discriminate a mean-subtracting bug: the running mean of [100, 102, 104]
    is 102, so subtracting it from -5 only pushes the result further
    negative -- the sign never flips either way. A small positive value
    below the mean does discriminate: dividing by scale alone keeps it
    positive, while subtracting the (much larger) mean first flips it
    negative."""
    scaler = ReturnScaler(gamma=0.99)
    scaler.update(torch.tensor([[100.0, 102.0, 104.0]]))

    assert scaler.normalize(torch.tensor([5.0]))[0].item() > 0.0


def test_state_round_trips_through_a_checkpoint() -> None:
    scaler = ReturnScaler(gamma=0.99)
    scaler.update(torch.tensor([[-10.0, 10.0]]))
    restored = ReturnScaler(gamma=0.99)

    restored.load_state_dict(scaler.state_dict())

    assert restored.scale == pytest.approx(scaler.scale)
