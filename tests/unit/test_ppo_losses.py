"""The clipped surrogate, checked at the points where clipping changes it."""

from __future__ import annotations

import pytest
import torch

from ppo.config import PPOConfig
from ppo.losses import ppo_losses
from tests.conftest import PINNED_ENCODER_REVISION


def _config() -> PPOConfig:
    return PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, ent_coef=0.0, vf_coef=0.0)


def test_the_ratio_is_exactly_one_when_logprob_old_came_from_these_logits() -> None:
    """The invariant the whole design turns on, at the level of one function."""
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 7)
    action = torch.zeros(1, 3, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

    output = ppo_losses(
        logits, torch.zeros(1, 3), action, logprob_old,
        advantage=torch.ones(1, 3), value_target=torch.zeros(1, 3), config=_config(),
    )

    assert output.max_abs_ratio_dev == pytest.approx(0.0, abs=1e-6)


def test_the_policy_loss_is_the_negative_advantage_when_the_ratio_is_one() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 7)
    action = torch.zeros(1, 3, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

    output = ppo_losses(
        logits, torch.zeros(1, 3), action, logprob_old,
        advantage=torch.full((1, 3), 2.0), value_target=torch.zeros(1, 3), config=_config(),
    )

    assert output.policy.item() == pytest.approx(-2.0)


def test_a_positive_advantage_is_clipped_at_one_plus_the_clip_range() -> None:
    """ratio = e^1 = 2.718, clip_range 0.2 -> the surrogate caps at 1.2 * A."""
    logits = torch.zeros(1, 1, 2)
    action = torch.zeros(1, 1, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1)[..., 0] - 1.0

    output = ppo_losses(
        logits, torch.zeros(1, 1), action, logprob_old,
        advantage=torch.ones(1, 1), value_target=torch.zeros(1, 1), config=_config(),
    )

    assert output.policy.item() == pytest.approx(-1.2)


def test_the_clip_fraction_counts_the_positions_where_clipping_bound() -> None:
    """4 positions, exactly 1 exceeds clip_range -> 0.25. Asymmetric on
    purpose: with an even 2-2 split the complement of the correct fraction is
    also 0.5, so a flipped comparison operator produces the same number and
    the test cannot tell them apart."""
    logits = torch.zeros(1, 4, 2)
    action = torch.zeros(1, 4, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1)[..., 0] - torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    output = ppo_losses(
        logits, torch.zeros(1, 4), action, logprob_old,
        advantage=torch.ones(1, 4), value_target=torch.zeros(1, 4), config=_config(),
    )

    assert output.clip_fraction == pytest.approx(0.25)


def test_the_value_loss_is_the_mean_squared_error_against_the_target() -> None:
    config = PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, ent_coef=0.0, vf_coef=1.0)
    logits = torch.zeros(1, 2, 2)
    action = torch.zeros(1, 2, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1)[..., 0]

    output = ppo_losses(
        logits, torch.zeros(1, 2), action, logprob_old,
        advantage=torch.zeros(1, 2), value_target=torch.full((1, 2), 3.0), config=config,
    )

    assert output.value.item() == pytest.approx(9.0)


def test_the_total_adds_the_scaled_value_loss_and_subtracts_the_scaled_entropy() -> None:
    """The one composition nothing else in this suite pins: every other test
    zeroes a coefficient or reads an unscaled component, and test_ppo_update
    only checks `total` for finiteness. Flipping `- ent_coef * entropy` to `+`
    turns the entropy bonus into a penalty that drives the policy
    deterministic -- the classic silent PPO bug -- and passes all of them.

    Uniform 2-way logits with logprob_old taken from those same logits give
    ratio 1, so with advantage 2.0 the policy loss is exactly -2.0; value 0
    against target 3.0 gives 9.0; entropy is ln 2. At vf_coef=0.5 and
    ent_coef=0.01 that is -2.0 + 4.5 - 0.006931472 = 2.493068528. The sign
    flip lands on 2.506931472 instead."""
    config = PPOConfig(
        frozen_encoder_revision=PINNED_ENCODER_REVISION, vf_coef=0.5, ent_coef=0.01
    )
    logits = torch.zeros(1, 1, 2)
    action = torch.zeros(1, 1, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1)[..., 0]

    output = ppo_losses(
        logits, torch.zeros(1, 1), action, logprob_old,
        advantage=torch.full((1, 1), 2.0), value_target=torch.full((1, 1), 3.0), config=config,
    )

    assert output.total.item() == pytest.approx(2.493068528, abs=1e-7)


def test_the_entropy_of_a_uniform_two_way_logit_is_log_two() -> None:
    config = PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, ent_coef=1.0, vf_coef=0.0)
    logits = torch.zeros(1, 1, 2)
    action = torch.zeros(1, 1, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1)[..., 0]

    output = ppo_losses(
        logits, torch.zeros(1, 1), action, logprob_old,
        advantage=torch.zeros(1, 1), value_target=torch.zeros(1, 1), config=config,
    )

    assert output.entropy.item() == pytest.approx(0.6931472, abs=1e-6)
