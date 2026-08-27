"""GAE, checked against arithmetic done by hand rather than against itself."""

from __future__ import annotations

import pytest
import torch

from ppo.gae import compute_gae


def test_gae_matches_a_hand_computed_three_step_trajectory() -> None:
    """gamma=0.5, lambda=1.0, one episode, rewards [1, 1, 1],
    values [0, 0, 0, 0]. With lambda=1 the advantage is the discounted
    return: A2 = 1, A1 = 1 + 0.5*1 = 1.5, A0 = 1 + 0.5*1.5 = 1.75."""
    reward = torch.tensor([[1.0, 1.0, 1.0]])
    value = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    episode_id = torch.zeros(1, 4, dtype=torch.int64)

    advantages, _ = compute_gae(reward, value, episode_id, gamma=0.5, gae_lambda=1.0)

    assert advantages[0].tolist() == pytest.approx([1.75, 1.5, 1.0])


def test_returns_are_advantages_plus_the_baseline_value() -> None:
    """`value` is distinct at every index on purpose: a constant `value`
    array (the brief's original [2, 2, 2, 2]) can't tell `value[:, :T]`
    (the correct baseline slice) apart from `value[:, 1:]` (the bootstrap
    slot, off by one) -- both slices read out identically, so a misaligned
    implementation would still pass. Distinct values make the two slices
    disagree, so the comparison actually exercises the alignment."""
    reward = torch.tensor([[1.0, 1.0, 1.0]])
    value = torch.tensor([[2.0, 3.0, 5.0, 7.0]])
    episode_id = torch.zeros(1, 4, dtype=torch.int64)

    advantages, returns = compute_gae(reward, value, episode_id, gamma=0.5, gae_lambda=1.0)

    assert returns[0].tolist() == pytest.approx((advantages[0] + value[0, :3]).tolist())


def test_gae_does_not_bootstrap_across_an_episode_boundary() -> None:
    """Step 1 ends an episode. Its advantage must be reward - value with no
    contribution from step 2, which belongs to a different episode."""
    reward = torch.tensor([[0.0, 1.0, 100.0]])
    value = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    episode_id = torch.tensor([[0, 0, 1, 1]])

    advantages, _ = compute_gae(reward, value, episode_id, gamma=0.5, gae_lambda=1.0)

    assert advantages[0, 1].item() == pytest.approx(1.0)


def test_gae_uses_the_bootstrap_slot_for_the_final_transition() -> None:
    """The last trained transition's next-state value comes from the extra
    slot, not from a zero."""
    reward = torch.tensor([[0.0]])
    value = torch.tensor([[0.0, 8.0]])
    episode_id = torch.zeros(1, 2, dtype=torch.int64)

    advantages, _ = compute_gae(reward, value, episode_id, gamma=0.5, gae_lambda=1.0)

    assert advantages[0, 0].item() == pytest.approx(4.0)


def test_gae_bootstraps_within_an_episode_across_its_own_final_observation() -> None:
    """Step 1 is the last observation of episode 0 -- a truncation, not a
    termination, since episode 1 begins fresh at step 2. The transition into
    step 1 (step 0 -> step 1) must still bootstrap from V(o_1): only the
    transition that actually spans the reset (step 1 -> step 2) may be cut.

    By hand: continues = [True, False]. running_1 = r1 - v1 = 2 - 4 = -2.
    delta_0 = r0 + gamma*v1*continues[0] - v0 = 0 + 0.5*4*1 - 0 = 2.
    running_0 = delta_0 + gamma*lambda*continues[0]*running_1
              = 2 + 0.5*1*1*(-2) = 1.0.
    An implementation that also zeroed continues[0] here -- treating step 1's
    being its episode's last observation as a reason not to bootstrap *into*
    it -- would instead give delta_0 = 0 - 0 = 0 and running_0 = 0.0."""
    reward = torch.tensor([[0.0, 2.0]])
    value = torch.tensor([[0.0, 4.0, 0.0]])
    episode_id = torch.tensor([[0, 0, 1]])

    advantages, _ = compute_gae(reward, value, episode_id, gamma=0.5, gae_lambda=1.0)

    assert advantages[0, 0].item() == pytest.approx(1.0)


def test_advantages_are_detached_even_when_value_requires_grad() -> None:
    """If the caller forgets to detach the critic's forward pass before
    calling compute_gae, the policy loss must still not be able to
    backpropagate through advantages into the critic -- that would silently
    train the critic to inflate its own advantages."""
    reward = torch.tensor([[1.0]])
    value = torch.tensor([[0.0, 0.0]], requires_grad=True)
    episode_id = torch.zeros(1, 2, dtype=torch.int64)

    advantages, _ = compute_gae(reward, value, episode_id, gamma=0.5, gae_lambda=1.0)

    assert advantages.requires_grad is False
