"""Generalized Advantage Estimation over one update's trained region.

`value` carries T+1 entries: the T trained positions plus the bootstrap slot.
Episode boundaries are read from `episode_id`, not from `done` alone -- a
respawned worker also produces a discontinuity, and it arrives as a new
episode id rather than as a done flag on the previous step.

Every episode in this environment ends by truncation, never termination: the
step budget runs out, but the MDP itself never reaches a terminal state. That
means V(s_T) must always be bootstrapped when the trained region simply stops
mid-episode, and must never be bootstrapped across the one transition that
actually spans a reset (the last observation of an old episode to the first
observation of the next). `continues` draws exactly that line: it is False
only on the transition whose next step starts a new episode id, so the
transition *into* an episode's final observation still bootstraps normally
from that observation's value -- it is a real, valid state, just the last one
this rollout happened to record."""

from __future__ import annotations

import torch


def compute_gae(
    reward: torch.Tensor,
    value: torch.Tensor,
    episode_id: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`reward` (B, T), `value` (B, T+1), `episode_id` (B, T+1).

    Returns (advantages, returns), both (B, T). `returns` is
    `advantages + value[:, :T]`, which is the standard value target.

    `value` is detached before any arithmetic: both outputs are targets for
    other losses (the policy loss uses `advantages`, the critic regression
    uses `returns`), never something to backpropagate through. If the caller
    passes a `value` that still carries a graph from the critic's forward
    pass, the policy loss must not be able to reach back into the critic
    through it and inflate advantages."""
    value = value.detach()
    steps = reward.shape[1]
    # continues[t] is 0 where step t+1 belongs to a different episode, which
    # is exactly where the bootstrap must not cross.
    continues = (episode_id[:, 1:] == episode_id[:, :-1]).to(reward.dtype)

    advantages = torch.zeros_like(reward)
    running = torch.zeros_like(reward[:, 0])
    for t in range(steps - 1, -1, -1):
        delta = reward[:, t] + gamma * value[:, t + 1] * continues[:, t] - value[:, t]
        running = delta + gamma * gae_lambda * continues[:, t] * running
        advantages[:, t] = running
    return advantages, advantages + value[:, :steps]
