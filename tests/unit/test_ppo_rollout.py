"""The rollout's ordering contract. Each of these guards a bug that produces
correctly-shaped tensors and a silently wrong model."""

from __future__ import annotations

import pytest
import torch

from ppo.rollout import collect_rollout

from .fakes import _rollout_harness


def test_the_cache_is_reset_after_the_terminal_step_not_before_the_next_one() -> None:
    """If reset ran first, the terminal observation would attend to a cleared
    cache and the last transition of every episode would be trained on the
    wrong context.

    `reset_calls_after_step` records `cache.advance_count` -- how many
    policy.step calls have completed -- at the moment reset() runs. Two
    steps (index 0 and the terminal index 1) must have completed their own
    policy.step by the time the terminal step's reset lands, so the
    correct value is 2. If reset ran before the terminal step's own
    policy.step instead, only step 0 would have completed and this would
    read 1 -- the mutation this test exists to catch."""
    harness = _rollout_harness(done_at_step=1)

    collect_rollout(**harness.kwargs(n_steps=3))

    assert harness.cache.reset_calls_after_step == [2]


def test_the_previous_action_becomes_episode_start_after_a_done() -> None:
    """Autoreset is next-step: the action taken at the terminal step is
    meaningless as context for the fresh episode that arrives at t+1."""
    harness = _rollout_harness(done_at_step=1)

    collect_rollout(**harness.kwargs(n_steps=3))

    assert harness.policy.prev_actions_seen[2].tolist() == [7, 7]


def test_the_previous_reward_is_zeroed_after_a_done() -> None:
    harness = _rollout_harness(done_at_step=1, reward=0.5)

    collect_rollout(**harness.kwargs(n_steps=3))

    assert harness.policy.prev_rewards_seen[2].tolist() == pytest.approx([0.0, 0.0])


def test_the_recorded_absolute_position_is_the_one_the_policy_used() -> None:
    """cache.abs_pos advances inside policy.step, so a snapshot taken after
    the call records the NEXT position and RoPE in the update no longer
    matches RoPE in the rollout."""
    harness = _rollout_harness(done_at_step=None)

    collect_rollout(**harness.kwargs(n_steps=3))
    chunk = harness.buffer.chunk(torch.tensor([0, 1]))

    assert chunk.abs_pos[0, harness.buffer.burn_in : harness.buffer.burn_in + 3].tolist() == [
        0,
        1,
        2,
    ]


def test_each_slots_reward_pays_for_the_action_recorded_in_the_previous_slot() -> None:
    """The alignment every downstream consumer has to know about, and which a
    constant reward makes invisible.

    Iteration t calls vec_env.step(prev_action), so the reward that comes back
    pays for a_{t-1}; the action sampled at o_t is then written to the SAME
    slot. With `reward = action + 1` and a scripted (3, 1, 5), the slots must
    read action (3, 1, 5) against reward (0+1, 3+1, 1+1) -- the initial
    prev_action is 0, and each later reward is one more than the PREVIOUS
    slot's action, never its own. compute_gae needs r(o_t, a_t), which
    therefore lives one slot forward."""
    harness = _rollout_harness(
        done_at_step=None, reward_from_action=True, action_script=(3, 1, 5)
    )

    collect_rollout(**harness.kwargs(n_steps=3))
    chunk = harness.buffer.chunk(torch.tensor([0, 1]))
    burn_in = harness.buffer.burn_in

    assert (
        chunk.action[0, burn_in : burn_in + 3].tolist(),
        chunk.reward[0, burn_in : burn_in + 3].tolist(),
    ) == ([3, 1, 5], [1.0, 4.0, 2.0])


def test_the_rollout_writes_exactly_n_steps_slots() -> None:
    harness = _rollout_harness(done_at_step=None)
    start = harness.buffer.write_cursor

    collect_rollout(**harness.kwargs(n_steps=3))

    assert harness.buffer.write_cursor - start == 3
