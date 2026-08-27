"""RolloutBuffer index arithmetic. Every silent bug in this trainer that is
not a ratio bug is an index bug."""

from __future__ import annotations

import pytest
import torch

from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from sequence_model.config import PolicyConfig


def _tiny_buffer() -> RolloutBuffer:
    """Helper, not a test: context_len 4 -> burn_in 3, n_steps 4, capacity 8."""
    return RolloutBuffer(
        config=PPOConfig(frozen_encoder_revision="x", n_steps=4),
        policy_config=PolicyConfig(
            d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1,
            d_ff=64, context_len=4, latent_dim=8, aux_state_dim=4,
        ),
        n_envs=2,
        device=torch.device("cpu"),
    )


def test_capacity_is_burn_in_plus_n_steps_plus_the_bootstrap_slot() -> None:
    assert _tiny_buffer().capacity == 8


def test_the_trained_slice_starts_after_the_burn_in_and_is_n_steps_long() -> None:
    buffer = _tiny_buffer()

    assert (buffer.trained_slice.start, buffer.trained_slice.stop) == (3, 7)


def test_chunk_returns_burn_in_plus_n_steps_plus_bootstrap_positions() -> None:
    buffer = _tiny_buffer()

    chunk = buffer.chunk(torch.tensor([0, 1]))

    assert chunk.latent.shape == (2, 8, 8)


def test_a_written_latent_is_readable_at_the_slot_it_was_written_to() -> None:
    buffer = _tiny_buffer()
    latent = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    buffer.write(slot=5, latent=latent, aux=torch.zeros(2, 4), action=torch.zeros(2, dtype=torch.int64),
                 prev_action=torch.zeros(2, dtype=torch.int64), prev_reward=torch.zeros(2),
                 reward=torch.zeros(2), done=torch.zeros(2, dtype=torch.bool),
                 episode_id=torch.zeros(2, dtype=torch.int64), abs_pos=torch.zeros(2, dtype=torch.int64),
                 logprob=torch.zeros(2), value=torch.zeros(2))

    assert buffer.chunk(torch.tensor([0, 1])).latent[1, 5].tolist() == pytest.approx(
        latent[1].to(torch.float16).float().tolist()
    )


def test_shift_moves_the_last_capacity_minus_n_steps_slots_to_the_front() -> None:
    """After a shift, the previous update's bootstrap observation must land at
    the first trained slot -- that is what makes every observation trained
    exactly once."""
    buffer = _tiny_buffer()
    marker = torch.full((2, 8), 9.0)
    buffer.write(slot=7, latent=marker, aux=torch.zeros(2, 4), action=torch.zeros(2, dtype=torch.int64),
                 prev_action=torch.zeros(2, dtype=torch.int64), prev_reward=torch.zeros(2),
                 reward=torch.zeros(2), done=torch.zeros(2, dtype=torch.bool),
                 episode_id=torch.zeros(2, dtype=torch.int64), abs_pos=torch.zeros(2, dtype=torch.int64),
                 logprob=torch.zeros(2), value=torch.zeros(2))

    buffer.shift()

    assert buffer.chunk(torch.tensor([0, 1])).latent[0, 3, 0].item() == pytest.approx(9.0)


def test_shift_sets_the_write_cursor_to_the_first_trained_slot_plus_one() -> None:
    buffer = _tiny_buffer()

    buffer.shift()

    assert buffer.write_cursor == 4
