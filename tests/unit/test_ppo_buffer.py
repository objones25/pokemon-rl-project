"""RolloutBuffer index arithmetic. Every silent bug in this trainer that is
not a ratio bug is an index bug."""

from __future__ import annotations

import pytest
import torch

from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from sequence_model.config import PolicyConfig
from tests.conftest import PINNED_ENCODER_REVISION


def _tiny_buffer(n_envs: int = 2) -> RolloutBuffer:
    """Helper, not a test: context_len 4 -> burn_in 3, n_steps 4, capacity 8."""
    return RolloutBuffer(
        config=PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, n_steps=4),
        policy_config=PolicyConfig(
            d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1,
            d_ff=64, context_len=4, latent_dim=8, aux_state_dim=4,
        ),
        n_envs=n_envs,
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


def test_chunk_returns_rows_in_the_requested_env_order_not_storage_order() -> None:
    """A subset-and-reorder request against 3 envs. If chunk() ever forgot to
    index by env_indices, this is the smallest case that would notice: with
    n_envs == len(env_indices) in identity order (the other tests), an
    unindexed return is indistinguishable from a correctly indexed one."""
    buffer = _tiny_buffer(n_envs=3)
    latents = torch.stack([torch.full((8,), float(env)) for env in range(3)])
    buffer.write(slot=0, latent=latents, aux=torch.zeros(3, 4), action=torch.zeros(3, dtype=torch.int64),
                 prev_action=torch.zeros(3, dtype=torch.int64), prev_reward=torch.zeros(3),
                 reward=torch.zeros(3), done=torch.zeros(3, dtype=torch.bool),
                 episode_id=torch.zeros(3, dtype=torch.int64), abs_pos=torch.zeros(3, dtype=torch.int64),
                 logprob=torch.zeros(3), value=torch.zeros(3))

    chunk = buffer.chunk(torch.tensor([2, 0]))

    assert chunk.latent[:, 0, 0].tolist() == pytest.approx([2.0, 0.0])


def test_written_latents_are_rounded_to_fp16_precision_not_stored_as_fp32() -> None:
    """0.1 is not exactly representable in fp16 or fp32, but the two roundings
    differ by ~2e-5 -- far outside pytest.approx's default tolerance -- so this
    is the smallest write that would notice fp32 storage silently replacing
    the fp16 budget the module exists for."""
    buffer = _tiny_buffer()
    latent = torch.full((2, 8), 0.1, dtype=torch.float32)
    buffer.write(slot=0, latent=latent, aux=torch.zeros(2, 4), action=torch.zeros(2, dtype=torch.int64),
                 prev_action=torch.zeros(2, dtype=torch.int64), prev_reward=torch.zeros(2),
                 reward=torch.zeros(2), done=torch.zeros(2, dtype=torch.bool),
                 episode_id=torch.zeros(2, dtype=torch.int64), abs_pos=torch.zeros(2, dtype=torch.int64),
                 logprob=torch.zeros(2), value=torch.zeros(2))

    stored = buffer.chunk(torch.tensor([0, 1])).latent[0, 0, 0].item()

    assert stored == pytest.approx(torch.tensor(0.1, dtype=torch.float16).item())


@pytest.mark.parametrize(
    "name",
    ["prev_action", "prev_reward", "abs_pos", "episode_id", "action", "reward", "done",
     "rollout_logprob", "rollout_value"],
)
def test_field_returns_exactly_what_the_chunks_matching_attribute_holds(name: str) -> None:
    """field() exists so a caller wanting one scalar column does not pay for a
    whole ChunkInputs -- whose fp32 latent copy is ~134 MB at production
    shapes. It must therefore agree with chunk() exactly, including the env
    reorder, or the two views of the buffer disagree."""
    buffer = _tiny_buffer(n_envs=3)
    latents = torch.stack([torch.full((8,), float(env)) for env in range(3)])
    buffer.write(slot=0, latent=latents, aux=torch.zeros(3, 4),
                 action=torch.tensor([1, 2, 3]), prev_action=torch.tensor([4, 5, 6]),
                 prev_reward=torch.tensor([0.5, 1.5, 2.5]), reward=torch.tensor([7.0, 8.0, 9.0]),
                 done=torch.tensor([True, False, True]), episode_id=torch.tensor([10, 11, 12]),
                 abs_pos=torch.tensor([13, 14, 15]), logprob=torch.tensor([-1.0, -2.0, -3.0]),
                 value=torch.tensor([0.25, 0.5, 0.75]))
    env_indices = torch.tensor([2, 0])

    field = buffer.field(name, env_indices)

    assert torch.equal(field, getattr(buffer.chunk(env_indices), name))


def test_field_does_not_build_the_fp32_latent_copy_a_chunk_would() -> None:
    """The whole point of field(): chunk().latent upcasts the fp16 store to
    fp32. If field went through chunk(), latent would come back fp32 here."""
    buffer = _tiny_buffer()

    assert buffer.field("latent", torch.tensor([0, 1])).dtype == torch.float16


def test_the_latent_storage_tensor_itself_is_fp16() -> None:
    """write()'s explicit .to(float16) cast means a value is fp16-rounded
    before it ever reaches the storage tensor, so the precision test above
    cannot notice the storage tensor's own dtype silently widening back to
    fp32 -- that mutation still rounds correctly and reads back correctly,
    it just doubles the 537 MB the module docstring budgets for. This test
    pins the storage dtype directly, which is the only way to catch that."""
    buffer = _tiny_buffer()

    assert buffer._latent.dtype == torch.float16
