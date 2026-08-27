import pytest
import torch

from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )


def test_empty_cache_has_no_valid_slots(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=3, device=torch.device("cpu"), dtype=torch.float32)

    assert cache.attention_mask().sum().item() == 0


def test_write_marks_exactly_one_slot_valid_per_env(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=3, device=torch.device("cpu"), dtype=torch.float32)
    k = torch.ones(3, 2, 1, 8)

    cache.write(layer=0, k_new=k, v_new=k)

    assert cache.attention_mask().sum().item() == 3


def test_write_returns_full_capacity_buffers(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=3, device=torch.device("cpu"), dtype=torch.float32)
    k = torch.ones(3, 2, 1, 8)

    k_all, v_all = cache.write(layer=0, k_new=k, v_new=k)

    assert (tuple(k_all.shape), tuple(v_all.shape)) == ((3, 2, 8, 8), (3, 2, 8, 8))


def test_write_places_the_new_key_at_the_current_write_slot(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=1, device=torch.device("cpu"), dtype=torch.float32)
    k_all, _ = cache.write(layer=0, k_new=torch.full((1, 2, 1, 8), 7.0), v_new=torch.zeros(1, 2, 1, 8))

    assert k_all[0, 0, 0].tolist() == [7.0] * 8


def test_advance_moves_the_write_slot_and_absolute_position(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=2, device=torch.device("cpu"), dtype=torch.float32)

    cache.advance()
    cache.advance()

    assert cache.abs_pos.tolist() == [2, 2]


def test_write_slot_wraps_around_at_capacity(tiny_config: PolicyConfig) -> None:
    """context_len=8, so the 9th write must land back in slot 0."""
    cache = RolloutCache.empty(tiny_config, n_envs=1, device=torch.device("cpu"), dtype=torch.float32)
    zeros = torch.zeros(1, 2, 1, 8)
    _fill_eight_slots(cache, zeros)

    k_all, _ = cache.write(layer=0, k_new=torch.full((1, 2, 1, 8), 9.0), v_new=zeros)

    assert k_all[0, 0, 0].tolist() == [9.0] * 8


def test_all_slots_valid_after_capacity_writes(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=1, device=torch.device("cpu"), dtype=torch.float32)
    _fill_eight_slots(cache, torch.zeros(1, 2, 1, 8))

    assert cache.attention_mask().sum().item() == 8


def test_reset_clears_validity_and_absolute_position_for_the_named_env(
    tiny_config: PolicyConfig,
) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=2, device=torch.device("cpu"), dtype=torch.float32)
    cache.write(layer=0, k_new=torch.ones(2, 2, 1, 8), v_new=torch.ones(2, 2, 1, 8))
    cache.advance()

    cache.reset(torch.tensor([True, False]))

    assert cache.attention_mask()[:, 0, 0].sum(dim=-1).tolist() == [0, 1]
    assert cache.abs_pos.tolist() == [0, 1]


def _fill_eight_slots(cache: RolloutCache, tensor: torch.Tensor) -> None:
    """Helper, not a test: eight writes take the tiny cache to capacity.
    Unrolled because test bodies may not contain loops."""
    for _ in range(8):
        cache.write(layer=0, k_new=tensor, v_new=tensor)
        cache.advance()
