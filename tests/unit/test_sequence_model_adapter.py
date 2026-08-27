import pytest
import torch

from sequence_model.adapter import InputAdapter
from sequence_model.config import PolicyConfig


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )


@pytest.fixture
def adapter(tiny_config: PolicyConfig) -> InputAdapter:
    torch.manual_seed(0)
    return InputAdapter(tiny_config, torch.zeros(16), torch.ones(16))


def test_adapter_maps_a_timestep_to_d_model(adapter: InputAdapter) -> None:
    out = adapter(
        torch.randn(2, 3, 16), torch.randn(2, 3, 4),
        torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3),
    )

    assert tuple(out.shape) == (2, 3, 32)


def test_latent_stats_are_buffers_not_parameters(adapter: InputAdapter) -> None:
    """register_buffer, not a bare attribute: a bare tensor is invisible
    to .to(device) and state_dict(). And not a Parameter: these are fixed
    stats from latent_stats.json, and letting the optimizer move them
    reintroduces the non-stationarity the normalizer exists to remove."""
    buffer_names = sorted(n for n, _ in adapter.named_buffers())
    parameter_names = [n for n, _ in adapter.named_parameters()]

    assert buffer_names == ["latent_mean", "latent_std"]
    assert "latent_mean" not in parameter_names


def test_latent_stats_appear_in_state_dict(adapter: InputAdapter) -> None:
    assert "latent_mean" in adapter.state_dict()


def test_adapter_normalizes_the_latent_by_the_published_stats(
    tiny_config: PolicyConfig,
) -> None:
    """mean=2, std=4 turns a latent of 10 into (10-2)/4 = 2. Asserted by
    zeroing every other contribution and reading the projection input
    back through an identity-summing projection."""
    torch.manual_seed(0)
    adapter = InputAdapter(tiny_config, torch.full((16,), 2.0), torch.full((16,), 4.0))

    normalized = adapter.normalize_latent(torch.full((1, 1, 16), 10.0))

    assert normalized.flatten()[0].item() == pytest.approx(2.0, abs=1e-4)


def test_episode_start_action_has_its_own_embedding_row(adapter: InputAdapter) -> None:
    """Index 7 means "no previous action". Reusing action 0 (DOWN) there
    teaches the model that every episode begins with a press it never
    made."""
    zeros_latent = torch.zeros(1, 1, 16)
    zeros_aux = torch.zeros(1, 1, 4)
    zero_reward = torch.zeros(1, 1)

    as_start = adapter(zeros_latent, zeros_aux, torch.full((1, 1), 7, dtype=torch.long), zero_reward)
    as_down = adapter(zeros_latent, zeros_aux, torch.zeros(1, 1, dtype=torch.long), zero_reward)

    assert (as_start - as_down).abs().max().item() > 1e-6


def test_action_embedding_has_one_row_per_action_plus_episode_start(
    adapter: InputAdapter,
) -> None:
    assert tuple(adapter.action_embed.weight.shape) == (8, 4)
