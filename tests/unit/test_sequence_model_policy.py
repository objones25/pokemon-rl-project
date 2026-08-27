import pytest
import torch

from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )


@pytest.fixture
def policy(tiny_config: PolicyConfig) -> RecurrentTransformerPolicy:
    torch.manual_seed(0)
    return RecurrentTransformerPolicy(tiny_config, torch.zeros(16), torch.ones(16))


def test_step_returns_one_logit_per_action_and_one_value(
    policy: RecurrentTransformerPolicy,
) -> None:
    cache = policy.new_cache(n_envs=3, device=torch.device("cpu"))

    out = policy.step(
        torch.randn(3, 16), torch.randn(3, 4),
        torch.full((3,), 7, dtype=torch.long), torch.zeros(3), cache,
    )

    assert (tuple(out.logits.shape), tuple(out.value.shape)) == ((3, 7), (3,))


def test_incremental_step_matches_full_chunk_forward(
    policy: RecurrentTransformerPolicy,
) -> None:
    """THE load-bearing test. Five tokens pushed through step() one at a
    time must equal the same five through forward_chunk(). Catches
    is_causal=True during decode, ring-buffer indexing errors, RoPE
    position drift, and mask errors, all at once."""
    torch.manual_seed(1)
    latent, aux, action, reward = _episode(seq_len=5, n_envs=2)
    cache = policy.new_cache(n_envs=2, device=torch.device("cpu"))

    stepped = _run_rollout(policy, latent, aux, action, reward, cache)
    chunked = policy.forward_chunk(
        latent, aux, action, reward,
        abs_pos=torch.arange(5).expand(2, 5),
        episode_id=torch.zeros(2, 5, dtype=torch.long),
        burn_in=0,
    )

    assert (stepped - chunked.logits).abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_forward_chunk_with_burn_in_matches_recorded_rollout_outputs(
    policy: RecurrentTransformerPolicy,
) -> None:
    """The correctness claim of the burn-in scheme. context_len is 8, so
    a 15-token sequence with burn_in=7 exercises the sliding window: the
    last token attends to positions 7..14 in both paths."""
    torch.manual_seed(2)
    latent, aux, action, reward = _episode(seq_len=15, n_envs=2)
    cache = policy.new_cache(n_envs=2, device=torch.device("cpu"))

    stepped = _run_rollout(policy, latent, aux, action, reward, cache)
    chunked = policy.forward_chunk(
        latent, aux, action, reward,
        abs_pos=torch.arange(15).expand(2, 15),
        episode_id=torch.zeros(2, 15, dtype=torch.long),
        burn_in=7,
    )

    assert (stepped[:, 7:] - chunked.logits).abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_forward_chunk_returns_only_the_gradient_region(
    policy: RecurrentTransformerPolicy,
) -> None:
    latent, aux, action, reward = _episode(seq_len=15, n_envs=2)

    chunked = policy.forward_chunk(
        latent, aux, action, reward,
        abs_pos=torch.arange(15).expand(2, 15),
        episode_id=torch.zeros(2, 15, dtype=torch.long),
        burn_in=7,
    )

    assert tuple(chunked.logits.shape) == (2, 8, 7)


def test_cache_reset_makes_step_independent_of_pre_reset_history(
    policy: RecurrentTransformerPolicy,
) -> None:
    torch.manual_seed(3)
    latent, aux, action, reward = _episode(seq_len=4, n_envs=1)
    dirty = policy.new_cache(n_envs=1, device=torch.device("cpu"))
    _run_rollout(policy, latent, aux, action, reward, dirty)
    dirty.reset(torch.tensor([True]))
    fresh = policy.new_cache(n_envs=1, device=torch.device("cpu"))
    first = (latent[:, 0], aux[:, 0], action[:, 0], reward[:, 0])

    after_reset = policy.step(*first, dirty).logits
    from_fresh = policy.step(*first, fresh).logits

    assert (after_reset - from_fresh).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_episode_mask_blocks_attention_across_a_boundary(
    policy: RecurrentTransformerPolicy,
) -> None:
    """Replacing every pre-boundary token with noise must leave the
    post-boundary outputs untouched."""
    torch.manual_seed(4)
    latent, aux, action, reward = _episode(seq_len=6, n_envs=1)
    abs_pos = torch.tensor([[0, 1, 2, 0, 1, 2]])
    episode_id = torch.tensor([[0, 0, 0, 1, 1, 1]])
    noisy = latent.clone()
    noisy[:, :3] = torch.randn(1, 3, 16)

    clean_out = policy.forward_chunk(latent, aux, action, reward, abs_pos, episode_id, burn_in=3)
    noisy_out = policy.forward_chunk(noisy, aux, action, reward, abs_pos, episode_id, burn_in=3)

    assert (clean_out.logits - noisy_out.logits).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_actor_head_is_a_bare_linear_emitting_raw_logits(
    policy: RecurrentTransformerPolicy,
) -> None:
    """A softmax or sigmoid before CrossEntropyLoss trains a broken model
    silently rather than crashing."""
    assert isinstance(policy.actor, torch.nn.Linear)
    assert policy.actor.bias is None


def test_every_parameter_receives_gradient(policy: RecurrentTransformerPolicy) -> None:
    torch.manual_seed(5)
    latent, aux, action, reward = _episode(seq_len=5, n_envs=2)

    out = policy.forward_chunk(
        latent, aux, action, reward,
        abs_pos=torch.arange(5).expand(2, 5),
        episode_id=torch.zeros(2, 5, dtype=torch.long),
        burn_in=0,
    )
    (out.logits.sum() + out.value.sum()).backward()

    ungrated = sorted(n for n, p in policy.named_parameters() if p.grad is None)
    assert ungrated == []


def test_output_projections_are_scaled_by_one_over_sqrt_two_n_layers(
    tiny_config: PolicyConfig,
) -> None:
    """n_layers=2, so the target std is 0.02 / sqrt(4) = 0.01. Keeps the
    residual stream's variance flat with depth."""
    torch.manual_seed(0)
    policy = RecurrentTransformerPolicy(tiny_config, torch.zeros(16), torch.ones(16))

    observed = policy.blocks[0].attention.o_proj.weight.std().item()

    assert observed == pytest.approx(0.01, rel=0.15)


def test_step_runs_under_bf16_autocast_with_a_matching_bf16_cache(
    policy: RecurrentTransformerPolicy,
) -> None:
    """The PPO rollout loop passes torch.bfloat16 explicitly for the KV
    cache, which only works inside a matching autocast context: outside
    one, _project's q comes out float32 (RMSNorm's fp32 weight promotes
    it) while cache.write casts K/V to the cache dtype, and SDPA raises a
    dtype-mismatch RuntimeError. This also gives RMSNorm's float32-mean-
    square/dtype-round-trip its first non-fp32 test coverage."""
    cache = policy.new_cache(n_envs=3, device=torch.device("cpu"), dtype=torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = policy.step(
            torch.randn(3, 16), torch.randn(3, 4),
            torch.full((3,), 7, dtype=torch.long), torch.zeros(3), cache,
        )

    assert (tuple(out.logits.shape), tuple(out.value.shape)) == ((3, 7), (3,))


def test_step_matches_chunk_forward_across_a_cache_reset_episode_boundary(
    policy: RecurrentTransformerPolicy,
) -> None:
    """The rollout-side reset (cache.reset) and the chunk-side episode mask
    are each tested alone; PPO actually executes their composition. Drives
    step() across a 6-token rollout with cache.reset() fired after token 2
    (as the rollout loop would on `done`), then compares against
    forward_chunk given abs_pos that restarts at 0 and episode_id that
    increments -- exactly what the rollout would have recorded. Masks.py
    notes this composition arises "roughly once per 160 chunks", which is
    exactly the frequency at which a bug here would never be noticed."""
    torch.manual_seed(6)
    latent, aux, action, reward = _episode(seq_len=6, n_envs=1)
    cache = policy.new_cache(n_envs=1, device=torch.device("cpu"))

    stepped = _run_rollout_with_reset_at(policy, latent, aux, action, reward, cache, reset_at=3)
    chunked = policy.forward_chunk(
        latent, aux, action, reward,
        abs_pos=torch.tensor([[0, 1, 2, 0, 1, 2]]),
        episode_id=torch.tensor([[0, 0, 0, 1, 1, 1]]),
        burn_in=0,
    )

    assert (stepped - chunked.logits).abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_step_output_is_not_an_inference_tensor(policy: RecurrentTransformerPolicy) -> None:
    """step() is deliberately @torch.no_grad(), NOT @torch.inference_mode():
    rollout outputs become inputs to forward_chunk during the PPO update, and
    an inference tensor raises "Inference tensors cannot be saved for
    backward" the moment it enters autograd -- at the first update, on a paid
    GPU. requires_grad is False either way, so is_inference() is the only
    check that distinguishes them."""
    cache = policy.new_cache(n_envs=3, device=torch.device("cpu"))

    out = policy.step(
        torch.randn(3, 16), torch.randn(3, 4),
        torch.full((3,), 7, dtype=torch.long), torch.zeros(3), cache,
    )

    assert out.logits.is_inference() is False
    assert out.value.is_inference() is False


def _episode(seq_len: int, n_envs: int) -> tuple[torch.Tensor, ...]:
    """Helper, not a test: a synthetic (n_envs, seq_len) rollout."""
    return (
        torch.randn(n_envs, seq_len, 16),
        torch.randn(n_envs, seq_len, 4),
        torch.randint(0, 7, (n_envs, seq_len)),
        torch.randn(n_envs, seq_len),
    )


def _run_rollout(policy, latent, aux, action, reward, cache) -> torch.Tensor:
    """Helper, not a test: drives step() across the sequence and stacks
    the logits into (n_envs, seq_len, action_dim)."""
    collected = [
        policy.step(latent[:, t], aux[:, t], action[:, t], reward[:, t], cache).logits
        for t in range(latent.shape[1])
    ]
    return torch.stack(collected, dim=1)


def _run_rollout_with_reset_at(policy, latent, aux, action, reward, cache, reset_at) -> torch.Tensor:
    """Helper, not a test: like _run_rollout, but fires cache.reset(...) for
    every env immediately after processing token index `reset_at - 1`, as
    the rollout loop would on receiving `done=True` for that transition."""
    collected = []
    for t in range(latent.shape[1]):
        out = policy.step(latent[:, t], aux[:, t], action[:, t], reward[:, t], cache)
        collected.append(out.logits)
        if t == reset_at - 1:
            cache.reset(torch.ones(latent.shape[0], dtype=torch.bool))
    return torch.stack(collected, dim=1)
