"""The update, and the invariant the whole design turns on."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import ppo.update as update_module
from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from ppo.normalizer import ReturnScaler
from ppo.update import run_update
from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy

# Captured once, before any test wraps ppo.update.ppo_losses in a spy, so every
# harness always wraps the real thing directly rather than a previous test's
# spy -- no unbounded nesting across the file's tests.
_REAL_PPO_LOSSES = update_module.ppo_losses


def _install_ppo_losses_spy(policy: RecurrentTransformerPolicy) -> dict:
    """Wraps the real ppo_losses so the harness can observe two things about
    run_update's OWN internal computation, not a hand reimplementation of it:
    every call's raw arguments, and -- for the first call only, while its
    loss tensors' graph is still alive -- an independently-computed split
    grad norm on those SAME tensors. Both of this spy's autograd.grad calls
    use retain_graph=True, so nothing here consumes graph run_update's own
    controller-addition-1 logic still needs a moment later."""
    state: dict = {"calls": [], "policy_grad_norm": None, "value_grad_norm": None}

    def spy(logits, value, action, logprob_old, advantage, value_target, config):
        state["calls"].append((logits, value, action, logprob_old, advantage, value_target, config))
        result = _REAL_PPO_LOSSES(
            logits, value, action, logprob_old, advantage, value_target, config
        )
        if state["policy_grad_norm"] is None:
            params = [p for p in policy.parameters() if p.requires_grad]
            policy_grads = torch.autograd.grad(
                result.policy, params, retain_graph=True, allow_unused=True
            )
            value_grads = torch.autograd.grad(
                config.vf_coef * result.value, params, retain_graph=True, allow_unused=True
            )
            state["policy_grad_norm"] = _independent_global_norm(policy_grads)
            state["value_grad_norm"] = _independent_global_norm(value_grads)
        return result

    update_module.ppo_losses = spy
    return state


def _independent_global_norm(grads: tuple[torch.Tensor | None, ...]) -> float:
    squared = [g.detach().float().pow(2).sum() for g in grads if g is not None]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


class CountingOptimizer:
    """Wraps a real AdamW so tests can assert step counts against a genuine
    optimizer rather than a bare Mock, whose auto-created attributes make a
    typo like `assert_called_once` silently pass forever."""

    def __init__(self, params) -> None:
        self._inner = torch.optim.AdamW(params, lr=1e-3)
        self.step_calls = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._inner.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self._inner.step()
        self.step_calls += 1


@dataclass
class _UpdateHarness:
    policy: RecurrentTransformerPolicy
    optimizer: CountingOptimizer
    buffer: RolloutBuffer
    scaler: ReturnScaler
    config: PPOConfig
    policy_config: PolicyConfig
    n_envs: int
    device: torch.device
    autocast_dtype: torch.dtype
    spy_state: dict

    def kwargs(self) -> dict:
        return {
            "policy": self.policy,
            "optimizer": self.optimizer,
            "scheduler": None,
            "buffer": self.buffer,
            "scaler": self.scaler,
            "config": self.config,
            "policy_config": self.policy_config,
            "n_envs": self.n_envs,
            "device": self.device,
            "autocast_dtype": self.autocast_dtype,
        }

    @property
    def advantage_std_seen(self) -> float:
        """The std of the advantage tensors ppo_losses actually received
        across epoch 1's minibatches, reassembled in minibatch order (which
        equals env-id order, per run_update's own pinned invariant).

        Reassembling the whole epoch rather than reading only the first call
        is deliberate: dividing one minibatch's advantages by its own (mean,
        std) forces that minibatch's std to ~1 regardless of whether
        normalization happened globally or locally -- a single-minibatch
        check cannot tell the two apart. Two independently-normalized groups
        do not recombine into a std of 1, though: each group's std used an
        (n-1) correction sized to that group, so pooling two such groups
        systematically undershoots 1. Reassembling pieces of one genuine
        global normalization has no such seam and reproduces the global
        tensor's std exactly. That gap is what makes this check discriminate."""
        n_minibatches = self.n_envs // self.config.minibatch_envs
        calls = self.spy_state["calls"][:n_minibatches]
        advantage = torch.cat([call[4] for call in calls], dim=0)
        return float(advantage.std())


def _update_harness(
    n_envs: int = 4,
    minibatch_envs: int = 2,
    n_epochs: int = 2,
    # Nonzero by default and deliberately so: it stands in for the staleness
    # the KV cache being carried across update boundaries always introduces
    # (see test_the_recomputed_logprobs_differ_from_the_rollout_recorded_ones).
    # A default of 0.0 would make the rollout-recorded logprob coincide with
    # the freshly recomputed one by construction, which would make
    # test_the_first_minibatch_of_epoch_one_has_a_ratio_of_exactly_one pass
    # even if run_update used the stale rollout-recorded logprob instead of
    # recomputing it -- exactly the bug that test exists to catch.
    rollout_logprob_offset: float = 0.1,
    nan_advantage: bool = False,
    max_nan: int = 3,
) -> _UpdateHarness:
    torch.manual_seed(0)
    device = torch.device("cpu")
    # bf16, matching CLAUDE.md's CPU/CUDA default (CPU autocast rejects fp32
    # outright). The harness's own _recompute_old call and run_update's
    # internal one both round through this same autocast context, so the two
    # stay bit-identical -- which is what the staleness test's 1e-5 tolerance
    # needs, not full fp32 precision.
    autocast_dtype = torch.bfloat16

    policy_config = PolicyConfig(
        d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1,
        d_ff=64, context_len=4, latent_dim=8, aux_state_dim=4,
    )
    config = PPOConfig(
        frozen_encoder_revision="x",
        n_steps=4,
        n_epochs=n_epochs,
        minibatch_envs=minibatch_envs,
        max_nan_minibatches_per_update=max_nan,
    )
    policy = RecurrentTransformerPolicy(policy_config, torch.zeros(8), torch.ones(8))
    buffer = RolloutBuffer(config, policy_config, n_envs, device)

    generator = torch.Generator().manual_seed(0)
    action_dim = policy_config.action_dim
    prev_action = torch.full((n_envs,), policy_config.episode_start_action, dtype=torch.int64)
    prev_reward = torch.zeros(n_envs)
    for slot in range(buffer.capacity):
        latent = torch.randn(n_envs, policy_config.latent_dim, generator=generator)
        aux = torch.randn(n_envs, policy_config.aux_state_dim, generator=generator)
        action = torch.randint(0, action_dim, (n_envs,), generator=generator)
        clean_reward = torch.rand(n_envs, generator=generator)
        # Only the STORED reward is corrupted -- prev_reward propagation stays
        # finite so a NaN GAE input doesn't also NaN the forward pass, which
        # would make the "non-finite loss" tests true for the wrong reason.
        stored_reward = clean_reward * float("nan") if nan_advantage else clean_reward
        buffer.write(
            slot=slot,
            latent=latent,
            aux=aux,
            action=action,
            prev_action=prev_action,
            prev_reward=prev_reward,
            reward=stored_reward,
            done=torch.zeros(n_envs, dtype=torch.bool),
            episode_id=torch.zeros(n_envs, dtype=torch.int64),
            abs_pos=torch.full((n_envs,), slot, dtype=torch.int64),
            logprob=torch.zeros(n_envs),
            value=torch.zeros(n_envs),
        )
        prev_action = action
        prev_reward = clean_reward

    env_order = torch.arange(n_envs, device=device)
    minibatches = env_order.split(minibatch_envs)
    true_logprob_old, _, _ = update_module._recompute_old(
        policy, buffer, minibatches, buffer.burn_in, buffer.trained_slice, autocast_dtype, device
    )
    buffer._logprob[:, buffer.trained_slice] = true_logprob_old - rollout_logprob_offset

    optimizer = CountingOptimizer(policy.parameters())
    spy_state = _install_ppo_losses_spy(policy)

    return _UpdateHarness(
        policy=policy,
        optimizer=optimizer,
        buffer=buffer,
        scaler=ReturnScaler(),
        config=config,
        policy_config=policy_config,
        n_envs=n_envs,
        device=device,
        autocast_dtype=autocast_dtype,
        spy_state=spy_state,
    )


def test_the_first_minibatch_of_epoch_one_has_a_ratio_of_exactly_one() -> None:
    """THE load-bearing test. pi_old is recomputed by a no_grad forward_chunk
    at update start, so before any optimizer step the ratio must be exactly 1.
    Using the rollout-recorded logprobs instead makes this fail."""
    harness = _update_harness()

    stats = run_update(**harness.kwargs())

    assert stats.max_abs_ratio_dev_epoch1_mb1 == pytest.approx(0.0, abs=1e-6)


def test_the_update_takes_one_optimizer_step_per_minibatch_per_epoch() -> None:
    harness = _update_harness(n_envs=4, minibatch_envs=2, n_epochs=3)

    run_update(**harness.kwargs())

    assert harness.optimizer.step_calls == 6


def test_the_recomputed_logprobs_differ_from_the_rollout_recorded_ones() -> None:
    """The staleness diagnostic that justified recomputing pi_old. It is
    reported, not asserted to be zero -- a nonzero value is the expected
    steady state, because the KV cache is carried across update boundaries."""
    harness = _update_harness(rollout_logprob_offset=0.25)

    stats = run_update(**harness.kwargs())

    assert stats.staleness_logprob_l1 == pytest.approx(0.25, abs=1e-5)


def test_a_non_finite_loss_skips_the_minibatch_without_stepping() -> None:
    harness = _update_harness(n_envs=2, minibatch_envs=2, n_epochs=1, nan_advantage=True)

    stats = run_update(**harness.kwargs())

    assert (stats.skipped_minibatches, harness.optimizer.step_calls) == (1, 0)


def test_too_many_non_finite_minibatches_abort_the_update() -> None:
    harness = _update_harness(
        n_envs=2, minibatch_envs=2, n_epochs=4, nan_advantage=True, max_nan=3
    )

    with pytest.raises(RuntimeError, match="non-finite loss in 3 minibatches"):
        run_update(**harness.kwargs())


def test_advantages_are_normalized_once_over_the_whole_update_batch() -> None:
    """Per-minibatch normalization would make each minibatch's targets depend
    on which envs happened to land in it."""
    harness = _update_harness(n_envs=4, minibatch_envs=2)

    run_update(**harness.kwargs())

    assert harness.advantage_std_seen == pytest.approx(1.0, abs=1e-3)


def test_the_split_grad_norms_match_an_independent_differentiation_of_the_same_tensors() -> None:
    """Controller addition 1. The harness's spy independently differentiates
    the exact loss.policy / loss.value tensors run_update produced for
    (epoch 1, minibatch 1), from outside, while their graph is still alive --
    not a hand reimplementation of the forward pass. run_update's own
    policy_grad_norm/value_grad_norm must match that independent
    differentiation of the same tensors."""
    harness = _update_harness()

    stats = run_update(**harness.kwargs())

    assert (stats.policy_grad_norm, stats.value_grad_norm) == pytest.approx(
        (harness.spy_state["policy_grad_norm"], harness.spy_state["value_grad_norm"])
    )
