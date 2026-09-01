"""The update, and the invariant the whole design turns on."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest
import torch

import ppo.update as update_module
from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from ppo.losses import ppo_losses
from ppo.normalizer import ReturnScaler
from ppo.update import run_update
from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy
from tests.conftest import PINNED_ENCODER_REVISION


def _install_ppo_losses_spy(policy: RecurrentTransformerPolicy, monkeypatch) -> dict:
    """Wraps the real ppo_losses so the harness can observe two things about
    run_update's OWN internal computation, not a hand reimplementation of it:
    every call's raw arguments, and -- for the first call only, while its
    loss tensors' graph is still alive -- an independently-computed split
    grad norm on those SAME tensors. Both of this spy's autograd.grad calls
    use retain_graph=True, so nothing here consumes graph run_update's own
    controller-addition-1 logic still needs a moment later.

    Installed through `monkeypatch`, so ppo.update.ppo_losses is restored at
    teardown. A bare module assignment leaves the spy in place for the rest of
    the session, which breaks "tests pass alone, in any order, in parallel"
    and makes the whole file's isolation depend on alphabetical ordering."""
    real_ppo_losses = update_module.ppo_losses
    state: dict = {"calls": [], "policy_grad_norm": None, "value_grad_norm": None}

    def spy(logits, value, action, logprob_old, advantage, value_target, config):
        state["calls"].append((logits, value, action, logprob_old, advantage, value_target, config))
        result = real_ppo_losses(
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

    monkeypatch.setattr("ppo.update.ppo_losses", spy)
    return state


def _install_compute_gae_spy(monkeypatch) -> dict:
    """Records the tensors run_update hands to compute_gae, so a test can pin
    what GAE actually consumed rather than only the numbers that fall out of
    it. Same monkeypatch-and-restore discipline as the ppo_losses spy."""
    real_compute_gae = update_module.compute_gae
    state: dict = {}

    def spy(reward, value, episode_id, gamma, gae_lambda):
        state["reward"] = reward.clone()
        state["value"] = value.clone()
        return real_compute_gae(reward, value, episode_id, gamma, gae_lambda)

    monkeypatch.setattr("ppo.update.compute_gae", spy)
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
    monkeypatch,
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
    # The ReturnScaler's scale at the moment run_update calls compute_gae.
    # 1.0 (an untouched scaler) leaves the normalized/raw units coincident,
    # which hides whether value_old is converted back out of normalized units
    # before being mixed with raw rewards.
    return_scale: float = 1.0,
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
        frozen_encoder_revision=PINNED_ENCODER_REVISION,
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
        # The reward stored at slot s pays for slot s-1's action, exactly as
        # the rollout writes it (see test_ppo_rollout.py's alignment test).
        # Deriving it from that action rather than drawing an independent
        # random number is what makes "which action did this reward pay for"
        # readable off the buffer, and therefore assertable downstream.
        clean_reward = prev_action.float() + 1.0
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
    spy_state = _install_ppo_losses_spy(policy, monkeypatch)

    scaler = ReturnScaler()
    # ReturnScaler.scale is sqrt(m2 / count); a large count keeps the update
    # this batch of returns performs from moving it materially, so the scale
    # run_update reads is the one this harness asked for.
    scaler.load_state_dict(
        {"count": 1e6, "mean": 0.0, "m2": return_scale * return_scale * 1e6}
    )

    return _UpdateHarness(
        policy=policy,
        optimizer=optimizer,
        buffer=buffer,
        scaler=scaler,
        config=config,
        policy_config=policy_config,
        n_envs=n_envs,
        device=device,
        autocast_dtype=autocast_dtype,
        spy_state=spy_state,
    )


def test_the_first_minibatch_of_epoch_one_has_a_ratio_of_exactly_one(monkeypatch) -> None:
    """THE load-bearing test. pi_old is recomputed by a no_grad forward_chunk
    at update start, so before any optimizer step the ratio must be exactly 1.
    Using the rollout-recorded logprobs instead makes this fail."""
    harness = _update_harness(monkeypatch)

    stats = run_update(**harness.kwargs())

    assert stats.max_abs_ratio_dev_epoch1_mb1 == pytest.approx(0.0, abs=1e-6)


def test_the_update_takes_one_optimizer_step_per_minibatch_per_epoch(monkeypatch) -> None:
    harness = _update_harness(monkeypatch, n_envs=4, minibatch_envs=2, n_epochs=3)

    run_update(**harness.kwargs())

    assert harness.optimizer.step_calls == 6


def test_the_recomputed_logprobs_differ_from_the_rollout_recorded_ones(monkeypatch) -> None:
    """The staleness diagnostic that justified recomputing pi_old. It is
    reported, not asserted to be zero -- a nonzero value is the expected
    steady state, because the KV cache is carried across update boundaries."""
    harness = _update_harness(monkeypatch, rollout_logprob_offset=0.25)

    stats = run_update(**harness.kwargs())

    assert stats.staleness_logprob_l1 == pytest.approx(0.25, abs=1e-5)


def test_a_non_finite_loss_skips_the_minibatch_without_stepping(monkeypatch) -> None:
    harness = _update_harness(monkeypatch, n_envs=2, minibatch_envs=2, n_epochs=1, nan_advantage=True)

    stats = run_update(**harness.kwargs())

    assert (stats.skipped_minibatches, harness.optimizer.step_calls) == (1, 0)


def test_too_many_non_finite_minibatches_abort_the_update(monkeypatch) -> None:
    harness = _update_harness(
        monkeypatch, n_envs=2, minibatch_envs=2, n_epochs=4, nan_advantage=True, max_nan=3
    )

    with pytest.raises(RuntimeError, match="non-finite loss in 3 minibatches"):
        run_update(**harness.kwargs())


def test_advantages_are_normalized_once_over_the_whole_update_batch(monkeypatch) -> None:
    """Per-minibatch normalization would make each minibatch's targets depend
    on which envs happened to land in it."""
    harness = _update_harness(monkeypatch, n_envs=4, minibatch_envs=2)

    run_update(**harness.kwargs())

    assert harness.advantage_std_seen == pytest.approx(1.0, abs=1e-3)


def test_the_split_grad_norms_match_an_independent_differentiation_of_the_same_tensors(monkeypatch) -> None:
    """Controller addition 1. The harness's spy independently differentiates
    the exact loss.policy / loss.value tensors run_update produced for
    (epoch 1, minibatch 1), from outside, while their graph is still alive --
    not a hand reimplementation of the forward pass. run_update's own
    policy_grad_norm/value_grad_norm must match that independent
    differentiation of the same tensors."""
    harness = _update_harness(monkeypatch)

    stats = run_update(**harness.kwargs())

    assert (stats.policy_grad_norm, stats.value_grad_norm) == pytest.approx(
        (harness.spy_state["policy_grad_norm"], harness.spy_state["value_grad_norm"])
    )


def test_grad_norm_max_captures_the_largest_minibatch_norm_not_just_the_last(
    monkeypatch,
) -> None:
    """train/grad_norm is overwritten every minibatch, so it reports only the
    LAST minibatch of the update -- a spike three minibatches earlier that a
    later, smaller-gradient minibatch happens to overwrite is invisible in
    that value. grad_norm_max exists to catch exactly that: the real
    clip_grad_norm_ still runs (so gradients are genuinely computed and
    clipped), but its reported return value is overridden with a controlled,
    non-monotonic sequence whose maximum is NOT the last call."""
    harness = _update_harness(monkeypatch, n_envs=4, minibatch_envs=2, n_epochs=2)
    real_clip = torch.nn.utils.clip_grad_norm_
    scripted = iter([1.0, 5.0, 2.0, 0.5])

    def fake_clip(parameters, max_norm):
        real_clip(parameters, max_norm)
        return next(scripted)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)

    stats = run_update(**harness.kwargs())

    assert (stats.grad_norm, stats.grad_norm_max) == pytest.approx((0.5, 5.0))


def test_gae_receives_the_reward_earned_by_each_trained_slots_own_action(monkeypatch) -> None:
    """compute_gae's delta_t = reward[t] + gamma*V[t+1] - V[t] is only correct
    when reward[t] is r(o_t, a_t). The rollout writes slot t as (o_t, a_t,
    reward for a_{t-1}), so the reward belonging to slot t's action lives at
    slot t+1. The harness stores `reward = previous slot's action + 1`,
    mirroring that exactly, so what GAE must see is each trained slot's OWN
    action plus one.

    Reading buffer[:, trained] instead trains the critic on a reward collected
    before o_t even existed and enters each action's own reward into its
    advantage at weight gamma*lambda rather than 1 -- with no crash, no shape
    change, and explained_variance still positive."""
    harness = _update_harness(monkeypatch)
    gae_spy = _install_compute_gae_spy(monkeypatch)
    trained = harness.buffer.trained_slice
    actions = harness.buffer.field("action", torch.arange(harness.n_envs))[:, trained]

    run_update(**harness.kwargs())

    assert torch.equal(gae_spy["reward"], actions.float() + 1.0)


def test_gae_receives_the_critics_value_converted_back_out_of_normalized_units(
    monkeypatch,
) -> None:
    """The critic regresses onto scaler.normalize(returns), so from update 1
    onward it emits values in units of returns/scale. Rewards are raw. Mixing
    the two inside GAE leaves the fixed point at delta_t ~= r_t*(1 - 1/scale):
    the baseline stops reducing variance and the effective value horizon
    collapses from gamma (~333 steps) to gamma*lambda (~15). Nothing shows it,
    because _explained_variance compares value_old to value_target and both
    are in the scaled space.

    value_old is recomputed here BEFORE run_update runs, while the weights are
    still the ones run_update will use for its own no_grad sweep."""
    harness = _update_harness(monkeypatch, return_scale=4.0)
    gae_spy = _install_compute_gae_spy(monkeypatch)
    minibatches = torch.arange(harness.n_envs).split(harness.config.minibatch_envs)
    _, value_old, _ = update_module._recompute_old(
        harness.policy, harness.buffer, minibatches, harness.buffer.burn_in,
        harness.buffer.trained_slice, harness.autocast_dtype, harness.device,
    )

    run_update(**harness.kwargs())

    assert gae_spy["value"].flatten().tolist() == pytest.approx(
        (value_old * 4.0).flatten().tolist()
    )


def test_a_skipped_non_finite_minibatch_is_logged_as_a_warning_with_its_reason(
    monkeypatch, caplog
) -> None:
    """A silently dropped minibatch is the trainer discarding real collected
    experience. On an unattended run the only other trace is a step count that
    quietly failed to advance, which nothing watches."""
    harness = _update_harness(
        monkeypatch, n_envs=2, minibatch_envs=2, n_epochs=1, nan_advantage=True
    )

    with caplog.at_level(logging.WARNING, logger="ppo.update"):
        run_update(**harness.kwargs())
    skipped = [r for r in caplog.records if r.message == "nan_minibatch_skipped"]

    assert [(r.levelname, r.skipped_this_update) for r in skipped] == [("WARNING", 1)]


def test_the_ppo_losses_spy_did_not_outlive_the_tests_that_installed_it() -> None:
    """Isolation, checked from the far end of the file. Every test above
    installs a spy over ppo.update.ppo_losses; a bare module assignment would
    leave the last one in place for the rest of the session, so any later test
    module -- in this run or a reordered/parallel one -- would silently
    exercise a spy instead of the real function. monkeypatch reverts it."""
    assert update_module.ppo_losses is ppo_losses
