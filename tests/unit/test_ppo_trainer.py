"""The outer loop: cadence, resume, and the abort guards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pytest
import torch

from pokemon_env.config import EnvConfig
from ppo import checkpoint as ppo_checkpoint
from ppo.config import PPOConfig
from ppo.normalizer import ReturnScaler
from ppo.trainer import PPODeps, run_training
from ppo.update import UpdateStats, run_update
from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy

from .fakes import FakeLatentEncoder, FakeVecEnv

_N_ENVS = 4
_MINIBATCH_ENVS = 2
_CONTEXT_LEN = 4
_BURN_IN = _CONTEXT_LEN - 1  # PPOConfig.burn_in(context_len)
_N_STEPS = 2


class FakeExperimentRun:
    """Hand-written fake typed against ExperimentRunLike. Records every
    logged dict and every (exit_code) `finish()` was called with, so tests
    can assert both without a Mock's auto-passing attributes."""

    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.finished_with: list[int] = []

    def log(self, metrics: dict) -> None:
        self.logged.append(metrics)

    def finish(self, exit_code: int = 0) -> None:
        self.finished_with.append(exit_code)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(exit_code=1 if exc_type is not None else 0)


@dataclass
class _TrainerHarness:
    deps: PPODeps
    vec_env: FakeVecEnv
    wandb_run: FakeExperimentRun
    directory: Path
    burn_in: int
    n_steps: int


def _stub_run_update(*, approx_kl: float, epoch1_dev: float):
    """A stand-in for `run_update` that skips the real forward/backward
    pass entirely, so the KL-abort and epoch-1-invariant tests can force an
    exact value onto exactly the two fields the trainer inspects -- wired in
    through `PPODeps.run_update`, never `mock.patch`."""

    def _run_update(
        policy, optimizer, scheduler, buffer, scaler, config, policy_config,
        n_envs, device, autocast_dtype,
    ) -> UpdateStats:
        return UpdateStats(
            policy_loss=0.0, value_loss=0.0, entropy=0.0, total_loss=0.0,
            clip_fraction=0.0, approx_kl=approx_kl,
            max_abs_ratio_dev_epoch1_mb1=epoch1_dev, max_abs_ratio_dev=epoch1_dev,
            explained_variance=0.0, staleness_logprob_l1=0.0, skipped_minibatches=0,
            grad_norm=0.0, policy_grad_norm=0.0, value_grad_norm=0.0,
        )

    return _run_update


def _trainer_harness(
    tmp_path: Path,
    *,
    checkpoint_every_updates: int = 25,
    forced_approx_kl: float | None = None,
    forced_epoch1_dev: float | None = None,
) -> _TrainerHarness:
    """A tiny real policy (matching the other ppo/ harnesses' shapes) plus a
    `FakeVecEnv`/`FakeLatentEncoder` and a `FakeExperimentRun`. `run_update`
    defaults to the real one; `forced_approx_kl`/`forced_epoch1_dev` swap in
    `_stub_run_update` instead, through `PPODeps.run_update`."""
    torch.manual_seed(0)
    policy_config = PolicyConfig(
        d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1,
        d_ff=64, context_len=_CONTEXT_LEN, latent_dim=8, aux_state_dim=4,
    )
    env_config = EnvConfig(n_envs=_N_ENVS)
    config = PPOConfig(
        frozen_encoder_revision="x",
        n_steps=_N_STEPS,
        n_epochs=1,
        minibatch_envs=_MINIBATCH_ENVS,
        checkpoint_dir=str(tmp_path),
        checkpoint_every_updates=checkpoint_every_updates,
    )
    policy = RecurrentTransformerPolicy(policy_config, torch.zeros(8), torch.ones(8))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    vec_env = FakeVecEnv(n_envs=_N_ENVS, aux_dim=policy_config.aux_state_dim, done_at_step=None)
    encoder = FakeLatentEncoder(latent_dim=policy_config.latent_dim, device=torch.device("cpu"))
    wandb_run = FakeExperimentRun()

    run_update_dep = run_update
    if forced_approx_kl is not None or forced_epoch1_dev is not None:
        run_update_dep = _stub_run_update(
            approx_kl=0.0 if forced_approx_kl is None else forced_approx_kl,
            epoch1_dev=0.0 if forced_epoch1_dev is None else forced_epoch1_dev,
        )

    deps = PPODeps(
        config=config,
        env_config=env_config,
        policy_config=policy_config,
        vec_env=vec_env,
        encoder=encoder,
        policy=policy,
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        autocast_dtype=torch.bfloat16,
        init_state_hash="deadbeef",
        git_commit="deadbeef",
        wandb_run=wandb_run,
        run_update=run_update_dep,
    )
    return _TrainerHarness(
        deps=deps, vec_env=vec_env, wandb_run=wandb_run, directory=tmp_path,
        burn_in=_BURN_IN, n_steps=_N_STEPS,
    )


def _write_seed_checkpoint(harness: _TrainerHarness, cache: object | None) -> None:
    """Writes a checkpoint directly (no `run_training` call), so the resume
    tests control exactly what `cache` the next `run_training` call finds --
    `None` for a checkpoint saved before the first rollout, or a real
    `RolloutCache` for one saved mid-run."""
    ppo_checkpoint.write_checkpoint(
        harness.directory, 0, 0, harness.deps.policy, harness.deps.optimizer, None,
        cache, harness.vec_env, ReturnScaler(), harness.deps.config,
        harness.deps.init_state_hash, None, "deadbeef",
    )


def test_the_warmup_runs_before_the_first_update(tmp_path) -> None:
    """burn_in observations must exist before update 0, or L varies and
    torch.compile recompiles on the second update. Capacity is
    burn_in + n_steps + 1: the warmup fills [0, burn_in] (burn_in + 1
    observations) and update 0's rollout writes n_steps more, landing
    exactly on the final slot."""
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=1)

    assert harness.vec_env.step_calls == harness.burn_in + 1 + harness.n_steps


def test_a_cold_start_resets_the_vec_env_exactly_once(tmp_path) -> None:
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=1)

    assert harness.vec_env.reset_calls == 1


def test_a_checkpoint_is_written_at_the_configured_cadence(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, checkpoint_every_updates=2)

    run_training(harness.deps, max_updates=4)

    assert len(list(tmp_path.glob("manifest_update*.json"))) == 2


def test_an_approx_kl_above_the_threshold_aborts_the_run(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, forced_approx_kl=1.0)

    with pytest.raises(RuntimeError, match="approx_kl"):
        run_training(harness.deps, max_updates=2)


def test_the_epoch_one_ratio_invariant_is_asserted_not_merely_logged(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, forced_epoch1_dev=0.01)

    with pytest.raises(AssertionError, match="epoch-1 minibatch-1 ratio"):
        run_training(harness.deps, max_updates=1)


def test_metrics_are_logged_once_per_update(tmp_path) -> None:
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=3)

    assert len(harness.wandb_run.logged) == 3


def test_the_wandb_run_is_finished_with_exit_code_zero_on_completion(tmp_path) -> None:
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=1)

    assert harness.wandb_run.finished_with == [0]


def test_the_wandb_run_is_finished_with_a_nonzero_exit_code_on_abort(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, forced_approx_kl=1.0)

    with pytest.raises(RuntimeError, match="approx_kl"):
        run_training(harness.deps, max_updates=1)

    assert harness.wandb_run.finished_with == [1]


def test_resuming_without_a_restored_cache_redoes_the_full_warmup(tmp_path) -> None:
    """`cache=None` is what a checkpoint saved before the first rollout (or
    one that deliberately dropped the cache) contains -- the run must warm
    up exactly as a cold start would, not skip straight to the trained
    region with an empty context."""
    seed_harness = _trainer_harness(tmp_path)
    _write_seed_checkpoint(seed_harness, cache=None)
    resumed_harness = _trainer_harness(tmp_path)

    run_training(resumed_harness.deps, max_updates=1)

    assert resumed_harness.vec_env.step_calls == resumed_harness.burn_in + 1 + resumed_harness.n_steps


def test_resuming_with_a_restored_cache_skips_the_burn_in_rebuild(tmp_path) -> None:
    """A restored cache already carries burn_in worth of real context, so
    redoing the burn_in-length rebuild would waste that many real (paid) env
    steps every single preemption. Only the one-step buffer seed is needed."""
    seed_harness = _trainer_harness(tmp_path)
    cache = seed_harness.deps.policy.new_cache(_N_ENVS, torch.device("cpu"), dtype=torch.bfloat16)
    _write_seed_checkpoint(seed_harness, cache=cache)
    resumed_harness = _trainer_harness(tmp_path)

    run_training(resumed_harness.deps, max_updates=1)

    assert resumed_harness.vec_env.step_calls == 1 + resumed_harness.n_steps


def test_resuming_with_a_restored_cache_does_not_reset_the_vec_env(tmp_path) -> None:
    """VecPokemonEnv.reset() reloads the emulator's init state -- calling it
    after a resume would discard the just-restored game position and every
    reward baseline describing progress from it."""
    seed_harness = _trainer_harness(tmp_path)
    cache = seed_harness.deps.policy.new_cache(_N_ENVS, torch.device("cpu"), dtype=torch.bfloat16)
    _write_seed_checkpoint(seed_harness, cache=cache)
    resumed_harness = _trainer_harness(tmp_path)

    run_training(resumed_harness.deps, max_updates=1)

    assert resumed_harness.vec_env.reset_calls == 0


def test_resuming_with_a_restored_cache_still_fills_the_buffer_before_the_update(tmp_path) -> None:
    """Skipping the burn_in rebuild must not also skip seeding the buffer's
    first trained slot -- by the time run_update is called, every update
    (cold-start or resumed) must have written the buffer up to capacity, or
    the trained region silently mis-pairs with stale/zeroed slots."""
    seed_harness = _trainer_harness(tmp_path)
    cache = seed_harness.deps.policy.new_cache(_N_ENVS, torch.device("cpu"), dtype=torch.bfloat16)
    _write_seed_checkpoint(seed_harness, cache=cache)
    resumed_harness = _trainer_harness(tmp_path)
    captured: dict = {}

    def _capturing_run_update(policy, optimizer, scheduler, buffer, *rest):
        captured["write_cursor"] = buffer.write_cursor
        captured["capacity"] = buffer.capacity
        return run_update(policy, optimizer, scheduler, buffer, *rest)

    resumed_harness.deps.run_update = _capturing_run_update

    run_training(resumed_harness.deps, max_updates=1)

    assert captured["write_cursor"] == captured["capacity"]
