"""The outer loop: cadence, resume, and the abort guards."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pytest
import torch

from pokemon_env.config import EnvConfig
from ppo import checkpoint as ppo_checkpoint
from ppo.config import PPOConfig
from ppo.normalizer import ReturnScaler
from ppo.trainer import (
    PPODeps,
    _record_best,
    _stalled_env_fraction,
    _visit_counts_as_image,
    run_training,
)
from ppo.update import UpdateStats, run_update
from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy
from tests.conftest import PINNED_ENCODER_REVISION

from .fakes import FakeLatentEncoder, FakeVecEnv

_N_ENVS = 4
_MINIBATCH_ENVS = 2
_CONTEXT_LEN = 4
_BURN_IN = _CONTEXT_LEN - 1  # PPOConfig.burn_in(context_len)
_N_STEPS = 2


class FakeExperimentRun:
    """Hand-written fake typed against ExperimentRunLike. Records every
    logged dict, every summary dict, and every (exit_code) `finish()` was
    called with, so tests can assert all three without a Mock's auto-passing
    attributes."""

    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.summaries: list[dict] = []
        self.finished_with: list[int] = []

    def log(self, metrics: dict) -> None:
        self.logged.append(metrics)

    def summary(self, metrics: dict) -> None:
        self.summaries.append(dict(metrics))

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


def _stub_run_update(
    *,
    approx_kl: float,
    epoch1_dev: float,
    forced_nan_abort: bool = False,
    # A single int is returned on every call; a list is consumed one value
    # per call (the last entry repeats once exhausted), so a test can
    # script a stall-then-recover-then-stall sequence across updates.
    minibatches_completed: int | list[int] = 24,
):
    """A stand-in for `run_update` that skips the real forward/backward
    pass entirely, so the KL-abort and epoch-1-invariant tests can force an
    exact value onto exactly the fields the trainer inspects -- wired in
    through `PPODeps.run_update`, never `mock.patch`. `forced_nan_abort`
    instead raises the same `RuntimeError` `run_update`'s own NaN-storm abort
    raises, at the default `max_nan_minibatches_per_update` threshold, so a
    test can prove that abort path reaches `run_training`'s log handler too."""
    calls = {"count": 0}

    def _run_update(
        policy, optimizer, scheduler, buffer, scaler, config, policy_config,
        n_envs, device, autocast_dtype,
    ) -> UpdateStats:
        if forced_nan_abort:
            raise RuntimeError(
                "non-finite loss in 3 minibatches of one update; aborting "
                "rather than stepping on corrupt gradients"
            )
        if isinstance(minibatches_completed, list):
            index = min(calls["count"], len(minibatches_completed) - 1)
            completed = minibatches_completed[index]
        else:
            completed = minibatches_completed
        calls["count"] += 1
        return UpdateStats(
            policy_loss=0.0, value_loss=0.0, entropy=0.0, total_loss=0.0,
            clip_fraction=0.0, approx_kl=approx_kl,
            max_abs_ratio_dev_epoch1_mb1=epoch1_dev, max_abs_ratio_dev=epoch1_dev,
            explained_variance=0.0, staleness_logprob_l1=0.0, skipped_minibatches=0,
            grad_norm=0.0, grad_norm_max=0.0, policy_grad_norm=0.0, value_grad_norm=0.0,
            target_kl_triggered=completed <= 1, minibatches_completed=completed,
            approx_kl_mean=approx_kl, clip_fraction_mean=0.0,
            ratio_abs_dev_p50=0.0, ratio_abs_dev_p95=0.0, ratio_abs_dev_p99=0.0,
            max_abs_ratio_dev_update=epoch1_dev, max_action_prob=0.0,
            raw_advantage_mean=0.0, raw_advantage_std=0.0, raw_advantage_abs_max=0.0,
            raw_advantage_top1_frac=0.0, raw_advantage_top1pct_frac=0.0,
        )

    return _run_update


def _trainer_harness(
    tmp_path: Path,
    *,
    checkpoint_every_updates: int = 25,
    artifact_every_updates: int = 25,
    forced_approx_kl: float | None = None,
    forced_epoch1_dev: float | None = None,
    forced_nan_abort: bool = False,
    forced_minibatches_completed: int | list[int] | None = None,
    target_kl: float | None = None,
    max_consecutive_stalled_updates: int = 10,
    steps_since_new_coord: int = 0,
    blackout_count: int = 0,
) -> _TrainerHarness:
    """A tiny real policy (matching the other ppo/ harnesses' shapes) plus a
    `FakeVecEnv`/`FakeLatentEncoder` and a `FakeExperimentRun`. `run_update`
    defaults to the real one; `forced_approx_kl`/`forced_epoch1_dev`/
    `forced_nan_abort`/`forced_minibatches_completed` swap in
    `_stub_run_update` instead, through `PPODeps.run_update`."""
    torch.manual_seed(0)
    policy_config = PolicyConfig(
        d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1,
        d_ff=64, context_len=_CONTEXT_LEN, latent_dim=8, aux_state_dim=4,
    )
    env_config = EnvConfig(n_envs=_N_ENVS)
    config = PPOConfig(
        frozen_encoder_revision=PINNED_ENCODER_REVISION,
        n_steps=_N_STEPS,
        n_epochs=1,
        minibatch_envs=_MINIBATCH_ENVS,
        checkpoint_dir=str(tmp_path),
        checkpoint_every_updates=checkpoint_every_updates,
        artifact_every_updates=artifact_every_updates,
        target_kl=target_kl,
        max_consecutive_stalled_updates=max_consecutive_stalled_updates,
    )
    policy = RecurrentTransformerPolicy(policy_config, torch.zeros(8), torch.ones(8))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    vec_env = FakeVecEnv(
        n_envs=_N_ENVS,
        aux_dim=policy_config.aux_state_dim,
        done_at_step=None,
        steps_since_new_coord=steps_since_new_coord,
        blackout_count=blackout_count,
    )
    encoder = FakeLatentEncoder(latent_dim=policy_config.latent_dim, device=torch.device("cpu"))
    wandb_run = FakeExperimentRun()

    run_update_dep = run_update
    if (
        forced_approx_kl is not None
        or forced_epoch1_dev is not None
        or forced_nan_abort
        or forced_minibatches_completed is not None
    ):
        run_update_dep = _stub_run_update(
            approx_kl=0.0 if forced_approx_kl is None else forced_approx_kl,
            epoch1_dev=0.0 if forced_epoch1_dev is None else forced_epoch1_dev,
            forced_nan_abort=forced_nan_abort,
            minibatches_completed=(
                24 if forced_minibatches_completed is None else forced_minibatches_completed
            ),
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


def test_a_rejected_minibatchs_approx_kl_does_not_abort_the_run_when_target_kl_is_set(
    tmp_path,
) -> None:
    """With target_kl enabled, a large approx_kl on the LAST-COMPUTED
    minibatch can describe one target_kl rejected -- never applied -- and
    no longer means the weights just moved that far. abort_approx_kl's old
    check must not fire off that value once target_kl is active, even
    when the value (1.0) would have aborted immediately under the old
    (target_kl=None) behavior in test_an_approx_kl_above_the_threshold_
    aborts_the_run above."""
    harness = _trainer_harness(
        tmp_path, forced_approx_kl=1.0, target_kl=0.02, forced_minibatches_completed=24
    )

    run_training(harness.deps, max_updates=2)

    # Both updates ran to completion (logged) rather than aborting partway.
    assert len(harness.wandb_run.logged) == 2


def test_updates_stalled_below_the_consecutive_threshold_do_not_abort(tmp_path) -> None:
    """minibatches_completed=1 (only the invariant-guaranteed first
    minibatch ever applies) on every update, but for fewer updates than
    max_consecutive_stalled_updates -- training continues."""
    harness = _trainer_harness(
        tmp_path,
        target_kl=0.02,
        max_consecutive_stalled_updates=3,
        forced_minibatches_completed=1,
    )

    run_training(harness.deps, max_updates=2)

    assert len(harness.wandb_run.logged) == 2


def test_updates_stalled_at_or_past_the_consecutive_threshold_abort(tmp_path) -> None:
    harness = _trainer_harness(
        tmp_path,
        target_kl=0.02,
        max_consecutive_stalled_updates=3,
        forced_minibatches_completed=1,
    )

    with pytest.raises(RuntimeError, match="minibatches_completed stayed at its floor"):
        run_training(harness.deps, max_updates=3)


def test_a_recovered_update_resets_the_consecutive_stall_counter(tmp_path) -> None:
    """Two stalls, then one healthy update, then three more stalls: with
    max_consecutive_stalled_updates=3, the counter must reset at the
    healthy update rather than accumulate across it -- 5 updates (2 stalls
    + 1 recovery + 2 more stalls) must not abort, since only 2 consecutive
    stalls exist after the reset; the 6th (3 consecutive again) does."""
    sequence = [1, 1, 24, 1, 1, 1]

    not_yet_enough = _trainer_harness(
        tmp_path / "not-yet-enough",
        target_kl=0.02,
        max_consecutive_stalled_updates=3,
        forced_minibatches_completed=sequence,
    )
    run_training(not_yet_enough.deps, max_updates=5)

    assert len(not_yet_enough.wandb_run.logged) == 5

    hits_threshold = _trainer_harness(
        tmp_path / "hits-threshold",
        target_kl=0.02,
        max_consecutive_stalled_updates=3,
        forced_minibatches_completed=sequence,
    )
    with pytest.raises(RuntimeError, match="minibatches_completed stayed at its floor"):
        run_training(hits_threshold.deps, max_updates=6)


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


def test_resuming_with_a_restored_cache_still_rebuilds_the_burn_in_prefix(tmp_path) -> None:
    """The buffer is not checkpointed, so a resume's [0, burn_in) slots are
    zero -- including their abs_pos and episode_id, which makes
    build_chunk_mask's window AND same-episode terms mask the prefix out
    entirely. Skipping the rebuild would train the first post-resume update on
    ~n_steps positions with near-zero context, while the behaviour policy that
    chose those actions had the restored cache's full context: pi_old would be
    recomputed under a different context than pi_behaviour."""
    seed_harness = _trainer_harness(tmp_path)
    cache = seed_harness.deps.policy.new_cache(_N_ENVS, torch.device("cpu"), dtype=torch.bfloat16)
    _write_seed_checkpoint(seed_harness, cache=cache)
    resumed_harness = _trainer_harness(tmp_path)

    run_training(resumed_harness.deps, max_updates=1)

    assert (
        resumed_harness.vec_env.step_calls
        == resumed_harness.burn_in + 1 + resumed_harness.n_steps
    )


def test_the_burn_in_prefix_a_resume_rebuilds_carries_real_absolute_positions(tmp_path) -> None:
    """The consequence the step count alone cannot show: with the prefix
    rebuilt, no trained position sees a zeroed abs_pos/episode_id slot, so
    build_chunk_mask's window admits the full context the behaviour policy
    had."""
    seed_harness = _trainer_harness(tmp_path)
    cache = seed_harness.deps.policy.new_cache(_N_ENVS, torch.device("cpu"), dtype=torch.bfloat16)
    _write_seed_checkpoint(seed_harness, cache=cache)
    resumed_harness = _trainer_harness(tmp_path)
    captured: dict = {}

    def _capturing_run_update(policy, optimizer, scheduler, buffer, *rest):
        captured["abs_pos"] = buffer.field("abs_pos", torch.tensor([0]))[0].tolist()
        return run_update(policy, optimizer, scheduler, buffer, *rest)

    resumed_harness.deps.run_update = _capturing_run_update

    run_training(resumed_harness.deps, max_updates=1)

    assert captured["abs_pos"] == [0, 1, 2, 3, 4, 5]


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


def test_the_leading_indicator_diagnostics_are_merged_into_the_logged_metrics(tmp_path) -> None:
    """RecurrentTransformerPolicy.diagnostics() exists solely for PPO, and the
    sequence-model spec is explicit that attention logit magnitude and residual
    norm move BEFORE loss and grad norm do. Unlogged, the run's earliest
    warning of divergence never reaches the dashboard."""
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=1)
    logged = harness.wandb_run.logged[0]

    assert sorted(k for k in logged if k.startswith(("attn/", "model/"))) == [
        "attn/dist_0",
        "attn/dist_1",
        "attn/dist_2-8",
        "attn/dist_257-1024",
        "attn/dist_65-256",
        "attn/dist_9-64",
        "attn/logit_max",
        "model/residual_norm",
    ]


def test_the_two_visual_artifacts_are_logged_on_the_artifact_cadence(tmp_path) -> None:
    """The exploration heatmap and the frame contact sheet: the two images
    that tell a human whether all 64 agents are stuck in the same menu,
    without reading a log line."""
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=1)
    logged = harness.wandb_run.logged[0]

    assert [type(logged["explore/heatmap"]).__name__, type(logged["env/contact_sheet"]).__name__] == [
        "Image",
        "Image",
    ]


def test_no_artifact_is_logged_on_an_update_off_the_artifact_cadence(tmp_path) -> None:
    """Rendering both images and a full attention matrix every update would
    cost far more than the diagnostics are worth; the cadence is the point."""
    harness = _trainer_harness(tmp_path, artifact_every_updates=2)

    run_training(harness.deps, max_updates=2)

    assert "explore/heatmap" not in harness.wandb_run.logged[1]


def test_rollout_and_update_time_are_logged_separately(tmp_path) -> None:
    """perf/iteration_s used to be one combined timer, so a slow iteration
    could not be attributed to either the env or the GPU-side update pass."""
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=1)
    logged = harness.wandb_run.logged[0]

    assert logged["perf/rollout_s"] >= 0.0
    assert logged["perf/update_s"] >= 0.0
    assert logged["perf/iteration_s"] == pytest.approx(
        logged["perf/rollout_s"] + logged["perf/update_s"]
    )


def test_checkpoint_write_time_is_logged_only_on_the_checkpoint_cadence(tmp_path) -> None:
    """perf/checkpoint_s folds into the SAME update's metrics dict rather
    than a second wandb_run.log() call, which would break the "one log()
    call per update" invariant on a checkpoint-cadence update."""
    harness = _trainer_harness(tmp_path, checkpoint_every_updates=2)

    run_training(harness.deps, max_updates=4)

    assert [
        "perf/checkpoint_s" in logged for logged in harness.wandb_run.logged
    ] == [True, False, True, False]
    assert len(harness.wandb_run.logged) == 4


def test_diagnostics_time_is_logged_only_on_the_artifact_cadence(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, artifact_every_updates=2)

    run_training(harness.deps, max_updates=4)

    assert [
        "perf/diagnostics_s" in logged for logged in harness.wandb_run.logged
    ] == [True, False, True, False]


def test_the_return_scalers_scale_is_logged(tmp_path) -> None:
    """The running std that rescales value targets -- if it shifts sharply,
    that is currently invisible even though it directly changes what the
    critic is regressed onto."""
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=1)
    logged = harness.wandb_run.logged[0]

    assert logged["train/return_scale"] > 0.0


def test_cumulative_wall_clock_hours_is_logged_and_increases_across_updates(tmp_path) -> None:
    """Nothing else on the dashboard answers "how many paid GPU-hours has
    this run cost so far"."""
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=2)
    logged = harness.wandb_run.logged

    assert logged[0]["perf/wall_clock_hours"] > 0.0
    assert logged[1]["perf/wall_clock_hours"] >= logged[0]["perf/wall_clock_hours"]


def test_respawn_total_and_delta_are_both_logged(tmp_path) -> None:
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=1)
    logged = harness.wandb_run.logged[0]

    assert (logged["env/worker_respawns_total"], logged["env/worker_respawns_delta"]) == (0.0, 0.0)


def test_stalled_frac_is_logged_as_zero_when_every_env_is_still_exploring(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, steps_since_new_coord=0)

    run_training(harness.deps, max_updates=1)
    logged = harness.wandb_run.logged[0]

    assert logged["env/stalled_frac"] == pytest.approx(0.0)


def test_stalled_frac_is_logged_as_one_when_every_env_found_nothing_new_this_rollout(
    tmp_path,
) -> None:
    """steps_since_new_coord reaching n_steps (_N_STEPS here) means the env
    found zero new coordinates across the entire rollout just collected --
    the menu-loop behaviour the contact sheet showed visually."""
    harness = _trainer_harness(tmp_path, steps_since_new_coord=_N_STEPS)

    run_training(harness.deps, max_updates=1)
    logged = harness.wandb_run.logged[0]

    assert logged["env/stalled_frac"] == pytest.approx(1.0)


def test_the_running_bests_are_written_to_the_wandb_summary(tmp_path) -> None:
    """Set explicitly rather than left as last-value: W&B's summary column
    otherwise shows whatever the final update logged, which on a 48-hour run
    is the least interesting number in the history."""
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=1)

    assert harness.wandb_run.summaries == [
        {"best/badges": 0.0, "best/unique_coords": 0.0, "best/reward_mean": 0.0}
    ]


def test_record_best_keeps_the_maximum_seen_rather_than_the_latest() -> None:
    """The pure part of the summary, where "best" is actually decided. Folding
    a WORSE second update in must not lower any of the three."""
    best: dict[str, float] = {}

    _record_best(
        best,
        {"progress/badges_max": 3.0, "explore/unique_coords_total": 40.0, "reward/mean": 0.5},
    )
    _record_best(
        best,
        {"progress/badges_max": 1.0, "explore/unique_coords_total": 90.0, "reward/mean": 0.25},
    )

    assert best == {"best/badges": 3.0, "best/unique_coords": 90.0, "best/reward_mean": 0.5}


def test_the_run_start_is_logged_with_the_three_config_dataclasses(tmp_path, caplog) -> None:
    """The JSON-lines record that says what this run actually was. Without it,
    a log file recovered from a preempted pod cannot be matched to the
    hyperparameters that produced it."""
    harness = _trainer_harness(tmp_path)

    with caplog.at_level(logging.INFO, logger="ppo.trainer"):
        run_training(harness.deps, max_updates=1)
    started = [r for r in caplog.records if r.message == "run_started"]

    assert [sorted(k for k in vars(started[0]) if k.endswith("_config")) for _ in started] == [
        ["env_config", "policy_config", "ppo_config"]
    ]


def test_an_abort_logs_an_error_carrying_the_traceback(tmp_path, caplog) -> None:
    """An unattended run that dies at hour 30 leaves only its log. Without
    exc_info the record says a run stopped, not which invariant broke."""
    harness = _trainer_harness(tmp_path, forced_approx_kl=1.0)

    with (
        caplog.at_level(logging.ERROR, logger="ppo.trainer"),
        pytest.raises(RuntimeError, match="approx_kl .* exceeded"),
    ):
        run_training(harness.deps, max_updates=1)
    aborted = [r for r in caplog.records if r.message == "training_aborted"]

    assert [r.exc_info[0] for r in aborted] == [RuntimeError]


def test_the_epoch_one_ratio_abort_also_logs_an_error_carrying_the_traceback(
    tmp_path, caplog
) -> None:
    """The other abort path -- an AssertionError, not a RuntimeError -- must
    reach the same handler, or the one failure the whole design turns on is
    the one that leaves no diagnosis behind."""
    harness = _trainer_harness(tmp_path, forced_epoch1_dev=0.01)

    with (
        caplog.at_level(logging.ERROR, logger="ppo.trainer"),
        pytest.raises(AssertionError, match="epoch-1 minibatch-1 ratio deviated"),
    ):
        run_training(harness.deps, max_updates=1)
    aborted = [r for r in caplog.records if r.message == "training_aborted"]

    assert [r.exc_info[0] for r in aborted] == [AssertionError]


def test_a_nan_storm_abort_also_logs_an_error_carrying_the_traceback(
    tmp_path, caplog
) -> None:
    """The third abort path -- run_update's own NaN-storm abort -- used to raise
    RuntimeError from outside the try block that wraps _check_abort_conditions,
    at src/ppo/trainer.py's `stats = deps.run_update(...)` call. Before the fix
    this propagates with no training_aborted record; the assertion on the log
    record (not just pytest.raises) is what would have caught that, since the
    exception already reaches the caller either way."""
    harness = _trainer_harness(tmp_path, forced_nan_abort=True)

    with (
        caplog.at_level(logging.ERROR, logger="ppo.trainer"),
        pytest.raises(RuntimeError, match="non-finite loss"),
    ):
        run_training(harness.deps, max_updates=1)
    aborted = [r for r in caplog.records if r.message == "training_aborted"]

    assert [r.exc_info[0] for r in aborted] == [RuntimeError]


def test_visit_counts_are_peak_normalized_so_a_single_visit_is_visible() -> None:
    """A plain uint32 -> uint8 cast renders a tile visited once as value 1,
    indistinguishable from black -- the heatmap would look empty exactly when
    exploration is only starting, which is when it is most worth looking at."""
    counts = np.array([[0, 1], [2, 4]], dtype=np.uint32)

    image = _visit_counts_as_image(counts)

    assert image.tolist() == [[0, 63], [127, 255]]


def test_an_all_zero_visit_map_stays_all_zero_rather_than_dividing_by_its_peak() -> None:
    counts = np.zeros((2, 2), dtype=np.uint32)

    image = _visit_counts_as_image(counts)

    assert image.tolist() == [[0, 0], [0, 0]]


def _stats_entry(steps_since_new_coord: int) -> dict:
    return {"steps_since_new_coord": steps_since_new_coord}


def test_stalled_env_fraction_counts_envs_that_found_nothing_new_this_rollout() -> None:
    """An env whose steps_since_new_coord has reached n_steps contributed
    zero new coordinates to the entire rollout just collected -- exactly the
    menu-loop behaviour the contact sheet showed visually."""
    stats = [_stats_entry(0), _stats_entry(500), _stats_entry(1024), _stats_entry(2000)]

    fraction = _stalled_env_fraction(stats, n_steps=1024)

    assert fraction == pytest.approx(0.5)


def test_stalled_env_fraction_is_zero_when_every_env_found_new_ground() -> None:
    stats = [_stats_entry(0), _stats_entry(100), _stats_entry(1023)]

    fraction = _stalled_env_fraction(stats, n_steps=1024)

    assert fraction == pytest.approx(0.0)


def test_blackout_count_total_is_logged(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, blackout_count=3)

    run_training(harness.deps, max_updates=1)
    logged = harness.wandb_run.logged[0]

    assert logged["env/blackout_count_total"] == pytest.approx(3.0 * _N_ENVS)
