"""The outer PPO loop.

Order per iteration: rollout -> update -> telemetry assembled -> cadence work
(diagnostics + artifacts every artifact_every_updates, checkpoint every
checkpoint_every_updates, both folding their own timing into the same
metrics dict) -> one wandb_run.log() call -> buffer shift. Checkpoint writing
happens before that single log() call specifically so perf/checkpoint_s
lands in the same update's row rather than needing a second log() call,
which would break "one log() per update."

The spec's periodic Hub snapshot is deliberately NOT here: it needs an upload
client and a credential path this trainer is not given, and
`hub_snapshot_every_updates` is consequently unread. Recorded as a deferral in
the design spec's handoff rather than half-built.

The buffer shift happens LAST, so a checkpoint written mid-iteration describes
a buffer state the next resume can reproduce by collecting n_steps fresh
observations.

The burn-in warmup runs on EVERY start, cold or resumed, restored KV cache or
not. The buffer is not checkpointed, so on a resume its [0, burn_in) slots are
zero -- and their abs_pos and episode_id are zero too, while the trained slots
carry real values, so build_chunk_mask's window AND same-episode terms mask
that prefix out entirely. The first post-resume update would then train n_steps
positions with near-zero context while the behaviour policy that chose those
actions had the restored cache's full context, so pi_old would be recomputed
under a different context than pi_behaviour. Rebuilding costs burn_in real env
steps (~0.6% of one episode) once per preemption, and the restored cache is
still doing work throughout -- policy.step serves from it while those
observations are collected.

A cold start also calls vec_env.reset() once, to get the emulator into its
initial state and start episode counting; a resume must NOT call it, because
VecPokemonEnv.reset() reloads the emulator's init state and would discard the
game position resume() just restored, along with every reward baseline that
describes progress from it."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
import wandb

from observability.tracking import ExperimentRunLike, NullExperimentRun
from pokemon_env.config import EnvConfig
from pokemon_env.telemetry import contact_sheet, exploration_heatmap, rollout_metrics
from ppo import checkpoint as ppo_checkpoint
from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from ppo.normalizer import ReturnScaler
from ppo.rollout import (
    CacheProtocol,
    EncoderProtocol,
    PolicyProtocol,
    RolloutState,
    VecEnvProtocol,
    collect_rollout,
)
from ppo.telemetry import update_metrics
from ppo.update import run_update
from sequence_model.config import PolicyConfig

if TYPE_CHECKING:
    from pokemon_env.vec_env import VecStep

logger = logging.getLogger(__name__)


class TrainerVecEnv(VecEnvProtocol, Protocol):
    """What run_training needs beyond collect_rollout's narrower surface.

    Declared here rather than widening `VecEnvProtocol`, whose documented job
    is "what collect_rollout needs" -- a rollout that could suddenly reach
    `stats()` or `last_components` would be a different, larger contract."""

    @property
    def last_step(self) -> VecStep | None: ...

    @property
    def last_components(self) -> dict[str, float]: ...

    @property
    def clip_fire_rate(self) -> float: ...

    def reset(self) -> VecStep: ...

    def stats(self) -> list[dict]: ...


class TrainerPolicy(PolicyProtocol, Protocol):
    """What run_training needs beyond the single `step` collect_rollout uses."""

    def new_cache(
        self, n_envs: int, device: torch.device, dtype: torch.dtype = ...
    ) -> CacheProtocol: ...

    def diagnostics(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        abs_pos: torch.Tensor,
        episode_id: torch.Tensor,
        layer: int = ...,
    ) -> dict[str, float]: ...

    def eval(self) -> TrainerPolicy: ...

    def train(self, mode: bool = ...) -> TrainerPolicy: ...


class OptimizerProtocol(Protocol):
    """Structural, not `torch.optim.Optimizer`, so a test double that wraps a
    real optimizer to count steps still satisfies it."""

    param_groups: list[dict]

    def zero_grad(self, set_to_none: bool = ...) -> None: ...

    def step(self) -> None: ...

    def state_dict(self) -> dict: ...

# W&B summary keys, and the per-update metric each one takes its running best
# from. Set explicitly rather than left as last-value: on a 48-hour run the
# final update's numbers are the least interesting ones in the history.
_BEST_SOURCES = {
    "best/badges": "progress/badges_max",
    "best/unique_coords": "explore/unique_coords_total",
    "best/reward_mean": "reward/mean",
}


@dataclass
class PPODeps:
    config: PPOConfig
    env_config: EnvConfig
    policy_config: PolicyConfig
    vec_env: TrainerVecEnv
    encoder: EncoderProtocol
    policy: TrainerPolicy
    optimizer: OptimizerProtocol
    scheduler: object | None
    device: torch.device
    autocast_dtype: torch.dtype
    init_state_hash: str
    git_commit: str
    wandb_run: ExperimentRunLike = field(default_factory=NullExperimentRun)
    run_update: Callable = run_update


def run_training(deps: PPODeps, max_updates: int | None = None) -> None:
    # Wraps validation too: the caller's W&B run already exists on the
    # dashboard by the time this is called, so even a config error must
    # still call finish() rather than leave that run marked "still running".
    with deps.wandb_run:
        deps.config.validate_against_n_envs(deps.env_config.n_envs)
        _run_training(deps, deps.config, max_updates)


def _run_training(deps: PPODeps, config: PPOConfig, max_updates: int | None) -> None:
    # Per-process, not cumulative across preemptions: time.monotonic() resets
    # on restart and no checkpoint state carries a prior segment's total. On a
    # resumed run this is "hours since this segment started," not "hours this
    # run has cost in total" -- still the number that answers "is this
    # iteration cadence normal" without reading a wall clock by hand.
    run_started = time.monotonic()
    logger.info(
        "run_started",
        extra={
            "ppo_config": asdict(config),
            "env_config": asdict(deps.env_config),
            "policy_config": asdict(deps.policy_config),
            "git_commit": deps.git_commit,
            "init_state_hash": deps.init_state_hash,
            "device": str(deps.device),
            "autocast_dtype": str(deps.autocast_dtype),
            "wandb_run_id": _run_id(deps.wandb_run),
        },
    )
    directory = Path(config.checkpoint_dir)
    scaler = ReturnScaler()
    buffer = RolloutBuffer(config, deps.policy_config, deps.env_config.n_envs, deps.device)
    cache = deps.policy.new_cache(deps.env_config.n_envs, deps.device, dtype=deps.autocast_dtype)
    generator = torch.Generator(device=deps.device).manual_seed(config.seed)

    resumed = ppo_checkpoint.resume(
        directory, deps.policy, deps.optimizer, deps.scheduler, deps.vec_env,
        scaler, deps.policy_config, config, deps.init_state_hash,
    )
    update = 0
    global_step = 0
    if resumed is not None:
        update, global_step = resumed.update + 1, resumed.global_step
        if resumed.cache is not None:
            cache = resumed.cache
        logger.info("resumed", extra={"update": update, "global_step": global_step})
    else:
        # Cold start only: a resume must not reload the emulator's init
        # state over the game position resume() just restored.
        deps.vec_env.reset()

    start_update = update
    state = RolloutState(
        prev_action=torch.full(
            (deps.env_config.n_envs,),
            deps.policy_config.episode_start_action,
            dtype=torch.int64,
            device=deps.device,
        ),
        prev_reward=torch.zeros(deps.env_config.n_envs, device=deps.device),
    )

    rollout_kwargs = {
        "vec_env": deps.vec_env, "encoder": deps.encoder, "policy": deps.policy,
        "cache": cache, "buffer": buffer, "generator": generator,
        "device": deps.device,
        "episode_start_action": deps.policy_config.episode_start_action,
        "autocast_dtype": deps.autocast_dtype,
    }

    # Unconditional, resume included: the buffer is not checkpointed, so its
    # burn-in prefix is zeroed on every start and a restored cache cannot fill
    # it. See this module's docstring.
    state = collect_rollout(state=state, n_steps=buffer.burn_in, **rollout_kwargs)
    logger.info("warmup_started", extra={"steps": buffer.burn_in})

    # The trained region's first slot. Every later rollout inherits this same
    # slot from the previous update's shift() instead.
    state = collect_rollout(state=state, n_steps=1, **rollout_kwargs)
    logger.info("warmup_complete", extra={"steps": buffer.write_cursor})

    best: dict[str, float] = {}
    consecutive_stalled = 0
    while max_updates is None or update < start_update + max_updates:
        if deps.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Split, not one combined timer: a slow iteration is otherwise
        # indistinguishable between the 64-subprocess env stalling and the
        # GPU-side forward/backward -- the "dataloader starvation" failure
        # mode a single perf/iteration_s number hides.
        respawns_before = _respawns(deps.vec_env)
        rollout_started = time.monotonic()
        state = collect_rollout(state=state, n_steps=config.n_steps, **rollout_kwargs)
        rollout_s = time.monotonic() - rollout_started
        global_step += config.n_steps * deps.env_config.n_envs

        # All four abort paths -- the epoch-1 ratio invariant, the
        # approx_kl threshold (target_kl disabled) or consecutive-stall
        # count (target_kl enabled, see _check_abort_conditions), and
        # run_update's own NaN-storm abort -- log with exc_info before
        # propagating: this is a 48-hour unattended run, and the
        # difference between "died at hour 30 on the ratio invariant" and
        # "died at hour 30 on KL" is the whole diagnosis. The exception
        # itself still propagates untouched.
        update_started = time.monotonic()
        try:
            stats = deps.run_update(
                deps.policy, deps.optimizer, deps.scheduler, buffer, scaler, config,
                deps.policy_config, deps.env_config.n_envs, deps.device, deps.autocast_dtype,
            )
            consecutive_stalled = _check_abort_conditions(stats, config, consecutive_stalled)
        except (AssertionError, RuntimeError):
            logger.exception(
                "training_aborted",
                extra={"update": update, "global_step": global_step},
            )
            raise
        update_s = time.monotonic() - update_started

        # collect_rollout just advanced the env n_steps times, so last_step
        # cannot be None here. Asserted rather than widening rollout_metrics
        # to accept an Optional it would only ever have to reject.
        last_step = deps.vec_env.last_step
        assert last_step is not None, (
            "vec_env.last_step is None after collect_rollout advanced the env"
        )
        respawns_total = _respawns(deps.vec_env)
        env_metrics = rollout_metrics(
            last_step, deps.vec_env.last_components,
            deps.vec_env.clip_fire_rate, respawns_total, deps.vec_env.stats(),
            respawns_delta=respawns_total - respawns_before,
        )
        metrics: dict = update_metrics(
            stats=stats,
            env_metrics=env_metrics,
            update=update,
            global_step=global_step,
            env_steps_this_update=config.n_steps * deps.env_config.n_envs,
            rollout_s=rollout_s,
            update_s=update_s,
            lr=_current_lr(deps.optimizer),
            peak_vram_gb=_peak_vram_gb(deps.device),
            return_scale=scaler.scale,
            wall_clock_hours=(time.monotonic() - run_started) / 3600.0,
        )
        _record_best(best, env_metrics)

        if update % config.artifact_every_updates == 0:
            # The leading indicators, and the two images. Both are far too
            # expensive for every update -- diagnostics materializes the full
            # attention matrix SDPA never forms -- and both move before loss
            # and grad norm do, which is the entire reason they exist.
            diagnostics_started = time.monotonic()
            metrics.update(
                _policy_diagnostics(deps.policy, buffer, config, deps.device, deps.autocast_dtype)
            )
            metrics["perf/diagnostics_s"] = time.monotonic() - diagnostics_started
            metrics.update(_artifacts(deps.vec_env, update))
            deps.wandb_run.summary(best)

        if update % config.checkpoint_every_updates == 0:
            checkpoint_started = time.monotonic()
            ppo_checkpoint.write_checkpoint(
                directory, update, global_step, deps.policy, deps.optimizer,
                deps.scheduler, cache, deps.vec_env, scaler, config,
                deps.init_state_hash, _run_id(deps.wandb_run), deps.git_commit,
            )
            metrics["perf/checkpoint_s"] = time.monotonic() - checkpoint_started

        deps.wandb_run.log(metrics)

        # LAST: a checkpoint above describes the pre-shift buffer state,
        # which the next resume reproduces by collecting n_steps fresh
        # observations.
        buffer.shift()
        update += 1

    logger.info("training_finished", extra={"update": update, "global_step": global_step})


def _check_abort_conditions(stats, config: PPOConfig, consecutive_stalled: int) -> int:
    """The conditions that stop the run. Extracted so the caller can wrap
    both in one handler that logs with a traceback before re-raising.
    Returns the (possibly reset or incremented) consecutive-stalled-update
    counter the caller must pass back in on the next call."""
    # An assert, not a metric: after recomputing pi_old the policy has not
    # changed at (epoch 1, minibatch 1), so any deviation is a real bug.
    assert stats.max_abs_ratio_dev_epoch1_mb1 < 1e-5, (
        f"epoch-1 minibatch-1 ratio deviated by "
        f"{stats.max_abs_ratio_dev_epoch1_mb1}; pi_old was not recomputed from "
        "the current weights"
    )

    if config.target_kl is None:
        if abs(stats.approx_kl) > config.abort_approx_kl:
            raise RuntimeError(
                f"approx_kl {stats.approx_kl} exceeded {config.abort_approx_kl}; "
                "aborting with the checkpoint intact"
            )
        return 0

    # With target_kl set, stats.approx_kl is ppo.update.run_update's
    # LAST-COMPUTED minibatch, which -- whenever target_kl_triggered is
    # True -- was rejected before backward()/step() and never applied. Its
    # magnitude describes what the update would have done, not what the
    # weights actually did, so comparing it against abort_approx_kl would
    # alarm on a mechanism that already did its job. Every APPLIED
    # minibatch, by construction, already satisfied
    # approx_kl <= 1.5*target_kl at the moment it applied, so
    # abort_approx_kl would be comparing that (small) number against a
    # threshold it was never going to reach anyway. Track sustained
    # inability to progress instead: minibatches_completed stuck at its
    # invariant-guaranteed floor of 1 for max_consecutive_stalled_updates
    # updates in a row means training has genuinely stalled.
    if stats.minibatches_completed <= 1:
        consecutive_stalled += 1
    else:
        consecutive_stalled = 0
    if consecutive_stalled >= config.max_consecutive_stalled_updates:
        raise RuntimeError(
            f"minibatches_completed stayed at its floor for {consecutive_stalled} "
            "consecutive updates; aborting with the checkpoint intact"
        )
    return consecutive_stalled


def _record_best(best: dict[str, float], env_metrics: dict[str, float]) -> None:
    """Folds this update's env metrics into the running bests, in place."""
    for target, source in _BEST_SOURCES.items():
        best[target] = max(best.get(target, env_metrics[source]), env_metrics[source])


def _policy_diagnostics(
    policy, buffer: RolloutBuffer, config: PPOConfig, device, autocast_dtype
) -> dict[str, float]:
    """Attention-logit magnitude, attention distance mass, and final-layer
    residual norm, on one minibatch. These move BEFORE loss and grad norm do,
    which is the whole reason the policy exposes them -- and why they are worth
    a forward pass that materializes the attention matrix SDPA never forms.

    Under the same autocast context as the training step, so the numbers
    describe that step rather than an fp32 approximation of it."""
    envs = torch.arange(min(config.minibatch_envs, buffer.n_envs), device=device)
    chunk = buffer.chunk(envs)
    was_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode(), torch.autocast(device.type, dtype=autocast_dtype):
            return policy.diagnostics(
                chunk.latent, chunk.aux_state, chunk.prev_action, chunk.prev_reward,
                chunk.abs_pos, chunk.episode_id, config.diagnostics_layer,
            )
    finally:
        policy.train(was_training)


def _artifacts(vec_env, update: int) -> dict:
    """The two images a human can sanity-check without reading a log line:
    where the agents have been, and what all of them are looking at right now.
    A raw ndarray is not rendered as an image by wandb -- it has to be wrapped
    in wandb.Image, and WandbRun.log swallows every exception by design, so a
    missing wrapper would fail silently rather than raise."""
    coord_keys = [key for entry in vec_env.stats() for key in entry["coord_keys"]]
    # Only called on the artifact cadence, always after a completed rollout.
    last_step = vec_env.last_step
    assert last_step is not None, (
        "vec_env.last_step is None when rendering artifacts"
    )
    return {
        "explore/heatmap": wandb.Image(
            _visit_counts_as_image(exploration_heatmap(coord_keys)), caption=f"update {update}"
        ),
        "env/contact_sheet": wandb.Image(
            contact_sheet(last_step.frames), caption=f"update {update}"
        ),
    }


def _visit_counts_as_image(heatmap: np.ndarray) -> np.ndarray:
    """uint32 visit counts -> uint8 grayscale, peak-normalized. A plain cast
    would render a tile visited once as value 1, which is indistinguishable
    from black -- the artifact would look empty exactly when exploration is
    only just starting, which is when it is most worth looking at."""
    peak = int(heatmap.max(initial=0))
    if peak == 0:
        return heatmap.astype(np.uint8)
    return (heatmap.astype(np.float32) * (255.0 / peak)).astype(np.uint8)


def _respawns(vec_env) -> int:
    return sum(getattr(backend, "respawns", 0) for backend in getattr(vec_env, "_backends", []))


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _peak_vram_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _run_id(run: ExperimentRunLike) -> str | None:
    return getattr(run, "run_id", None)
