"""The outer PPO loop.

Order per iteration: rollout -> update -> telemetry -> cadence work
(checkpoint, artifacts, hub snapshot) -> buffer shift.

The buffer shift happens LAST, so a checkpoint written mid-iteration describes
a buffer state the next resume can reproduce by collecting n_steps fresh
observations.

Resume has two sub-cases, not one. Without a restored KV cache (a checkpoint
saved before the first rollout, or one that deliberately dropped the cache)
the run redoes the full burn_in-length warmup, exactly like a cold start --
the cache has no real context to serve from. With a restored cache, that
context already exists, so only ONE fresh observation is needed to seed the
buffer's first trained slot (buffer.write_cursor is moved to burn_in first,
so that observation lands in the right place) -- redoing the full rebuild
there would waste burn_in real (paid) env steps on every preemption.

A cold start also calls vec_env.reset() once, to get the emulator into its
initial state and start episode counting; a resume must NOT call it, because
VecPokemonEnv.reset() reloads the emulator's init state and would discard the
game position resume() just restored, along with every reward baseline that
describes progress from it."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from observability.tracking import ExperimentRunLike, NullExperimentRun
from pokemon_env.config import EnvConfig
from pokemon_env.telemetry import rollout_metrics
from ppo import checkpoint as ppo_checkpoint
from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from ppo.normalizer import ReturnScaler
from ppo.rollout import RolloutState, collect_rollout
from ppo.telemetry import update_metrics
from ppo.update import run_update
from sequence_model.config import PolicyConfig

logger = logging.getLogger(__name__)


@dataclass
class PPODeps:
    config: PPOConfig
    env_config: EnvConfig
    policy_config: PolicyConfig
    vec_env: object
    encoder: object
    policy: object
    optimizer: object
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
    warmup_needed = True
    if resumed is not None:
        update, global_step = resumed.update + 1, resumed.global_step
        if resumed.cache is not None:
            cache, warmup_needed = resumed.cache, False
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

    if warmup_needed:
        # No real context to serve from -- rebuild it from real env steps
        # before any position has a full context window.
        state = collect_rollout(state=state, n_steps=buffer.burn_in, **rollout_kwargs)
        logger.info("warmup_started", extra={"steps": buffer.burn_in})
    else:
        # The restored cache already carries burn_in worth of real context;
        # only the buffer needs seeding, so the cursor is moved to where the
        # single observation below must land.
        buffer.write_cursor = buffer.burn_in

    # The trained region's first slot, supplied by either the warmup above or
    # (on resume) the restored cache's existing history. Every later rollout
    # inherits this same slot from the previous update's shift() instead.
    state = collect_rollout(state=state, n_steps=1, **rollout_kwargs)
    logger.info("warmup_complete", extra={"steps": buffer.write_cursor})

    while max_updates is None or update < start_update + max_updates:
        started = time.monotonic()
        if deps.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        state = collect_rollout(state=state, n_steps=config.n_steps, **rollout_kwargs)
        global_step += config.n_steps * deps.env_config.n_envs

        stats = deps.run_update(
            deps.policy, deps.optimizer, deps.scheduler, buffer, scaler, config,
            deps.policy_config, deps.env_config.n_envs, deps.device, deps.autocast_dtype,
        )

        # An assert, not a metric: after recomputing pi_old the policy has not
        # changed at (epoch 1, minibatch 1), so any deviation is a real bug
        # and the run must stop.
        assert stats.max_abs_ratio_dev_epoch1_mb1 < 1e-5, (
            f"epoch-1 minibatch-1 ratio deviated by "
            f"{stats.max_abs_ratio_dev_epoch1_mb1}; pi_old was not recomputed from "
            "the current weights"
        )
        if abs(stats.approx_kl) > config.abort_approx_kl:
            raise RuntimeError(
                f"approx_kl {stats.approx_kl} exceeded {config.abort_approx_kl}; "
                "aborting with the checkpoint intact"
            )

        elapsed = time.monotonic() - started
        env_metrics = rollout_metrics(
            deps.vec_env.last_step, deps.vec_env.last_components,
            deps.vec_env.clip_fire_rate, _respawns(deps.vec_env), deps.vec_env.stats(),
        )
        deps.wandb_run.log(
            update_metrics(
                stats, env_metrics, update, global_step, elapsed,
                config.n_steps * deps.env_config.n_envs / max(elapsed, 1e-9),
                _current_lr(deps.optimizer), _peak_vram_gb(deps.device),
            )
        )

        if update % config.checkpoint_every_updates == 0:
            ppo_checkpoint.write_checkpoint(
                directory, update, global_step, deps.policy, deps.optimizer,
                deps.scheduler, cache, deps.vec_env, scaler, config,
                deps.init_state_hash, _run_id(deps.wandb_run), deps.git_commit,
            )

        # LAST: a checkpoint just written above describes the pre-shift
        # buffer state, which the next resume reproduces by collecting
        # n_steps fresh observations.
        buffer.shift()
        update += 1

    logger.info("training_finished", extra={"update": update, "global_step": global_step})


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
