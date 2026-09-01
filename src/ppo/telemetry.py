"""Per-update scalars and the W&B config.

Numbers moving over time go to W&B; events with a cause go to the JSON-lines
log. Nothing is emitted per env step -- 65,536 of those per update would be a
self-inflicted outage."""

from __future__ import annotations

from dataclasses import asdict

from pokemon_env.config import EnvConfig
from ppo.config import PPOConfig
from ppo.update import UpdateStats
from sequence_model.config import PolicyConfig

# Every metric this trainer logs is a per-update scalar, so one axis covers
# them all. Declared via define_metric so log() never passes step=: wandb
# silently drops a log whose step is below its current one, and the axis
# travelling as an ordinary field sidesteps that entirely.
STEP_METRICS: dict[str, str] = {
    "loss/*": "train/update",
    "ratio/*": "train/update",
    "value/*": "train/update",
    "train/*": "train/update",
    "staleness/*": "train/update",
    "reward/*": "train/update",
    "env/*": "train/update",
    "progress/*": "train/update",
    "explore/*": "train/update",
    "episode/*": "train/update",
    "attn/*": "train/update",
    "model/*": "train/update",
    "perf/*": "train/update",
    "system/*": "train/update",
}


def update_metrics(
    stats: UpdateStats,
    env_metrics: dict[str, float],
    update: int,
    global_step: int,
    env_steps_this_update: int,
    rollout_s: float,
    update_s: float,
    lr: float,
    peak_vram_gb: float,
    return_scale: float,
    wall_clock_hours: float,
) -> dict[str, float]:
    """One update's scalars, plus the caller's already-assembled env/policy
    metrics merged in unchanged. `train/update` is the value STEP_METRICS
    declares as every family's x-axis -- callers must never also pass
    `step=` to `log()`.

    Both split gradient norms are logged alongside the total: the policy and
    critic share a trunk, so a single clipped global `train/grad_norm` can
    hide a large value-loss gradient consuming the whole clip budget and
    shrinking the policy gradient toward nothing.

    `rollout_s`/`update_s` are kept separate rather than pre-summed by the
    caller, and `perf/rollout_env_steps_per_sec` is computed from `rollout_s`
    alone: a slow iteration is otherwise indistinguishable between the
    64-subprocess env stalling and the GPU-side forward/backward, exactly
    the "dataloader starvation" failure mode a combined timer hides."""
    iteration_s = rollout_s + update_s
    metrics = {
        "train/update": float(update),
        "train/env_step": float(global_step),
        "train/lr": float(lr),
        "train/grad_norm": stats.grad_norm,
        "train/grad_norm_max": stats.grad_norm_max,
        "train/policy_grad_norm": stats.policy_grad_norm,
        "train/value_grad_norm": stats.value_grad_norm,
        "train/skipped_minibatches": float(stats.skipped_minibatches),
        "train/return_scale": float(return_scale),
        "loss/policy": stats.policy_loss,
        "loss/value": stats.value_loss,
        "loss/entropy": stats.entropy,
        "loss/total": stats.total_loss,
        "ratio/max_abs_dev_epoch1_mb1": stats.max_abs_ratio_dev_epoch1_mb1,
        "ratio/max_abs_dev": stats.max_abs_ratio_dev,
        "ratio/clip_fraction": stats.clip_fraction,
        "ratio/approx_kl": stats.approx_kl,
        "staleness/logprob_l1": stats.staleness_logprob_l1,
        "value/explained_variance": stats.explained_variance,
        "perf/rollout_s": float(rollout_s),
        "perf/update_s": float(update_s),
        "perf/iteration_s": float(iteration_s),
        "perf/env_steps_per_sec": float(env_steps_this_update) / max(iteration_s, 1e-9),
        "perf/rollout_env_steps_per_sec": float(env_steps_this_update) / max(rollout_s, 1e-9),
        "perf/wall_clock_hours": float(wall_clock_hours),
        "system/peak_vram_gb": float(peak_vram_gb),
    }
    metrics.update(env_metrics)
    return metrics


def wandb_config(
    ppo_config: PPOConfig,
    env_config: EnvConfig,
    policy_config: PolicyConfig,
    gate_results: dict,
    git_commit: str,
    gpu_name: str,
    torch_version: str,
) -> dict:
    """The three dataclasses plus provenance. Gate results are included so the
    chosen SDPA backend and the measured throughput are part of the run record
    rather than a number in someone's terminal scrollback.

    Nothing here reads the environment: a W&B config is readable by everyone
    with project access, and a credential in it is a credential published.
    The dict is built exclusively from the three dataclasses' own fields
    (via asdict), the caller-supplied gate_results mapping, and the explicit
    provenance arguments below -- there is no path from os.environ or any
    other ambient source into this dict."""
    config: dict = {}
    for prefix, dataclass_instance in (
        ("ppo", ppo_config),
        ("env", env_config),
        ("policy", policy_config),
    ):
        for key, value in asdict(dataclass_instance).items():
            config[f"{prefix}/{key}"] = value
    for key, value in gate_results.items():
        config[f"gate/{key}"] = value
    config["run/git_commit"] = git_commit
    config["run/gpu"] = gpu_name
    config["run/torch_version"] = torch_version
    return config
