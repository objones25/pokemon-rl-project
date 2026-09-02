"""Metric assembly. Instrumentation bugs are invisible in every test that does
not look at the output."""

from __future__ import annotations

from dataclasses import fields

import pytest

from pokemon_env.config import EnvConfig
from ppo.config import PPOConfig
from ppo.telemetry import STEP_METRICS, update_metrics, wandb_config
from ppo.update import UpdateStats
from sequence_model.config import PolicyConfig
from tests.conftest import PINNED_ENCODER_REVISION


def _stats_kwargs(**overrides: object) -> dict:
    """Builds a default update_metrics() call. An override matching an
    UpdateStats field name replaces that field; any other override replaces a
    top-level update_metrics keyword."""
    stats_field_names = {field.name for field in fields(UpdateStats)}
    stats_defaults: dict = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 0.3,
        "total_loss": 0.4,
        "clip_fraction": 0.05,
        "approx_kl": 0.01,
        "max_abs_ratio_dev_epoch1_mb1": 0.0,
        "max_abs_ratio_dev": 0.02,
        "explained_variance": 0.5,
        "staleness_logprob_l1": 0.01,
        "skipped_minibatches": 0,
        "grad_norm": 1.0,
        "grad_norm_max": 1.5,
        "policy_grad_norm": 0.6,
        "value_grad_norm": 0.4,
        "target_kl_triggered": False,
        "minibatches_completed": 24,
        "approx_kl_mean": 0.011,
        "clip_fraction_mean": 0.06,
        "ratio_abs_dev_p50": 0.005,
        "ratio_abs_dev_p95": 0.03,
        "ratio_abs_dev_p99": 0.08,
        "max_abs_ratio_dev_update": 0.15,
        "max_action_prob": 0.4,
        "raw_advantage_mean": 0.0,
        "raw_advantage_std": 1.2,
        "raw_advantage_abs_max": 5.0,
        "raw_advantage_top1_frac": 0.05,
        "raw_advantage_top1pct_frac": 0.2,
    }
    stats_overrides = {key: value for key, value in overrides.items() if key in stats_field_names}
    top_overrides = {
        key: value for key, value in overrides.items() if key not in stats_field_names
    }
    stats_defaults.update(stats_overrides)
    stats = UpdateStats(**stats_defaults)
    kwargs: dict = {
        "stats": stats,
        "env_metrics": {},
        "update": 1,
        "global_step": 1024,
        "env_steps_this_update": 6553,
        "rollout_s": 7.0,
        "update_s": 3.0,
        "lr": 3e-4,
        "peak_vram_gb": 12.0,
        "return_scale": 1.0,
        "wall_clock_hours": 0.5,
    }
    kwargs.update(top_overrides)
    return kwargs


def _config_kwargs() -> dict:
    return {
        "ppo_config": PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION),
        "env_config": EnvConfig(),
        "policy_config": PolicyConfig(),
        "gate_results": {
            "sdpa_backend": "flash_attention_2",
            "throughput_env_steps_per_sec": 5000.0,
        },
        "git_commit": "deadbeef",
        "gpu_name": "NVIDIA A100",
        "torch_version": "2.13.0",
    }


def test_every_metric_family_declares_an_x_axis() -> None:
    """Without define_metric, wandb uses its own internal step counter and a
    resumed run's points land at the wrong x."""
    assert set(STEP_METRICS.values()) == {"train/update"}


def test_update_metrics_carries_the_x_axis_value_as_a_field() -> None:
    """log() must never pass step=; the axis travels as a normal metric."""
    metrics = update_metrics(**_stats_kwargs(update=7))

    assert metrics["train/update"] == pytest.approx(7.0)


def test_update_metrics_reports_the_epoch_one_ratio_deviation() -> None:
    metrics = update_metrics(**_stats_kwargs(max_abs_ratio_dev_epoch1_mb1=0.0))

    assert metrics["ratio/max_abs_dev_epoch1_mb1"] == pytest.approx(0.0)


def test_update_metrics_merges_the_env_metrics_unchanged() -> None:
    metrics = update_metrics(**_stats_kwargs(env_metrics={"reward/mean": 0.25}))

    assert metrics["reward/mean"] == pytest.approx(0.25)


def test_update_metrics_maps_every_stats_field_without_transposition() -> None:
    """Each UpdateStats/call field gets its own distinct value, so a bug that
    swaps two fields (e.g. loss/policy <-> loss/value, or the two split grad
    norms) shows up as a wrong number rather than passing by coincidence."""
    stats = UpdateStats(
        policy_loss=1.0,
        value_loss=2.0,
        entropy=3.0,
        total_loss=4.0,
        clip_fraction=5.0,
        approx_kl=6.0,
        max_abs_ratio_dev_epoch1_mb1=7.0,
        max_abs_ratio_dev=8.0,
        explained_variance=9.0,
        staleness_logprob_l1=10.0,
        skipped_minibatches=11,
        grad_norm=12.0,
        grad_norm_max=21.0,
        policy_grad_norm=13.0,
        value_grad_norm=14.0,
        target_kl_triggered=True,
        minibatches_completed=24,
        approx_kl_mean=25.0,
        clip_fraction_mean=26.0,
        ratio_abs_dev_p50=27.0,
        ratio_abs_dev_p95=28.0,
        ratio_abs_dev_p99=29.0,
        max_abs_ratio_dev_update=30.0,
        max_action_prob=31.0,
        raw_advantage_mean=32.0,
        raw_advantage_std=33.0,
        raw_advantage_abs_max=34.0,
        raw_advantage_top1_frac=35.0,
        raw_advantage_top1pct_frac=36.0,
    )

    metrics = update_metrics(
        stats=stats,
        env_metrics={},
        update=15,
        global_step=16,
        env_steps_this_update=1000,
        rollout_s=8.0,
        update_s=2.0,
        lr=19.0,
        peak_vram_gb=20.0,
        return_scale=22.0,
        wall_clock_hours=23.0,
    )

    assert metrics == pytest.approx(
        {
            "train/update": 15.0,
            "train/env_step": 16.0,
            "train/lr": 19.0,
            "train/grad_norm": 12.0,
            "train/grad_norm_max": 21.0,
            "train/policy_grad_norm": 13.0,
            "train/value_grad_norm": 14.0,
            "train/skipped_minibatches": 11.0,
            "train/minibatches_completed": 24.0,
            "train/target_kl_triggered": 1.0,
            "train/return_scale": 22.0,
            "loss/policy": 1.0,
            "loss/value": 2.0,
            "loss/entropy": 3.0,
            "loss/total": 4.0,
            "loss/max_action_prob": 31.0,
            "ratio/max_abs_dev_epoch1_mb1": 7.0,
            "ratio/max_abs_dev": 8.0,
            "ratio/clip_fraction": 5.0,
            "ratio/approx_kl": 6.0,
            "ratio/approx_kl_mean": 25.0,
            "ratio/clip_fraction_mean": 26.0,
            "ratio/abs_dev_p50": 27.0,
            "ratio/abs_dev_p95": 28.0,
            "ratio/abs_dev_p99": 29.0,
            "ratio/max_abs_dev_update": 30.0,
            "staleness/logprob_l1": 10.0,
            "value/explained_variance": 9.0,
            "advantage/raw_mean": 32.0,
            "advantage/raw_std": 33.0,
            "advantage/raw_abs_max": 34.0,
            "advantage/top1_frac": 35.0,
            "advantage/top1pct_frac": 36.0,
            "perf/rollout_s": 8.0,
            "perf/update_s": 2.0,
            "perf/iteration_s": 10.0,
            "perf/env_steps_per_sec": 100.0,
            "perf/rollout_env_steps_per_sec": 125.0,
            "perf/wall_clock_hours": 23.0,
            "system/peak_vram_gb": 20.0,
        }
    )


def test_update_metrics_splits_rollout_and_update_time() -> None:
    """perf/iteration_s used to be one combined timer around both the
    rollout and the update pass, so a slow iteration could not be attributed
    to either the 64-subprocess env or the GPU-side forward/backward."""
    metrics = update_metrics(**_stats_kwargs(rollout_s=6.0, update_s=4.0))

    assert (
        metrics["perf/rollout_s"],
        metrics["perf/update_s"],
        metrics["perf/iteration_s"],
    ) == (pytest.approx(6.0), pytest.approx(4.0), pytest.approx(10.0))


def test_update_metrics_reports_rollout_only_throughput_separately_from_overall() -> None:
    """A copy-paste bug that divided env_steps_this_update by iteration_s for
    BOTH throughput fields would make these two numbers coincide even though
    rollout_s and update_s differ -- exactly what would hide an env-side
    bottleneck behind a normal-looking overall throughput number."""
    metrics = update_metrics(
        **_stats_kwargs(env_steps_this_update=1000, rollout_s=5.0, update_s=5.0)
    )

    assert (
        metrics["perf/rollout_env_steps_per_sec"],
        metrics["perf/env_steps_per_sec"],
    ) == (pytest.approx(200.0), pytest.approx(100.0))


def test_update_metrics_reports_the_return_scalers_scale() -> None:
    """The running std that rescales value targets and advantages -- if it
    shifts sharply (e.g. after a badge unlock changes the reward
    distribution), that is currently invisible even though it directly
    changes what the critic is regressed onto."""
    metrics = update_metrics(**_stats_kwargs(return_scale=3.5))

    assert metrics["train/return_scale"] == pytest.approx(3.5)


def test_update_metrics_reports_cumulative_wall_clock_hours() -> None:
    """Nothing else on the dashboard answers 'how many paid GPU-hours has
    this run cost so far'."""
    metrics = update_metrics(**_stats_kwargs(wall_clock_hours=12.25))

    assert metrics["perf/wall_clock_hours"] == pytest.approx(12.25)


def test_no_secret_shaped_key_reaches_the_wandb_config() -> None:
    """A W&B config is readable by everyone with project access."""
    config = wandb_config(**_config_kwargs())
    suspicious = [
        key
        for key in config
        if any(word in key.lower() for word in ("token", "key", "secret", "password"))
    ]

    assert suspicious == []


def test_the_wandb_config_contains_exactly_the_expected_keys() -> None:
    """Computed independently of wandb_config's own asdict() call, from the
    dataclass definitions directly -- so an accidentally-merged extra source
    (os.environ, a credentials object) shows up as an unexpected key even if
    none of its names happen to contain a suspicious substring."""
    kwargs = _config_kwargs()
    config = wandb_config(**kwargs)

    expected = (
        {f"ppo/{field.name}" for field in fields(PPOConfig)}
        | {f"env/{field.name}" for field in fields(EnvConfig)}
        | {f"policy/{field.name}" for field in fields(PolicyConfig)}
        | {f"gate/{key}" for key in kwargs["gate_results"]}
        | {"run/git_commit", "run/gpu", "run/torch_version"}
    )

    assert set(config) == expected


def test_the_wandb_config_records_gate_results_under_the_gate_prefix() -> None:
    config = wandb_config(**_config_kwargs())

    assert config["gate/sdpa_backend"] == "flash_attention_2"


def test_the_wandb_config_records_the_pinned_encoder_revision() -> None:
    config = wandb_config(**_config_kwargs())

    assert config["ppo/frozen_encoder_revision"] == PINNED_ENCODER_REVISION
