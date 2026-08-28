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
        "policy_grad_norm": 0.6,
        "value_grad_norm": 0.4,
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
        "iteration_s": 10.0,
        "env_steps_per_sec": 6553.6,
        "lr": 3e-4,
        "peak_vram_gb": 12.0,
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
        policy_grad_norm=13.0,
        value_grad_norm=14.0,
    )

    metrics = update_metrics(
        stats=stats,
        env_metrics={},
        update=15,
        global_step=16,
        iteration_s=17.0,
        env_steps_per_sec=18.0,
        lr=19.0,
        peak_vram_gb=20.0,
    )

    assert metrics == pytest.approx(
        {
            "train/update": 15.0,
            "train/env_step": 16.0,
            "train/lr": 19.0,
            "train/grad_norm": 12.0,
            "train/policy_grad_norm": 13.0,
            "train/value_grad_norm": 14.0,
            "train/skipped_minibatches": 11.0,
            "loss/policy": 1.0,
            "loss/value": 2.0,
            "loss/entropy": 3.0,
            "loss/total": 4.0,
            "ratio/max_abs_dev_epoch1_mb1": 7.0,
            "ratio/max_abs_dev": 8.0,
            "ratio/clip_fraction": 5.0,
            "ratio/approx_kl": 6.0,
            "staleness/logprob_l1": 10.0,
            "value/explained_variance": 9.0,
            "perf/iteration_s": 17.0,
            "perf/env_steps_per_sec": 18.0,
            "system/peak_vram_gb": 20.0,
        }
    )


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
