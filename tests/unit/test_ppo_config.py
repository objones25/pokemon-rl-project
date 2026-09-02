"""PPOConfig loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ppo.config import PPOConfig, load_config
from tests.conftest import PINNED_ENCODER_REVISION

# Resolved from this file rather than the process cwd, so the shipped-config
# test below passes when the suite is run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_config_reads_every_field_from_yaml(tmp_path) -> None:
    path = tmp_path / "ppo.yaml"
    path.write_text(
        f"n_steps: 512\nn_epochs: 2\nfrozen_encoder_revision: {PINNED_ENCODER_REVISION}\n"
    )

    config = load_config(path)

    assert (config.n_steps, config.n_epochs, config.frozen_encoder_revision) == (
        512,
        2,
        PINNED_ENCODER_REVISION,
    )


def test_load_config_rejects_an_unknown_field(tmp_path) -> None:
    path = tmp_path / "ppo.yaml"
    path.write_text(f"frozen_encoder_revision: {PINNED_ENCODER_REVISION}\nnot_a_field: 3\n")

    with pytest.raises(ValueError, match="unknown config field"):
        load_config(path)


def test_config_rejects_a_missing_frozen_encoder_revision() -> None:
    with pytest.raises(ValueError, match="frozen_encoder_revision must be pinned"):
        PPOConfig()


@pytest.mark.parametrize("revision", ["main", "master", "MAIN"])
def test_config_rejects_a_branch_head_as_the_frozen_encoder_revision(revision: str) -> None:
    """A branch head resolves at download time, and the checkpoint manifest
    records only the name -- so a mid-run push to the encoder repo changes the
    features underneath the agent AND stays invisible to the next resume."""
    with pytest.raises(ValueError, match="is a branch head, not a pin"):
        PPOConfig(frozen_encoder_revision=revision)


@pytest.mark.parametrize("revision", ["abc123", "v1.0", "0123456789abcdef0123456789abcdef0123456"])
def test_config_rejects_a_frozen_encoder_revision_that_is_not_a_full_commit_sha(
    revision: str,
) -> None:
    """Tags and short shas name a moving or ambiguous target; only 40 hex
    characters name one immutable tree."""
    with pytest.raises(ValueError, match="is not a resolved commit sha"):
        PPOConfig(frozen_encoder_revision=revision)


def test_config_accepts_a_forty_character_hex_commit_sha() -> None:
    config = PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION)

    assert config.frozen_encoder_revision == PINNED_ENCODER_REVISION


def test_the_shipped_ppo_config_pins_the_encoder_to_a_resolved_commit() -> None:
    """configs/ppo.yaml shipped `main` -- PPOConfig accepted it, and the pin
    the rest of the design depends on was defeated by the config file itself."""
    config = load_config(_REPO_ROOT / "configs" / "ppo.yaml")

    assert config.frozen_encoder_revision == "9db5cb99991fd976501fca533e976ecad815b321"


def test_config_rejects_an_n_envs_not_divisible_by_minibatch_envs() -> None:
    with pytest.raises(ValueError, match="minibatch_envs=7 does not divide"):
        PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, minibatch_envs=7).validate_against_n_envs(64)


def test_burn_in_is_one_less_than_the_context_length() -> None:
    config = PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION)

    assert config.burn_in(context_len=1024) == 1023


def test_buffer_capacity_is_burn_in_plus_n_steps_plus_one_bootstrap_slot() -> None:
    config = PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, n_steps=1024)

    assert config.buffer_capacity(context_len=1024) == 2048


def test_config_rejects_n_steps_less_than_one() -> None:
    with pytest.raises(ValueError, match="n_steps=0 must be at least 1"):
        PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, n_steps=0)


def test_config_rejects_n_epochs_less_than_one() -> None:
    with pytest.raises(ValueError, match="n_epochs=0 must be at least 1"):
        PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, n_epochs=0)


def test_config_rejects_minibatch_envs_less_than_one() -> None:
    with pytest.raises(ValueError, match="minibatch_envs=0 must be at least 1"):
        PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, minibatch_envs=0)


@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_config_rejects_gamma_not_strictly_between_zero_and_one(gamma: float) -> None:
    with pytest.raises(ValueError, match="must lie in"):
        PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, gamma=gamma)


def test_validate_against_n_envs_succeeds_when_divisible() -> None:
    config = PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, minibatch_envs=8)

    assert config.validate_against_n_envs(64) is None


def test_config_rejects_a_negative_lr_decay_steps() -> None:
    with pytest.raises(ValueError, match="lr_decay_steps=-1 must be at least 0"):
        PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, lr_decay_steps=-1)


@pytest.mark.parametrize("lr_floor_ratio", [-0.1, 1.1])
def test_config_rejects_an_lr_floor_ratio_outside_zero_one(lr_floor_ratio: float) -> None:
    with pytest.raises(ValueError, match="lr_floor_ratio=.* must lie in \\[0, 1\\]"):
        PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, lr_floor_ratio=lr_floor_ratio)


@pytest.mark.parametrize("lr_floor_ratio", [0.0, 1.0])
def test_config_accepts_lr_floor_ratio_at_either_boundary(lr_floor_ratio: float) -> None:
    config = PPOConfig(
        frozen_encoder_revision=PINNED_ENCODER_REVISION, lr_floor_ratio=lr_floor_ratio
    )

    assert config.lr_floor_ratio == pytest.approx(lr_floor_ratio)


def test_lr_decay_steps_and_floor_ratio_default_to_decay_disabled() -> None:
    """0 decay steps is the sentinel the scheduler reads as "stay constant
    past warmup" -- existing configs/tests that don't mention these fields
    must keep behaving exactly as before this feature existed."""
    config = PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION)

    assert (config.lr_decay_steps, config.lr_floor_ratio) == (0, 0.1)


@pytest.mark.parametrize("target_kl", [0.0, -0.02])
def test_config_rejects_a_non_positive_target_kl(target_kl: float) -> None:
    with pytest.raises(ValueError, match="target_kl=.* must be > 0, or None to disable"):
        PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, target_kl=target_kl)


def test_target_kl_defaults_to_disabled() -> None:
    config = PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION)

    assert config.target_kl is None


def test_config_accepts_a_positive_target_kl() -> None:
    config = PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, target_kl=0.02)

    assert config.target_kl == pytest.approx(0.02)
