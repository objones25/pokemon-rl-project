"""PPOConfig loading and validation."""

from __future__ import annotations

import pytest

from ppo.config import PPOConfig, load_config


def test_load_config_reads_every_field_from_yaml(tmp_path) -> None:
    path = tmp_path / "ppo.yaml"
    path.write_text("n_steps: 512\nn_epochs: 2\nfrozen_encoder_revision: abc123\n")

    config = load_config(path)

    assert (config.n_steps, config.n_epochs, config.frozen_encoder_revision) == (512, 2, "abc123")


def test_load_config_rejects_an_unknown_field(tmp_path) -> None:
    path = tmp_path / "ppo.yaml"
    path.write_text("frozen_encoder_revision: abc123\nnot_a_field: 3\n")

    with pytest.raises(ValueError, match="unknown config field"):
        load_config(path)


def test_config_rejects_a_missing_frozen_encoder_revision() -> None:
    with pytest.raises(ValueError, match="frozen_encoder_revision must be pinned"):
        PPOConfig()


def test_config_rejects_an_n_envs_not_divisible_by_minibatch_envs() -> None:
    with pytest.raises(ValueError, match="minibatch_envs=7 does not divide"):
        PPOConfig(frozen_encoder_revision="abc123", minibatch_envs=7).validate_against_n_envs(64)


def test_burn_in_is_one_less_than_the_context_length() -> None:
    config = PPOConfig(frozen_encoder_revision="abc123")

    assert config.burn_in(context_len=1024) == 1023


def test_buffer_capacity_is_burn_in_plus_n_steps_plus_one_bootstrap_slot() -> None:
    config = PPOConfig(frozen_encoder_revision="abc123", n_steps=1024)

    assert config.buffer_capacity(context_len=1024) == 2048


def test_config_rejects_n_steps_less_than_one() -> None:
    with pytest.raises(ValueError, match="n_steps=0 must be at least 1"):
        PPOConfig(frozen_encoder_revision="abc123", n_steps=0)


def test_config_rejects_n_epochs_less_than_one() -> None:
    with pytest.raises(ValueError, match="n_epochs=0 must be at least 1"):
        PPOConfig(frozen_encoder_revision="abc123", n_epochs=0)


def test_config_rejects_minibatch_envs_less_than_one() -> None:
    with pytest.raises(ValueError, match="minibatch_envs=0 must be at least 1"):
        PPOConfig(frozen_encoder_revision="abc123", minibatch_envs=0)


@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_config_rejects_gamma_not_strictly_between_zero_and_one(gamma: float) -> None:
    with pytest.raises(ValueError, match="must lie in"):
        PPOConfig(frozen_encoder_revision="abc123", gamma=gamma)


def test_validate_against_n_envs_succeeds_when_divisible() -> None:
    config = PPOConfig(frozen_encoder_revision="abc123", minibatch_envs=8)

    assert config.validate_against_n_envs(64) is None
