import pytest

from pokemon_env.config import EnvConfig, load_config


def test_load_config_reads_yaml_values(tmp_path) -> None:
    path = tmp_path / "env.yaml"
    path.write_text("n_envs: 8\naction_freq: 16\npress_frames: 4\n")

    config = load_config(path)

    assert (config.n_envs, config.action_freq, config.press_frames) == (8, 16, 4)


def test_load_config_rejects_an_unknown_field(tmp_path) -> None:
    """A typo'd key would otherwise be silently ignored and the run would use
    the default, which is indistinguishable from the setting having no effect."""
    path = tmp_path / "env.yaml"
    path.write_text("n_env: 8\n")

    with pytest.raises(ValueError, match=r"unknown config field\(s\): \['n_env'\]"):
        load_config(path)


def test_config_rejects_press_frames_that_leave_no_release_window() -> None:
    """The step is press -> tick(press_frames) -> release -> tick(action_freq -
    press_frames - 1) -> tick(1). With press_frames = action_freq - 1 the
    release tick count is 0, so the button is never released and every
    subsequent action is entered with it still held."""
    with pytest.raises(ValueError, match="press_frames=23 leaves 0 frames"):
        EnvConfig(action_freq=24, press_frames=23)


def test_config_defaults_match_the_reference_implementation() -> None:
    config = EnvConfig()

    assert (config.n_envs, config.action_freq, config.max_steps) == (64, 24, 163_840)


def test_idle_penalty_weight_defaults_to_zero() -> None:
    """Opt-in like target_kl/lr_decay_steps: a new tunable reward term
    defaults off so every existing default-EnvConfig test keeps its current
    behavior, and configs/pokemon_env.yaml is what actually turns it on for
    a real run."""
    config = EnvConfig()

    assert config.idle_penalty_weight == pytest.approx(0.0)


def test_battle_incentive_weights_default_to_zero() -> None:
    """Same opt-in pattern: damage/battle_win/catch/money all default off."""
    config = EnvConfig()

    assert (
        config.damage_weight,
        config.battle_win_weight,
        config.catch_weight,
        config.money_weight,
    ) == (0.0, 0.0, 0.0, 0.0)


def test_low_hp_penalty_weight_defaults_to_zero() -> None:
    """Opt-in like every other reward weight added since idle_penalty_weight:
    defaults off so every existing default-EnvConfig test keeps its current
    behavior."""
    config = EnvConfig()

    assert config.low_hp_penalty_weight == pytest.approx(0.0)


def test_low_hp_threshold_defaults_to_a_quarter_health() -> None:
    config = EnvConfig()

    assert config.low_hp_threshold == pytest.approx(0.25)


def test_low_hp_threshold_rejects_zero() -> None:
    """A zero threshold divides by zero computing the penalty's severity
    ramp (rewards.py's (threshold - fraction) / threshold) -- reject it at
    construction, not with a ZeroDivisionError deep in a training step."""
    with pytest.raises(ValueError, match=r"low_hp_threshold=0.0 must be in \(0, 1\]"):
        EnvConfig(low_hp_threshold=0.0)


def test_low_hp_threshold_rejects_a_value_above_one() -> None:
    with pytest.raises(ValueError, match=r"low_hp_threshold=1.5 must be in \(0, 1\]"):
        EnvConfig(low_hp_threshold=1.5)
