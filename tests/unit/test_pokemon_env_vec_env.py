import numpy as np
import pytest

from pokemon_env import ram
from pokemon_env.config import EnvConfig
from pokemon_env.session import EnvSession
from pokemon_env.vec_env import InProcessBackend, VecPokemonEnv

from .fakes import FakeEmulator


@pytest.fixture
def vec_env() -> VecPokemonEnv:
    config = EnvConfig(n_envs=3, max_steps=2)
    backends = [
        InProcessBackend(EnvSession(FakeEmulator(), config, init_state=b"init"))
        for _ in range(3)
    ]
    return VecPokemonEnv(backends, config)


def _two_badge_emulator() -> FakeEmulator:
    """A FakeEmulator whose badge byte has popcount 2, so badge_weight (1.00)
    * 2 = 2.00 comfortably exceeds the reward accumulator's 1.0 clip cap --
    the static default FakeEmulator() never produces a clipped step at all."""
    memory = bytearray(0x10000)
    memory[ram.BADGES_ADDR] = 0b11
    return FakeEmulator(memory=memory)


@pytest.fixture
def two_badge_vec_env() -> VecPokemonEnv:
    config = EnvConfig(n_envs=3, max_steps=2)
    backends = [
        InProcessBackend(EnvSession(_two_badge_emulator(), config, init_state=b"init"))
        for _ in range(3)
    ]
    return VecPokemonEnv(backends, config)


def test_step_returns_batched_arrays_with_the_encoder_frame_shape(vec_env) -> None:
    vec_env.reset()

    result = vec_env.step(np.zeros(3, dtype=np.int64))

    assert (result.frames.shape, result.frames.dtype.name) == ((3, 1, 144, 160), "uint8")


def test_step_returns_one_aux_row_per_env(vec_env) -> None:
    vec_env.reset()

    result = vec_env.step(np.zeros(3, dtype=np.int64))

    assert (result.aux.shape, result.aux.dtype.name) == ((3, 32), "float32")


def test_done_is_reported_on_the_terminal_step_not_deferred(vec_env) -> None:
    """max_steps=2, so the second step is terminal. The sequence-model spec's
    cache.reset(done) contract requires reset to run AFTER the step whose
    transition ended the episode -- so done must be true on that step."""
    vec_env.reset()
    vec_env.step(np.zeros(3, dtype=np.int64))

    second = vec_env.step(np.zeros(3, dtype=np.int64))

    assert second.done.tolist() == [True, True, True]


def test_the_step_after_done_returns_the_reset_observation(vec_env) -> None:
    """Next-step autoreset: done at t returns the TERMINAL observation, and
    the reset observation arrives at t+1 with an incremented episode_id."""
    vec_env.reset()
    vec_env.step(np.zeros(3, dtype=np.int64))
    vec_env.step(np.zeros(3, dtype=np.int64))

    after = vec_env.step(np.zeros(3, dtype=np.int64))

    assert (after.episode_id.tolist(), after.done.tolist()) == ([1, 1, 1], [False, False, False])


def test_reset_starts_every_env_at_episode_zero(vec_env) -> None:
    result = vec_env.reset()

    assert result.episode_id.tolist() == [0, 0, 0]


def test_step_rejects_a_wrong_length_action_array(vec_env) -> None:
    vec_env.reset()

    with pytest.raises(ValueError, match="actions has length 2, expected 3"):
        vec_env.step(np.zeros(2, dtype=np.int64))


def test_clip_fire_rate_reports_the_fraction_of_clipped_steps(vec_env) -> None:
    """Above roughly 0.1% the reward weights are miscalibrated and the
    ordering between achievements is being flattened."""
    vec_env.reset()
    vec_env.step(np.zeros(3, dtype=np.int64))

    assert vec_env.clip_fire_rate == pytest.approx(0.0)


def test_clip_fire_rate_counts_steps_whose_reward_was_clipped(two_badge_vec_env) -> None:
    """2 badges * badge_weight 1.00 = 2.00, past the reward clip's 1.0 cap, so
    every env's first step is clipped. reset() also counts toward the
    denominator -- 3 clipped observations out of 6 total collected."""
    two_badge_vec_env.reset()

    two_badge_vec_env.step(np.zeros(3, dtype=np.int64))

    assert two_badge_vec_env.clip_fire_rate == pytest.approx(0.5)


def test_last_components_reports_the_mean_reward_breakdown_across_envs(
    two_badge_vec_env,
) -> None:
    two_badge_vec_env.reset()

    two_badge_vec_env.step(np.zeros(3, dtype=np.int64))

    assert two_badge_vec_env.last_components == pytest.approx(
        {"badges": 2.0, "events": 0.0, "explore": 0.3, "heal": 0.0, "levels": 0.0}
    )


def test_state_dict_round_trips_the_per_env_step_counters(vec_env) -> None:
    """Asserts the per-env counters actually moved across, not just that a
    version constant matches -- the version would compare equal against a
    restore that loaded nothing."""
    vec_env.reset()
    vec_env.step(np.zeros(3, dtype=np.int64))
    state = vec_env.state_dict()

    config = EnvConfig(n_envs=3, max_steps=2)
    restored = VecPokemonEnv(
        [
            InProcessBackend(EnvSession(FakeEmulator(), config, init_state=b"init"))
            for _ in range(3)
        ],
        config,
    )
    restored.load_state_dict(state)

    assert [b["session"]["step_count"] for b in restored.state_dict()["backends"]] == [1, 1, 1]


def test_load_state_dict_rejects_a_stale_aux_state_version(vec_env) -> None:
    """A policy trained against layout v1 and fed v2 data reads different
    signals from the same indices, with no crash and no shape error."""
    vec_env.reset()
    state = vec_env.state_dict()
    state["aux_state_version"] = 99

    with pytest.raises(ValueError, match="AUX_STATE_VERSION=99"):
        vec_env.load_state_dict(state)


def test_load_state_dict_rejects_a_different_env_count(vec_env) -> None:
    vec_env.reset()
    state = vec_env.state_dict()
    state["backends"] = state["backends"][:2]

    with pytest.raises(ValueError, match="cannot be redistributed"):
        vec_env.load_state_dict(state)


def test_load_state_dict_rejects_a_stale_schema_version(vec_env) -> None:
    """The version was written but never read until now. An unvalidated
    version field is worse than none -- it reads as protection while a layout
    change resumes silently against fields that no longer mean the same."""
    vec_env.reset()
    state = vec_env.state_dict()
    state["schema_version"] = 99

    with pytest.raises(ValueError, match="schema_version=99"):
        vec_env.load_state_dict(state)
