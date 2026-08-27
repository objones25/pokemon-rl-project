import numpy as np
import pytest

from checkpointing.io import find_latest_checkpoint, load_checkpoint, save_checkpoint
from pokemon_env.checkpoint import (
    ENV_CHECKPOINT_PATTERN,
    build_env_checkpoint_state,
    restore_env_checkpoint,
)
from pokemon_env.config import EnvConfig
from pokemon_env.session import EnvSession
from pokemon_env.vec_env import InProcessBackend, VecPokemonEnv

from .fakes import FakeEmulator


def _vec_env() -> VecPokemonEnv:
    """Helper, not a test."""
    config = EnvConfig(n_envs=2, max_steps=8)
    return VecPokemonEnv(
        [
            InProcessBackend(EnvSession(FakeEmulator(), config, init_state=b"init"))
            for _ in range(2)
        ],
        config,
    )


def test_state_records_the_update_number() -> None:
    vec_env = _vec_env()
    vec_env.reset()

    state = build_env_checkpoint_state(update=5, vec_env=vec_env, init_state_hash="abc")

    assert state["update"] == 5


def test_restore_rejects_a_changed_init_state() -> None:
    """A different starting state invalidates every reward baseline in the
    checkpoint -- the max_historical values describe progress from a game
    position the new state does not share."""
    vec_env = _vec_env()
    vec_env.reset()
    state = build_env_checkpoint_state(update=1, vec_env=vec_env, init_state_hash="abc")

    with pytest.raises(ValueError, match="init.state changed"):
        restore_env_checkpoint(_vec_env(), state, init_state_hash="def")


def test_restore_round_trips_through_the_shared_checkpoint_io(tmp_path) -> None:
    """Asserts a per-env counter that actually moved across, not a version
    constant compared against itself -- that would also pass against a
    restore_env_checkpoint that loaded nothing at all. A freshly constructed
    _vec_env() never has reset() or step() called on it, so its backends sit
    at step_count=0; only a real load_state_dict call can produce 2 here."""
    vec_env = _vec_env()
    vec_env.reset()
    vec_env.step(np.zeros(2, dtype=np.int64))
    vec_env.step(np.zeros(2, dtype=np.int64))
    state = build_env_checkpoint_state(update=3, vec_env=vec_env, init_state_hash="abc")
    path = tmp_path / "env_update00000003.pt"

    save_checkpoint(path, state)
    restored = _vec_env()
    restore_env_checkpoint(restored, load_checkpoint(path), init_state_hash="abc")

    assert [b["session"]["step_count"] for b in restored.state_dict()["backends"]] == [2, 2]


def test_the_checkpoint_pattern_does_not_collide_with_the_other_runs(tmp_path) -> None:
    """The PPO run, the policy checkpoints and the pretraining run may share
    one network volume. A pattern that globbed the others would prune their
    resume points."""
    (tmp_path / "env_update00000100.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000900.pt").write_bytes(b"")

    latest = find_latest_checkpoint(tmp_path, pattern=ENV_CHECKPOINT_PATTERN)

    assert latest == tmp_path / "env_update00000100.pt"


def test_the_pattern_finds_nothing_when_only_pretraining_checkpoints_exist(tmp_path) -> None:
    """The prior test alone is not decisive: 'checkpoint_step...' sorts before
    'env_update...' lexicographically ('c' < 'e'), so find_latest_checkpoint
    would return the env file even if ENV_CHECKPOINT_PATTERN also matched
    checkpoint_step*.pt files -- the exclusion would never actually be
    exercised. A directory holding only a pretraining checkpoint is the case
    that only passes if the pattern truly excludes it."""
    (tmp_path / "checkpoint_step00000900.pt").write_bytes(b"")

    latest = find_latest_checkpoint(tmp_path, pattern=ENV_CHECKPOINT_PATTERN)

    assert latest is None
