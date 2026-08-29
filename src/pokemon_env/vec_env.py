"""Parent-side vectorization over N env backends.

Autoreset is next-step, deliberately: done=True at step t returns the TERMINAL
observation, and the reset observation arrives at t+1. This exists to satisfy
the sequence-model spec's cache.reset(done) ordering contract -- reset must run
AFTER the step whose transition ended the episode, or the final transition of
every episode attends to a cleared cache. Making it structural here means the
trainer cannot get the ordering wrong.

The backend Protocol has two implementations: InProcessBackend (here, for
tests and debugging -- roughly 64x too slow for a real run) and
SubprocessBackend (subprocess_backend.py, what production uses)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pokemon_env.aux_state import AUX_STATE_DIM, AUX_STATE_VERSION
from pokemon_env.config import EnvConfig
from pokemon_env.emulator import SCREEN_HEIGHT, SCREEN_WIDTH
from pokemon_env.session import EnvSession, StepResult

# Bumped 1->2 because EnvSession.load_state_dict now reads
# state["episode_lengths"] directly (Task 2); a checkpoint written before that
# change has no such key and would otherwise fail with a bare KeyError instead
# of this module's legible version-mismatch error.
VEC_ENV_SCHEMA_VERSION = 2


class EnvBackend(Protocol):
    def send_reset(self) -> None: ...
    def send_step(self, action: int) -> None: ...
    def recv(self) -> StepResult: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def stats(self) -> dict: ...
    def close(self) -> None: ...


class InProcessBackend:
    """Drives an EnvSession directly. Exists so vec_env logic -- autoreset
    ordering, episode_id monotonicity, batching -- is testable without
    spawning processes.

    Has no concurrency to exploit -- it drives one EnvSession in the parent
    process -- so send_step/send_reset just run the work eagerly and stash
    the StepResult; recv() returns and clears it. The stash also enforces
    the send/recv call-order invariant SubprocessBackend depends on for real
    concurrency: a second send before a recv, or a recv with nothing
    pending, is a programmer error in the dispatch loop, not something that
    should silently succeed with a stale value."""

    def __init__(self, session: EnvSession) -> None:
        self._session = session
        self._pending: StepResult | None = None

    def _dispatch(self, result: StepResult) -> None:
        if self._pending is not None:
            raise RuntimeError(
                "InProcessBackend: send_step/send_reset called while a previous "
                "dispatch has not been recv()'d yet -- sends and recvs must "
                "alternate one-for-one"
            )
        self._pending = result

    def send_reset(self) -> None:
        self._dispatch(self._session.reset())

    def send_step(self, action: int) -> None:
        self._dispatch(self._session.step(action))

    def recv(self) -> StepResult:
        if self._pending is None:
            raise RuntimeError(
                "InProcessBackend: recv() called with no matching "
                "send_step/send_reset"
            )
        result, self._pending = self._pending, None
        return result

    def state_dict(self) -> dict:
        """Same envelope as SubprocessBackend's, so a checkpoint is portable
        between the two backends. The parent-side counters are constant here:
        an in-process backend has no worker to lose and never respawns."""
        return {
            "session": self._session.state_dict(),
            "respawns": 0,
            "episode_offset": 0,
            "last_episode_id": -1,
        }

    def load_state_dict(self, state: dict) -> None:
        self._session.load_state_dict(state["session"])

    def stats(self) -> dict:
        return self._session.stats()

    def close(self) -> None:
        self._session.close()


@dataclass(frozen=True)
class VecStep:
    frames: np.ndarray  # (N, 1, 144, 160) uint8
    aux: np.ndarray  # (N, 32) float32
    reward: np.ndarray  # (N,) float32
    done: np.ndarray  # (N,) bool
    episode_id: np.ndarray  # (N,) int64


class VecPokemonEnv:
    def __init__(self, backends: Sequence[EnvBackend], config: EnvConfig) -> None:
        if len(backends) != config.n_envs:
            raise ValueError(
                f"got {len(backends)} backends for n_envs={config.n_envs}; "
                "the batch dimension must match the configured env count"
            )
        self._backends = backends
        self._config = config
        self._needs_reset = np.zeros(config.n_envs, dtype=bool)
        self._last_components: dict[str, float] = {}
        self._last_step: VecStep | None = None
        self._clipped_steps = 0
        self._total_steps = 0

    @property
    def n_envs(self) -> int:
        return self._config.n_envs

    @property
    def last_components(self) -> dict[str, float]:
        """Mean reward per component over the most recent vector step."""
        return dict(self._last_components)

    @property
    def last_step(self) -> VecStep | None:
        """The most recent VecStep, for end-of-update telemetry that needs
        the raw reward/done array alongside the mean components above."""
        return self._last_step

    @property
    def clip_fire_rate(self) -> float:
        if self._total_steps == 0:
            return 0.0
        return self._clipped_steps / self._total_steps

    def reset(self) -> VecStep:
        self._needs_reset[:] = False
        for backend in self._backends:
            backend.send_reset()
        return self._collect([backend.recv() for backend in self._backends])

    def step(self, actions: np.ndarray) -> VecStep:
        if len(actions) != self._config.n_envs:
            raise ValueError(
                f"actions has length {len(actions)}, expected {self._config.n_envs}"
            )
        for backend, action, needs_reset in zip(
            self._backends, actions, self._needs_reset, strict=True
        ):
            if needs_reset:
                backend.send_reset()
            else:
                backend.send_step(int(action))
        results = [backend.recv() for backend in self._backends]
        self._needs_reset = np.array([result.done for result in results], dtype=bool)
        return self._collect(results)

    def _collect(self, results: list[StepResult]) -> VecStep:
        self._total_steps += len(results)
        self._clipped_steps += sum(1 for result in results if result.clipped)
        self._last_components = _mean_components(results)

        frames = np.empty(
            (len(results), 1, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8
        )
        aux = np.empty((len(results), AUX_STATE_DIM), dtype=np.float32)
        for i, result in enumerate(results):
            frames[i, 0] = result.frame
            aux[i] = result.aux

        step = VecStep(
            frames=frames,
            aux=aux,
            reward=np.array([r.reward for r in results], dtype=np.float32),
            done=np.array([r.done for r in results], dtype=bool),
            episode_id=np.array([r.episode_id for r in results], dtype=np.int64),
        )
        self._last_step = step
        return step

    def stats(self) -> list[dict]:
        """One dict per env, in env order. Called once per PPO update."""
        return [backend.stats() for backend in self._backends]

    def state_dict(self) -> dict:
        return {
            "schema_version": VEC_ENV_SCHEMA_VERSION,
            "aux_state_version": AUX_STATE_VERSION,
            "needs_reset": self._needs_reset.tolist(),
            "backends": [backend.state_dict() for backend in self._backends],
        }

    def load_state_dict(self, state: dict) -> None:
        # Checked, not merely written. An unvalidated version field is worse
        # than none: it reads as protection while a schema change resumes
        # silently against a mismatched layout.
        if state["schema_version"] != VEC_ENV_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint has schema_version={state['schema_version']}, this build "
                f"is {VEC_ENV_SCHEMA_VERSION}. The vec-env state layout changed, so "
                "per-env state would be restored into fields that no longer mean the "
                "same thing."
            )
        if state["aux_state_version"] != AUX_STATE_VERSION:
            raise ValueError(
                f"checkpoint has AUX_STATE_VERSION={state['aux_state_version']}, "
                f"this build is {AUX_STATE_VERSION}. The aux vector's slot layout "
                "changed, so a policy trained against the old one would be silently "
                "reading different signals from the same indices."
            )
        if len(state["backends"]) != self._config.n_envs:
            raise ValueError(
                f"checkpoint holds {len(state['backends'])} envs but this run has "
                f"{self._config.n_envs}; per-env state cannot be redistributed"
            )
        self._needs_reset = np.array(state["needs_reset"], dtype=bool)
        for backend, backend_state in zip(self._backends, state["backends"], strict=True):
            backend.load_state_dict(backend_state)

    def close(self) -> None:
        for backend in self._backends:
            backend.close()


def _mean_components(results: list[StepResult]) -> dict[str, float]:
    """Helper: mean of each reward component across envs, for telemetry."""
    keys = {key for result in results for key in result.components}
    return {
        key: sum(result.components.get(key, 0.0) for result in results) / len(results)
        for key in sorted(keys)
    }
