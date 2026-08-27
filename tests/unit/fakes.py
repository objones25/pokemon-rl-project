"""Hand-written test doubles, importable by any test module.

Separate from conftest.py so tests can construct these directly rather than
only receiving them as fixtures -- the vectorized env tests need one fake per
env, which a single fixture cannot supply."""

from __future__ import annotations

import numpy as np

from pokemon_env.aux_state import AUX_STATE_DIM
from pokemon_env.session import StepResult


class FakeEmulator:
    """Hand-written fake typed against the Emulator Protocol, per CLAUDE.md's
    preference for fakes over mock.patch. Records every call so tests can
    assert the exact button/tick sequence."""

    def __init__(self, memory: bytearray | None = None, frame: np.ndarray | None = None) -> None:
        self.memory = bytearray(0x10000) if memory is None else memory
        self.frame = np.zeros((144, 160), dtype=np.uint8) if frame is None else frame
        self.calls: list[tuple] = []
        self.state = b"fake-emulator-state"
        self.closed = False

    def tick(self, count: int, render: bool) -> bool:
        self.calls.append(("tick", count, render))
        return True

    def button_press(self, button: str) -> None:
        self.calls.append(("press", button))

    def button_release(self, button: str) -> None:
        self.calls.append(("release", button))

    def read_memory(self, addr: int) -> int:
        return self.memory[addr]

    def screen_frame(self) -> np.ndarray:
        return self.frame.copy()

    def save_state(self) -> bytes:
        return self.state

    def load_state(self, state: bytes) -> None:
        self.state = state
        self.calls.append(("load_state", len(state)))

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    """Hand-written fake typed against the `EnvBackend` Protocol (`vec_env.py`),
    for VecPokemonEnv tests that need one fake per env without a real
    EnvSession or subprocess worker behind it."""

    def __init__(
        self,
        coord_keys: tuple[int, ...] = (),
        badges: int = 0,
        event_flags: int = 0,
        step_count: int = 0,
    ) -> None:
        self._coord_keys = coord_keys
        self._badges = badges
        self._event_flags = event_flags
        self._step_count = step_count

    def _result(self) -> StepResult:
        return StepResult(
            frame=np.zeros((144, 160), dtype=np.uint8),
            aux=np.zeros(AUX_STATE_DIM, dtype=np.float32),
            reward=0.0,
            done=False,
            episode_id=0,
            components={},
            clipped=False,
        )

    def reset(self) -> StepResult:
        return self._result()

    def step(self, action: int) -> StepResult:
        return self._result()

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def close(self) -> None:
        pass

    def stats(self) -> dict:
        return {
            "coord_keys": list(self._coord_keys),
            "badges": self._badges,
            "event_flags": self._event_flags,
            "step_count": self._step_count,
            "episode_lengths": [],
        }
