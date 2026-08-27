"""Hand-written test doubles, importable by any test module.

Separate from conftest.py so tests can construct these directly rather than
only receiving them as fixtures -- the vectorized env tests need one fake per
env, which a single fixture cannot supply."""

from __future__ import annotations

import numpy as np


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
