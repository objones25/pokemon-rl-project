"""The boundary that makes everything else testable.

PyBoy needs a real commercial ROM, which is gitignored and can never exist in
CI. Behind this Protocol, a FakeEmulator over a synthetic 64 KB bytearray
covers ram.py, aux_state.py, rewards.py and session.py -- where essentially
every game-specific bug will live -- with no ROM and no PyBoy.

Verified against PyBoy 2.x: window='null' (not 'headless' or 'dummy', both
removed in 2.0.0); screen.ndarray is (144, 160, 4) uint8 RGBA; tick(count,
render) renders only the LAST frame of the tick; button_press/button_release
take lowercase strings; load_state needs seek(0) first."""

from __future__ import annotations

import io
from typing import Protocol

import numpy as np

SCREEN_HEIGHT = 144
SCREEN_WIDTH = 160


class Emulator(Protocol):
    def tick(self, count: int, render: bool) -> bool: ...
    def button_press(self, button: str) -> None: ...
    def button_release(self, button: str) -> None: ...
    def read_memory(self, addr: int) -> int: ...
    def screen_frame(self) -> np.ndarray: ...
    def save_state(self) -> bytes: ...
    def load_state(self, state: bytes) -> None: ...
    def close(self) -> None: ...


class PyBoyEmulator:
    """Adapter over the real thing. Constructed only inside a worker process."""

    def __init__(self, rom_path: str) -> None:
        from pyboy import PyBoy

        self._pyboy = PyBoy(
            rom_path,
            window="null",
            sound_emulated=False,
            log_level="ERROR",
        )

    def tick(self, count: int, render: bool) -> bool:
        return self._pyboy.tick(count, render)

    def button_press(self, button: str) -> None:
        self._pyboy.button_press(button)

    def button_release(self, button: str) -> None:
        self._pyboy.button_release(button)

    def read_memory(self, addr: int) -> int:
        return self._pyboy.memory[addr]

    def screen_frame(self) -> np.ndarray:
        """(144, 160) uint8. The Game Boy is monochrome so channel 0 of the
        RGBA buffer is the grayscale image.

        ascontiguousarray is load-bearing twice: the channel slice is a
        non-contiguous view, and screen.ndarray references a live backing
        buffer that the next tick overwrites. Returning the view would hand
        callers a frame that silently changes underneath them."""
        return np.ascontiguousarray(self._pyboy.screen.ndarray[:, :, 0])

    def save_state(self) -> bytes:
        buffer = io.BytesIO()
        self._pyboy.save_state(buffer)
        return buffer.getvalue()

    def load_state(self, state: bytes) -> None:
        buffer = io.BytesIO(state)
        buffer.seek(0)  # PyBoy reads from the current position, not from 0
        self._pyboy.load_state(buffer)

    def close(self) -> None:
        self._pyboy.stop(save=False)
