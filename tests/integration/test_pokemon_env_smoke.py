"""Opt-in integration tests against the real ROM and a real PyBoy.

Run with:
    uv run pytest -m slow tests/integration/test_pokemon_env_smoke.py -v

Auto-skipped when the ROM is absent, so a fresh checkout never fails. The ROM
is gitignored and must never be committed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pokemon_env.emulator import PyBoyEmulator

pytestmark = pytest.mark.slow

_ROM = Path("Pokemon Red.gb")

_needs_rom = pytest.mark.skipif(
    not _ROM.exists(), reason=f"{_ROM} not present; it is gitignored and must be supplied locally"
)


@_needs_rom
def test_save_state_is_small_enough_to_checkpoint_all_64_envs() -> None:
    """THE measurement the design spec makes task one. Measured: one state is
    167,677 B (~164 KiB) -- the earlier ~50 KB estimate was 3.3x low. 64 of
    them is 10.73 MB, 4.0% of the sequence model's 256 MiB KV cache, so
    "save both" holds: the cache-vs-emulator-state decision this measurement
    pivots on. The 256 KiB bound is only ~1.5x the measured size -- it is a
    regression guard against PyBoy's state growing, not a generous margin,
    so re-cost this before adding envs, frame history, or a second emulator
    per env."""
    emulator = PyBoyEmulator(str(_ROM))
    emulator.tick(60, False)

    state = emulator.save_state()
    emulator.close()

    assert len(state) < 256 * 1024


@_needs_rom
def test_screen_frame_matches_the_encoder_input_contract() -> None:
    """(144, 160) uint8. GrayscaleResNetEncoder rejects a transposed
    (N, 1, 160, 144), so getting this backwards is caught -- but only after
    the frame has already been through IPC."""
    emulator = PyBoyEmulator(str(_ROM))
    emulator.tick(60, True)

    frame = emulator.screen_frame()
    emulator.close()

    assert (frame.shape, frame.dtype.name) == ((144, 160), "uint8")


@_needs_rom
def test_screen_frame_returns_a_copy_not_a_live_view() -> None:
    """screen.ndarray references a buffer the next tick overwrites. A view
    would mean every frame in a rollout silently became the newest one."""
    emulator = PyBoyEmulator(str(_ROM))
    emulator.tick(60, True)
    first = emulator.screen_frame()
    before = first.copy()

    emulator.tick(600, True)
    emulator.close()

    assert (first == before).all()
