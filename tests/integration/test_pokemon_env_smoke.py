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
    """THE measurement the design spec makes task one. The whole
    cache-vs-emulator-state decision pivots on it: if 64 emulator states are
    negligible beside the sequence model's 256 MiB KV cache, both get saved
    together. The 256 KiB bound is 5x the ~50 KB estimate -- generous enough
    not to be brittle, tight enough that a wildly larger state fails loudly."""
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
