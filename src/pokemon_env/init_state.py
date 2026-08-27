"""Produces artifacts/init.state by replaying a committed button script.

Nothing ROM-derived enters git: the script is reviewable text, the ROM is the
one you already own, and the result lands in a gitignored directory. The
alternative of committing or downloading a third-party .state means shipping a
ROM memory dump with provenance we do not control; booting from scratch means
all 64 envs burn the intro, title and naming screens on every reset, learning
a button dance that has nothing to do with the task.

The script below advances past the title, intro cutscene and naming screens to
the first controllable overworld step. Frame counts are generous -- the intro
animations are long and unskippable, and overshooting a menu is far cheaper
than landing in one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pokemon_env.emulator import Emulator


@dataclass(frozen=True)
class ButtonPress:
    """`button=None` is a deliberate wait: the intro has unskippable
    animations, and pressing through them advances past the naming screen."""

    button: str | None
    frames: int


INTRO_SCRIPT: tuple[ButtonPress, ...] = (
    ButtonPress(None, 600),  # boot + Game Freak logo
    ButtonPress("start", 10),  # title screen
    ButtonPress(None, 180),
    ButtonPress("a", 10),  # NEW GAME
    ButtonPress(None, 120),
    ButtonPress("a", 10),  # through Oak's intro
    ButtonPress(None, 240),
    ButtonPress("a", 10),
    ButtonPress(None, 240),
    ButtonPress("a", 10),  # accept the default player name
    ButtonPress(None, 180),
    ButtonPress("a", 10),
    ButtonPress(None, 240),
    ButtonPress("a", 10),  # accept the default rival name
    ButtonPress(None, 600),  # the rest of the intro cutscene
    ButtonPress("a", 10),
    ButtonPress(None, 300),
)


def generate_init_state(emulator: Emulator, script: tuple[ButtonPress, ...]) -> bytes:
    """Replays `script` against a freshly booted emulator and returns the
    resulting save state."""
    for press in script:
        if press.frames < 1:
            raise ValueError(f"frames={press.frames} must be at least 1")
        if press.button is None:
            emulator.tick(press.frames, False)
            continue
        emulator.button_press(press.button)
        emulator.tick(press.frames, False)
        emulator.button_release(press.button)
        emulator.tick(1, False)
    return emulator.save_state()


def state_hash(state: bytes) -> str:
    """Recorded in checkpoints so a resume detects that init.state changed
    underneath it. A different starting state invalidates every reward
    baseline the checkpoint holds."""
    return hashlib.sha256(state).hexdigest()
