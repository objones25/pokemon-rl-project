import pytest

from pokemon_env.init_state import (
    INTRO_SCRIPT,
    ButtonPress,
    generate_init_state,
    state_hash,
)


def test_generate_init_state_returns_the_emulator_state(fake_emulator) -> None:
    fake_emulator.state = b"post-intro"

    result = generate_init_state(fake_emulator, (ButtonPress(button="a", frames=2),))

    assert result == b"post-intro"


def test_a_press_ticks_then_releases(fake_emulator) -> None:
    generate_init_state(fake_emulator, (ButtonPress(button="start", frames=3),))

    assert fake_emulator.calls == [
        ("press", "start"),
        ("tick", 3, False),
        ("release", "start"),
        ("tick", 1, False),
    ]


def test_a_wait_ticks_without_touching_any_button(fake_emulator) -> None:
    """A None button is a deliberate wait -- the intro has long unskippable
    animations, and pressing through them advances past the naming screen."""
    generate_init_state(fake_emulator, (ButtonPress(button=None, frames=60),))

    assert fake_emulator.calls == [("tick", 60, False)]


def test_the_intro_script_is_not_empty() -> None:
    """An empty script would produce a boot-screen state that loads fine and
    leaves all 64 agents stuck at the title."""
    assert len(INTRO_SCRIPT) > 0


def test_state_hash_is_stable_for_identical_bytes() -> None:
    assert state_hash(b"abc") == state_hash(b"abc")


def test_state_hash_differs_for_different_bytes() -> None:
    """Recorded in checkpoints so a resume detects that init.state changed
    underneath it -- a different starting state invalidates every reward
    baseline in the checkpoint."""
    assert state_hash(b"abc") != state_hash(b"abd")


def test_generate_init_state_rejects_a_non_positive_frame_count(fake_emulator) -> None:
    with pytest.raises(ValueError, match="frames=0 must be at least 1"):
        generate_init_state(fake_emulator, (ButtonPress(button="a", frames=0),))
