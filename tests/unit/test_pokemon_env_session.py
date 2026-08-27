import pytest

from pokemon_env.config import EnvConfig
from pokemon_env.session import ACTION_DIM, BUTTONS, EnvSession


@pytest.fixture
def session(fake_emulator) -> EnvSession:
    return EnvSession(fake_emulator, EnvConfig(max_steps=4), init_state=b"init")


def test_there_are_seven_actions_matching_the_policy_config() -> None:
    assert (len(BUTTONS), ACTION_DIM) == (7, 7)


def test_step_presses_holds_releases_and_renders_only_the_last_frame(
    fake_emulator, session
) -> None:
    """press -> tick(8, False) -> release -> tick(15, False) -> tick(1, True),
    totalling the 24 frames of frame-skip. Rendering only the final frame is
    PyBoy's documented performance guidance."""
    session.reset()
    fake_emulator.calls.clear()

    session.step(4)  # 'a'

    assert fake_emulator.calls == [
        ("press", "a"),
        ("tick", 8, False),
        ("release", "a"),
        ("tick", 15, False),
        ("tick", 1, True),
    ]


def test_reset_loads_the_init_state(fake_emulator, session) -> None:
    session.reset()

    assert fake_emulator.state == b"init"


def test_done_fires_exactly_at_the_step_budget(session) -> None:
    session.reset()
    session.step(0)
    session.step(0)
    session.step(0)

    final = session.step(0)

    assert (final.done, final.episode_id) == (True, 0)


def test_reset_increments_the_episode_id(session) -> None:
    session.reset()

    second = session.reset()

    assert second.episode_id == 1


def test_reset_returns_step_count_to_zero(session) -> None:
    session.reset()
    session.step(0)
    session.reset()

    result = session.step(0)

    assert result.done is False


def test_step_rejects_an_out_of_range_action(session) -> None:
    session.reset()

    with pytest.raises(ValueError, match="action=7 is outside"):
        session.step(7)


def test_frame_has_the_encoder_input_shape(session) -> None:
    result = session.reset()

    assert (result.frame.shape, result.frame.dtype.name) == ((144, 160), "uint8")
