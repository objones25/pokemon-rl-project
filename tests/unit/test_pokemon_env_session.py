from unittest.mock import patch

import pytest

from pokemon_env import ram
from pokemon_env.config import EnvConfig
from pokemon_env.session import ACTION_DIM, BUTTONS, EnvSession

from .fakes import FakeEmulator


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


def test_step_reads_badge_count_and_event_flag_count_exactly_once_each(session) -> None:
    """rewards.step() and aux_state.build_aux_state() each independently need
    these two RAM-derived values. EnvSession.step() must compute each once
    and thread the result to both, not let each side re-read RAM for a value
    that has not changed since the tick just completed -- event_flag_count in
    particular is a 311-byte popcount, not a single-byte read."""
    session.reset()

    with (
        patch.object(ram, "badge_count", wraps=ram.badge_count) as badge_spy,
        patch.object(ram, "event_flag_count", wraps=ram.event_flag_count) as event_spy,
    ):
        session.step(0)

    assert (badge_spy.call_count, event_spy.call_count) == (1, 1)


def test_frame_has_the_encoder_input_shape(session) -> None:
    result = session.reset()

    assert (result.frame.shape, result.frame.dtype.name) == ((144, 160), "uint8")


def test_stats_reports_the_coordinate_keys_the_accumulator_has_seen() -> None:
    session = EnvSession(FakeEmulator(), EnvConfig(), init_state=b"")
    session.reset()
    session.step(0)

    assert session.stats()["coord_keys"] == [ram.coord_key(0, 0, 0)]


def test_stats_reports_the_length_of_a_completed_episode() -> None:
    session = EnvSession(FakeEmulator(), EnvConfig(max_steps=2), init_state=b"")
    session.reset()
    session.step(0)
    session.step(0)
    session.reset()

    assert session.stats()["episode_lengths"] == [2]


def test_stats_drains_the_episode_length_history_so_lengths_are_not_double_counted() -> None:
    session = EnvSession(FakeEmulator(), EnvConfig(max_steps=2), init_state=b"")
    session.reset()
    session.step(0)
    session.step(0)
    session.reset()
    session.stats()

    assert session.stats()["episode_lengths"] == []


def test_state_dict_round_trips_the_episode_length_history() -> None:
    """The design spec's §11 item 5 is what justifies bumping
    VEC_ENV_SCHEMA_VERSION in a later task: episode-length history must
    survive a save/restore cycle through session state_dict, not just live in
    memory for the current process's lifetime."""
    session = EnvSession(FakeEmulator(), EnvConfig(max_steps=2), init_state=b"init")
    session.reset()
    session.step(0)
    session.step(0)
    session.reset()
    state = session.state_dict()

    restored = EnvSession(FakeEmulator(), EnvConfig(max_steps=2), init_state=b"init")
    restored.load_state_dict(state)

    assert restored.stats()["episode_lengths"] == [2]


def test_state_dict_snapshot_is_unaffected_by_a_later_reset_on_the_same_session() -> None:
    """state_dict() must copy episode_lengths, not alias the live list.
    InProcessBackend hands this dict straight back to callers with no
    serialization boundary in between, so a caller holding an earlier
    snapshot must not see it mutate underneath them when the session that
    produced it keeps running."""
    session = EnvSession(FakeEmulator(), EnvConfig(max_steps=2), init_state=b"init")
    session.reset()
    session.step(0)
    session.step(0)
    session.reset()
    snapshot = session.state_dict()

    session.step(0)
    session.step(0)
    session.reset()

    assert snapshot["episode_lengths"] == [2]
