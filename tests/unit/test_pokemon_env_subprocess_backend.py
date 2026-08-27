from collections.abc import Iterator

import numpy as np
import pytest

from pokemon_env.config import EnvConfig
from pokemon_env.session import EnvSession
from pokemon_env.subprocess_backend import Command, FrameBuffer, handle_command

from .fakes import FakeEmulator


@pytest.fixture
def frame_buffer() -> Iterator[FrameBuffer]:
    """Iterator, not FrameBuffer: a yield fixture's annotation describes what
    the function returns, and a type checker flags the mismatch."""
    buffer = FrameBuffer.create(n_envs=3)
    yield buffer
    buffer.close()
    buffer.unlink()


@pytest.fixture
def emulator() -> FakeEmulator:
    return FakeEmulator()


@pytest.fixture
def session(emulator: FakeEmulator) -> EnvSession:
    return EnvSession(emulator, EnvConfig(n_envs=3, max_steps=4), init_state=b"init")


def test_frame_buffer_has_one_slot_per_env(frame_buffer) -> None:
    assert (frame_buffer.array.shape, frame_buffer.array.dtype.name) == ((3, 144, 160), "uint8")


def test_frame_buffer_slots_are_independent(frame_buffer) -> None:
    """Workers write disjoint slices with no locking, so a shape or stride
    error that made them overlap would silently mix envs' observations."""
    frame_buffer.array[0] = 7
    frame_buffer.array[1] = 9

    assert (int(frame_buffer.array[0, 0, 0]), int(frame_buffer.array[1, 0, 0])) == (7, 9)


def test_frame_buffer_attaches_to_an_existing_block_by_name(frame_buffer) -> None:
    frame_buffer.array[2] = 5
    attached = FrameBuffer.attach(frame_buffer.name, n_envs=3)

    value = int(attached.array[2, 0, 0])
    attached.close()

    assert value == 5


def test_handle_reset_writes_the_frame_into_the_shared_slot(
    frame_buffer, emulator, session
) -> None:
    """The fake is injected as its own fixture rather than reached for through
    session._emulator -- a test that pokes a private attribute breaks on any
    refactor that renames it."""
    emulator.frame = np.full((144, 160), 3, dtype=np.uint8)

    handle_command(session, Command.RESET, None, frame_buffer.array[1])

    assert int(frame_buffer.array[1, 0, 0]) == 3


def test_handle_command_writes_only_its_own_slot(
    frame_buffer, emulator, session
) -> None:
    """Workers write disjoint slices with no locking. A slot-index error would
    silently overwrite a neighbouring env's observation."""
    emulator.frame = np.full((144, 160), 3, dtype=np.uint8)

    handle_command(session, Command.RESET, None, frame_buffer.array[1])

    assert (int(frame_buffer.array[0, 0, 0]), int(frame_buffer.array[2, 0, 0])) == (0, 0)


def test_handle_reset_returns_the_payload_without_the_frame(
    frame_buffer, session
) -> None:
    """The frame goes through shared memory; everything else is small enough
    to ride the pipe. Including the frame in the payload would reintroduce the
    1.5 GB-per-rollout pickling cost the shared block exists to remove."""
    payload = handle_command(session, Command.RESET, None, frame_buffer.array[0])

    assert "frame" not in payload


def test_handle_step_advances_the_episode(frame_buffer, session) -> None:
    handle_command(session, Command.RESET, None, frame_buffer.array[0])

    payload = handle_command(session, Command.STEP, 3, frame_buffer.array[0])

    assert (payload["episode_id"], payload["done"]) == (0, False)


def test_handle_command_rejects_an_unknown_command(frame_buffer, session) -> None:
    with pytest.raises(ValueError, match="unknown command"):
        handle_command(session, "NOT_A_COMMAND", None, frame_buffer.array[0])
