from collections.abc import Iterator

import numpy as np
import pytest

from pokemon_env.config import EnvConfig
from pokemon_env.session import EnvSession
from pokemon_env.subprocess_backend import (
    Command,
    FrameBuffer,
    SubprocessBackend,
    handle_command,
    worker_main,
)

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


class FakeConnection:
    """Stands in for a multiprocessing Pipe end. `poll_results` and
    `responses` are consumed in order, so a test scripts the exact sequence
    the backend will observe."""

    def __init__(self, responses: list, poll_results: list[bool] | None = None) -> None:
        self.sent: list = []
        self.responses = list(responses)
        self.poll_results = list(poll_results) if poll_results is not None else []
        self.closed = False

    def send(self, obj: object) -> None:
        self.sent.append(obj)

    def poll(self, timeout: float | None = None) -> bool:
        return self.poll_results.pop(0) if self.poll_results else True

    def recv(self):
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.terminated = False
        self.joined = False

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.joined = True


def _ok(index: int = 0) -> tuple:
    """Helper, not a test: a well-formed worker response."""
    return (
        "ok",
        {
            "aux": np.zeros(32, dtype=np.float32),
            "reward": 0.0,
            "done": False,
            "episode_id": index,
            "components": {},
            "clipped": False,
        },
    )


def _backend(frame_buffer, responses, poll_results=None, index=0):
    """Helper, not a test: a SubprocessBackend wired to scripted fakes.
    Returns (backend, connection, process) so tests can assert on all three."""
    connection = FakeConnection(responses, poll_results)
    process = FakeProcess()

    def fake_spawn(shm_name, idx, config, rom_path, init_state):
        return connection, process

    backend = SubprocessBackend(
        index=index,
        shm_name=frame_buffer.name,
        config=EnvConfig(n_envs=3, max_steps=8),
        rom_path="unused.gb",
        init_state=b"init",
        frame_slot=frame_buffer.array[index],
        spawn_worker=fake_spawn,
    )
    return backend, connection, process


def test_step_sends_the_step_command_with_its_action(frame_buffer) -> None:
    backend, connection, _ = _backend(frame_buffer, [_ok(), _ok()])
    backend.reset()

    backend.step(3)

    assert connection.sent[-1] == (Command.STEP, 3)


def test_call_raises_timeout_naming_the_env_index(frame_buffer) -> None:
    """A 24-frame tick takes about 1ms, so a 60s silence is a hang, not
    slowness. The message must name the env so an operator knows which of
    64 workers died."""
    backend, _, _ = _backend(frame_buffer, [_ok()], poll_results=[False], index=2)

    with pytest.raises(TimeoutError, match="env 2 did not answer"):
        backend.reset()


def test_a_worker_error_response_raises_naming_the_env(frame_buffer) -> None:
    backend, _, _ = _backend(frame_buffer, [("error", "ValueError: boom")], index=1)

    with pytest.raises(RuntimeError, match="env 1 worker failed"):
        backend.reset()


def test_step_respawns_and_forces_done_when_the_worker_times_out(
    frame_buffer,
) -> None:
    """A dead worker must not take the run down. It respawns from init.state
    and forces done=True so the trainer resets its memory for that env --
    a respawned worker shares no history with the old one."""
    backend, _, _ = _backend(
        frame_buffer, [_ok(), _ok()], poll_results=[True, False, True]
    )
    backend.reset()

    result = backend.step(0)

    assert (result.done, result.reward) == (True, 0.0)


def test_a_respawn_increments_the_respawn_counter(frame_buffer) -> None:
    """Respawn rate is a logged leading indicator of memory pressure, so the
    counter must actually move."""
    backend, _, _ = _backend(
        frame_buffer, [_ok(), _ok()], poll_results=[True, False, True]
    )
    backend.reset()
    backend.step(0)

    assert backend.respawns == 1


def test_a_respawn_terminates_a_process_that_is_still_alive(frame_buffer) -> None:
    """A hung worker is alive but unresponsive; leaving it running leaks a
    process and its emulator for the rest of the run."""
    backend, _, process = _backend(
        frame_buffer, [_ok(), _ok()], poll_results=[True, False, True]
    )
    backend.reset()
    backend.step(0)

    assert process.terminated is True


def test_close_sends_the_close_command(frame_buffer) -> None:
    backend, connection, _ = _backend(frame_buffer, [_ok()])

    backend.close()

    assert connection.sent[-1] == (Command.CLOSE, None)


def test_close_terminates_a_worker_that_will_not_exit(frame_buffer) -> None:
    backend, _, process = _backend(frame_buffer, [_ok()])

    backend.close()

    assert (process.joined, process.terminated) == (True, True)


def test_state_dict_returns_the_workers_state(frame_buffer) -> None:
    backend, _, _ = _backend(frame_buffer, [("ok", {"state": {"step_count": 7}})])

    assert backend.state_dict() == {"step_count": 7}


def test_to_result_takes_its_frame_from_the_shared_slot(frame_buffer) -> None:
    """The frame never rides the pipe -- the backend reads it from its own
    shared-memory slice. If it read the payload instead, the 1.5 GB per
    rollout the shared block exists to remove would come straight back."""
    frame_buffer.array[0] = 77
    backend, _, _ = _backend(frame_buffer, [_ok()])

    result = backend.reset()

    assert int(result.frame[0, 0]) == 77


def test_worker_main_uses_the_injected_emulator_factory(frame_buffer) -> None:
    """worker_main drives one command then exits on CLOSE. With a fake
    factory this runs the real loop body with no process and no ROM."""
    emulator = FakeEmulator()
    connection = FakeConnection([(Command.RESET, None), (Command.CLOSE, None)])

    worker_main(
        connection,
        frame_buffer.name,
        0,
        EnvConfig(n_envs=3, max_steps=8),
        "unused.gb",
        b"init",
        emulator_factory=lambda rom_path: emulator,
    )

    assert emulator.closed is True


def test_worker_main_writes_only_into_its_own_slot(frame_buffer) -> None:
    """THE index-binding test, and the reason this task exists at all.

    worker_main writes `buffer.array[index]` while the parent reads its own
    `frame_slot`. If those ever disagreed, environment A would silently
    receive environment B's screen -- slices are deliberately unlocked
    because disjoint, so a mix-up raises nothing.

    This could not be tested before the emulator became injectable: the
    integration test cannot catch it either, because `frames[i]` is copied
    FROM `buffer.array[i]`, so comparing them is a tautology that holds even
    under a swap, and a distinct-frame count still passes when unwritten
    slots keep their zero-initialised bytes.

    Here the fake's frame is a known constant, so the assertion pins which
    slot received it AND that the neighbours were untouched."""
    emulator = FakeEmulator(frame=np.full((144, 160), 99, dtype=np.uint8))
    connection = FakeConnection([(Command.RESET, None), (Command.CLOSE, None)])

    worker_main(
        connection,
        frame_buffer.name,
        1,  # writes slot 1 only
        EnvConfig(n_envs=3, max_steps=8),
        "unused.gb",
        b"init",
        emulator_factory=lambda rom_path: emulator,
    )

    written = [int(frame_buffer.array[i, 0, 0]) for i in range(3)]
    assert written == [0, 99, 0]


def test_worker_main_reports_an_exception_back_to_the_parent(frame_buffer) -> None:
    """A worker that dies silently leaves the parent blocked until its 60s
    timeout. Errors must travel back over the pipe instead."""
    connection = FakeConnection([(Command.STEP, 99), (Command.CLOSE, None)])

    worker_main(
        connection,
        frame_buffer.name,
        0,
        EnvConfig(n_envs=3, max_steps=8),
        "unused.gb",
        b"init",
        emulator_factory=lambda rom_path: FakeEmulator(),
    )

    assert connection.sent[0][0] == "error"
