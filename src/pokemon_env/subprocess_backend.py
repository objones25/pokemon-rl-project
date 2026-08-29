"""64 subprocess workers, each owning one PyBoy emulator.

PyBoy is GIL-bound, so threads buy no parallelism. Frames cross into the
parent through one SharedMemory block -- a frame is 23,040 B, so 64 of them is
1.47 MB per vector step and 1.5 GB of pickling per 1024-step rollout, roughly
19% of the 8.0 s rollout budget. Shared memory makes it a memcpy. No locking:
slices are disjoint and the parent joins all responses before reading.

Spawn, not fork: fork would duplicate the parent's CUDA context into every
worker, which CUDA does not support."""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from enum import StrEnum
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Protocol

import numpy as np

from pokemon_env.config import EnvConfig
from pokemon_env.emulator import SCREEN_HEIGHT, SCREEN_WIDTH, Emulator, PyBoyEmulator
from pokemon_env.session import EnvSession, StepResult
from pokemon_env.vec_env import VecPokemonEnv

logger = logging.getLogger(__name__)


class WorkerConnection(Protocol):
    """The pipe end this module actually uses, as a Protocol rather than
    `multiprocessing.connection.Connection`.

    Typing against the concrete class made the injected `spawn_worker`
    untypable: a hand-written fake cannot be a `Connection`, so every test
    that scripts one was a type error even though the injection point exists
    precisely so tests can supply a fake. `Connection` satisfies this
    structurally, so production is unaffected."""

    def send(self, obj: Any) -> None: ...
    def poll(self, timeout: float | None = ...) -> bool: ...
    def recv(self) -> Any: ...
    def close(self) -> None: ...


class WorkerProcess(Protocol):
    """Likewise for the process handle: only these three methods are used, and
    `BaseProcess` satisfies them structurally."""

    def is_alive(self) -> bool: ...
    def terminate(self) -> None: ...
    def join(self, timeout: float | None = ...) -> None: ...


class Command(StrEnum):
    RESET = "RESET"
    STEP = "STEP"
    STATE_DICT = "STATE_DICT"
    LOAD_STATE = "LOAD_STATE"
    STATS = "STATS"
    CLOSE = "CLOSE"


class FrameBuffer:
    """One (n_envs, 144, 160) uint8 block. `create` in the parent, `attach` by
    name in each worker."""

    def __init__(self, shm: SharedMemory, n_envs: int, owner: bool) -> None:
        self._shm = shm
        self._owner = owner
        self.array = np.ndarray(
            (n_envs, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8, buffer=shm.buf
        )

    @classmethod
    def create(cls, n_envs: int) -> FrameBuffer:
        shm = SharedMemory(create=True, size=n_envs * SCREEN_HEIGHT * SCREEN_WIDTH)
        return cls(shm, n_envs, owner=True)

    @classmethod
    def attach(cls, name: str, n_envs: int) -> FrameBuffer:
        return cls(SharedMemory(name=name), n_envs, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    def close(self) -> None:
        # Drop the ndarray's reference to the buffer first: SharedMemory.close()
        # raises BufferError while an exported memoryview is still alive.
        del self.array
        self._shm.close()

    def unlink(self) -> None:
        """Parent only. Releases the OS-level block.

        Guarded rather than merely documented: unlink destroys the block for
        every attached process, so a worker calling it would take all 64 envs'
        frame slots with it."""
        if not self._owner:
            raise RuntimeError(
                "unlink() may only be called by the process that created the block; "
                "calling it from a worker destroys every env's frame slot"
            )
        self._shm.unlink()


def _payload(result: StepResult) -> dict:
    """Everything except the frame, which travels through shared memory."""
    return {
        "aux": result.aux,
        "reward": result.reward,
        "done": result.done,
        "episode_id": result.episode_id,
        "components": result.components,
        "clipped": result.clipped,
    }


def handle_command(
    session: EnvSession, command: str, argument: object, frame_slot: np.ndarray
) -> dict:
    """Pure dispatch, so the worker's behaviour is testable without spawning.
    Writes the frame into `frame_slot` and returns the pipe payload."""
    if command == Command.RESET:
        result = session.reset()
    elif command == Command.STEP:
        result = session.step(int(argument))  # type: ignore[arg-type]
    elif command == Command.STATE_DICT:
        return {"state": session.state_dict()}
    elif command == Command.LOAD_STATE:
        session.load_state_dict(argument)  # type: ignore[arg-type]
        return {"ok": True}
    elif command == Command.STATS:
        return {"stats": session.stats()}
    else:
        raise ValueError(f"unknown command {command!r}")

    frame_slot[:] = result.frame
    return _payload(result)


def worker_main(
    conn: WorkerConnection,
    shm_name: str,
    index: int,
    config: EnvConfig,
    rom_path: str,
    init_state: bytes,
    emulator_factory: Callable[[str], Emulator] = PyBoyEmulator,
) -> None:
    """Worker entry point. Owns exactly one emulator for its lifetime."""
    buffer = FrameBuffer.attach(shm_name, config.n_envs)
    session = EnvSession(emulator_factory(rom_path), config, init_state)
    try:
        while True:
            command, argument = conn.recv()
            if command == Command.CLOSE:
                break
            try:
                conn.send(("ok", handle_command(session, command, argument, buffer.array[index])))
            except Exception as error:  # must reach the parent, not die silently
                logger.exception("worker_command_failed", extra={"command": str(command)})
                conn.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        session.close()
        buffer.close()
        conn.close()


SpawnWorker = Callable[
    [str, int, EnvConfig, str, bytes], tuple[WorkerConnection, WorkerProcess]
]


def spawn_real_worker(
    shm_name: str,
    index: int,
    config: EnvConfig,
    rom_path: str,
    init_state: bytes,
) -> tuple[WorkerConnection, WorkerProcess]:
    """The production spawn. Module-level so `spawn` can pickle it by
    reference, and injectable so SubprocessBackend's timeout, error-routing
    and respawn logic can be tested without a real process or a ROM."""
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe()
    process = context.Process(
        target=worker_main,
        args=(child_conn, shm_name, index, config, rom_path, init_state),
        daemon=True,
    )
    process.start()
    child_conn.close()
    return parent_conn, process


class SubprocessBackend:
    """Parent-side handle on one worker. Respawns it from init.state on death
    or timeout rather than taking the whole run down.

    step/reset are two-phase: send_step()/send_reset() dispatch the command
    and return immediately, catching only a failure of the send itself.
    recv() blocks for the reply. VecPokemonEnv relies on this split to fire
    every backend's send before waiting on any reply -- see
    docs/superpowers/specs/2026-08-29-vec-env-step-concurrency-design.md."""

    def __init__(
        self,
        index: int,
        shm_name: str,
        config: EnvConfig,
        rom_path: str,
        init_state: bytes,
        frame_slot: np.ndarray,
        spawn_worker: SpawnWorker = spawn_real_worker,
    ) -> None:
        self._index = index
        self._shm_name = shm_name
        self._config = config
        self._rom_path = rom_path
        self._init_state = init_state
        self._frame_slot = frame_slot
        self._spawn_worker = spawn_worker
        self._respawns = 0
        # A respawned worker gets a fresh EnvSession whose episode_id restarts
        # at 0, but VecStep's contract is that episode_id is MONOTONIC per env
        # -- the transformer uses it to detect episode boundaries, and a
        # repeated id merges two distinct episodes in its attention mask. The
        # offset keeps the parent-visible sequence climbing across respawns.
        self._episode_offset = 0
        self._last_episode_id = -1
        # Two-phase dispatch state. None means "no send is outstanding";
        # recv() clears it back to None as soon as it reads it, so a stray
        # second recv() (or a recv() with no matching send) is caught here
        # rather than silently blocking on a pipe read that was never primed.
        self._pending_is_step: bool | None = None
        self._send_failed = False
        self._spawn()

    @property
    def respawns(self) -> int:
        return self._respawns

    def _spawn(self) -> None:
        self._conn, self._process = self._spawn_worker(
            self._shm_name, self._index, self._config, self._rom_path, self._init_state
        )

    def _call(self, command: Command, argument: object = None) -> dict:
        """Synchronous send+recv, for commands that are never dispatched
        two-phase (STATE_DICT, LOAD_STATE, STATS -- once per PPO update, not
        once per step) and for the internal reset used to recover after a
        respawn."""
        self._conn.send((command, argument))
        if not self._conn.poll(self._config.worker_timeout_s):
            raise TimeoutError(
                f"env {self._index} did not answer {command} within "
                f"{self._config.worker_timeout_s}s; a 24-frame tick takes about 1ms, "
                "so this is a hang, not slowness"
            )
        status, payload = self._conn.recv()
        if status == "error":
            raise RuntimeError(f"env {self._index} worker failed: {payload}")
        return payload

    def _restart(self) -> StepResult:
        """Respawn from init.state, NOT from the last checkpoint's emulator
        state: that state pairs with a checkpoint-time reward baseline and
        coord set, so restoring it against a current-time accumulator would
        silently re-earn rewards for progress already banked.

        Ends in `_reset_once`, deliberately NOT a two-phase send_reset()+
        recv(): `_reset_once` recovers by calling back into here, so
        recovering twice would recurse forever on a worker that dies on
        every spawn. An unattended run that spins silently is worse than one
        that dies with a clear error."""
        self._respawns += 1
        self._episode_offset = self._last_episode_id + 1
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=5)
        self._close_connection()
        self._spawn()
        return self._reset_once()

    def _close_connection(self) -> None:
        """Release the old pipe's file descriptors. Respawns are expected over
        a multi-day run, so leaking one FD pair per respawn eventually
        exhausts the parent's descriptor table."""
        try:
            self._conn.close()
        except OSError:  # obs: allow LOG007 -- the pipe is already closed; nothing to report
            pass

    def _to_result(self, payload: dict) -> StepResult:
        episode_id = int(payload["episode_id"]) + self._episode_offset
        self._last_episode_id = max(self._last_episode_id, episode_id)
        return StepResult(
            frame=self._frame_slot,
            aux=payload["aux"],
            reward=payload["reward"],
            done=payload["done"],
            episode_id=episode_id,
            components=payload["components"],
            clipped=payload["clipped"],
        )

    def _reset_once(self) -> StepResult:
        """Bare reset with no recovery. Used by `_restart` so recovery cannot
        recurse."""
        return self._to_result(self._call(Command.RESET))

    def _dispatch(self, command: Command, argument: object, is_step: bool, label: str) -> None:
        if self._pending_is_step is not None:
            raise RuntimeError(
                f"env {self._index}: {label} called while a previous dispatch has "
                "not been recv()'d yet -- sends and recvs must alternate one-for-one"
            )
        self._pending_is_step = is_step
        try:
            self._conn.send((command, argument))
            self._send_failed = False
        except (BrokenPipeError, OSError):
            # The worker is already dead. Swallowed here, not raised: this
            # runs inside VecPokemonEnv's dispatch loop, which has no
            # try/except, so raising would abort dispatch to every backend
            # after this one before recv() is even called on the ones that
            # already sent fine. recv() surfaces it instead, on the same
            # _restart() path a recv()-side timeout already takes.
            self._send_failed = True

    def send_reset(self) -> None:
        """VecPokemonEnv routes every autoreset through here, so a worker that
        dies on an episode boundary must be recovered via recv() below --
        otherwise it propagates out of VecPokemonEnv.step() and kills the
        run."""
        self._dispatch(Command.RESET, None, is_step=False, label="send_reset")

    def send_step(self, action: int) -> None:
        self._dispatch(Command.STEP, action, is_step=True, label="send_step")

    def recv(self) -> StepResult:
        if self._pending_is_step is None:
            raise RuntimeError(
                f"env {self._index}: recv() called with no matching "
                "send_step/send_reset"
            )
        is_step = self._pending_is_step
        self._pending_is_step = None
        command_name = "STEP" if is_step else "RESET"
        try:
            if self._send_failed:
                raise BrokenPipeError(
                    f"env {self._index}: send failed before the worker could reply"
                )
            if not self._conn.poll(self._config.worker_timeout_s):
                raise TimeoutError(
                    f"env {self._index} did not answer {command_name} within "
                    f"{self._config.worker_timeout_s}s; a 24-frame tick takes about "
                    "1ms, so this is a hang, not slowness"
                )
            status, payload = self._conn.recv()
        except (TimeoutError, EOFError, BrokenPipeError):
            restarted = self._restart()
            if not is_step:
                return restarted
            # Force the episode boundary so the trainer resets its KV cache
            # for this env; a respawned worker shares no history with the
            # old one.
            return StepResult(
                frame=restarted.frame,
                aux=restarted.aux,
                reward=0.0,
                done=True,
                episode_id=restarted.episode_id,
                components={},
                clipped=False,
            )
        if status == "error":
            # A software bug in this project's own reward/observation code,
            # not a process failure -- the worker is still alive and
            # replied. Raised here, outside the except block above, so it
            # can never be caught by the respawn path and silently
            # discarded as an elevated respawn count. See "Adjacent bug
            # found during review" in the spec.
            raise RuntimeError(f"env {self._index} worker failed: {payload}")
        return self._to_result(payload)

    def state_dict(self) -> dict:
        """The worker's session state plus the parent-side bookkeeping the
        worker cannot know about. `episode_offset` in particular must survive:
        without it, a resume restarts the visible episode sequence from the
        restored session's own counter and breaks monotonicity across the
        crash."""
        return {
            "session": self._call(Command.STATE_DICT)["state"],
            "respawns": self._respawns,
            "episode_offset": self._episode_offset,
            "last_episode_id": self._last_episode_id,
        }

    def load_state_dict(self, state: dict) -> None:
        self._respawns = state["respawns"]
        self._episode_offset = state["episode_offset"]
        self._last_episode_id = state["last_episode_id"]
        self._call(Command.LOAD_STATE, state["session"])

    def stats(self) -> dict:
        """One round trip per update, not per step. The payload is ~1.3 KB per
        env, against the 168 KB a STATE_DICT round trip ships to extract the
        same coordinates."""
        return self._call(Command.STATS)["stats"]

    def close(self) -> None:
        try:
            self._conn.send((Command.CLOSE, None))
        except (BrokenPipeError, OSError):  # obs: allow LOG007 -- worker already gone at shutdown
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
        self._close_connection()


def build_subprocess_vec_env(config: EnvConfig) -> tuple[VecPokemonEnv, FrameBuffer]:
    """Production constructor. The caller must keep the FrameBuffer alive for
    the env's lifetime and call unlink() after close()."""
    from pathlib import Path

    # Preflight the ROM before spawning. Without this, a wrong path spawns 64
    # processes that each die inside PyBoyEmulator.__init__, and the parent
    # sees only a bare EOFError with no hint of the real cause.
    if not Path(config.rom_path).exists():
        raise FileNotFoundError(
            f"ROM not found at {config.rom_path!r}. It is gitignored and must be "
            "supplied locally; every worker would otherwise fail to construct its "
            "emulator and the parent would report only a broken pipe."
        )
    init_state = Path(config.init_state_path).read_bytes()
    buffer = FrameBuffer.create(config.n_envs)
    backends = [
        SubprocessBackend(
            index=i,
            shm_name=buffer.name,
            config=config,
            rom_path=config.rom_path,
            init_state=init_state,
            frame_slot=buffer.array[i],
        )
        for i in range(config.n_envs)
    ]
    return VecPokemonEnv(backends, config), buffer
