"""64 subprocess workers, each owning one PyBoy emulator.

PyBoy is GIL-bound, so threads buy no parallelism. Frames cross into the
parent through one SharedMemory block -- a frame is 23,040 B, so 64 of them is
1.47 MB per vector step and 1.5 GB of pickling per 1024-step rollout, roughly
19% of the 8.0 s rollout budget. Shared memory makes it a memcpy. No locking:
slices are disjoint and the parent joins all responses before reading.

Spawn, not fork: fork would duplicate the parent's CUDA context into every
worker, which CUDA does not support."""

from __future__ import annotations

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
            except Exception as error:  # noqa: BLE001 -- must reach the parent, not die silently
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
    or timeout rather than taking the whole run down."""

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
        self._spawn()

    @property
    def respawns(self) -> int:
        return self._respawns

    def _spawn(self) -> None:
        self._conn, self._process = self._spawn_worker(
            self._shm_name, self._index, self._config, self._rom_path, self._init_state
        )

    def _call(self, command: Command, argument: object = None) -> dict:
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

        Ends in `_reset_once`, deliberately NOT `reset`. `reset` recovers by
        calling back into here, so recovering twice would recurse forever on a
        worker that dies on every spawn. An unattended run that spins silently
        is worse than one that dies with a clear error."""
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
        except OSError:
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

    def reset(self) -> StepResult:
        """VecPokemonEnv routes every autoreset through here, so a worker that
        dies on an episode boundary must be recovered here too -- otherwise it
        propagates out of VecPokemonEnv.step() and kills the run."""
        try:
            return self._reset_once()
        except (TimeoutError, RuntimeError, EOFError, BrokenPipeError):
            return self._restart()

    def step(self, action: int) -> StepResult:
        try:
            payload = self._call(Command.STEP, action)
        except (TimeoutError, RuntimeError, EOFError, BrokenPipeError):
            restarted = self._restart()
            # Force the episode boundary so the trainer resets its KV cache for
            # this env; a respawned worker shares no history with the old one.
            return StepResult(
                frame=restarted.frame,
                aux=restarted.aux,
                reward=0.0,
                done=True,
                episode_id=restarted.episode_id,
                components={},
                clipped=False,
            )
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

    def close(self) -> None:
        try:
            self._conn.send((Command.CLOSE, None))
        except (BrokenPipeError, OSError):
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
