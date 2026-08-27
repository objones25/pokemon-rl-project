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
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from pokemon_env.config import EnvConfig
from pokemon_env.emulator import SCREEN_HEIGHT, SCREEN_WIDTH, Emulator, PyBoyEmulator
from pokemon_env.session import EnvSession, StepResult
from pokemon_env.vec_env import VecPokemonEnv


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
        """Parent only. Releases the OS-level block."""
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
    conn: Connection,
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
    [str, int, EnvConfig, str, bytes], tuple[Connection, BaseProcess]
]


def spawn_real_worker(
    shm_name: str,
    index: int,
    config: EnvConfig,
    rom_path: str,
    init_state: bytes,
) -> tuple[Connection, BaseProcess]:
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
        silently re-earn rewards for progress already banked."""
        self._respawns += 1
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=5)
        self._spawn()
        return self.reset()

    def _to_result(self, payload: dict) -> StepResult:
        return StepResult(
            frame=self._frame_slot,
            aux=payload["aux"],
            reward=payload["reward"],
            done=payload["done"],
            episode_id=payload["episode_id"],
            components=payload["components"],
            clipped=payload["clipped"],
        )

    def reset(self) -> StepResult:
        return self._to_result(self._call(Command.RESET))

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
        return self._call(Command.STATE_DICT)["state"]

    def load_state_dict(self, state: dict) -> None:
        self._call(Command.LOAD_STATE, state)

    def close(self) -> None:
        try:
            self._conn.send((Command.CLOSE, None))
        except (BrokenPipeError, OSError):
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()


def build_subprocess_vec_env(config: EnvConfig) -> tuple[VecPokemonEnv, FrameBuffer]:
    """Production constructor. The caller must keep the FrameBuffer alive for
    the env's lifetime and call unlink() after close()."""
    from pathlib import Path

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
