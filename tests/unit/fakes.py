"""Hand-written test doubles, importable by any test module.

Separate from conftest.py so tests can construct these directly rather than
only receiving them as fixtures -- the vectorized env tests need one fake per
env, which a single fixture cannot supply."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pokemon_env.aux_state import AUX_STATE_DIM
from pokemon_env.emulator import SCREEN_HEIGHT, SCREEN_WIDTH
from pokemon_env.session import StepResult
from pokemon_env.vec_env import VecStep
from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from ppo.rollout import RolloutState
from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig
from sequence_model.policy import StepOutput


class FakeEmulator:
    """Hand-written fake typed against the Emulator Protocol, per CLAUDE.md's
    preference for fakes over mock.patch. Records every call so tests can
    assert the exact button/tick sequence."""

    def __init__(self, memory: bytearray | None = None, frame: np.ndarray | None = None) -> None:
        self.memory = bytearray(0x10000) if memory is None else memory
        self.frame = np.zeros((144, 160), dtype=np.uint8) if frame is None else frame
        self.calls: list[tuple] = []
        self.state = b"fake-emulator-state"
        self.closed = False

    def tick(self, count: int, render: bool) -> bool:
        self.calls.append(("tick", count, render))
        return True

    def button_press(self, button: str) -> None:
        self.calls.append(("press", button))

    def button_release(self, button: str) -> None:
        self.calls.append(("release", button))

    def read_memory(self, addr: int) -> int:
        return self.memory[addr]

    def screen_frame(self) -> np.ndarray:
        return self.frame.copy()

    def save_state(self) -> bytes:
        return self.state

    def load_state(self, state: bytes) -> None:
        self.state = state
        self.calls.append(("load_state", len(state)))

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    """Hand-written fake typed against the `EnvBackend` Protocol (`vec_env.py`),
    for VecPokemonEnv tests that need one fake per env without a real
    EnvSession or subprocess worker behind it."""

    def __init__(
        self,
        coord_keys: tuple[int, ...] = (),
        badges: int = 0,
        event_flags: int = 0,
        step_count: int = 0,
    ) -> None:
        self._coord_keys = coord_keys
        self._badges = badges
        self._event_flags = event_flags
        self._step_count = step_count

    def _result(self) -> StepResult:
        # reset/step/state_dict/load_state_dict/close and this helper exist
        # only to satisfy EnvBackend's structural Protocol surface -- no test
        # calls them today; only stats() is exercised.
        return StepResult(
            frame=np.zeros((144, 160), dtype=np.uint8),
            aux=np.zeros(AUX_STATE_DIM, dtype=np.float32),
            reward=0.0,
            done=False,
            episode_id=0,
            components={},
            clipped=False,
        )

    def reset(self) -> StepResult:
        return self._result()

    def step(self, action: int) -> StepResult:
        return self._result()

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def close(self) -> None:
        pass

    def stats(self) -> dict:
        return {
            "coord_keys": list(self._coord_keys),
            "badges": self._badges,
            "event_flags": self._event_flags,
            "step_count": self._step_count,
            "episode_lengths": [],
        }


class FakeVecEnv:
    """Hand-written fake typed against the `VecEnvProtocol` `collect_rollout`
    consumes (`ppo/rollout.py`). Scripts a `done` flag at one fixed step
    index and otherwise returns fixed-shape zero frames/aux, so rollout
    ordering tests only need to reason about the contracts under test, never
    real env dynamics.

    Later tasks (checkpoint resume, W&B stats, clip-rate telemetry) extend
    this same class with `state_dict`/`load_state_dict`, `stats()`,
    `last_step`, and `clip_fire_rate` -- kept to exactly the Protocol's
    surface plus the internal step counter those additions will read, so
    nothing here has to be forked."""

    def __init__(
        self,
        n_envs: int,
        aux_dim: int,
        done_at_step: int | None,
        reward: float = 0.0,
    ) -> None:
        self.n_envs = n_envs
        self._aux_dim = aux_dim
        self._done_at_step = done_at_step
        self._reward = reward
        self._step_count = 0

    def step(self, actions: np.ndarray) -> VecStep:
        done = np.full(self.n_envs, self._step_count == self._done_at_step, dtype=bool)
        step = VecStep(
            frames=np.zeros((self.n_envs, 1, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8),
            aux=np.zeros((self.n_envs, self._aux_dim), dtype=np.float32),
            reward=np.full(self.n_envs, self._reward, dtype=np.float32),
            done=done,
            episode_id=np.zeros(self.n_envs, dtype=np.int64),
        )
        self._step_count += 1
        return step


class FakeLatentEncoder:
    """Hand-written fake typed against the `EncoderProtocol` `collect_rollout`
    consumes. Ignores frame content -- rollout ordering tests only need a
    fixed-shape latent, never real vision features."""

    def __init__(self, latent_dim: int, device: torch.device) -> None:
        self._latent_dim = latent_dim
        self._device = device

    def encode(self, frames: np.ndarray) -> torch.Tensor:
        return torch.zeros(frames.shape[0], self._latent_dim, device=self._device)


class RecordingCache:
    """Wraps a real `RolloutCache`, typed against the `CacheProtocol`
    `collect_rollout` consumes. `advance()` -- which `RecordingPolicy` calls
    to reproduce the real `policy.step`'s cache mutation without a real
    transformer -- increments `advance_count`, and `reset()` records that
    count whenever it is called with any done env.

    Recording `advance_count` rather than an independently incremented loop
    counter is deliberate: it is what lets a reset-ordering test distinguish
    "reset ran after this step's own policy.step" from "reset ran before
    it" within the SAME loop iteration. A counter that only tracked which
    iteration reset() landed on would still read as unchanged if reset were
    moved earlier within that same iteration -- the exact bug this cache
    exists to catch -- because reset() still runs exactly once per
    iteration either way."""

    def __init__(self, inner: RolloutCache) -> None:
        self._inner = inner
        self.reset_calls_after_step: list[int] = []
        self.advance_count = 0

    @property
    def abs_pos(self) -> torch.Tensor:
        return self._inner.abs_pos

    def advance(self) -> None:
        self._inner.advance()
        self.advance_count += 1

    def reset(self, done: torch.Tensor) -> None:
        self._inner.reset(done)
        if bool(done.any()):
            self.reset_calls_after_step.append(self.advance_count)


class RecordingPolicy:
    """Hand-written fake typed against the `PolicyProtocol` `collect_rollout`
    consumes. Records the `prev_action`/`prev_reward` it was called with --
    exactly the surface the episode-boundary ordering contracts assert
    against -- and advances the cache the way the real `policy.step` does,
    since the abs_pos-snapshot-before-advance ordering is itself under test.
    Returns uniform logits (all-zero, so softmax is uniform) and zero
    values."""

    def __init__(self, action_dim: int) -> None:
        self._action_dim = action_dim
        self.prev_actions_seen: list[torch.Tensor] = []
        self.prev_rewards_seen: list[torch.Tensor] = []

    def step(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        cache: RecordingCache,
    ) -> StepOutput:
        self.prev_actions_seen.append(prev_action.clone())
        self.prev_rewards_seen.append(prev_reward.clone())
        cache.advance()
        n_envs = latent.shape[0]
        return StepOutput(
            logits=torch.zeros(n_envs, self._action_dim),
            value=torch.zeros(n_envs),
        )


@dataclass
class _RolloutHarness:
    """Bundles one scripted rollout scenario's fakes plus a `.kwargs()`
    builder, so each ordering-contract test only states what it varies
    (done step, reward) and gets `collect_rollout`'s full argument list."""

    vec_env: FakeVecEnv
    encoder: FakeLatentEncoder
    policy: RecordingPolicy
    cache: RecordingCache
    buffer: RolloutBuffer
    state: RolloutState
    generator: torch.Generator
    device: torch.device
    episode_start_action: int
    autocast_dtype: torch.dtype

    def kwargs(self, n_steps: int) -> dict:
        return {
            "vec_env": self.vec_env,
            "encoder": self.encoder,
            "policy": self.policy,
            "cache": self.cache,
            "buffer": self.buffer,
            "state": self.state,
            "n_steps": n_steps,
            "generator": self.generator,
            "device": self.device,
            "episode_start_action": self.episode_start_action,
            "autocast_dtype": self.autocast_dtype,
        }


def _rollout_harness(done_at_step: int | None, reward: float = 0.0) -> _RolloutHarness:
    """One fixed 2-env scenario shared by `test_ppo_rollout.py`: a `done`
    flag scripted at `done_at_step` (or never, if None), a tiny buffer/cache
    sized so 3 written slots land at `[burn_in, burn_in + 3)`, and a
    `.kwargs()` builder."""
    device = torch.device("cpu")
    n_envs = 2
    policy_config = PolicyConfig(
        d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1,
        d_ff=64, context_len=4, latent_dim=8, aux_state_dim=4,
    )
    ppo_config = PPOConfig(frozen_encoder_revision="x", n_steps=3)
    buffer = RolloutBuffer(ppo_config, policy_config, n_envs, device)
    buffer.write_cursor = buffer.burn_in

    return _RolloutHarness(
        vec_env=FakeVecEnv(
            n_envs=n_envs,
            aux_dim=policy_config.aux_state_dim,
            done_at_step=done_at_step,
            reward=reward,
        ),
        encoder=FakeLatentEncoder(latent_dim=policy_config.latent_dim, device=device),
        policy=RecordingPolicy(action_dim=policy_config.action_dim),
        cache=RecordingCache(
            RolloutCache.empty(policy_config, n_envs, device, dtype=torch.bfloat16)
        ),
        buffer=buffer,
        state=RolloutState(
            prev_action=torch.zeros(n_envs, dtype=torch.int64),
            prev_reward=torch.zeros(n_envs),
        ),
        generator=torch.Generator().manual_seed(0),
        device=device,
        episode_start_action=policy_config.episode_start_action,
        autocast_dtype=torch.bfloat16,
    )
