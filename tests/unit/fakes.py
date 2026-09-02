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
from pokemon_env.session import ACTION_DIM, StepResult
from pokemon_env.vec_env import VecStep
from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from ppo.rollout import RolloutState
from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig
from sequence_model.policy import StepOutput
from tests.conftest import PINNED_ENCODER_REVISION


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
        self._pending: StepResult | None = None

    def _result(self) -> StepResult:
        # send_reset/send_step/recv/state_dict/load_state_dict/close and this
        # helper exist only to satisfy EnvBackend's structural Protocol surface
        # -- no test calls them today; only stats() is exercised.
        return StepResult(
            frame=np.zeros((144, 160), dtype=np.uint8),
            aux=np.zeros(AUX_STATE_DIM, dtype=np.float32),
            reward=0.0,
            done=False,
            episode_id=0,
            components={},
            clipped=False,
        )

    def send_reset(self) -> None:
        self._pending = self._result()

    def send_step(self, action: int) -> None:
        self._pending = self._result()

    def recv(self) -> StepResult:
        result, self._pending = self._pending, None
        assert result is not None, "recv() called without a prior send_reset()/send_step()"
        return result

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


class RecordingBackend:
    """Hand-written fake typed against `EnvBackend`, logging every
    send_step/send_reset/recv call into a list shared across every backend
    in one VecPokemonEnv. Exists to prove VecPokemonEnv issues every send_*
    before any recv() -- the property the concurrent-dispatch fix exists to
    establish. Nothing about that ordering is visible from the returned
    StepResults themselves, so this is the only way to test it."""

    def __init__(
        self, index: int, call_log: list[str], fails_send_step: bool = False
    ) -> None:
        self._index = index
        self._call_log = call_log
        self._pending: StepResult | None = None
        # Stands in for SubprocessBackend.send_step() swallowing a
        # BrokenPipeError/OSError from conn.send(): the call still happens
        # (logged below) but produces no real dispatch. recv() below mirrors
        # the real backend's recovery -- it never raises from a swallowed
        # send, it returns a valid StepResult (there, via a respawn) -- so
        # this fake falls back to a fresh result instead of the missing one.
        self._fails_send_step = fails_send_step

    def _result(self) -> StepResult:
        return StepResult(
            frame=np.zeros((144, 160), dtype=np.uint8),
            aux=np.zeros(AUX_STATE_DIM, dtype=np.float32),
            reward=0.0,
            done=False,
            episode_id=0,
            components={},
            clipped=False,
        )

    def send_reset(self) -> None:
        self._call_log.append(f"send:{self._index}")
        self._pending = self._result()

    def send_step(self, action: int) -> None:
        self._call_log.append(f"send:{self._index}")
        if not self._fails_send_step:
            self._pending = self._result()

    def recv(self) -> StepResult:
        self._call_log.append(f"recv:{self._index}")
        result = self._pending if self._pending is not None else self._result()
        self._pending = None
        return result

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def stats(self) -> dict:
        return {}

    def close(self) -> None:
        pass


class FakeVecEnv:
    """Hand-written fake typed against the `VecEnvProtocol` `collect_rollout`
    consumes (`ppo/rollout.py`). Scripts a `done` flag at one fixed step
    index and otherwise returns fixed-shape zero frames/aux, so rollout
    ordering tests only need to reason about the contracts under test, never
    real env dynamics.

    Task 14 extends this same class with `reset()`, `stats()`, `last_step`,
    `last_components`, and `clip_fire_rate` -- kept to exactly the Protocol's
    surface `ppo.trainer` reads, so nothing here has to be forked.

    Also mirrors two real behaviours a rollout-ordering test can otherwise
    miss entirely: `VecPokemonEnv`'s own next-step autoreset (an env pending
    reset never looks at its action value) and `EnvSession.step`'s action-
    range validation (every other env raises on an action outside
    `[0, ACTION_DIM)`, exactly like the real emulator-backed session). Without
    both, a fake that accepts any action silently passes tests a real env
    would crash on."""

    def __init__(
        self,
        n_envs: int,
        aux_dim: int,
        done_at_step: int | None,
        reward: float = 0.0,
        reward_from_action: bool = False,
    ) -> None:
        self.n_envs = n_envs
        self._aux_dim = aux_dim
        self._done_at_step = done_at_step
        self._reward = reward
        # A constant reward makes the buffer's reward slot indistinguishable
        # from every other reward slot, so no test could see which action a
        # stored reward actually pays for. With reward_from_action, step()
        # returns `action + 1` for the action it was just handed -- injective
        # over the 7-way action space, so the alignment is readable off the
        # buffer directly.
        self._reward_from_action = reward_from_action
        # step_calls doubles as the done_at_step script cursor and the count
        # a trainer test asserts against -- one field, not two that could
        # silently drift apart.
        self.step_calls = 0
        self.reset_calls = 0
        self._last_step: VecStep | None = None
        # Mirrors VecPokemonEnv._needs_reset: which envs the NEXT step() call
        # must route through an implicit reset rather than validating.
        self._needs_reset = np.zeros(n_envs, dtype=bool)
        self.actions_seen: list[np.ndarray] = []

    def reset(self) -> VecStep:
        self.reset_calls += 1
        self._needs_reset[:] = False
        self._last_step = VecStep(
            frames=np.zeros((self.n_envs, 1, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8),
            aux=np.zeros((self.n_envs, self._aux_dim), dtype=np.float32),
            reward=np.zeros(self.n_envs, dtype=np.float32),
            done=np.zeros(self.n_envs, dtype=bool),
            episode_id=np.zeros(self.n_envs, dtype=np.int64),
        )
        return self._last_step

    def step(self, actions: np.ndarray) -> VecStep:
        self.actions_seen.append(actions.copy())
        for action, pending_reset in zip(actions, self._needs_reset, strict=True):
            if not pending_reset and not (0 <= action < ACTION_DIM):
                raise ValueError(f"action={action} is outside [0, {ACTION_DIM})")
        done = np.full(self.n_envs, self.step_calls == self._done_at_step, dtype=bool)
        reward = (
            actions.astype(np.float32) + 1.0
            if self._reward_from_action
            else np.full(self.n_envs, self._reward, dtype=np.float32)
        )
        step = VecStep(
            frames=np.zeros((self.n_envs, 1, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8),
            aux=np.zeros((self.n_envs, self._aux_dim), dtype=np.float32),
            reward=reward,
            done=done,
            episode_id=np.zeros(self.n_envs, dtype=np.int64),
        )
        self._needs_reset = done.copy()
        self.step_calls += 1
        self._last_step = step
        return step

    @property
    def last_step(self) -> VecStep | None:
        """The most recent VecStep from either `reset()` or `step()`, for
        end-of-update telemetry."""
        return self._last_step

    @property
    def last_components(self) -> dict[str, float]:
        """No reward components modeled here -- `pokemon_env.telemetry`'s own
        tests cover the real per-component mean; this fake only needs a dict
        `rollout_metrics` can merge."""
        return {}

    @property
    def clip_fire_rate(self) -> float:
        return 0.0

    def stats(self) -> list[dict]:
        """One dict per env, matching `FakeBackend.stats()`'s shape --
        `rollout_metrics` reads `coord_keys`/`badges`/`event_flags`/
        `episode_lengths` from every entry."""
        return [
            {
                "coord_keys": [],
                "badges": 0,
                "event_flags": 0,
                "step_count": self.step_calls,
                "episode_lengths": [],
            }
            for _ in range(self.n_envs)
        ]

    def state_dict(self) -> dict:
        """Round-trips the one piece of internal state a checkpoint test can
        observe drifting: `step_calls`. Real shape (schema_version,
        aux_state_version, per-backend state) is `VecPokemonEnv`'s job and is
        exercised against the real class in test_pokemon_env_checkpoint.py --
        this fake only needs to prove `ppo.checkpoint` round-trips whatever
        `vec_env.state_dict()` hands it."""
        return {"step_count": self.step_calls}

    def load_state_dict(self, state: dict) -> None:
        self.step_calls = state["step_count"]


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
    values -- unless `action_script` is given, in which case call `i` returns
    logits that put all of the probability mass on `action_script[i]`, so the
    action landing in each buffer slot is a known constant rather than a draw
    from a seeded generator. That is what lets a reward-alignment test assert
    exact numbers instead of a relation between two tensors."""

    def __init__(self, action_dim: int, action_script: tuple[int, ...] | None = None) -> None:
        self._action_dim = action_dim
        self._action_script = action_script
        self.prev_actions_seen: list[torch.Tensor] = []
        self.prev_rewards_seen: list[torch.Tensor] = []

    def _logits(self, n_envs: int, call_index: int) -> torch.Tensor:
        if self._action_script is None:
            return torch.zeros(n_envs, self._action_dim)
        # -1e4 rather than -inf: softmax of (0, -1e4) is exactly (1.0, 0.0) in
        # float32, and multinomial then samples the scripted index with
        # certainty, while -inf would make log_softmax produce a NaN gradient
        # path if this fake is ever reused under autograd.
        logits = torch.full((n_envs, self._action_dim), -1e4)
        logits[:, self._action_script[call_index]] = 0.0
        return logits

    def step(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        cache: RecordingCache,
    ) -> StepOutput:
        call_index = len(self.prev_actions_seen)
        self.prev_actions_seen.append(prev_action.clone())
        self.prev_rewards_seen.append(prev_reward.clone())
        cache.advance()
        n_envs = latent.shape[0]
        return StepOutput(
            logits=self._logits(n_envs, call_index),
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


def _rollout_harness(
    done_at_step: int | None,
    reward: float = 0.0,
    reward_from_action: bool = False,
    action_script: tuple[int, ...] | None = None,
    start_at_episode_start: bool = False,
) -> _RolloutHarness:
    """One fixed 2-env scenario shared by `test_ppo_rollout.py`: a `done`
    flag scripted at `done_at_step` (or never, if None), a tiny buffer/cache
    sized so 3 written slots land at `[burn_in, burn_in + 3)`, and a
    `.kwargs()` builder.

    `start_at_episode_start` scripts the state every cold start and resume
    actually constructs -- `RolloutState.prev_action` initialized to
    `episode_start_action` -- rather than the default `0` every other test in
    this file uses. The default `0` is a real action, so it can never see the
    bug where that sentinel reaches `vec_env.step()` on the very first
    iteration."""
    device = torch.device("cpu")
    n_envs = 2
    policy_config = PolicyConfig(
        d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1,
        d_ff=64, context_len=4, latent_dim=8, aux_state_dim=4,
    )
    ppo_config = PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, n_steps=3)
    buffer = RolloutBuffer(ppo_config, policy_config, n_envs, device)
    buffer.write_cursor = buffer.burn_in
    initial_action = (
        policy_config.episode_start_action if start_at_episode_start else 0
    )

    return _RolloutHarness(
        vec_env=FakeVecEnv(
            n_envs=n_envs,
            aux_dim=policy_config.aux_state_dim,
            done_at_step=done_at_step,
            reward=reward,
            reward_from_action=reward_from_action,
        ),
        encoder=FakeLatentEncoder(latent_dim=policy_config.latent_dim, device=device),
        policy=RecordingPolicy(
            action_dim=policy_config.action_dim, action_script=action_script
        ),
        cache=RecordingCache(
            RolloutCache.empty(policy_config, n_envs, device, dtype=torch.bfloat16)
        ),
        buffer=buffer,
        state=RolloutState(
            prev_action=torch.full((n_envs,), initial_action, dtype=torch.int64),
            prev_reward=torch.zeros(n_envs),
        ),
        generator=torch.Generator().manual_seed(0),
        device=device,
        episode_start_action=policy_config.episode_start_action,
        autocast_dtype=torch.bfloat16,
    )
