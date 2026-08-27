"""One environment: one emulator, one reward accumulator, one episode counter.

Deliberately knows nothing about processes or vectorization -- it is a plain
object driven by whichever backend owns it, which is what lets the whole
step/reset/reward path be tested against a FakeEmulator with no ROM.

Autoreset is NOT handled here. The session reports done and waits; VecPokemonEnv
owns the next-step autoreset that satisfies the sequence-model spec's
cache.reset(done) ordering contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokemon_env.aux_state import ExplorationCounters, build_aux_state
from pokemon_env.config import EnvConfig
from pokemon_env.emulator import Emulator
from pokemon_env.rewards import RewardAccumulator

# Index order is the action space. Must stay stable across checkpoints: a
# reordering silently remaps every action the policy has learned.
BUTTONS = ("down", "left", "right", "up", "a", "b", "start")
ACTION_DIM = len(BUTTONS)


@dataclass(frozen=True)
class StepResult:
    frame: np.ndarray  # (144, 160) uint8
    aux: np.ndarray  # (32,) float32
    reward: float
    done: bool
    episode_id: int
    components: dict[str, float]
    clipped: bool


class EnvSession:
    def __init__(self, emulator: Emulator, config: EnvConfig, init_state: bytes) -> None:
        self._emulator = emulator
        self._config = config
        self._init_state = init_state
        self._rewards = RewardAccumulator(config)
        self._step_count = 0
        self._episode_id = -1  # first reset() makes it 0

    def reset(self) -> StepResult:
        self._emulator.load_state(self._init_state)
        self._rewards.reset(self._emulator)
        self._step_count = 0
        self._episode_id += 1
        return self._observe(reward=0.0, clipped=False, components={})

    def step(self, action: int) -> StepResult:
        if not 0 <= action < ACTION_DIM:
            raise ValueError(f"action={action} is outside [0, {ACTION_DIM})")

        button = BUTTONS[action]
        self._emulator.button_press(button)
        self._emulator.tick(self._config.press_frames, False)
        self._emulator.button_release(button)
        self._emulator.tick(self._config.release_frames, False)
        self._emulator.tick(1, True)  # only the final frame is rendered

        self._step_count += 1
        breakdown = self._rewards.step(self._emulator)
        return self._observe(
            reward=breakdown.reward,
            clipped=breakdown.clipped,
            components=breakdown.components,
        )

    def _observe(self, reward: float, clipped: bool, components: dict[str, float]) -> StepResult:
        exploration = ExplorationCounters(
            coords_seen=self._rewards.coords_seen,
            steps_since_new_coord=self._rewards.steps_since_new_coord,
            maps_visited=self._rewards.maps_visited,
        )
        return StepResult(
            frame=self._emulator.screen_frame(),
            aux=build_aux_state(
                self._emulator, self._step_count, exploration, self._config.max_steps
            ),
            reward=reward,
            done=self._step_count >= self._config.max_steps,
            episode_id=self._episode_id,
            components=components,
            clipped=clipped,
        )

    def state_dict(self) -> dict:
        return {
            "emulator": self._emulator.save_state(),
            "rewards": self._rewards.state_dict(),
            "step_count": self._step_count,
            "episode_id": self._episode_id,
        }

    def load_state_dict(self, state: dict) -> None:
        self._emulator.load_state(state["emulator"])
        self._rewards.load_state_dict(state["rewards"])
        self._step_count = state["step_count"]
        self._episode_id = state["episode_id"]

    def close(self) -> None:
        self._emulator.close()
