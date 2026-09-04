"""Environment configuration, loaded from configs/pokemon_env.yaml via
config_io.load_dataclass_config.

Reward weights are initial guesses from the design spec, chosen so a normal
step's reward lands well inside [0, 1] and the clip fires only on genuine
outliers. They are the parameters most likely to need tuning against the
first run's reward histogram."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config_io import load_dataclass_config


@dataclass(frozen=True)
class EnvConfig:
    rom_path: str = "Pokemon Red.gb"
    init_state_path: str = "artifacts/init.state"
    n_envs: int = 64
    action_freq: int = 24
    press_frames: int = 8
    max_steps: int = 163_840
    worker_timeout_s: float = 60.0
    badge_weight: float = 1.00
    heal_weight: float = 0.50
    explore_weight: float = 0.30
    event_weight: float = 0.10
    level_weight: float = 0.05
    # Opt-in like target_kl/lr_decay_steps: 0.0 disables it, reproducing the
    # old no-penalty behavior exactly. configs/pokemon_env.yaml sets the real
    # value; see rewards.py's RewardAccumulator.step for what it penalizes
    # and why it has to sit outside the monotone-gain formula.
    idle_penalty_weight: float = 0.0

    def __post_init__(self) -> None:
        release_frames = self.action_freq - self.press_frames - 1
        if release_frames < 1:
            raise ValueError(
                f"press_frames={self.press_frames} leaves {release_frames} frames for the "
                f"release window at action_freq={self.action_freq}; the button would never "
                "be released and every later action would be entered with it still held"
            )
        if self.press_frames < 1:
            raise ValueError(f"press_frames={self.press_frames} must be at least 1")
        if self.n_envs < 1:
            raise ValueError(f"n_envs={self.n_envs} must be at least 1")

    @property
    def release_frames(self) -> int:
        """Frames to tick after releasing the button, before the final rendered
        frame. press + release + 1 == action_freq."""
        return self.action_freq - self.press_frames - 1


def load_config(path: str | Path) -> EnvConfig:
    return load_dataclass_config(EnvConfig, path)
