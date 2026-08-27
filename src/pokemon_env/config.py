"""Environment configuration, loaded from configs/pokemon_env.yaml.
Mirrors contrastive_pretrain.config's dataclass + yaml.safe_load pattern.

Reward weights are initial guesses from the design spec, chosen so a normal
step's reward lands well inside [0, 1] and the clip fires only on genuine
outliers. They are the parameters most likely to need tuning against the
first run's reward histogram."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EnvConfig:
    rom_path: str = "Pokemon Red.gb"
    init_state_path: str = "artifacts/init.state"
    frozen_encoder_repo_id: str = "objones25/pokemon-contrastive-encoder"
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
    seed: int = 0

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
    data = yaml.safe_load(Path(path).read_text()) or {}
    valid_fields = {f.name for f in fields(EnvConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")
    return EnvConfig(**data)
