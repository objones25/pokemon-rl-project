"""Structured metrics and the two visual artifacts a human can sanity-check
without reading logs.

Per CLAUDE.md: every pipeline component emits JSON-lines progress and a live
W&B run, and anything that discards data says why -- here that is the
clip-fire rate, since clipping is the one place this component drops signal."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from pokemon_env.vec_env import VecStep


def contact_sheet(frames: np.ndarray) -> np.ndarray:
    """(N, 1, 144, 160) uint8 -> one tiled grayscale image.

    The fastest way to see that all 64 agents are stuck in the same menu.
    Grid is the smallest square that fits N, laid out row-major (env 0 at
    top-left, env 1 to its right, wrapping to the next row); unused tiles
    stay black."""
    n_envs, _, height, width = frames.shape
    side = math.ceil(math.sqrt(n_envs))
    sheet = np.zeros((side * height, side * width), dtype=np.uint8)
    for i in range(n_envs):
        row, column = divmod(i, side)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = frames[
            i, 0
        ]
    return sheet


def exploration_heatmap(
    coord_keys: Iterable[int], height: int = 256, width: int = 256
) -> np.ndarray:
    """Visit counts over all envs, projected into one image.

    ram.coord_key packs (map_id << 16) | (x << 8) | y, each field a uint8, so
    unpacking here must mirror that exactly: map_id = (key >> 16) & 0xFF,
    x = (key >> 8) & 0xFF, y = key & 0xFF. Maps are laid out on a grid by id
    rather than by true world position -- the reference implementation's
    global_map.py has the real projection, and swapping it in later changes
    only this function."""
    heatmap = np.zeros((height, width), dtype=np.uint32)
    maps_per_row = max(width // 16, 1)
    for key in coord_keys:
        map_id = (key >> 16) & 0xFF
        x = (key >> 8) & 0xFF
        y = key & 0xFF
        origin_row = (map_id // maps_per_row) * 16
        origin_column = (map_id % maps_per_row) * 16
        row = (origin_row + y % 16) % height
        column = (origin_column + x % 16) % width
        heatmap[row, column] += 1
    return heatmap


def rollout_metrics(
    step: VecStep,
    components: dict[str, float],
    clip_fire_rate: float,
    respawns: int,
) -> dict[str, float]:
    """Flat scalar dict, ready for wandb.log and for a JSON-lines record."""
    metrics = {
        "reward/mean": float(step.reward.mean()),
        "reward/max": float(step.reward.max()),
        "env/clip_fire_rate": float(clip_fire_rate),
        "env/worker_respawns": float(respawns),
        "env/episodes_finished": float(step.done.sum()),
    }
    for name, value in components.items():
        metrics[f"reward/{name}"] = float(value)
    return metrics
