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


MAP_TILE = 64
MAPS_SHOWN = 12


def exploration_heatmap(
    coord_keys: Iterable[int], height: int = 256, width: int = 256
) -> np.ndarray:
    """Visit counts over all envs, projected into one image.

    ram.coord_key packs (map_id << 16) | (x << 8) | y, each field a uint8, so
    unpacking here mirrors that exactly.

    Coordinates render at their true (x, y) inside a per-map tile. The earlier
    version folded x and y mod 16, which collided most distinct coordinates in
    the same map -- the artifact looked plausible and showed almost nothing.
    Only the top MAPS_SHOWN maps by unique-coordinate count get a tile: a
    64x64 tile holds Pokemon Red's largest map, and a 256x256 image holds
    twelve of them at 4 per row with room for the labels a caller may draw."""
    counts: dict[int, int] = {}
    for key in coord_keys:
        counts[key] = counts.get(key, 0) + 1

    unique_per_map: dict[int, int] = {}
    for key in counts:
        map_id = (key >> 16) & 0xFF
        unique_per_map[map_id] = unique_per_map.get(map_id, 0) + 1

    ranked = sorted(unique_per_map, key=lambda m: (-unique_per_map[m], m))[:MAPS_SHOWN]
    tile_index = {map_id: position for position, map_id in enumerate(ranked)}
    maps_per_row = max(width // MAP_TILE, 1)

    heatmap = np.zeros((height, width), dtype=np.uint32)
    for key, count in counts.items():
        map_id = (key >> 16) & 0xFF
        position = tile_index.get(map_id)
        if position is None:
            continue
        origin_row = (position // maps_per_row) * MAP_TILE
        origin_column = (position % maps_per_row) * MAP_TILE
        row = origin_row + min((key & 0xFF), MAP_TILE - 1)
        column = origin_column + min(((key >> 8) & 0xFF), MAP_TILE - 1)
        if row < height and column < width:
            heatmap[row, column] += count
    return heatmap


def rollout_metrics(
    step: VecStep,
    components: dict[str, float],
    clip_fire_rate: float,
    respawns: int,
    stats: list[dict],
    respawns_delta: int = 0,
) -> dict[str, float]:
    """Flat scalar dict, ready for wandb.log and for a JSON-lines record.

    `stats` is VecPokemonEnv.stats() -- one entry per env. Unique coordinates
    are counted across the union of all envs, not summed per env: two envs that
    walked the same route have explored one route, and summing would report
    twice the exploration that happened.

    `respawns` is cumulative since the worker process started (it survives
    resume, via SubprocessBackend's own checkpoint state) -- a burst of
    respawns in one update only shows up as a slope change in that monotonic
    curve. `respawns_delta` is this update's own count, so a crash loop is a
    visible spike rather than a slope a human has to notice.

    Assumes `stats` is non-empty: `max(...)`/`sum(...) / len(stats)` below
    would raise on an empty list. EnvConfig validates n_envs >= 1, and `stats`
    has one entry per env, so this holds for every real caller."""
    unique_coords = {key for entry in stats for key in entry["coord_keys"]}
    unique_maps = {(key >> 16) & 0xFF for key in unique_coords}
    lengths = [length for entry in stats for length in entry["episode_lengths"]]
    stalls = [entry["steps_since_new_coord"] for entry in stats]

    metrics = {
        "reward/mean": float(step.reward.mean()),
        "reward/max": float(step.reward.max()),
        "env/clip_fire_rate": float(clip_fire_rate),
        "env/worker_respawns_total": float(respawns),
        "env/worker_respawns_delta": float(respawns_delta),
        "env/episodes_finished": float(sum(len(entry["episode_lengths"]) for entry in stats)),
        "progress/badges_max": float(max(entry["badges"] for entry in stats)),
        "progress/badges_mean": float(
            sum(entry["badges"] for entry in stats) / len(stats)
        ),
        "progress/event_flags_max": float(max(entry["event_flags"] for entry in stats)),
        "explore/unique_coords_total": float(len(unique_coords)),
        "explore/unique_maps": float(len(unique_maps)),
        "episode/length_mean": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "env/steps_since_new_coord_mean": float(sum(stalls) / len(stalls)),
        "env/steps_since_new_coord_max": float(max(stalls)),
    }
    for name, value in components.items():
        metrics[f"reward/{name}"] = float(value)
    return metrics
