"""The 32-d RAM-derived state vector that rides alongside each frame.

Width is fixed at 32 by the merged PolicyConfig.aux_state_dim -- changing it
changes the model, so the layout is versioned instead. Thirty real signals,
one reserved.

Every slot is normalized to [0, 1], clamped, then mapped 2x-1 into [-1, 1].
The centering is interface fit against the merged InputAdapter, whose proj is
nn.Linear(..., bias=False): a block of inputs with mean 0.5 becomes a fixed
offset vector the model must absorb with real capacity."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pokemon_env import ram
from pokemon_env.emulator import Emulator

AUX_STATE_VERSION = 1
AUX_STATE_DIM = 32
RESERVED_SLOT = 31

_COORD_SATURATION = 20_000
_STUCK_SATURATION = 1_000
_MAX_MAPS = 255


@dataclass(frozen=True)
class ExplorationCounters:
    """Env-side counters the RAM cannot supply."""

    coords_seen: int
    steps_since_new_coord: int
    maps_visited: int


def build_aux_state(
    mem: Emulator,
    step_count: int,
    exploration: ExplorationCounters,
    max_steps: int,
) -> np.ndarray:
    """(32,) float32 in [-1, 1]. See the design spec's slot table."""
    raw = np.zeros(AUX_STATE_DIM, dtype=np.float32)

    # Slots past party_size hold whatever the last Pokemon to occupy them left
    # behind: Pokemon Red does not clear a slot when one is deposited or
    # released. Reading all six unconditionally shows the policy a party the
    # agent does not have -- five phantom Pokemon with levels and HP.
    # Clamped because RAM holds out-of-range values mid-write, and a
    # party_size of 255 would otherwise slice past the six real slots.
    live = min(ram.party_size(mem), ram.PARTY_SLOTS)

    raw[0] = ram.party_size(mem) / ram.PARTY_SLOTS
    raw[1:7] = [
        (level / ram.MAX_LEVEL if i < live else 0.0)
        for i, level in enumerate(ram.party_levels(mem))
    ]
    party_hp = ram.party_hp(mem)
    raw[7:13] = [
        (current / maximum if i < live and maximum > 0 else 0.0)
        for i, (current, maximum) in enumerate(party_hp)
    ]
    raw[13] = ram.badge_count(mem) / ram.MAX_BADGES
    raw[14] = ram.event_flag_count(mem) / ram.EVENT_FLAG_COUNT

    x, y, map_id = ram.game_coords(mem)
    raw[15] = map_id / 255.0
    raw[16] = x / 255.0
    raw[17] = y / 255.0
    raw[18] = 1.0 if ram.in_battle(mem) else 0.0
    raw[19] = ram.read_money(mem) / ram.MAX_MONEY
    raw[20:26] = [level / ram.MAX_LEVEL for level in ram.opponent_levels(mem)]
    # Live slots only, and deliberately NOT ram.aggregate_hp_fraction, which
    # sums all six. A stale full-health Pokemon in a vacated slot would
    # otherwise mask the live one being nearly dead. That function is left
    # alone because rewards.py's healing term is built on it -- changing it
    # here would silently alter reward semantics, which is a separate
    # decision from fixing what the policy observes.
    live_hp = party_hp[:live]
    live_max = sum(maximum for _, maximum in live_hp)
    raw[26] = (sum(current for current, _ in live_hp) / live_max) if live_max else 0.0

    raw[27] = math.log1p(exploration.coords_seen) / math.log1p(_COORD_SATURATION)
    raw[28] = step_count / max_steps
    raw[29] = min(exploration.steps_since_new_coord, _STUCK_SATURATION) / _STUCK_SATURATION
    raw[30] = exploration.maps_visited / _MAX_MAPS

    # Clamp BEFORE centering: RAM holds out-of-range values mid-write (a level
    # of 255), and an unclamped 2x-1 would inject a large outlier into a value
    # head the architecture plan warns is hypersensitive to input scale.
    np.clip(raw, 0.0, 1.0, out=raw)
    centered = raw * 2.0 - 1.0

    # The reserved slot is written as literal 0.0, not run through the
    # centering -- a "constant 0" centered becomes -1.0, which is a strong
    # constant signal rather than the absence of one.
    centered[RESERVED_SLOT] = 0.0
    return centered
