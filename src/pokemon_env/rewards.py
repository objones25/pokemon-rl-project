"""The architecture plan's section-4 reward, implemented literally.

    total_t = sum(w_i * c_i(t))            # every c_i monotone cumulative
    r_t     = clip(max(0, total_t - M), 0, 1)
    M       = max(M, total_t)              # the max_historical baseline

Every component is a running maximum or a monotone count, so cycles pay
nothing -- the deposit/withdraw exploit section 4 names, and equally walking
back and forth over known ground.

Clipping is an outlier guard, not the normalizer. Weights are chosen so a
normal step lands well inside the range; hard-clipping raw weights of the
reference implementation's scale (badge 10, event 4) would collapse both to
exactly 1.0 and leave the agent unable to tell a gym badge from a door."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pokemon_env import ram
from pokemon_env.config import EnvConfig
from pokemon_env.emulator import Emulator

# heal is the one component this module's own docstring is wrong about:
# "every component is a running maximum or a monotone count, so cycles pay
# nothing" holds for badges/events/coords (each bounded by finite game
# state) but not for _update_healing's raw total_healing, which grows once
# per HP recovery with no limit on how many deposit/withdraw HP cycles one
# episode can contain. A real training run hit exactly this: reward/heal
# grew past every other reward component combined over ~9M steps in one
# episode while badges stayed at zero -- the policy had found repeated
# damage/heal cycling was strictly easier reward than actual progress.
#
# Capped here, on the WEIGHTED contribution rather than the raw
# accumulator, so retuning heal_weight later cannot silently raise how much
# healing can ever be worth. 0.2 -- a fifth of one badge at this config's
# default badge_weight=1.0 -- still rewards genuine recovery without ever
# letting it compete with real progress. Once total_healing's weighted
# value crosses this, repeated cycles pay the same nothing every other
# component's cycles already do, restoring the invariant above.
_HEAL_CONTRIBUTION_CEILING = 0.2


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    clipped: bool
    components: dict[str, float]
    # Raw (unweighted) counts this step already read from RAM, exposed so
    # EnvSession can hand them to build_aux_state instead of it re-deriving
    # the same values with a second RAM read -- event_flag_count in
    # particular is a 311-byte popcount, not a single-byte read.
    badge_count: int
    event_flag_count: int


@dataclass
class _State:
    max_total: float = 0.0
    explore_sum: float = 0.0
    total_healing: float = 0.0
    base_event_flags: int = 0
    last_hp_fraction: float = 0.0
    last_party_size: int = 0
    steps_since_new_coord: int = 0
    seen_coords: set[int] = field(default_factory=set)
    seen_maps: set[int] = field(default_factory=set)


class RewardAccumulator:
    """One per env. Owns the max_historical baseline and the coordinate set."""

    def __init__(self, config: EnvConfig) -> None:
        self._config = config
        self._state = _State()

    @property
    def coords_seen(self) -> int:
        return len(self._state.seen_coords)

    @property
    def maps_visited(self) -> int:
        return len(self._state.seen_maps)

    def coord_keys(self) -> list[int]:
        """The packed coordinate keys themselves, not just the count.

        Sorted so the value is stable across runs -- the set's iteration
        order is not, and an unstable order would make the exploration
        heatmap artifact differ between two runs over identical states.
        A method rather than a property because it materializes a list
        that can reach tens of thousands of entries."""
        return sorted(self._state.seen_coords)

    @property
    def steps_since_new_coord(self) -> int:
        return self._state.steps_since_new_coord

    def reset(self, mem: Emulator) -> None:
        """Captures the event flags init.state already has set, so the agent
        is not paid on step one for progress it did not make."""
        self._state = _State(
            base_event_flags=ram.event_flag_count(mem),
            last_hp_fraction=ram.aggregate_hp_fraction(mem),
            last_party_size=ram.party_size(mem),
        )

    def step(self, mem: Emulator) -> RewardBreakdown:
        self._update_healing(mem)
        self._update_exploration(mem)

        badge_count = ram.badge_count(mem)
        event_flag_count = ram.event_flag_count(mem)

        components = {
            "badges": self._config.badge_weight * badge_count,
            "heal": min(
                self._config.heal_weight * self._state.total_healing, _HEAL_CONTRIBUTION_CEILING
            ),
            "explore": self._config.explore_weight * self._state.explore_sum,
            "events": self._config.event_weight * self._event_score(mem, event_flag_count),
            "levels": self._config.level_weight * ram.level_score(mem),
        }
        total = sum(components.values())

        gain = max(0.0, total - self._state.max_total)
        self._state.max_total = max(self._state.max_total, total)
        return RewardBreakdown(
            reward=min(gain, 1.0),
            clipped=gain > 1.0,
            components=components,
            badge_count=badge_count,
            event_flag_count=event_flag_count,
        )

    def _event_score(self, mem: Emulator, event_flag_count: int) -> int:
        return max(
            event_flag_count - self._state.base_event_flags - int(ram.museum_ticket_set(mem)),
            0,
        )

    def _update_healing(self, mem: Emulator) -> None:
        """Squared, so a full heal is worth far more than a trickle. Skipped
        when party size changed: HP fraction rises when a healthy Pokemon
        joins, and crediting that would pay for catching things twice."""
        current = ram.aggregate_hp_fraction(mem)
        size = ram.party_size(mem)
        if current > self._state.last_hp_fraction and size == self._state.last_party_size:
            delta = current - self._state.last_hp_fraction
            self._state.total_healing += delta * delta
        self._state.last_hp_fraction = current
        self._state.last_party_size = size

    def _update_exploration(self, mem: Emulator) -> None:
        """Only outside battle -- in battle the position bytes are stale, so
        recording them credits tiles never actually walked.

        The k-th newly discovered coordinate earns 1/sqrt(k), section 4's
        'decaying scalar reward'. The reference implementation's flat 0.1
        does not decay."""
        if ram.in_battle(mem):
            self._state.steps_since_new_coord += 1
            return

        x, y, map_id = ram.game_coords(mem)
        key = ram.coord_key(x, y, map_id)
        if key in self._state.seen_coords:
            self._state.steps_since_new_coord += 1
            return

        self._state.seen_coords.add(key)
        self._state.seen_maps.add(map_id)
        self._state.explore_sum += 1.0 / math.sqrt(len(self._state.seen_coords))
        self._state.steps_since_new_coord = 0

    def state_dict(self) -> dict:
        """Coordinates leave as a sorted list of ints, not a set: the
        checkpoint is loaded with torch.load(weights_only=True), which will
        not restore a set or a tuple-keyed dict."""
        return {
            "max_total": self._state.max_total,
            "explore_sum": self._state.explore_sum,
            "total_healing": self._state.total_healing,
            "base_event_flags": self._state.base_event_flags,
            "last_hp_fraction": self._state.last_hp_fraction,
            "last_party_size": self._state.last_party_size,
            "steps_since_new_coord": self._state.steps_since_new_coord,
            "seen_coords": sorted(self._state.seen_coords),
            "seen_maps": sorted(self._state.seen_maps),
        }

    def load_state_dict(self, state: dict) -> None:
        self._state = _State(
            max_total=state["max_total"],
            explore_sum=state["explore_sum"],
            total_healing=state["total_healing"],
            base_event_flags=state["base_event_flags"],
            last_hp_fraction=state["last_hp_fraction"],
            last_party_size=state["last_party_size"],
            steps_since_new_coord=state["steps_since_new_coord"],
            seen_coords=set(state["seen_coords"]),
            seen_maps=set(state["seen_maps"]),
        )
