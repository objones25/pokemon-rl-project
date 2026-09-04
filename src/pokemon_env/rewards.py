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

# Same reversible-cycle risk as heal, mirrored to the opponent's side: a
# trainer can heal their own Pokemon mid-battle (some carry Potions), which
# would otherwise let repeated chip-damage-then-enemy-heal cycles farm this
# forever. Capped on the weighted contribution for the same reason heal is.
_DAMAGE_CONTRIBUTION_CEILING = 0.2

# A released-then-recaught Pokemon (via Bill's PC) would otherwise reopen
# the identical cycle heal_weight was capped for after run 1 -- party_size
# can decrease as well as increase. Capped on the weighted contribution.
_CATCH_CONTRIBUTION_CEILING = 0.3

# How many steps a stalled battle gets before it counts as idling. A real
# turn is several env-steps of menu navigation (open FIGHT, pick a move)
# before HP actually moves; a 0-grace trigger here would tax the mechanical
# overhead of fighting, not genuine stalling. Run 9's discovered exploit sat
# at this for tens of thousands of steps (docs/ppo-experiment-history.md) --
# 8 is nowhere near that scale.
_BATTLE_STALL_GRACE_STEPS = 8

# How many steps of walking without a NEW coordinate the overworld gets
# before it counts as idling. A 0-grace trigger (this project's original
# design) is structurally broken, not just mistuned: explore_weight/sqrt(k)
# decays toward 0 as k grows while a flat per-step penalty does not, so
# ANY nonzero idle_penalty_weight eventually makes continued exploration
# net-negative even when the agent is behaving perfectly -- confirmed
# directly, not just in theory: run 9's own numbers put a healthy discovery
# every ~15 steps at ~0.0055 reward against 15 steps of 0.002 penalty
# (0.03), already net-negative at the very weight this project shipped
# with. Persistent cross-episode exploration (this project's other recent
# change) makes this worse over a run's lifetime, not better: later
# episodes must walk back through already-fully-seen ground to reach the
# frontier, which is legitimate progress that finds nothing new for a
# while. 1000 matches aux_state.py's own _STUCK_SATURATION -- the same
# point where the model's "how stuck am I" observation already reads
# maximally stuck -- so the reward penalty and the model's own perception
# of stalling now agree on what counts as stalled.
_OVERWORLD_STALL_GRACE_STEPS = 1000


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
    total_damage: float = 0.0
    battle_sum: float = 0.0
    wins_since_badge: int = 0
    last_badge_count: int = 0
    total_catches: int = 0
    base_event_flags: int = 0
    last_hp_fraction: float = 0.0
    last_party_size: int = 0
    # 1.0, not 0.0: a genuinely fresh opponent (wild encounter, or a
    # trainer's next Pokemon switched in) always starts at full HP, so this
    # is the value that makes the very first in-battle read never look like
    # a spurious drop.
    last_opponent_hp_fraction: float = 1.0
    blackout_count: int = 0
    pending_blackout_recovery: bool = False
    steps_since_new_coord: int = 0
    steps_since_battle_progress: int = 0
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

    @property
    def blackout_count(self) -> int:
        return self._state.blackout_count

    def reset(self, mem: Emulator) -> None:
        """Captures the event flags init.state already has set, so the agent
        is not paid on step one for progress it did not make.

        Exploration (seen_coords/seen_maps/explore_sum) and blackout_count
        persist across this reset; everything else does not. blackout_count
        is pure telemetry describing the policy's risk behavior across the
        whole training run -- the same category env/worker_respawns_total is
        already in -- not reward-affecting state that needs to match a
        genuine game-state reset, unlike badges/HP/party/events, which
        really do come back to their init values because the emulator
        genuinely reloads init_state on every reset (cold start and every
        later autoreset alike). EnvSession.reset() is one code path for both
        the very first cold start and every later autoreset at max_steps.

        max_total is rebased to exactly the persisted explore contribution
        (not left at 0) -- otherwise the fresh episode's `total` would sit
        above a stale 0 baseline and manufacture a reward on step one that
        nothing there actually earned."""
        explore_weight = self._config.explore_weight
        persisted_explore_sum = self._state.explore_sum
        persisted_blackout_count = self._state.blackout_count
        self._state = _State(
            base_event_flags=ram.event_flag_count(mem),
            last_hp_fraction=ram.live_party_hp_fraction(mem),
            last_party_size=ram.party_size(mem),
            max_total=explore_weight * persisted_explore_sum,
            explore_sum=persisted_explore_sum,
            blackout_count=persisted_blackout_count,
            seen_coords=self._state.seen_coords,
            seen_maps=self._state.seen_maps,
        )

    def step(self, mem: Emulator) -> RewardBreakdown:
        # Read once, used by both _update_catches (against the OLD value)
        # and _update_healing (whose own exemption needs the same OLD
        # value) -- last_party_size is advanced exactly once, at the end of
        # this method, after both have read it.
        party_size = ram.party_size(mem)
        in_battle = ram.in_battle(mem)
        badge_count = ram.badge_count(mem)

        self._update_catches(party_size)
        self._update_healing(mem, party_size)
        self._update_exploration(mem, in_battle)
        self._update_battle_progress(mem, in_battle)
        self._update_badge_tier(badge_count)

        event_flag_count = ram.event_flag_count(mem)

        components = {
            "badges": self._config.badge_weight * badge_count,
            "heal": min(
                self._config.heal_weight * self._state.total_healing, _HEAL_CONTRIBUTION_CEILING
            ),
            "explore": self._config.explore_weight * self._state.explore_sum,
            "events": self._config.event_weight * self._event_score(mem, event_flag_count),
            "levels": self._config.level_weight * ram.level_score(mem),
            "damage": min(
                self._config.damage_weight * self._state.total_damage,
                _DAMAGE_CONTRIBUTION_CEILING,
            ),
            "battle_won": self._config.battle_win_weight * self._state.battle_sum,
            "catch": min(
                self._config.catch_weight * self._state.total_catches,
                _CATCH_CONTRIBUTION_CEILING,
            ),
            "money": self._config.money_weight * (ram.read_money(mem) / ram.MAX_MONEY),
        }
        total = sum(components.values())

        gain = max(0.0, total - self._state.max_total)
        self._state.max_total = max(self._state.max_total, total)
        self._state.last_party_size = party_size

        # A genuinely separate, additive term -- not folded into `components`
        # above, because `total`/`gain`/`max_total` must stay computed from
        # only the monotone components. Section 4's positive-delta rule is
        # about never paying for a reversible cycle (deposit/withdraw,
        # badge loss); it says nothing against a per-step cost that can
        # never be profitably reversed, and the architecture plan's own
        # clip range for the advantage estimator is [-1, 1], not [0, 1].
        #
        # Two trigger conditions, each with its own grace window: a stalled
        # position outside battle, or a stalled battle (run 9 found the
        # entire population sitting at the FIGHT menu forever once
        # position-based idling was penalized with no grace, since battle
        # was the only place left exempt -- docs/ppo-experiment-history.md,
        # run 10). The overworld grace window (_OVERWORLD_STALL_GRACE_STEPS)
        # was added after that: a 0-grace trigger there fights
        # explore_weight/sqrt(k)'s own decay and is net-negative for
        # healthy exploration regardless of idle_penalty_weight's
        # magnitude, not just at the value this project first shipped.
        idle_penalty = (
            self._config.idle_penalty_weight
            if (
                (not in_battle and self._state.steps_since_new_coord > _OVERWORLD_STALL_GRACE_STEPS)
                or (in_battle and self._state.steps_since_battle_progress > _BATTLE_STALL_GRACE_STEPS)
            )
            else 0.0
        )
        return RewardBreakdown(
            reward=min(gain, 1.0) - idle_penalty,
            clipped=gain > 1.0,
            components={**components, "idle": -idle_penalty},
            badge_count=badge_count,
            event_flag_count=event_flag_count,
        )

    def _event_score(self, mem: Emulator, event_flag_count: int) -> int:
        return max(
            event_flag_count - self._state.base_event_flags - int(ram.museum_ticket_set(mem)),
            0,
        )

    def _update_catches(self, party_size: int) -> None:
        """A new Pokemon in the party, same monotone-count shape as
        badges/events. Summed rather than incremented by 1, though only one
        catch can happen per env-step in practice -- last_party_size is
        advanced by the caller (step), once, after this and _update_healing
        have both read the OLD value."""
        if party_size > self._state.last_party_size:
            self._state.total_catches += party_size - self._state.last_party_size

    def _update_healing(self, mem: Emulator, party_size: int) -> None:
        """Squared, so a full heal is worth far more than a trickle. Skipped
        when party size changed: HP fraction rises when a healthy Pokemon
        joins, and crediting that would pay for catching things twice.

        Also detects a blackout (live-party HP hitting exactly 0 after
        being above 0) and excludes the forced full-heal that follows one
        from this credit -- see docs/superpowers/specs/
        2026-09-05-battle-reward-redesign-design.md Part 3. The exemption
        consumes on ANY current > last transition while pending, regardless
        of party_size, so it can't get stuck open by an unrelated
        party-size change landing on the exact recovery step -- an earlier
        draft of this logic had exactly that bug, gating the flag's
        clearing on the same party_size check the ordinary heal-crediting
        path needs."""
        current = ram.live_party_hp_fraction(mem)
        if current == 0.0 and self._state.last_hp_fraction > 0.0:
            self._state.blackout_count += 1
            self._state.pending_blackout_recovery = True
        elif current > self._state.last_hp_fraction:
            if self._state.pending_blackout_recovery:
                self._state.pending_blackout_recovery = False
            elif party_size == self._state.last_party_size:
                delta = current - self._state.last_hp_fraction
                self._state.total_healing += delta * delta
        self._state.last_hp_fraction = current

    def _update_exploration(self, mem: Emulator, in_battle: bool) -> None:
        """Only outside battle -- in battle the position bytes are stale, so
        recording them credits tiles never actually walked.

        The k-th newly discovered coordinate earns 1/sqrt(k), section 4's
        'decaying scalar reward'. The reference implementation's flat 0.1
        does not decay."""
        if in_battle:
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

    def _update_battle_progress(self, mem: Emulator, in_battle: bool) -> None:
        """Damage dealt (opponent HP fraction dropping, same squared-delta
        shape as _update_healing) and a decaying battle-won bonus (the
        i-th win since the last badge is worth 1/sqrt(i), the same shape
        explore_sum already uses -- a rising->falling edge at exactly 0,
        not a level check, since the faint/switch animation holds the
        opponent at 0 HP for many env-steps and a level check would pay
        every one of them). steps_since_battle_progress is
        steps_since_new_coord's battle-side twin, closing the loophole run
        9 found: idle_penalty exempting battle turns meant sitting at the
        FIGHT menu forever was the only place left that cost nothing.

        Only meaningful in battle -- these bytes are stale otherwise, same
        guard as _update_exploration's position check. max_hp == 0 means
        the battle struct has not been populated yet (just-opened battle);
        treated as no-progress-yet, not as a fainted opponent, or the
        placeholder zero would misread as an instant win."""
        if not in_battle:
            self._state.steps_since_battle_progress = 0
            return

        max_hp = ram.read_u16_be(mem, ram.ENEMY_MON_MAX_HP_ADDR)
        if max_hp == 0:
            self._state.steps_since_battle_progress += 1
            return

        current = ram.read_u16_be(mem, ram.ENEMY_MON_HP_ADDR) / max_hp
        last = self._state.last_opponent_hp_fraction
        if current < last:
            delta = last - current
            self._state.total_damage += delta * delta
            if current == 0.0:
                self._state.wins_since_badge += 1
                self._state.battle_sum += 1.0 / math.sqrt(self._state.wins_since_badge)
            self._state.steps_since_battle_progress = 0
        elif current > last:
            self._state.steps_since_battle_progress = 0
        else:
            self._state.steps_since_battle_progress += 1
        self._state.last_opponent_hp_fraction = current

    def _update_badge_tier(self, badge_count: int) -> None:
        """Run AFTER _update_battle_progress each step, not before: if a
        badge and the win that earns it land on the same step, the win
        must still be attributed to the tier it happened in. Only the
        NEXT win starts the fresh post-badge decay curve. battle_sum
        itself is never reset -- only wins_since_badge, the exponent
        driving future additions -- so this can't manufacture a spurious
        reward spike; see docs/superpowers/specs/
        2026-09-05-battle-reward-redesign-design.md Part 1."""
        if badge_count > self._state.last_badge_count:
            self._state.wins_since_badge = 0
        self._state.last_badge_count = badge_count

    def state_dict(self) -> dict:
        """Coordinates leave as a sorted list of ints, not a set: the
        checkpoint is loaded with torch.load(weights_only=True), which will
        not restore a set or a tuple-keyed dict."""
        return {
            "max_total": self._state.max_total,
            "explore_sum": self._state.explore_sum,
            "total_healing": self._state.total_healing,
            "total_damage": self._state.total_damage,
            "battle_sum": self._state.battle_sum,
            "wins_since_badge": self._state.wins_since_badge,
            "last_badge_count": self._state.last_badge_count,
            "total_catches": self._state.total_catches,
            "base_event_flags": self._state.base_event_flags,
            "last_hp_fraction": self._state.last_hp_fraction,
            "last_party_size": self._state.last_party_size,
            "last_opponent_hp_fraction": self._state.last_opponent_hp_fraction,
            "blackout_count": self._state.blackout_count,
            "pending_blackout_recovery": self._state.pending_blackout_recovery,
            "steps_since_new_coord": self._state.steps_since_new_coord,
            "steps_since_battle_progress": self._state.steps_since_battle_progress,
            "seen_coords": sorted(self._state.seen_coords),
            "seen_maps": sorted(self._state.seen_maps),
        }

    def load_state_dict(self, state: dict) -> None:
        self._state = _State(
            max_total=state["max_total"],
            explore_sum=state["explore_sum"],
            total_healing=state["total_healing"],
            total_damage=state["total_damage"],
            battle_sum=state["battle_sum"],
            wins_since_badge=state["wins_since_badge"],
            last_badge_count=state["last_badge_count"],
            total_catches=state["total_catches"],
            base_event_flags=state["base_event_flags"],
            last_hp_fraction=state["last_hp_fraction"],
            last_party_size=state["last_party_size"],
            last_opponent_hp_fraction=state["last_opponent_hp_fraction"],
            blackout_count=state["blackout_count"],
            pending_blackout_recovery=state["pending_blackout_recovery"],
            steps_since_new_coord=state["steps_since_new_coord"],
            steps_since_battle_progress=state["steps_since_battle_progress"],
            seen_coords=set(state["seen_coords"]),
            seen_maps=set(state["seen_maps"]),
        )
