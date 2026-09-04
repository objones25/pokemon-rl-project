# Battle Reward Redesign — Design Spec

Follow-up to `docs/2026-09-05-battle-reward-hacking-audit.md`, which
diagnosed run 11 (`cq14kskq`): the agent converged on grinding wild
battles as a terminal strategy — `reward/battle_won` (uncapped) reached
123.6 by update 46 (~261 wins/env) while `reward/money` fell
monotonically from ~$3175 to ~$129 and `explore/unique_coords_total`
flatlined. The audit traced this to `battle_win_weight` being the only
reward component in `rewards.py` with a flat, non-decaying marginal value
on an easy, infinitely-repeatable action, and to a second effect the
money crash exposed: nothing distinguishes a skilled win from a reckless
one, and a blackout's forced full-heal currently *pays* the `heal`
reward it should instead be exempt from.

Explicitly rejected during brainstorming: a hard cap on `battle_won`
(punishes a core, intended gameplay loop) and a discrete blackout penalty
(risks teaching battle-avoidance rather than risk-management). The
design below instead makes battling's *reward shape* mirror exploration's
already-proven decay, and adds a continuous, proactive pressure that
fires on the *state* of being hurt, never on the *act* of fighting.

## Goals, restated

- Exploration and badges must remain the dominant long-run incentive.
  Battling and leveling support that goal; they must never be able to
  out-compete it structurally, at any weight.
- Winning battles, including many wins in a row, must stay legitimately
  rewarding — this is normal Pokémon play, not a bug to punish.
- Teach "heal before you're in danger," continuously and proactively —
  not "don't fight," and not only after a blackout already happened.
- Blackouts get logged. Their forced heal is never rewarded. No separate
  punishment for the blackout event itself.

## Part 1 — Battle wins decay per badge tier

**Current state** (`rewards.py`): `_State.total_battles_won: int`
increments by 1 on every opponent-fainted edge; `components["battle_won"]
= battle_win_weight * total_battles_won` — flat, unbounded.

**New state:**

```python
battle_sum: float = 0.0          # replaces total_battles_won
wins_since_badge: int = 0
last_badge_count: int = 0
```

**Mechanism**, mirroring `explore_sum`/`steps_since_new_coord` exactly
(same decaying-running-sum shape, same "the k-th thing this tier is worth
1/sqrt(k)" rule already validated by `_update_exploration`): the i-th win
*since the last badge* (1-indexed, `i = wins_since_badge` after
incrementing) adds `1.0 / sqrt(i)` to `battle_sum`.
`components["battle_won"] = battle_win_weight * battle_sum` (no cap —
the decay bounds the *marginal* value, which is what actually matters;
see the audit's boxed rule).

```python
def _update_battle_progress(self, mem, in_battle):
    ...
    if current < last:
        ...
        if current == 0.0:
            self._state.wins_since_badge += 1
            self._state.battle_sum += 1.0 / math.sqrt(self._state.wins_since_badge)
        ...
```

**Badge-tier reset**, a new step run *after* `_update_battle_progress`
each step:

```python
def _update_badge_tier(self, badge_count: int) -> None:
    if badge_count > self._state.last_badge_count:
        self._state.wins_since_badge = 0
    self._state.last_badge_count = badge_count
```

Ordering matters and is deliberate: because this runs *after* battle
progress, the win that itself earns a badge (if the game processes the
badge on the same step the opponent faints — timing not verified either
way) is counted under the *old* tier's index, and only the *next* win
starts the fresh post-badge decay curve. This also means
`_update_badge_tier` must read `badge_count` fresh each step regardless
of battle state — it isn't gated on `in_battle`.

**Why this can't manufacture a spurious reward on reset**: `battle_sum`
itself never decreases or resets — only `wins_since_badge` (the exponent
driving *new* additions) does. The very first win after a badge adds a
fresh `1/sqrt(1) = 1.0` to whatever `battle_sum` already was, which is
by construction always a positive marginal addition — exactly the "gain"
the monotone-total formula is built to reward, no special-casing needed
in `step()`'s existing `gain = max(0, total - max_total)` logic.

**Persistence across episode auto-resets**: `battle_sum` /
`wins_since_badge` / `last_badge_count` do **not** persist — matching
`total_healing`/`total_damage`/`total_catches`, not the exploration
exception. Badges themselves reset to 0 when the emulator reloads
`init_state`, so a fresh episode's `last_badge_count` starting at 0 is
already correct without special-casing; there is nothing to carry over.

## Part 2 — Continuous low-HP pressure, and a shared-bug fix it depends on

**The bug found while designing this**: `ram.aggregate_hp_fraction()`
sums HP across **all six party slots**, including slots the player has
never used. Pokémon Red never clears a slot when a party member is
deposited or released, so a vacated slot's stale full-HP leftover value
can mask a live Pokémon actually being near death.
`pokemon_env/aux_state.py` already discovered this and works around it
inline (its own comment: *"Live slots only, and deliberately NOT
ram.aggregate_hp_fraction, which sums all six... A stale full-health
Pokemon in a vacated slot would otherwise mask the live one being nearly
dead"*) — but the correct computation only exists duplicated inline
there, not as a reusable reader, and `rewards.py`'s own `_update_healing`
still uses the flawed version. A safety-critical low-HP signal must not
inherit that flaw.

**Fix**: promote the correct computation to `ram.py`:

```python
def live_party_hp_fraction(mem: Emulator) -> float:
    """Party-wide health in [0, 1], live slots only. Unlike
    aggregate_hp_fraction, a vacated slot's stale leftover HP (the game
    never clears a slot when a party member is deposited or released)
    cannot mask a live Pokemon actually being near death. Same nan
    guard as aggregate_hp_fraction: 0.0 rather than nan with no live
    max HP to divide by."""
    size = min(party_size(mem), PARTY_SLOTS)
    slots = party_hp(mem)[:size]
    total_max = sum(maximum for _, maximum in slots)
    if total_max == 0:
        return 0.0
    return sum(current for current, _ in slots) / total_max
```

- `aux_state.py`'s `raw[26]` line calls this instead of its inline
  duplicate.
- `rewards.py`'s `_update_healing` switches from `aggregate_hp_fraction`
  to `live_party_hp_fraction` for its own `current`/`last_hp_fraction`
  tracking — fixing a latent bug in the existing `heal` reward as a
  side effect, not just enabling the new signal.
- `ram.aggregate_hp_fraction` becomes dead code once both callers move
  off it (confirm via grep before deleting; no other caller is known at
  spec time).

**The new component**, in `EnvConfig`:

```python
low_hp_penalty_weight: float = 0.0    # opt-in, same pattern as idle_penalty_weight
low_hp_threshold: float = 0.25        # requested value: 25%, not the 50% first floated
```

A continuous ramp, not a cliff, so the pressure is gradable rather than a
step discontinuity PPO's advantage estimation would otherwise have to
absorb in one bucket:

```python
live_fraction = ram.live_party_hp_fraction(mem)
if live_fraction < self._config.low_hp_threshold:
    severity = (self._config.low_hp_threshold - live_fraction) / self._config.low_hp_threshold
    low_hp_penalty = self._config.low_hp_penalty_weight * severity
else:
    low_hp_penalty = 0.0
```

`severity` is 0 at exactly the threshold and 1.0 at 0% HP — the penalty
scales smoothly from nothing to full strength as the party gets closer
to fainting entirely. Applied unconditionally, in battle or the
overworld — a hurt team wandering around ungoverned by anything gets the
same nudge as a hurt team still fighting, because the lesson is "go heal
your team," not "don't fight while hurt."

Structurally identical to `idle_penalty`: a genuinely separate, additive
term applied outside the `min(gain, 1.0)` monotone-gain formula (which
still cannot express a continuous negative signal), surfaced as its own
`components["low_hp"] = -low_hp_penalty` entry for observability. Total
reward becomes `min(gain, 1.0) - idle_penalty - low_hp_penalty`; the two
penalties are orthogonal and both apply if both conditions hold (e.g.
stalled *and* hurt), which is correct — they're unrelated failure modes.

## Part 3 — Blackout: log it, exempt its heal, punish nothing directly

**New state:**

```python
blackout_count: int = 0
pending_blackout_recovery: bool = False
```

**Detection and heal-exemption**, both inside `_update_healing` (the one
place `live_party_hp_fraction`'s current-vs-last transition is already
being tracked, so this is the natural home rather than a third parallel
tracker):

```python
def _update_healing(self, mem, party_size):
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
```

Worked through deliberately as a small state machine because the
black-screen/teleport/heal sequence spans several env-steps at
`action_freq=24`, not one, and an earlier draft of this logic had a real
bug worth naming: gating the flag's *clearing* on `party_size ==
last_party_size` (the same guard the ordinary heal-crediting branch
needs) meant that if the party size ever happened to differ on the exact
recovery step, `pending_blackout_recovery` would never clear and would
silently exempt a later, unrelated heal too. The version above decouples
the two concerns: consuming the flag (the `if
self._state.pending_blackout_recovery` branch) happens on *any*
`current > last` transition while the flag is set, regardless of party
size; the party-size guard only applies to the ordinary crediting path,
once the flag is confirmed clear. A blackout can only be *counted* once
per transition (`current == 0.0 and last_hp_fraction > 0.0`, which
requires `last_hp_fraction` to have been above 0, so a second
consecutive 0.0-reading step falls through to the `elif` and does
nothing, correctly).

**Telemetry**: `RewardAccumulator.blackout_count` property (mirroring
`coords_seen`/`maps_visited`), threaded through `EnvSession.stats()` the
same way `steps_since_new_coord` already is, aggregated in
`pokemon_env.telemetry.rollout_metrics` as `env/blackout_count_total` /
`env/blackout_count_delta` — mirroring the existing
`worker_respawns_total`/`worker_respawns_delta` pair already in this
codebase, not inventing a new telemetry shape.

**Persistence across episode auto-resets**: `blackout_count` **does**
persist, unlike everything else new in this spec — it's pure telemetry
describing the policy's risk behavior across the whole training run, the
same category `worker_respawns_total` is already in ("cumulative since
the worker process started... survives resume"), not reward-affecting
state that needs to match a genuine game-state reset. `pending_blackout_recovery`
does not persist (resets to `False` fresh each episode) — the recovery
sequence is a handful of steps long, so a reset landing exactly inside
one is negligible and not worth special-casing.

**No separate blackout punishment.** Between the continuous low-HP
pressure (which had every opportunity to already correct course before
a blackout happened) and losing half the wallet (already a real, if
currently unlabeled, consequence), a third punishment on top is the more
likely lever to overcorrect into "the agent learns not to fight" — the
exact failure this design is built to avoid.

## What changes

- `src/pokemon_env/ram.py`: add `live_party_hp_fraction`; remove
  `aggregate_hp_fraction` once both callers move off it.
- `src/pokemon_env/aux_state.py`: `raw[26]` calls the new reader instead
  of its inline duplicate.
- `src/pokemon_env/config.py`: add `low_hp_penalty_weight: float = 0.0`,
  `low_hp_threshold: float = 0.25`.
- `src/pokemon_env/rewards.py`: `_State` gains `battle_sum`,
  `wins_since_badge`, `last_badge_count`, `blackout_count`,
  `pending_blackout_recovery`; loses `total_battles_won`.
  `_update_battle_progress` changes its win-crediting to the decayed
  sum; new `_update_badge_tier`; `_update_healing` switches HP readers
  and gains the blackout detection/exemption; `step()` computes and
  subtracts `low_hp_penalty` alongside `idle_penalty`, adds a
  `blackout_count` property, and threads the new fields through
  `state_dict()`/`load_state_dict()`.
- `src/pokemon_env/session.py`: `stats()` gains `blackout_count`.
- `src/pokemon_env/telemetry.py`: `rollout_metrics` gains
  `env/blackout_count_total` / `env/blackout_count_delta`, threaded the
  same way `worker_respawns_total`/`_delta` already are (a
  `blackout_delta`-style parameter into `rollout_metrics`, and a running
  total the caller tracks — mirror `_respawns`'s exact pattern in
  `ppo/trainer.py`).
- `configs/pokemon_env.yaml`: set real values for `low_hp_penalty_weight`
  and confirm `low_hp_threshold: 0.25`, with the same reasoned-not-tuned
  documentation style every other weight in this file already has.
- `tests/unit/fakes.py`: `FakeBackend`/`FakeVecEnv` stats dicts gain
  `blackout_count` (same mechanical update as the `steps_since_new_coord`
  rollout from two sessions ago).

## Testing

TDD throughout, per this project's convention — each item below gets a
red test before the corresponding code.

**Battle-win decay / badge reset** (`test_pokemon_env_rewards.py`):
- Second win in the same tier pays `battle_win_weight / sqrt(2)` as its
  own component delta (mirrors `test_exploration_decays_as_one_over_root_k`'s
  structure exactly).
- A badge earned mid-run resets the decay: the next win after a badge
  pays a full `battle_win_weight / sqrt(1)`, not a continuation of the
  pre-badge decay curve and not zero.
- The win that itself earns the badge is attributed to the *old* tier
  (counted before the reset takes effect), proven by checking
  `wins_since_badge` from the *next* win onward starts at 1, not 2.
- Many wins within one tier still produce a bounded-growth, not-flat
  `battle_won` value (sanity check against the exact failure mode from
  run 11 — assert the 250th win's marginal contribution is small
  relative to the 1st, unlike the old flat design).

**Low-HP penalty + the `live_party_hp_fraction` bug fix**
(`test_pokemon_env_ram.py` for the reader, `test_pokemon_env_rewards.py`
for the penalty):
- `live_party_hp_fraction` ignores a vacated slot's stale full-HP
  leftover value when computing the live-only fraction — the regression
  test for the bug this spec fixes, constructed the same way
  `aux_state.py`'s own tests already prove this for slot 26.
- No penalty at or above the 25% threshold; a smoothly scaling penalty
  below it, pinned at an exact value partway down (e.g. 10% HP) and at
  the 0%-HP boundary (`severity == 1.0`).
- Applies identically in battle and out of it (unlike `idle_penalty`,
  which is deliberately battle/overworld-asymmetric).
- Composes additively with `idle_penalty` when both conditions hold in
  the same step.

**Blackout** (`test_pokemon_env_rewards.py`):
- HP dropping to exactly 0 after being above 0 increments
  `blackout_count` by exactly 1.
- Staying at 0 for several consecutive steps (simulating the
  black-screen/teleport animation) does not double-count.
- The recovery step (HP jumping back to 1.0) earns no `heal` credit.
- A *later*, genuine heal (after the blackout-recovery step has already
  consumed the exemption) earns `heal` credit normally — proving the
  exemption is one-shot, not a permanent hole in the heal reward.
- `blackout_count` persists across `RewardAccumulator.reset()`, mirrored
  against the existing exploration-persistence tests
  (`test_reset_persists_exploration_across_an_episode_boundary` and
  siblings).

**Telemetry plumbing**
(`test_pokemon_env_telemetry.py`, `test_pokemon_env_session.py`,
`test_ppo_trainer.py`): mechanical extensions following the exact
pattern the `steps_since_new_coord` → `env/stalled_frac` rollout already
established two sessions ago — `stats()` includes the new field, and
`rollout_metrics` aggregates it correctly across envs.

## Non-goals

- Does not touch `level_weight`, `damage_weight`, `catch_weight`,
  `explore_weight`, `badge_weight`, or their existing caps — none are
  implicated in run 11's failure, and CLAUDE.md's own convention is
  against moving parts that aren't broken.
- Does not add a discrete blackout punishment, per the explicit design
  decision above.
- Does not change `idle_penalty_weight` or its grace window (fixed last
  session) — `low_hp_penalty` is a new, independent mechanism, not a
  retuning of that one.
- Does not attempt to detect or discourage PP exhaustion directly (the
  Struggle/recoil mechanic named in the audit) — the low-HP signal is a
  proxy for the same underlying "you're in danger, retreat" lesson
  without needing a second RAM-derived signal, per explicit direction
  during brainstorming ("I don't want to investigate pp levels").
- Does not attempt to verify battle-outcome/badge RAM update ordering
  empirically (whether a badge and the win that earns it land in the
  same env-step) — the design is correct either way (see Part 1), so
  this is left unverified rather than requiring a live-ROM investigation
  this spec doesn't otherwise need.
