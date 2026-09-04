# Battle Reward Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix run 11's battle-win reward hacking (agent grinding wild battles as a terminal strategy) without a hard cap or a discrete blackout punishment — decay `battle_won`'s marginal value per badge tier (the same tool `explore_sum` already uses), add a continuous low-HP proactive penalty, and log/exempt blackouts.

**Architecture:** All three pieces live in `RewardAccumulator` (`src/pokemon_env/rewards.py`), the existing per-env reward state machine. A new shared RAM reader (`ram.live_party_hp_fraction`) fixes a latent bug (`aggregate_hp_fraction` can be masked by a vacated party slot's stale HP) that both the existing `heal` reward and the new low-HP signal would otherwise inherit. Blackout telemetry threads through the same `stats()` → `rollout_metrics` pipeline `steps_since_new_coord` already uses.

**Tech Stack:** Python 3.12, pytest (strict config, 93% branch coverage floor), `FakeEmulator` (no real ROM in unit tests).

**Spec:** `docs/superpowers/specs/2026-09-05-battle-reward-redesign-design.md`

## Global Constraints

- TDD throughout: write the failing test, watch it fail for the right reason, then implement. No production code without a red test first.
- Every new `EnvConfig` weight defaults to `0.0` (opt-in), matching `idle_penalty_weight`/`damage_weight`/etc. — no existing default-`EnvConfig()` test may change behavior.
- `ruff check` clean and the full `tests/unit` suite green (821+ passing, 93%+ coverage) before any commit.
- No `mock.patch` — hand-written fakes (`FakeEmulator`, `FakeBackend`, `FakeVecEnv`) only, per this project's existing convention.
- Prove each new/changed test can fail: break the code it covers, confirm red, revert, before moving on. State which check you did in the task's final step.
- Commit after every task (small, working, tested increments) — never bundle two tasks into one commit.

---

### Task 1: `ram.live_party_hp_fraction`

**Files:**
- Modify: `src/pokemon_env/ram.py` (add function, near `aggregate_hp_fraction` at line 102)
- Test: `tests/unit/test_pokemon_env_ram.py`

**Interfaces:**
- Produces: `ram.live_party_hp_fraction(mem: Emulator) -> float` — party HP fraction across only the occupied party slots (`[0, party_size)`), `0.0` when there's no live max HP to divide by (mirrors `aggregate_hp_fraction`'s own nan guard). Used by Tasks 2, 5, 6.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_pokemon_env_ram.py`, near the existing `aggregate_hp_fraction` tests:

```python
def test_live_party_hp_fraction_is_zero_when_party_is_empty(fake_emulator) -> None:
    """All-zero memory is the pre-game state (party_size=0). Dividing by a
    zero live max would produce nan, which propagates silently through
    the whole aux vector and every reward that reads this."""
    assert ram.live_party_hp_fraction(fake_emulator) == pytest.approx(0.0)


def test_live_party_hp_fraction_ignores_a_vacated_slot(fake_emulator) -> None:
    """The regression test for the bug this function fixes: Pokemon Red
    never clears a slot when a party member is deposited or released, so
    slot 1 here still holds a stale, full-HP Pokemon the player does not
    have. aggregate_hp_fraction would average it in and read a healthy
    0.7; only the live slot 0 (10/100, critically low) should count."""
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 10
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + ram.PARTY_STRIDE + 1] = 40
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + ram.PARTY_STRIDE + 1] = 40

    assert ram.live_party_hp_fraction(fake_emulator) == pytest.approx(0.1)


def test_live_party_hp_fraction_sums_across_live_slots_only(fake_emulator) -> None:
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 2
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 30
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 60
    fake_emulator.memory[ram.PARTY_HP_BASE + ram.PARTY_STRIDE + 1] = 10
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + ram.PARTY_STRIDE + 1] = 40

    assert ram.live_party_hp_fraction(fake_emulator) == pytest.approx(0.4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_pokemon_env_ram.py -q --no-cov -k live_party_hp_fraction`
Expected: FAIL — `AttributeError: module 'pokemon_env.ram' has no attribute 'live_party_hp_fraction'`

- [ ] **Step 3: Implement**

In `src/pokemon_env/ram.py`, add directly after `aggregate_hp_fraction` (after line 110):

```python
def live_party_hp_fraction(mem: Emulator) -> float:
    """Party-wide health in [0, 1], live slots only. Unlike
    aggregate_hp_fraction, a vacated slot's stale leftover HP (the game
    never clears a slot when a party member is deposited or released)
    cannot mask a live Pokemon actually being near death. Same nan guard
    as aggregate_hp_fraction: 0.0 rather than nan with no live max HP to
    divide by."""
    size = min(party_size(mem), PARTY_SLOTS)
    slots = party_hp(mem)[:size]
    total_max = sum(maximum for _, maximum in slots)
    if total_max == 0:
        return 0.0
    return sum(current for current, _ in slots) / total_max
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_pokemon_env_ram.py -q --no-cov`
Expected: all pass (29 existing + 3 new = 32)

- [ ] **Step 5: Prove the vacated-slot test can fail**

Temporarily change `size = min(party_size(mem), PARTY_SLOTS)` to `size = PARTY_SLOTS` (i.e. simulate the bug — sum all six slots unconditionally), rerun
`uv run pytest tests/unit/test_pokemon_env_ram.py -q --no-cov -k test_live_party_hp_fraction_ignores_a_vacated_slot`, confirm it fails (reads `0.7`-ish instead of `0.1`), then revert the change.

- [ ] **Step 6: Commit**

```bash
git add src/pokemon_env/ram.py tests/unit/test_pokemon_env_ram.py
git commit -m "feat(pokemon_env): add live_party_hp_fraction, live-slots-only HP reader

Fixes the same masking bug aux_state.py already worked around inline
(a vacated party slot's stale HP can hide a live Pokemon near death) --
promoted to a shared reader so rewards.py can use it too."
```

---

### Task 2: `aux_state.py` uses the shared reader

**Files:**
- Modify: `src/pokemon_env/aux_state.py:84-92`
- Test: `tests/unit/test_pokemon_env_aux_state.py`

**Interfaces:**
- Consumes: `ram.live_party_hp_fraction` from Task 1.

- [ ] **Step 1: Confirm existing coverage still describes the behavior**

Run: `uv run pytest tests/unit/test_pokemon_env_aux_state.py -q --no-cov -k slot_26 or -k hp` and read the matching test(s) — this task changes *how* `raw[26]` is computed, not what it computes, so the existing test(s) for that slot should already pin the expected value and don't need new assertions. If no such test exists, add one first (pin `raw[26]`'s value for a live-slots-vs-vacated-slot scenario matching Task 1's regression test), confirm it passes against the *current* inline implementation, before changing the implementation.

- [ ] **Step 2: Replace the inline computation**

In `src/pokemon_env/aux_state.py`, replace lines 84-92:

```python
    # Live slots only, and deliberately NOT ram.aggregate_hp_fraction, which
    # sums all six. A stale full-health Pokemon in a vacated slot would
    # otherwise mask the live one being nearly dead. That function is left
    # alone because rewards.py's healing term is built on it -- changing it
    # here would silently alter reward semantics, which is a separate
    # decision from fixing what the policy observes.
    live_hp = party_hp[:live]
    live_max = sum(maximum for _, maximum in live_hp)
    raw[26] = (sum(current for current, _ in live_hp) / live_max) if live_max else 0.0
```

with:

```python
    # ram.live_party_hp_fraction re-reads party_size/party_hp rather than
    # reusing the `live`/`party_hp` locals above -- a handful of extra
    # single-byte reads, not the event_flag_count-scale cost that justifies
    # threading a value through elsewhere in this module. rewards.py's heal
    # reward and low-HP penalty use the same reader now (see
    # docs/superpowers/specs/2026-09-05-battle-reward-redesign-design.md),
    # so this observation and those reward terms agree on what "how hurt is
    # my team" means.
    raw[26] = ram.live_party_hp_fraction(mem)
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/unit/test_pokemon_env_aux_state.py -q --no-cov`
Expected: all pass, unchanged count — this step changes an implementation detail, not behavior, so no test count should change.

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest tests/unit -q`
Expected: same pass count as before this task, coverage floor still met.

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_env/aux_state.py
git commit -m "refactor(pokemon_env): aux_state raw[26] uses the shared live-HP reader

No behavior change -- same computation, now backed by
ram.live_party_hp_fraction (Task 1) instead of a duplicated inline
version, so rewards.py's upcoming use of the same reader (Task 5) can't
silently drift from what the policy observes."
```

---

### Task 3: `EnvConfig` gains the low-HP config fields

**Files:**
- Modify: `src/pokemon_env/config.py`
- Test: `tests/unit/test_pokemon_env_config.py`

**Interfaces:**
- Produces: `EnvConfig.low_hp_penalty_weight: float` (default `0.0`), `EnvConfig.low_hp_threshold: float` (default `0.25`). Used by Task 6.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_pokemon_env_config.py`, near the existing `idle_penalty_weight` default test:

```python
def test_low_hp_penalty_weight_defaults_to_zero() -> None:
    """Opt-in like every other reward weight added since idle_penalty_weight:
    defaults off so every existing default-EnvConfig test keeps its current
    behavior."""
    config = EnvConfig()

    assert config.low_hp_penalty_weight == pytest.approx(0.0)


def test_low_hp_threshold_defaults_to_a_quarter_health() -> None:
    config = EnvConfig()

    assert config.low_hp_threshold == pytest.approx(0.25)


def test_low_hp_threshold_rejects_zero() -> None:
    """A zero threshold divides by zero computing the penalty's severity
    ramp (rewards.py's (threshold - fraction) / threshold) -- reject it at
    construction, not with a ZeroDivisionError deep in a training step."""
    with pytest.raises(ValueError, match=r"low_hp_threshold=0.0 must be in \(0, 1\]"):
        EnvConfig(low_hp_threshold=0.0)


def test_low_hp_threshold_rejects_a_value_above_one() -> None:
    with pytest.raises(ValueError, match=r"low_hp_threshold=1.5 must be in \(0, 1\]"):
        EnvConfig(low_hp_threshold=1.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_pokemon_env_config.py -q --no-cov -k low_hp`
Expected: FAIL — `AttributeError`/`TypeError: unexpected keyword argument 'low_hp_threshold'` on the first two, collection error or wrong-exception failures on the validation ones.

- [ ] **Step 3: Implement**

In `src/pokemon_env/config.py`, add after `money_weight: float = 0.0` (line 43):

```python
    # Continuous, proactive pressure while the party is hurt -- see
    # rewards.py's RewardAccumulator._low_hp_penalty. Opt-in like every
    # other weight above; 0.25 (not the 50% first considered) is the
    # threshold below which it starts ramping up.
    low_hp_penalty_weight: float = 0.0
    low_hp_threshold: float = 0.25
```

In `__post_init__` (after the existing `n_envs` check, before the method ends):

```python
        if not 0.0 < self.low_hp_threshold <= 1.0:
            raise ValueError(f"low_hp_threshold={self.low_hp_threshold} must be in (0, 1]")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_pokemon_env_config.py -q --no-cov`
Expected: all pass.

- [ ] **Step 5: Prove the validation tests can fail**

Temporarily comment out the new `__post_init__` check, rerun the two rejection tests, confirm both fail (`EnvConfig(low_hp_threshold=0.0)` no longer raises), then restore the check.

- [ ] **Step 6: Commit**

```bash
git add src/pokemon_env/config.py tests/unit/test_pokemon_env_config.py
git commit -m "feat(pokemon_env): add low_hp_penalty_weight/low_hp_threshold config

Opt-in, defaults matching every other reward weight added this week.
Validates threshold is in (0, 1] at construction rather than risking a
ZeroDivisionError inside a training step."
```

---

### Task 4: Battle-win decay, reset per badge tier

**Files:**
- Modify: `src/pokemon_env/rewards.py`
- Test: `tests/unit/test_pokemon_env_rewards.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks (uses existing `ram.badge_count`, `ram.read_u16_be`, `ram.ENEMY_MON_HP_ADDR`/`ENEMY_MON_MAX_HP_ADDR`).
- Produces: `_State.battle_sum: float`, `_State.wins_since_badge: int`, `_State.last_badge_count: int` (replaces `_State.total_battles_won`). `components["battle_won"]` now reflects the decayed sum. Used by Task 6 (no direct dependency, but shares `step()`).

This task changes `_State`, `_update_battle_progress`, adds `_update_badge_tier`, and reorders `step()` to compute `badge_count` before the `_update_*` calls so `_update_badge_tier` can use it. `state_dict`/`load_state_dict` are updated in the same task since they're mechanically tied to `_State`'s field list — splitting them into a separate task would leave an intermediate broken checkpoint format.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_pokemon_env_rewards.py`, near the existing `test_a_second_battle_won_against_a_new_opponent_pays_again` test (reuse the `_set_opponent_hp` helper already defined there):

```python
def test_the_second_win_in_one_tier_pays_the_decayed_value(fake_emulator) -> None:
    """Mirrors test_exploration_decays_as_one_over_root_k's structure
    exactly -- the 2nd win, like the 2nd new coordinate, is worth
    weight/sqrt(2), not another full weight."""
    win_accumulator = RewardAccumulator(EnvConfig(battle_win_weight=0.5))
    win_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=10)
    win_accumulator.step(fake_emulator)  # first win
    _set_opponent_hp(fake_emulator, current=8, max_hp=8)
    win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=8)

    second_win = win_accumulator.step(fake_emulator)

    assert second_win.reward == pytest.approx(0.5 / math.sqrt(2))


def test_a_badge_resets_the_win_decay_to_a_fresh_curve(fake_emulator) -> None:
    """The whole point of Part 1 of the redesign: a badge does not erase
    battle_sum (which would risk a spurious reward spike), it resets which
    exponent the NEXT win uses -- so the first win after a badge pays a
    full fresh weight/sqrt(1), not a continuation of the pre-badge decay."""
    win_accumulator = RewardAccumulator(EnvConfig(battle_win_weight=0.5))
    win_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=10)
    win_accumulator.step(fake_emulator)  # first win, tier 1
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0001  # badge earned
    win_accumulator.step(fake_emulator)  # processes the badge, no HP change this step
    _set_opponent_hp(fake_emulator, current=8, max_hp=8)
    win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=8)

    first_win_of_new_tier = win_accumulator.step(fake_emulator)

    assert first_win_of_new_tier.reward == pytest.approx(0.5)


def test_the_win_that_earns_a_badge_is_attributed_to_the_old_tier(fake_emulator) -> None:
    """If the badge flag and the fainting edge land on the exact same
    step, that win must still count as the old tier's Nth win, not reset
    itself out of existence -- proven by checking the tier only resets
    for whatever comes NEXT.

    badge_weight=0.0 isolates battle_won's reward from the badge's own:
    _update_badge_tier reads ram.badge_count directly, unaffected by
    badge_weight, so this doesn't change what tier-reset behavior is
    under test -- it only removes components["badges"]'s own +1.0 jump
    (which would otherwise land on the exact same step as the win, since
    the badge flag is deliberately set there too, and clip .reward to a
    contaminated 1.0 instead of the intended 0.5 signal)."""
    win_accumulator = RewardAccumulator(EnvConfig(battle_win_weight=0.5, badge_weight=0.0))
    win_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=10)
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0001  # badge lands same step as the win
    same_step_win = win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=8, max_hp=8)
    win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=8)

    next_win = win_accumulator.step(fake_emulator)

    assert (same_step_win.reward, next_win.reward) == (
        pytest.approx(0.5),          # 1st win of tier 1, badge or not
        pytest.approx(0.5),          # 1st win of the NEW tier, freshly reset
    )


def test_many_wins_in_one_tier_decay_instead_of_staying_flat(fake_emulator) -> None:
    """The exact failure mode from run 11, made unrepresentable: the 50th
    win's own marginal contribution must be small relative to the 1st,
    unlike the old flat battle_win_weight * total_battles_won design."""
    win_accumulator = RewardAccumulator(EnvConfig(battle_win_weight=0.5))
    win_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    last_reward = None
    for i in range(50):
        _set_opponent_hp(fake_emulator, current=10, max_hp=10)
        win_accumulator.step(fake_emulator)
        _set_opponent_hp(fake_emulator, current=0, max_hp=10)
        last_reward = win_accumulator.step(fake_emulator).reward

    assert last_reward == pytest.approx(0.5 / math.sqrt(50))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_pokemon_env_rewards.py -q --no-cov -k "tier or decayed_value or old_tier"`
Expected: FAIL — the first two/three on wrong reward values (still flat `0.5` per win under the current design instead of the decayed values), the badge-reset test failing because nothing resets yet.

- [ ] **Step 3: Update `_State`**

In `src/pokemon_env/rewards.py`, replace line 102 (`total_battles_won: int = 0`) with:

```python
    battle_sum: float = 0.0
    wins_since_badge: int = 0
    last_badge_count: int = 0
```

- [ ] **Step 4: Update `_update_battle_progress` and add `_update_badge_tier`**

Replace the win-crediting block inside `_update_battle_progress` (currently):

```python
            if current == 0.0:
                self._state.total_battles_won += 1
```

with:

```python
            if current == 0.0:
                self._state.wins_since_badge += 1
                self._state.battle_sum += 1.0 / math.sqrt(self._state.wins_since_badge)
```

Add a new method directly after `_update_battle_progress`:

```python
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
```

`_update_battle_progress`'s docstring currently reads exactly:

```python
        """Damage dealt (opponent HP fraction dropping, same squared-delta
        shape as _update_healing) and a battle-won bonus (a rising->falling
        edge at exactly 0, not a level check -- the faint/switch animation
        holds the opponent at 0 HP for many env-steps, and a level check
        would pay every one of them). steps_since_battle_progress is
        steps_since_new_coord's battle-side twin, closing the loophole run
        9 found: idle_penalty exempting battle turns meant sitting at the
        FIGHT menu forever was the only place left that cost nothing.

        Only meaningful in battle -- these bytes are stale otherwise, same
        guard as _update_exploration's position check. max_hp == 0 means
        the battle struct has not been populated yet (just-opened battle);
        treated as no-progress-yet, not as a fainted opponent, or the
        placeholder zero would misread as an instant win."""
```

Replace the whole docstring with:

```python
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
```

- [ ] **Step 5: Wire `_update_badge_tier` into `step()`**

In `step()`, `badge_count` is currently computed *after* the `_update_*` calls. Move it before them and call the new method. Replace:

```python
        party_size = ram.party_size(mem)
        in_battle = ram.in_battle(mem)

        self._update_catches(party_size)
        self._update_healing(mem, party_size)
        self._update_exploration(mem, in_battle)
        self._update_battle_progress(mem, in_battle)

        badge_count = ram.badge_count(mem)
        event_flag_count = ram.event_flag_count(mem)
```

with:

```python
        party_size = ram.party_size(mem)
        in_battle = ram.in_battle(mem)
        badge_count = ram.badge_count(mem)

        self._update_catches(party_size)
        self._update_healing(mem, party_size)
        self._update_exploration(mem, in_battle)
        self._update_battle_progress(mem, in_battle)
        self._update_badge_tier(badge_count)

        event_flag_count = ram.event_flag_count(mem)
```

And update the `components["battle_won"]` line:

```python
            "battle_won": self._config.battle_win_weight * self._state.battle_sum,
```

- [ ] **Step 6: Update `state_dict`/`load_state_dict`**

In `state_dict()`, replace `"total_battles_won": self._state.total_battles_won,` with:

```python
            "battle_sum": self._state.battle_sum,
            "wins_since_badge": self._state.wins_since_badge,
            "last_badge_count": self._state.last_badge_count,
```

In `load_state_dict()`, replace `total_battles_won=state["total_battles_won"],` with:

```python
            battle_sum=state["battle_sum"],
            wins_since_badge=state["wins_since_badge"],
            last_badge_count=state["last_badge_count"],
```

- [ ] **Step 7: Fix the one pre-existing test the decay changes**

Run: `uv run pytest tests/unit/test_pokemon_env_rewards.py -q --no-cov -k battle_won`

Two of the three existing battle-win tests are unaffected: `test_a_fainted_opponent_earns_the_battle_win_bonus` and `test_the_battle_win_bonus_does_not_repeat_while_the_same_faint_persists` both only ever earn a *single* win (`wins_since_badge` reaches 1), and `0.5 / sqrt(1) == 0.5` — identical to the old flat value, so both still pass unchanged.

`test_a_second_battle_won_against_a_new_opponent_pays_again` earns a *second* win and currently asserts:

```python
    assert second_win.reward == pytest.approx(0.5)
```

Under the decay, the second win in the same tier pays `0.5 / sqrt(2)`, not a flat `0.5`. Update the assertion:

```python
    assert second_win.reward == pytest.approx(0.5 / math.sqrt(2))
```

Also update that test's docstring line "Same shape as a second badge: the monotone-gain formula only pays the marginal increase" to instead explain the decay is now the reason the second win pays less, not (only) the marginal-gain mechanic — e.g. replace the docstring with:

```python
    """The second win in a tier pays its own decayed marginal value
    (weight/sqrt(2)), proving total_battles_won's successor (battle_sum)
    is a real running sum that keeps growing across wins, not a one-shot
    flag consumed by the first win."""
```

- [ ] **Step 8: Fix `test_state_dict_round_trips_the_accumulator` and any other `total_battles_won` reference**

Run: `grep -rn "total_battles_won" tests/` — update any remaining reference (there should be none in production code after Step 6, but a checkpoint round-trip test may construct a raw state dict by hand). If found, replace with `battle_sum`/`wins_since_badge`/`last_badge_count` equivalents.

- [ ] **Step 9: Run the full unit suite**

Run: `uv run pytest tests/unit -q`
Expected: all pass, coverage floor met. `grep -rn "total_battles_won" src/ tests/` returns nothing.

- [ ] **Step 10: Prove the badge-reset test can fail**

Temporarily change `_update_badge_tier` to never reset (`if False:` instead of the real condition), rerun `test_a_badge_resets_the_win_decay_to_a_fresh_curve`, confirm it fails (the post-badge win pays a heavily decayed value instead of a fresh `0.5`), then restore the real condition.

- [ ] **Step 11: Commit**

```bash
git add src/pokemon_env/rewards.py tests/unit/test_pokemon_env_rewards.py
git commit -m "feat(pokemon_env): decay battle_won per badge tier instead of a flat count

Run 11 (wandb cq14kskq, docs/2026-09-05-battle-reward-hacking-audit.md):
battle_win_weight was the only reward component with a flat,
non-decaying marginal value on an easy repeatable action -- ~261
wins/env by update 46, still climbing. Gives it the same decaying-sum
shape explore_sum already uses (i-th win since the last badge worth
1/sqrt(i)), resetting the exponent -- never the accumulated sum itself,
so it can't manufacture a spurious reward spike -- each time a badge is
earned. No hard cap, per explicit design direction: winning stays
legitimately rewarding, it just stops being a flat-value infinite
grind."
```

---

### Task 5: Blackout detection, heal exemption, and the HP-reader switch

**Files:**
- Modify: `src/pokemon_env/rewards.py`
- Test: `tests/unit/test_pokemon_env_rewards.py`

**Interfaces:**
- Consumes: `ram.live_party_hp_fraction` (Task 1).
- Produces: `RewardAccumulator.blackout_count: int` property. Used by Task 7.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_pokemon_env_rewards.py`:

```python
def test_hp_dropping_to_zero_is_counted_as_a_blackout(fake_emulator) -> None:
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 50
    accumulator = RewardAccumulator(EnvConfig())
    accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 0

    accumulator.step(fake_emulator)

    assert accumulator.blackout_count == 1


def test_hp_staying_at_zero_does_not_double_count_the_blackout(fake_emulator) -> None:
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 50
    accumulator = RewardAccumulator(EnvConfig())
    accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 0
    accumulator.step(fake_emulator)

    accumulator.step(fake_emulator)  # still 0 -- simulates the black-screen animation

    assert accumulator.blackout_count == 1


def test_the_blackout_recovery_heal_earns_no_heal_credit(fake_emulator) -> None:
    heal_accumulator = RewardAccumulator(EnvConfig(heal_weight=1.0))
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 50
    heal_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 0
    heal_accumulator.step(fake_emulator)  # blackout
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 100  # forced full heal at the Center

    recovery = heal_accumulator.step(fake_emulator)

    assert recovery.components["heal"] == pytest.approx(0.0)


def test_a_later_genuine_heal_earns_credit_normally_after_the_exemption_is_spent(
    fake_emulator,
) -> None:
    heal_accumulator = RewardAccumulator(EnvConfig(heal_weight=1.0))
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 50
    heal_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 0
    heal_accumulator.step(fake_emulator)  # blackout
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 100
    heal_accumulator.step(fake_emulator)  # the exempted recovery
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 60  # took damage
    heal_accumulator.step(fake_emulator)
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 90  # a real, later heal

    genuine_heal = heal_accumulator.step(fake_emulator)

    assert genuine_heal.components["heal"] == pytest.approx(0.3 * 0.3)


def test_blackout_count_persists_across_an_episode_boundary(fake_emulator) -> None:
    accumulator = RewardAccumulator(EnvConfig())
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 50
    accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 0
    accumulator.step(fake_emulator)  # blackout, count -> 1

    accumulator.reset(fake_emulator)  # a fresh episode -- badges/heal/etc. all reset

    assert accumulator.blackout_count == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_pokemon_env_rewards.py -q --no-cov -k "blackout or exemption_is_spent"`
Expected: FAIL — `AttributeError: 'RewardAccumulator' object has no attribute 'blackout_count'` on the first four; the persistence test fails once `blackout_count` exists but before `reset()` is updated to persist it.

- [ ] **Step 3: Update `_State`**

Add two fields after `last_opponent_hp_fraction` (around line 111):

```python
    blackout_count: int = 0
    pending_blackout_recovery: bool = False
```

- [ ] **Step 4: Add the `blackout_count` property**

Directly after the existing `steps_since_new_coord` property (around line 145):

```python
    @property
    def blackout_count(self) -> int:
        return self._state.blackout_count
```

- [ ] **Step 5: Switch `_update_healing`'s HP reader and add blackout detection**

Replace `_update_healing` entirely:

```python
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
```

- [ ] **Step 6: Switch `reset()`'s HP reader and persist `blackout_count`**

In `reset()`, replace:

```python
        explore_weight = self._config.explore_weight
        persisted_explore_sum = self._state.explore_sum
        self._state = _State(
            base_event_flags=ram.event_flag_count(mem),
            last_hp_fraction=ram.aggregate_hp_fraction(mem),
            last_party_size=ram.party_size(mem),
            max_total=explore_weight * persisted_explore_sum,
            explore_sum=persisted_explore_sum,
            seen_coords=self._state.seen_coords,
            seen_maps=self._state.seen_maps,
        )
```

with:

```python
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
```

`reset()`'s full docstring currently reads exactly:

```python
        """Captures the event flags init.state already has set, so the agent
        is not paid on step one for progress it did not make.

        Exploration (seen_coords/seen_maps/explore_sum) persists across this
        reset; everything else does not. EnvSession.reset() is one code path
        for both the very first cold start and every later autoreset at
        max_steps, and the emulator genuinely reloads init_state on both --
        badges/HP/party/events really do come back to their init values, so
        max_total (which those feed into) must reset with them, or a fresh
        episode would need to out-earn every badge the last one banked just
        to see reward again. Exploration is different: the coordinates
        themselves are still the same physical tiles, seen for real, so
        forgetting them on every one of the ~160-update resets this project's
        fixed max_steps produces (run 9, docs/ppo-experiment-history.md) pays
        the decaying 1/sqrt(k) bonus at full price for the same starting
        area every time instead of ever converging on genuinely new ground.

        max_total is rebased to exactly the persisted explore contribution
        (not left at 0) -- otherwise the fresh episode's `total` would sit
        above a stale 0 baseline and manufacture a reward on step one that
        nothing there actually earned."""
```

Replace the whole docstring with:

```python
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
```

- [ ] **Step 7: Update `state_dict`/`load_state_dict`**

Add `"blackout_count": self._state.blackout_count,` and `"pending_blackout_recovery": self._state.pending_blackout_recovery,` to `state_dict()`'s return dict, and the matching `blackout_count=state["blackout_count"],` / `pending_blackout_recovery=state["pending_blackout_recovery"],` to `load_state_dict()`'s `_State(...)` construction.

- [ ] **Step 8: Remove the now-dead `aggregate_hp_fraction`**

Run: `grep -rn "aggregate_hp_fraction" src/ tests/` — confirm the only remaining references are `ram.py`'s own definition and its two tests (`test_aggregate_hp_fraction_is_zero_when_max_hp_is_zero`, `test_aggregate_hp_fraction_sums_across_the_party` in `tests/unit/test_pokemon_env_ram.py`). Delete the function from `src/pokemon_env/ram.py` (lines 102-110) and both tests.

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_pokemon_env_rewards.py tests/unit/test_pokemon_env_ram.py -q --no-cov`
Expected: all pass.

- [ ] **Step 10: Run the full unit suite**

Run: `uv run pytest tests/unit -q`
Expected: all pass, coverage floor met.

- [ ] **Step 11: Prove the exemption test can fail**

Temporarily remove the `if self._state.pending_blackout_recovery:` branch entirely (so the recovery jump always falls through to ordinary crediting), rerun `test_the_blackout_recovery_heal_earns_no_heal_credit`, confirm it fails (heal credit is nonzero), then restore.

- [ ] **Step 12: Commit**

```bash
git add src/pokemon_env/rewards.py src/pokemon_env/ram.py tests/unit/test_pokemon_env_rewards.py tests/unit/test_pokemon_env_ram.py
git commit -m "feat(pokemon_env): log blackouts, exempt their forced heal from reward

A blackout (all live party HP hitting 0) is detected the same way a
battle win already is -- a falling edge, guarded against double-counting
across the several env-steps the black-screen/teleport sequence spans.
The forced full-heal that follows is excluded from the heal reward,
which was previously paying it in full -- the exact effect the audit
(docs/2026-09-05-battle-reward-hacking-audit.md) found explained why
reward/heal saturated its cap so early in run 11.

_update_healing and reset() both switch from ram.aggregate_hp_fraction
to live_party_hp_fraction (Task 1), fixing a latent masking bug as a
side effect; aggregate_hp_fraction is now dead code and removed.

No separate blackout punishment, per explicit design direction --
logging and the heal exemption are the whole fix here; Task 6 adds the
proactive pressure meant to prevent reaching this state at all."
```

---

### Task 6: Continuous low-HP penalty

**Files:**
- Modify: `src/pokemon_env/rewards.py`
- Test: `tests/unit/test_pokemon_env_rewards.py`

**Interfaces:**
- Consumes: `ram.live_party_hp_fraction` (Task 1), `EnvConfig.low_hp_penalty_weight`/`low_hp_threshold` (Task 3).
- Produces: `components["low_hp"]` in every `RewardBreakdown`.

- [ ] **Step 1: Write the failing tests**

```python
def test_no_low_hp_penalty_at_or_above_the_threshold(fake_emulator) -> None:
    accumulator = RewardAccumulator(EnvConfig(low_hp_penalty_weight=0.1))
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 25  # exactly the 25% default threshold
    accumulator.reset(fake_emulator)

    result = accumulator.step(fake_emulator)

    assert result.components["low_hp"] == pytest.approx(0.0)


def test_low_hp_penalty_scales_with_severity_below_the_threshold(fake_emulator) -> None:
    """At 10% HP against a 25% threshold, severity = (0.25-0.10)/0.25 = 0.6."""
    accumulator = RewardAccumulator(EnvConfig(low_hp_penalty_weight=0.1))
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 10
    accumulator.reset(fake_emulator)

    result = accumulator.step(fake_emulator)

    assert result.components["low_hp"] == pytest.approx(-0.1 * 0.6)


def test_low_hp_penalty_is_at_full_severity_when_fully_fainted(fake_emulator) -> None:
    accumulator = RewardAccumulator(EnvConfig(low_hp_penalty_weight=0.1))
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 50
    accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 0

    result = accumulator.step(fake_emulator)

    assert result.components["low_hp"] == pytest.approx(-0.1)


def test_low_hp_penalty_applies_in_battle_the_same_as_the_overworld(fake_emulator) -> None:
    accumulator = RewardAccumulator(EnvConfig(low_hp_penalty_weight=0.1))
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 10
    accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1

    result = accumulator.step(fake_emulator)

    assert result.components["low_hp"] == pytest.approx(-0.1 * 0.6)


def test_low_hp_penalty_composes_with_idle_penalty(fake_emulator) -> None:
    accumulator = RewardAccumulator(
        EnvConfig(low_hp_penalty_weight=0.1, idle_penalty_weight=0.01)
    )
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 10
    accumulator.reset(fake_emulator)
    accumulator.step(fake_emulator)  # registers (0, 0, 0) as seen

    results = [accumulator.step(fake_emulator) for _ in range(1002)]  # past the idle grace window

    assert results[-1].reward == pytest.approx(0.0 - 0.01 - 0.1 * 0.6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_pokemon_env_rewards.py -q --no-cov -k low_hp`
Expected: FAIL — `KeyError: 'low_hp'` (the component doesn't exist yet).

- [ ] **Step 3: Add `_low_hp_penalty` and wire it into `step()`**

Add a new method, e.g. directly after `_event_score`:

```python
    def _low_hp_penalty(self, mem: Emulator) -> float:
        """Continuous, not a cliff: ramps from 0 at the threshold to full
        strength at 0% HP, so the pressure is gradable rather than a step
        discontinuity PPO's advantage estimation would otherwise have to
        absorb in one bucket. Applies whether in battle or the overworld --
        a hurt team wandering around ungoverned by anything gets the same
        nudge as a hurt team still fighting; the lesson is "go heal your
        team," not "don't fight while hurt.\""""
        threshold = self._config.low_hp_threshold
        live_fraction = ram.live_party_hp_fraction(mem)
        if live_fraction >= threshold:
            return 0.0
        severity = (threshold - live_fraction) / threshold
        return self._config.low_hp_penalty_weight * severity
```

In `step()`, add the computation alongside `idle_penalty` and fold it into the returned `RewardBreakdown`. Replace:

```python
        return RewardBreakdown(
            reward=min(gain, 1.0) - idle_penalty,
            clipped=gain > 1.0,
            components={**components, "idle": -idle_penalty},
            badge_count=badge_count,
            event_flag_count=event_flag_count,
        )
```

with:

```python
        low_hp_penalty = self._low_hp_penalty(mem)
        return RewardBreakdown(
            reward=min(gain, 1.0) - idle_penalty - low_hp_penalty,
            clipped=gain > 1.0,
            components={**components, "idle": -idle_penalty, "low_hp": -low_hp_penalty},
            badge_count=badge_count,
            event_flag_count=event_flag_count,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_pokemon_env_rewards.py -q --no-cov`
Expected: all pass.

- [ ] **Step 5: Fix the exact-dict-equality test in `test_pokemon_env_vec_env.py`**

Run: `uv run pytest tests/unit -q 2>&1 | tail -30` — expect `test_last_components_reports_the_mean_reward_breakdown_across_envs` to fail (new `"low_hp"` key not in its expected dict), matching the exact pattern hit twice already this week. Open `tests/unit/test_pokemon_env_vec_env.py`, add `"low_hp": 0.0` to the expected dict literal.

- [ ] **Step 6: Run the full unit suite**

Run: `uv run pytest tests/unit -q`
Expected: all pass, coverage floor met.

- [ ] **Step 7: Prove the severity-scaling test can fail**

Temporarily change `severity = (threshold - live_fraction) / threshold` to `severity = 1.0` (flatten the ramp into a cliff), rerun `test_low_hp_penalty_scales_with_severity_below_the_threshold`, confirm it fails (`-0.1` instead of `-0.06`), then restore.

- [ ] **Step 8: Commit**

```bash
git add src/pokemon_env/rewards.py tests/unit/test_pokemon_env_rewards.py tests/unit/test_pokemon_env_vec_env.py
git commit -m "feat(pokemon_env): continuous low-HP penalty, 25% threshold

Proactive, not reactive: ramps smoothly from 0 at low_hp_threshold to
full strength at 0% HP, active in battle and the overworld alike --
the lesson is 'go heal your team,' not 'don't fight while hurt,' per
explicit design direction to avoid teaching battle-avoidance. A
genuinely separate additive term, same pattern as idle_penalty."
```

---

### Task 7: `blackout_count` threads through `stats()`

**Files:**
- Modify: `src/pokemon_env/session.py`
- Modify: `tests/unit/fakes.py`
- Test: `tests/unit/test_pokemon_env_session.py`

**Interfaces:**
- Consumes: `RewardAccumulator.blackout_count` (Task 5).
- Produces: `EnvSession.stats()["blackout_count"]`, `FakeBackend`/`FakeVecEnv` stats dicts carrying the same key. Used by Task 8.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_pokemon_env_session.py`, near the existing `stats()` tests:

```python
def test_stats_reports_the_blackout_count(fake_emulator) -> None:
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 50
    session = EnvSession(fake_emulator, EnvConfig(), init_state=b"")
    session.reset()
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 0

    session.step(0)

    assert session.stats()["blackout_count"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_pokemon_env_session.py -q --no-cov -k blackout`
Expected: FAIL — `KeyError: 'blackout_count'`

- [ ] **Step 3: Implement**

In `src/pokemon_env/session.py`, `stats()`, add `"blackout_count": self._rewards.blackout_count,` to the returned dict (after `"steps_since_new_coord"`).

- [ ] **Step 4: Update `FakeBackend`/`FakeVecEnv`**

In `tests/unit/fakes.py`:

- `FakeBackend.__init__`: add a `blackout_count: int = 0` parameter, stored as `self._blackout_count = blackout_count`.
- `FakeBackend.stats()`: add `"blackout_count": self._blackout_count,` to the returned dict.
- `FakeVecEnv.__init__`: add a `blackout_count: int = 0` parameter, stored as `self._blackout_count = blackout_count`.
- `FakeVecEnv.stats()`: add `"blackout_count": self._blackout_count,` to each entry's dict, and update its docstring's field list (`coord_keys`/`badges`/`event_flags`/`episode_lengths`/`steps_since_new_coord`) to also name `blackout_count`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_pokemon_env_session.py -q --no-cov`
Expected: all pass.

- [ ] **Step 6: Run the full unit suite**

Run: `uv run pytest tests/unit -q`
Expected: all pass — this step's `fakes.py` changes are additive (new optional constructor params, new dict key), so no other test should break, but confirm.

- [ ] **Step 7: Commit**

```bash
git add src/pokemon_env/session.py tests/unit/fakes.py tests/unit/test_pokemon_env_session.py
git commit -m "feat(pokemon_env): thread blackout_count through EnvSession.stats()

Mechanical extension of the same rollout steps_since_new_coord already
got two sessions ago -- FakeBackend/FakeVecEnv gain the matching
optional constructor param so downstream telemetry tests (Task 8/9) can
exercise a nonzero blackout count."
```

---

### Task 8: `env/blackout_count_total` telemetry

**Files:**
- Modify: `src/pokemon_env/telemetry.py`
- Test: `tests/unit/test_pokemon_env_telemetry.py`

**Interfaces:**
- Consumes: `stats` entries carrying `"blackout_count"` (Task 7).
- Produces: `rollout_metrics(...)["env/blackout_count_total"]`. Used by Task 9 (no code change needed there beyond what already calls `rollout_metrics`, since this is summed from the same `stats` list already passed in).

Deliberately no `_delta` companion, unlike `worker_respawns_total`/`_delta`: `respawns_before`/`respawns_total` are cheap in-process attribute reads on `vec_env._backends`, so calling `_respawns` twice per update costs nothing; `blackout_count` lives inside each subprocess worker's `RewardAccumulator` and only reaches the parent through `stats()`'s IPC round trip to all 64 workers. Computing a delta the same way would mean a second full `stats()` call every update purely for this metric — `env/blackout_count_total` alone (matching how `explore/unique_coords_total` also has no delta companion) avoids that cost.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_pokemon_env_telemetry.py`, near the `episodes_finished` test — update the shared `_empty_stats`/`_stats_with_steps_since_new_coord` helpers first (add `"blackout_count": 0` to each dict they build), then add:

```python
def test_rollout_metrics_sums_blackout_count_across_envs() -> None:
    stats = [
        {
            "coord_keys": [],
            "badges": 0,
            "event_flags": 0,
            "step_count": 0,
            "episode_lengths": [],
            "steps_since_new_coord": 0,
            "blackout_count": count,
        }
        for count in (2, 0, 5)
    ]

    metrics = rollout_metrics(_vec_step(3), components={}, clip_fire_rate=0.0, respawns=0, stats=stats)

    assert metrics["env/blackout_count_total"] == pytest.approx(7.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_pokemon_env_telemetry.py -q --no-cov -k blackout`
Expected: FAIL — `KeyError: 'env/blackout_count_total'`

- [ ] **Step 3: Implement**

In `src/pokemon_env/telemetry.py`, `rollout_metrics`, add a line computing the sum alongside `stalls` (near `stalls = [entry["steps_since_new_coord"] for entry in stats]`):

```python
    blackouts = sum(entry["blackout_count"] for entry in stats)
```

Add to the `metrics` dict (near the other `env/` entries):

```python
        "env/blackout_count_total": float(blackouts),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_pokemon_env_telemetry.py -q --no-cov`
Expected: all pass — this also requires every pre-existing `rollout_metrics` test using `_empty_stats`/`_stats_with_steps_since_new_coord` or an inline stats dict literal to already carry `"blackout_count"` from Step 1's helper update; fix any remaining inline dict literal the same way (`"blackout_count": 0`).

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest tests/unit -q`
Expected: all pass, coverage floor met.

- [ ] **Step 6: Commit**

```bash
git add src/pokemon_env/telemetry.py tests/unit/test_pokemon_env_telemetry.py
git commit -m "feat(pokemon_env): surface env/blackout_count_total telemetry

Summed across envs from the same stats() call rollout_metrics already
receives -- no delta companion (unlike worker_respawns_total/_delta):
computing one would need a second per-update IPC round trip to all 64
subprocess workers purely for this metric, unlike the cheap in-process
respawn counter."
```

---

### Task 9: `ppo/trainer.py` picks up the new telemetry automatically — verify, no code change expected

**Files:**
- Test: `tests/unit/test_ppo_trainer.py`

**Interfaces:**
- Consumes: `rollout_metrics(...)["env/blackout_count_total"]` (Task 8), already merged into `env_metrics` and logged wholesale by `update_metrics` (`src/ppo/telemetry.py`) — the same pass-through that carried `env/stalled_frac` and `env/steps_since_new_coord_mean` onto the dashboard with no `ppo/trainer.py` changes when those were added.

This task exists to *prove* the pass-through actually works end to end through `FakeVecEnv`, not to add new production code — `rollout_metrics`'s return dict already flows into `run_training`'s single `wandb_run.log()` call unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_ppo_trainer.py`. First add a `blackout_count: int = 0` parameter to `_trainer_harness` (mirroring the existing `steps_since_new_coord` parameter), threaded to `FakeVecEnv(..., blackout_count=blackout_count)`:

```python
def test_blackout_count_total_is_logged(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, blackout_count=3)

    run_training(harness.deps, max_updates=1)
    logged = harness.wandb_run.logged[0]

    assert logged["env/blackout_count_total"] == pytest.approx(3.0 * _N_ENVS)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_ppo_trainer.py -q --no-cov -k blackout_count_total`
Expected: FAIL — `TypeError: _trainer_harness() got an unexpected keyword argument 'blackout_count'`

- [ ] **Step 3: Add the harness parameter**

In `tests/unit/test_ppo_trainer.py`, `_trainer_harness`'s signature currently ends with:

```python
    target_kl: float | None = None,
    max_consecutive_stalled_updates: int = 10,
    steps_since_new_coord: int = 0,
) -> _TrainerHarness:
```

Add `blackout_count: int = 0` after `steps_since_new_coord`:

```python
    target_kl: float | None = None,
    max_consecutive_stalled_updates: int = 10,
    steps_since_new_coord: int = 0,
    blackout_count: int = 0,
) -> _TrainerHarness:
```

Further down the same function, `FakeVecEnv` is constructed as:

```python
    vec_env = FakeVecEnv(
        n_envs=_N_ENVS,
        aux_dim=policy_config.aux_state_dim,
        done_at_step=None,
        steps_since_new_coord=steps_since_new_coord,
    )
```

Add the new parameter to that call:

```python
    vec_env = FakeVecEnv(
        n_envs=_N_ENVS,
        aux_dim=policy_config.aux_state_dim,
        done_at_step=None,
        steps_since_new_coord=steps_since_new_coord,
        blackout_count=blackout_count,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_ppo_trainer.py -q --no-cov -k blackout_count_total`
Expected: PASS — confirms the metric reaches `wandb_run.log()` with no `src/ppo/trainer.py` change needed, since `rollout_metrics`'s dict already flows through unchanged.

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest tests/unit -q`
Expected: all pass, coverage floor met.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_ppo_trainer.py
git commit -m "test(ppo): prove env/blackout_count_total reaches wandb_run.log()

No src/ppo/trainer.py change needed -- rollout_metrics's dict already
flows through update_metrics unchanged, the same pass-through that
carried env/stalled_frac onto the dashboard with no trainer.py edit
when it was added. This is the end-to-end proof, not new plumbing."
```

---

### Task 10: Production config values

**Files:**
- Modify: `configs/pokemon_env.yaml`

- [ ] **Step 1: Set the real values**

Append to `configs/pokemon_env.yaml`, after the existing `idle_penalty_weight` block:

```yaml
# Run 11 (2026-09-05, wandb cq14kskq, docs/2026-09-05-battle-reward-hacking-audit.md
# and docs/superpowers/specs/2026-09-05-battle-reward-redesign-design.md) found
# battle_win_weight's flat, uncapped marginal value let the agent grind wild
# battles as a terminal strategy -- ~261 wins/env by update 46 while
# exploration flatlined and reward/money crashed from repeated blackouts.
# battle_won now decays per badge tier (see rewards.py) rather than being
# capped -- this section is the OTHER half of that fix: a continuous,
# proactive penalty for a hurt team, active whether battling or not, meant
# to prevent ever reaching a blackout rather than punishing one after the
# fact. 0.25 -- not the 50% first considered -- is the threshold below
# which it starts ramping up. low_hp_penalty_weight itself is a first
# guess, not yet validated by a live run -- re-tune against the next run's
# reward/low_hp trajectory alongside env/blackout_count_total, the same
# way every other weight in this file has been.
low_hp_penalty_weight: 0.05
low_hp_threshold: 0.25
```

- [ ] **Step 2: Verify the config loads**

Run:
```bash
uv run python3 -c "
from pokemon_env.config import load_config
cfg = load_config('configs/pokemon_env.yaml')
print(cfg.low_hp_penalty_weight, cfg.low_hp_threshold)
"
```
Expected output: `0.05 0.25`

- [ ] **Step 3: Run the full unit suite one final time**

Run: `uv run pytest tests/unit -q`
Expected: all pass, coverage floor met, matching or exceeding the pre-plan baseline (821 passed, 95.19% coverage).

- [ ] **Step 4: `ruff check` the full set of touched files**

Run:
```bash
uv run ruff check src/pokemon_env/ram.py src/pokemon_env/aux_state.py src/pokemon_env/config.py \
  src/pokemon_env/rewards.py src/pokemon_env/session.py src/pokemon_env/telemetry.py \
  tests/unit/fakes.py tests/unit/test_pokemon_env_ram.py tests/unit/test_pokemon_env_aux_state.py \
  tests/unit/test_pokemon_env_config.py tests/unit/test_pokemon_env_rewards.py \
  tests/unit/test_pokemon_env_session.py tests/unit/test_pokemon_env_telemetry.py \
  tests/unit/test_pokemon_env_vec_env.py tests/unit/test_ppo_trainer.py
```
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add configs/pokemon_env.yaml
git commit -m "chore(pokemon_env): set low_hp_penalty_weight=0.05, low_hp_threshold=0.25

First guess, reasoned but not yet tuned against a live run -- 0.25 per
explicit direction (50% was considered too high). Completes the battle
reward redesign from docs/superpowers/specs/
2026-09-05-battle-reward-redesign-design.md."
```

---

## Post-implementation

Update `docs/ppo-experiment-history.md` with a "Fix" entry under Run 11 summarizing what shipped (mirroring the style of every prior fix entry in that file), once all ten tasks are committed. Not its own task here since it has no tests of its own and depends on the whole plan being done first — do it as a final wrap-up step, not a numbered task.
