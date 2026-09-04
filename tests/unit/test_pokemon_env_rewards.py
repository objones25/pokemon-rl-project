import math

import pytest

from pokemon_env import ram
from pokemon_env.config import EnvConfig
from pokemon_env.rewards import RewardAccumulator


@pytest.fixture
def accumulator(fake_emulator) -> RewardAccumulator:
    acc = RewardAccumulator(EnvConfig())
    acc.reset(fake_emulator)
    return acc


def _set_coord(emulator, x: int, y: int, map_id: int) -> None:
    """Helper, not a test."""
    emulator.memory[ram.X_POS_ADDR] = x
    emulator.memory[ram.Y_POS_ADDR] = y
    emulator.memory[ram.MAP_ID_ADDR] = map_id


def test_a_level_that_rises_then_falls_earns_nothing_the_second_time(
    fake_emulator, accumulator
) -> None:
    """The cycle exploit section 4 names: deposit and withdraw a Pokemon to
    farm the same level reward forever. max_historical makes it pay once."""
    fake_emulator.memory[ram.PARTY_LEVEL_BASE] = 20
    accumulator.step(fake_emulator)
    fake_emulator.memory[ram.PARTY_LEVEL_BASE] = 5
    accumulator.step(fake_emulator)
    fake_emulator.memory[ram.PARTY_LEVEL_BASE] = 20

    second_time = accumulator.step(fake_emulator)

    assert second_time.reward == pytest.approx(0.0)


def test_revisiting_a_coordinate_earns_nothing(fake_emulator, accumulator) -> None:
    _set_coord(fake_emulator, 5, 5, 1)
    first = accumulator.step(fake_emulator)
    _set_coord(fake_emulator, 6, 6, 1)
    accumulator.step(fake_emulator)
    _set_coord(fake_emulator, 5, 5, 1)

    revisit = accumulator.step(fake_emulator)

    assert (first.reward > 0.0, revisit.reward) == (True, pytest.approx(0.0))


def test_the_first_new_coordinate_earns_the_full_explore_weight(
    fake_emulator, accumulator
) -> None:
    """k=1 -> 1/sqrt(1) = 1.0, times explore_weight 0.30."""
    _set_coord(fake_emulator, 5, 5, 1)

    result = accumulator.step(fake_emulator)

    assert result.reward == pytest.approx(0.30)


def test_exploration_decays_as_one_over_root_k(fake_emulator, accumulator) -> None:
    """Section 4 requires a decaying exploration bonus; the reference
    implementation's flat 0.1 does not decay. The 4th new coordinate is worth
    0.30/sqrt(4) = 0.15."""
    _set_coord(fake_emulator, 1, 1, 1)
    accumulator.step(fake_emulator)
    _set_coord(fake_emulator, 2, 2, 1)
    accumulator.step(fake_emulator)
    _set_coord(fake_emulator, 3, 3, 1)
    accumulator.step(fake_emulator)
    _set_coord(fake_emulator, 4, 4, 1)

    fourth = accumulator.step(fake_emulator)

    assert fourth.reward == pytest.approx(0.30 / math.sqrt(4))


def test_coordinates_are_not_recorded_during_battle(fake_emulator, accumulator) -> None:
    """In battle the position bytes are stale, so recording them would credit
    exploration for tiles never actually walked."""
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_coord(fake_emulator, 5, 5, 1)

    result = accumulator.step(fake_emulator)

    assert (result.reward, accumulator.coords_seen) == (pytest.approx(0.0), 0)


def test_walking_over_known_ground_pays_no_idle_penalty_within_the_grace_window(
    fake_emulator,
) -> None:
    """A 0-grace trigger is structurally broken, not just mistuned:
    explore_weight/sqrt(k) decays toward 0 as k grows while a flat per-step
    penalty does not, so it eventually makes healthy exploration
    net-negative regardless of idle_penalty_weight's magnitude -- confirmed
    against run 9's own numbers, not just in theory (docs/ppo-experiment-
    history.md). The grace window means genuinely normal walking between
    two already-known points -- even for hundreds of steps -- costs
    nothing."""
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    _set_coord(fake_emulator, 5, 5, 1)
    idle_accumulator.step(fake_emulator)  # discovers (5, 5, 1)

    results = [idle_accumulator.step(fake_emulator) for _ in range(500)]  # revisits, well within grace

    assert all(result.reward == pytest.approx(0.0) for result in results)


def test_a_stall_well_past_the_grace_window_pays_the_idle_penalty(fake_emulator) -> None:
    """The 2026-09-04 investigation (docs/ppo-experiment-history.md, run 9)
    found the majority of envs stuck in menus for thousands of steps with
    zero cost. idle_penalty_weight is the fix -- applied outside the
    monotone-gain formula, since that formula floors every step at 0 and
    cannot express a cost (see RewardAccumulator.step) -- once a stall
    genuinely outlasts the grace window."""
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    _set_coord(fake_emulator, 5, 5, 1)
    idle_accumulator.step(fake_emulator)  # discovers (5, 5, 1)

    results = [idle_accumulator.step(fake_emulator) for _ in range(1002)]  # past the 1000-step grace

    assert results[-1].reward == pytest.approx(-0.01)


def test_the_idle_penalty_does_not_apply_the_step_a_new_coordinate_is_found(
    fake_emulator,
) -> None:
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    _set_coord(fake_emulator, 5, 5, 1)

    result = idle_accumulator.step(fake_emulator)

    assert result.reward == pytest.approx(0.30)


def test_the_idle_penalty_does_not_apply_during_battle(fake_emulator) -> None:
    """Battle turns never move the player -- position bytes are stale there
    (see _update_exploration) -- so they always look like a step with no new
    coordinate. Penalizing them would tax the agent for fighting, exactly
    the opposite of what this project wants."""
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1

    result = idle_accumulator.step(fake_emulator)

    assert result.reward == pytest.approx(0.0)


def test_the_idle_penalty_is_reported_as_its_own_reward_component(fake_emulator) -> None:
    """Observability first: a term that suppresses reward must be visible on
    its own W&B panel (reward/idle), not silently folded into the total."""
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    _set_coord(fake_emulator, 5, 5, 1)
    idle_accumulator.step(fake_emulator)  # discovers (5, 5, 1)

    results = [idle_accumulator.step(fake_emulator) for _ in range(1002)]  # past the grace window

    assert results[-1].components["idle"] == pytest.approx(-0.01)


def test_the_idle_penalty_does_not_suppress_a_genuine_gain_in_the_same_step(
    fake_emulator,
) -> None:
    """A badge earned while genuinely stalled (past the grace window) must
    still pay in full -- the idle penalty is a separate, additive term, not
    something that erodes max_total or the gain computation itself."""
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    idle_accumulator.step(fake_emulator)  # registers (0, 0, 0) as seen
    for _ in range(1001):
        idle_accumulator.step(fake_emulator)  # same coordinate, past the grace window
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0001

    result = idle_accumulator.step(fake_emulator)  # still stalled, plus a badge

    assert result.reward == pytest.approx(1.00 - 0.01)


def _set_opponent_hp(emulator, current: int, max_hp: int) -> None:
    """Helper, not a test. Both fields are u16 big-endian; only the low byte
    is set since these tests never need values above 255."""
    emulator.memory[ram.ENEMY_MON_HP_ADDR + 1] = current
    emulator.memory[ram.ENEMY_MON_MAX_HP_ADDR + 1] = max_hp


def test_idle_penalty_does_not_apply_within_the_battle_stall_grace_window(
    fake_emulator,
) -> None:
    """A real battle turn takes several env-steps of menu navigation (open
    FIGHT, pick a move) before HP actually changes -- a 0-grace trigger
    exactly like the overworld one would tax an agent for the mechanical
    overhead of fighting, not just for genuinely stalling."""
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    idle_accumulator.step(fake_emulator)

    result = idle_accumulator.step(fake_emulator)  # HP unchanged, still within grace

    assert result.reward == pytest.approx(0.0)


def test_idle_penalty_applies_once_a_battle_stalls_past_the_grace_window(
    fake_emulator,
) -> None:
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)

    results = [idle_accumulator.step(fake_emulator) for _ in range(10)]  # past the grace window

    assert results[-1].reward == pytest.approx(-0.01)


def test_a_change_in_opponent_hp_resets_the_battle_stall_counter(fake_emulator) -> None:
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    for _ in range(10):
        idle_accumulator.step(fake_emulator)  # now well past the grace window
    _set_opponent_hp(fake_emulator, current=5, max_hp=10)
    idle_accumulator.step(fake_emulator)  # HP just changed -- counter resets

    result = idle_accumulator.step(fake_emulator)  # one step since, within grace again

    assert result.reward == pytest.approx(0.0)


def test_damage_dealt_pays_the_squared_hp_fraction_drop(fake_emulator) -> None:
    """Mirrors _update_healing's own squared-delta shape exactly, applied to
    the opponent's HP instead of the player's."""
    damage_accumulator = RewardAccumulator(EnvConfig(damage_weight=0.1))
    damage_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    damage_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=5, max_hp=10)

    result = damage_accumulator.step(fake_emulator)

    assert result.components["damage"] == pytest.approx(0.1 * 0.5**2)


def test_damage_dealt_is_capped_like_heal(fake_emulator) -> None:
    damage_accumulator = RewardAccumulator(EnvConfig(damage_weight=100.0))
    damage_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    damage_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=10)

    result = damage_accumulator.step(fake_emulator)

    assert result.components["damage"] == pytest.approx(0.2)


def test_a_new_opponents_fresh_hp_is_not_recorded_as_damage(fake_emulator) -> None:
    """HP rising -- a fainted opponent replaced by a fresh one -- must never
    register as damage -- only a genuine drop does. Banks a real faint first
    (total_damage=1.0) so the assertion actually distinguishes "unaffected"
    from "always zero"."""
    damage_accumulator = RewardAccumulator(EnvConfig(damage_weight=0.1))
    damage_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    damage_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=10)
    damage_accumulator.step(fake_emulator)  # a real faint: total_damage becomes 1.0
    _set_opponent_hp(fake_emulator, current=8, max_hp=8)  # a fresh opponent, full HP

    result = damage_accumulator.step(fake_emulator)

    assert result.components["damage"] == pytest.approx(0.1)


def test_damage_is_not_recorded_outside_battle(fake_emulator) -> None:
    """Position-style staleness guard: opponent HP bytes mean nothing once
    the battle has ended."""
    damage_accumulator = RewardAccumulator(EnvConfig(damage_weight=1.0))
    damage_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    damage_accumulator.step(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 0
    _set_opponent_hp(fake_emulator, current=0, max_hp=10)

    result = damage_accumulator.step(fake_emulator)

    assert result.components["damage"] == pytest.approx(0.0)


def test_a_fainted_opponent_earns_the_battle_win_bonus(fake_emulator) -> None:
    win_accumulator = RewardAccumulator(EnvConfig(battle_win_weight=0.5))
    win_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=10)

    result = win_accumulator.step(fake_emulator)

    assert result.components["battle_won"] == pytest.approx(0.5)


def test_the_battle_win_bonus_does_not_repeat_while_the_same_faint_persists(
    fake_emulator,
) -> None:
    """The faint/switch animation holds the opponent at 0 HP for many
    env-steps -- a rising-edge detector, not a level check, or every one of
    those steps would pay the bonus again."""
    win_accumulator = RewardAccumulator(EnvConfig(battle_win_weight=0.5))
    win_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=10)
    win_accumulator.step(fake_emulator)  # the win, banked into max_total

    still_fainted = win_accumulator.step(fake_emulator)  # opponent HP unchanged at 0

    assert (still_fainted.reward, still_fainted.components["battle_won"]) == (
        pytest.approx(0.0),
        pytest.approx(0.5),  # unchanged from the win step -- not double-counted
    )


def test_a_second_battle_won_against_a_new_opponent_pays_again(fake_emulator) -> None:
    """The second win in a tier pays its own decayed marginal value
    (weight/sqrt(2)), proving total_battles_won's successor (battle_sum)
    is a real running sum that keeps growing across wins, not a one-shot
    flag consumed by the first win."""
    win_accumulator = RewardAccumulator(EnvConfig(battle_win_weight=0.5))
    win_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    _set_opponent_hp(fake_emulator, current=10, max_hp=10)
    win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=10)
    win_accumulator.step(fake_emulator)  # first win, banked into max_total
    _set_opponent_hp(fake_emulator, current=8, max_hp=8)  # a fresh opponent
    win_accumulator.step(fake_emulator)
    _set_opponent_hp(fake_emulator, current=0, max_hp=8)

    second_win = win_accumulator.step(fake_emulator)

    assert second_win.reward == pytest.approx(0.5 / math.sqrt(2))


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


def test_catching_a_pokemon_pays_the_catch_bonus(fake_emulator) -> None:
    """PARTY_SIZE_ADDR is set to the starting party (1, the starter) BEFORE
    reset() -- exactly like a real init.state -- so reset() baselines
    last_party_size at 1, not FakeEmulator's default 0. Baselining against
    the synthetic 0 would count the starter itself as a catch."""
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    catch_accumulator = RewardAccumulator(EnvConfig(catch_weight=0.1))
    catch_accumulator.reset(fake_emulator)
    catch_accumulator.step(fake_emulator)
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 2

    result = catch_accumulator.step(fake_emulator)

    assert result.components["catch"] == pytest.approx(0.1)


def test_the_catch_bonus_is_capped(fake_emulator) -> None:
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    catch_accumulator = RewardAccumulator(EnvConfig(catch_weight=100.0))
    catch_accumulator.reset(fake_emulator)
    catch_accumulator.step(fake_emulator)
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 2

    result = catch_accumulator.step(fake_emulator)

    assert result.components["catch"] == pytest.approx(0.3)


def test_earning_money_pays_the_money_component(fake_emulator) -> None:
    money_accumulator = RewardAccumulator(EnvConfig(money_weight=1.0))
    money_accumulator.reset(fake_emulator)

    result = money_accumulator.step(fake_emulator)

    assert result.components["money"] == pytest.approx(0.0)


def test_a_money_gain_pays_a_component_normalized_by_max_money(fake_emulator) -> None:
    """Checked against components["money"] directly, not result.reward: any
    weight big enough to make a $50 gain readable on its own would also
    blow straight through the per-step clip(..., 0, 1) every component
    shares -- this is the same normalization aux_state.py already uses."""
    money_accumulator = RewardAccumulator(EnvConfig(money_weight=1.0))
    money_accumulator.reset(fake_emulator)
    money_accumulator.step(fake_emulator)  # banks the starting (zero) money
    fake_emulator.memory[ram.MONEY_ADDRS[2]] = 0x50  # BCD 50

    result = money_accumulator.step(fake_emulator)

    assert result.components["money"] == pytest.approx(50.0 / ram.MAX_MONEY)


def test_spending_money_never_produces_negative_reward(fake_emulator) -> None:
    """total/gain/max_total's own max(0, ...) already protects this -- money
    going down just means gain stays 0, the same as any other component
    that regresses, never a negative contribution."""
    money_accumulator = RewardAccumulator(EnvConfig(money_weight=1.0))
    money_accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.MONEY_ADDRS[2]] = 0x50
    money_accumulator.step(fake_emulator)
    fake_emulator.memory[ram.MONEY_ADDRS[2]] = 0x10  # spent some

    result = money_accumulator.step(fake_emulator)

    assert result.reward == pytest.approx(0.0)


def test_a_badge_earns_exactly_the_clip_cap(fake_emulator, accumulator) -> None:
    """In-battle isolates the badge component from the first-coordinate
    exploration credit that the all-zero fake memory would otherwise fire
    at (0, 0, 0): without it, badge (1.00) + explore (0.30) clip to the same
    1.0 that badge alone should produce, so the test would pass for any
    badge_weight >= 0.70. Asserting clipped is False as well closes the top:
    it fails a badge_weight above 1.00 too, where the pre-clip gain exceeds
    1.0 and clipping (not badge_weight) starts doing the work."""
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0001

    result = accumulator.step(fake_emulator)

    assert (result.reward, result.clipped) == (pytest.approx(1.0), False)


def test_a_step_crossing_several_components_clips_to_exactly_one(
    fake_emulator, accumulator
) -> None:
    """Beating a gym fires badge + several events + level-ups at once, sums
    past 1.0, and the excess is discarded. The clipped flag is what makes the
    clip-fire rate observable."""
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0011
    fake_emulator.memory[ram.PARTY_LEVEL_BASE] = 40

    result = accumulator.step(fake_emulator)

    assert (result.reward, result.clipped) == (pytest.approx(1.0), True)


def test_reward_is_never_negative_when_progress_is_lost(fake_emulator, accumulator) -> None:
    """Section 4's positive-delta rule is about not paying for reversible
    cycles (deposit/withdraw, badge loss), not a blanket ban on any negative
    reward -- idle_penalty_weight (tested separately) is a deliberate,
    bounded exception to that, since idling can never be profitably
    reversed. Losing a badge (an impossible-but-cheap guard) must still
    earn 0 under the monotone-gain formula, not a negative number, with
    idle_penalty_weight at its default of 0."""
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0001
    accumulator.step(fake_emulator)
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0000

    result = accumulator.step(fake_emulator)

    assert result.reward == pytest.approx(0.0)


def test_healing_is_ignored_when_party_size_changed(fake_emulator, accumulator) -> None:
    """HP fraction rises when a healthy Pokemon joins the party. Crediting
    that as healing pays the agent for catching things, twice."""
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 10
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    accumulator.step(fake_emulator)
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 2
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 100

    result = accumulator.step(fake_emulator)

    assert result.components["heal"] == pytest.approx(0.0)


def test_a_single_modest_heal_earns_its_full_uncapped_value(fake_emulator) -> None:
    """A genuine, unremarkable recovery must still be rewarded exactly as
    before -- the cap exists for repeated farming past the ceiling, not for
    an ordinary single heal nowhere near it.

    PARTY_SIZE_ADDR is set to 1 BEFORE reset() -- exactly like a real
    init.state -- since live_party_hp_fraction (unlike the old
    aggregate_hp_fraction) reads only live slots, and FakeEmulator's
    default party_size of 0 would read every slot's HP as 0 regardless of
    what this test writes to PARTY_HP_BASE."""
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    accumulator = RewardAccumulator(EnvConfig())
    accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 0
    # Matches last_hp_fraction=0.0 from reset() exactly (0 == 0, not >), so
    # this step registers no gain -- the heal measured below is isolated to
    # the second step alone.
    accumulator.step(fake_emulator)
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 30

    result = accumulator.step(fake_emulator)

    # heal_weight(0.5) * delta(0.3)^2 = 0.045, well under the 0.2 ceiling.
    assert result.components["heal"] == pytest.approx(0.045)


def test_repeated_deposit_withdraw_healing_cycles_are_capped_not_farmable(
    fake_emulator,
) -> None:
    """The exact exploit this module's docstring says every component
    defends against ('the deposit/withdraw exploit section 4 names') --
    except _update_healing's raw accumulator has no ceiling, so unlike every
    other component, repeated cycles here keep paying forever. A real
    training run hit this: reward/heal grew past every other reward
    component combined over ~9M steps while badges stayed at zero. After
    enough cycles, the heal component must stop growing.

    PARTY_SIZE_ADDR is set to 1 BEFORE reset() -- see
    test_a_single_modest_heal_earns_its_full_uncapped_value's docstring for
    why live_party_hp_fraction needs this where aggregate_hp_fraction did
    not. HP dips to 1, not 0, each cycle, so it never trips blackout
    detection."""
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    accumulator = RewardAccumulator(EnvConfig())
    accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    result = None
    for _ in range(50):
        fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 1
        accumulator.step(fake_emulator)
        fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 100
        result = accumulator.step(fake_emulator)

    assert result is not None  # range(50) always runs at least once
    assert result.components["heal"] == pytest.approx(0.2)


def test_the_heal_contribution_ceiling_does_not_scale_with_heal_weight(
    fake_emulator,
) -> None:
    """The ceiling bounds heal's contribution to `total` directly, not the
    raw total_healing accumulator -- so retuning heal_weight later cannot
    silently raise how much healing can ever be worth relative to a badge.

    PARTY_SIZE_ADDR is set to 1 BEFORE reset() -- see
    test_a_single_modest_heal_earns_its_full_uncapped_value's docstring for
    why live_party_hp_fraction needs this where aggregate_hp_fraction did
    not."""
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    acc = RewardAccumulator(EnvConfig(heal_weight=5.0))
    acc.reset(fake_emulator)
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 100
    result = None
    for _ in range(50):
        fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 1
        acc.step(fake_emulator)
        fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 100
        result = acc.step(fake_emulator)

    assert result is not None  # range(50) always runs at least once
    assert result.components["heal"] == pytest.approx(0.2)


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


def test_base_event_flags_captured_at_reset_earn_nothing(fake_emulator) -> None:
    """init.state already has flags set. Without subtracting the baseline the
    agent is paid on step one for progress it did not make."""
    fake_emulator.memory[ram.EVENT_FLAGS_START] = 0xFF
    accumulator = RewardAccumulator(EnvConfig())
    accumulator.reset(fake_emulator)

    result = accumulator.step(fake_emulator)

    assert result.components["events"] == pytest.approx(0.0)


def test_steps_since_new_coord_counts_up_between_discoveries(
    fake_emulator, accumulator
) -> None:
    _set_coord(fake_emulator, 5, 5, 1)
    accumulator.step(fake_emulator)
    accumulator.step(fake_emulator)
    accumulator.step(fake_emulator)

    assert accumulator.steps_since_new_coord == 2


def test_state_dict_round_trips_the_accumulator(fake_emulator, accumulator) -> None:
    """Resume correctness: a restored accumulator must not re-pay for
    progress already banked."""
    _set_coord(fake_emulator, 5, 5, 1)
    accumulator.step(fake_emulator)
    state = accumulator.state_dict()

    restored = RewardAccumulator(EnvConfig())
    restored.load_state_dict(state)
    _set_coord(fake_emulator, 5, 5, 1)
    replayed = restored.step(fake_emulator)

    assert (replayed.reward, restored.coords_seen) == (pytest.approx(0.0), 1)


def test_coord_keys_returns_every_visited_coordinate_key(fake_emulator) -> None:
    accumulator = RewardAccumulator(EnvConfig())
    accumulator.reset(fake_emulator)
    _set_coord(fake_emulator, 3, 4, 38)

    accumulator.step(fake_emulator)

    assert accumulator.coord_keys() == [ram.coord_key(3, 4, 38)]


def test_reset_persists_exploration_across_an_episode_boundary(
    fake_emulator, accumulator
) -> None:
    """Run 9 (docs/ppo-experiment-history.md) found all 64 envs autoreset in
    perfect lockstep every 163,840 steps -- wiping seen_coords on every one
    of those resets means the decaying explore bonus pays full price again
    for the same starting area every ~160 updates, forever, instead of
    pushing toward genuinely new ground across the run. A coordinate seen in
    the episode before a reset must still count as seen after it -- which
    also proves max_total was correctly rebased to the persisted explore
    contribution, not left at 0: a stale 0 baseline would manufacture a
    reward here that nothing actually earned."""
    _set_coord(fake_emulator, 5, 5, 1)
    accumulator.step(fake_emulator)
    accumulator.reset(fake_emulator)

    revisit = accumulator.step(fake_emulator)  # (5, 5, 1) again, from a fresh episode

    assert revisit.reward == pytest.approx(0.0)


def test_reset_still_pays_full_price_for_ground_new_to_the_whole_run(
    fake_emulator, accumulator
) -> None:
    """The decaying 1/sqrt(k) bonus must keep counting from where the prior
    episode left off, not restart at k=1 -- otherwise every reset cheapens
    what "new" is worth instead of the run converging on real exploration."""
    _set_coord(fake_emulator, 1, 1, 1)
    accumulator.step(fake_emulator)
    _set_coord(fake_emulator, 2, 2, 1)
    accumulator.step(fake_emulator)
    _set_coord(fake_emulator, 3, 3, 1)
    accumulator.step(fake_emulator)
    accumulator.reset(fake_emulator)
    _set_coord(fake_emulator, 4, 4, 1)

    fourth_ever = accumulator.step(fake_emulator)

    assert fourth_ever.reward == pytest.approx(0.30 / math.sqrt(4))


def test_reset_still_clears_badges_and_healing_like_the_real_emulator_reload_does(
    fake_emulator, accumulator
) -> None:
    """Only exploration persists. Badges/HP/party come back from
    EnvSession.reset()'s real emulator.load_state(init_state) at their
    init-state values every episode -- carrying max_total over as well would
    make a fresh episode need to out-earn every badge the last one banked
    just to see reward again."""
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0001
    accumulator.step(fake_emulator)  # episode 1 earns the badge
    accumulator.reset(fake_emulator)
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0000  # the real reload would clear this too
    accumulator.step(fake_emulator)  # baseline step: badges back to 0, matches max_total

    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0001
    fresh_badge = accumulator.step(fake_emulator)  # earning the "same" badge again in episode 2

    assert fresh_badge.reward == pytest.approx(1.00)


def test_a_restored_accumulator_still_pays_for_genuinely_new_ground(
    fake_emulator, accumulator
) -> None:
    """The test above passes even if explore_sum is dropped on restore: a
    zeroed explore_sum still leaves the recomputed total BELOW the restored
    max_total, so the gain is 0 either way and the drop is invisible.

    Stepping onto an unseen coordinate separates them. With explore_sum
    intact the second discovery earns 0.30/sqrt(2); with it dropped the
    recomputed total never climbs back above max_total and the reward is 0 --
    an agent that silently stops being paid for exploring."""
    _set_coord(fake_emulator, 5, 5, 1)
    accumulator.step(fake_emulator)
    restored = RewardAccumulator(EnvConfig())
    restored.load_state_dict(accumulator.state_dict())

    _set_coord(fake_emulator, 6, 6, 1)
    discovered = restored.step(fake_emulator)

    assert discovered.reward == pytest.approx(0.30 / math.sqrt(2))
