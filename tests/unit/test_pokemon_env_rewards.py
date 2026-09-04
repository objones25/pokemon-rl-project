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


def test_a_step_that_finds_no_new_coordinate_pays_the_idle_penalty(fake_emulator) -> None:
    """The 2026-09-04 investigation (docs/ppo-experiment-history.md, run 9)
    found the majority of envs stuck in menus for thousands of steps with
    zero cost. idle_penalty_weight is the fix -- applied outside the
    monotone-gain formula, since that formula floors every step at 0 and
    cannot express a cost (see RewardAccumulator.step)."""
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    _set_coord(fake_emulator, 5, 5, 1)
    idle_accumulator.step(fake_emulator)  # discovers (5, 5, 1)
    _set_coord(fake_emulator, 6, 6, 1)
    idle_accumulator.step(fake_emulator)  # discovers (6, 6, 1)
    _set_coord(fake_emulator, 5, 5, 1)

    revisit = idle_accumulator.step(fake_emulator)  # no new coordinate

    assert revisit.reward == pytest.approx(-0.01)


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
    idle_accumulator.step(fake_emulator)
    _set_coord(fake_emulator, 6, 6, 1)
    idle_accumulator.step(fake_emulator)
    _set_coord(fake_emulator, 5, 5, 1)

    revisit = idle_accumulator.step(fake_emulator)

    assert revisit.components["idle"] == pytest.approx(-0.01)


def test_the_idle_penalty_does_not_suppress_a_genuine_gain_in_the_same_step(
    fake_emulator,
) -> None:
    """A badge earned while not standing on new ground must still pay in
    full -- the idle penalty is a separate, additive term, not something
    that erodes max_total or the gain computation itself."""
    idle_accumulator = RewardAccumulator(EnvConfig(idle_penalty_weight=0.01))
    idle_accumulator.reset(fake_emulator)
    idle_accumulator.step(fake_emulator)  # registers (0, 0, 0) as seen
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0001

    result = idle_accumulator.step(fake_emulator)  # same coordinate again, plus a badge

    assert result.reward == pytest.approx(1.00 - 0.01)


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


def test_a_single_modest_heal_earns_its_full_uncapped_value(
    fake_emulator, accumulator
) -> None:
    """A genuine, unremarkable recovery must still be rewarded exactly as
    before -- the cap exists for repeated farming past the ceiling, not for
    an ordinary single heal nowhere near it."""
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
    fake_emulator, accumulator
) -> None:
    """The exact exploit this module's docstring says every component
    defends against ('the deposit/withdraw exploit section 4 names') --
    except _update_healing's raw accumulator has no ceiling, so unlike every
    other component, repeated cycles here keep paying forever. A real
    training run hit this: reward/heal grew past every other reward
    component combined over ~9M steps while badges stayed at zero. After
    enough cycles, the heal component must stop growing."""
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
    silently raise how much healing can ever be worth relative to a badge."""
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
