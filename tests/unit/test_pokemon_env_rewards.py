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
    """Section 4 forbids penalties. Losing a badge (an impossible-but-cheap
    guard) must earn 0, not a negative number."""
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
