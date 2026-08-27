import numpy as np
import pytest

from pokemon_env import ram
from pokemon_env.aux_state import (
    AUX_STATE_DIM,
    AUX_STATE_VERSION,
    RESERVED_SLOT,
    ExplorationCounters,
    build_aux_state,
)

_NO_EXPLORATION = ExplorationCounters(coords_seen=0, steps_since_new_coord=0, maps_visited=0)


def _build(fake_emulator, step_count: int = 0, exploration=_NO_EXPLORATION) -> np.ndarray:
    """Helper, not a test: the common call shape."""
    return build_aux_state(fake_emulator, step_count, exploration, max_steps=163_840)


def test_aux_state_has_the_width_the_policy_config_fixes(fake_emulator) -> None:
    result = _build(fake_emulator)

    assert (result.shape, result.dtype.name) == ((AUX_STATE_DIM,), "float32")


def test_every_slot_is_in_range_for_zeroed_memory(fake_emulator) -> None:
    result = _build(fake_emulator)

    assert bool(((result >= -1.0) & (result <= 1.0)).all()) is True


def test_every_slot_is_in_range_for_all_ones_memory(fake_emulator) -> None:
    """RAM holds out-of-range garbage during transitions -- a level of 255
    mid-write. Unclamped, 2x-1 injects a large outlier into a value network
    the architecture plan calls hypersensitive to input scale."""
    fake_emulator.memory = bytearray(b"\xff" * 0x10000)

    result = _build(fake_emulator)

    assert bool(((result >= -1.0) & (result <= 1.0)).all()) is True


def test_the_reserved_slot_is_exactly_zero_not_the_centered_value(fake_emulator) -> None:
    """A 'constant 0' run through the 2x-1 centering emits -1.0, which is a
    strong constant signal rather than the absence of one."""
    result = _build(fake_emulator)

    assert result[RESERVED_SLOT].item() == pytest.approx(0.0)


def test_badge_slot_is_centered(fake_emulator) -> None:
    """3 badges of 8 -> 0.375 raw -> 2*0.375-1 = -0.25 after centering."""
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0111

    result = _build(fake_emulator)

    assert result[13].item() == pytest.approx(-0.25)


def test_in_battle_slot_is_plus_one_when_in_battle(fake_emulator) -> None:
    fake_emulator.memory[ram.IN_BATTLE_ADDR] = 1

    result = _build(fake_emulator)

    assert result[18].item() == pytest.approx(1.0)


def test_episode_progress_slot_tracks_step_count(fake_emulator) -> None:
    """Quarter-way through the episode: 0.25 raw -> -0.5 centered.

    Deliberately not the half-way point: at step_count/max_steps == 0.5, the
    correct "fraction elapsed" formula and a reversed "fraction remaining"
    bug (1 - x) both center to 0.0 -- a fixed point that can't tell the two
    formulas apart. The quarter point gives -0.5 for the correct formula and
    +0.5 for the reversed one, so the test actually discriminates."""
    result = _build(fake_emulator, step_count=40_960)

    assert result[28].item() == pytest.approx(-0.5)


def test_aux_state_version_is_recorded_for_checkpoint_validation() -> None:
    """A policy trained against layout v1 and fed v2 data is silently wrong in
    exactly the way a PolicyConfig mismatch is -- no crash, no shape error."""
    assert AUX_STATE_VERSION == 1


def _set_party(emulator, size: int, levels: list[int]) -> None:
    """Helper, not a test: writes a party of `size` members while leaving
    higher slots holding whatever `levels` says -- which is how Pokemon Red
    actually behaves after a deposit, since it does not clear the vacated
    slot."""
    emulator.memory[ram.PARTY_SIZE_ADDR] = size
    for i, level in enumerate(levels):
        emulator.memory[ram.PARTY_LEVEL_BASE + ram.PARTY_STRIDE * i] = level


def test_level_slots_past_the_party_size_are_zero(fake_emulator) -> None:
    """Pokemon Red does not clear a slot when a Pokemon leaves the party, so
    after depositing five the struct still holds their levels. Reading all six
    unconditionally shows the policy a party the agent does not have."""
    _set_party(fake_emulator, size=1, levels=[20, 20, 20, 20, 20, 20])

    result = _build(fake_emulator)

    assert result[2:7].tolist() == [-1.0, -1.0, -1.0, -1.0, -1.0]


def test_the_live_level_slot_still_reports_its_pokemon(fake_emulator) -> None:
    """The masking must not blank the slots that ARE live: level 20 of 100 is
    0.2 raw, which centers to -0.6."""
    _set_party(fake_emulator, size=1, levels=[20, 20, 20, 20, 20, 20])

    result = _build(fake_emulator)

    assert result[1].item() == pytest.approx(-0.6)


def test_hp_slots_past_the_party_size_are_zero(fake_emulator) -> None:
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 30
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 60
    fake_emulator.memory[ram.PARTY_HP_BASE + ram.PARTY_STRIDE + 1] = 40
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + ram.PARTY_STRIDE + 1] = 40

    result = _build(fake_emulator)

    assert (result[7].item(), result[8].item()) == (pytest.approx(0.0), pytest.approx(-1.0))


def test_aggregate_hp_ignores_slots_past_the_party_size(fake_emulator) -> None:
    """Slot 26. A stale full-health Pokemon in an inactive slot would otherwise
    mask the live one being nearly dead -- 30/60 = 0.5 diluted to 70/100 = 0.7."""
    fake_emulator.memory[ram.PARTY_SIZE_ADDR] = 1
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 30
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 60
    fake_emulator.memory[ram.PARTY_HP_BASE + ram.PARTY_STRIDE + 1] = 40
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + ram.PARTY_STRIDE + 1] = 40

    result = _build(fake_emulator)

    assert result[26].item() == pytest.approx(0.0)


def test_a_garbage_party_size_cannot_read_past_the_party_struct(fake_emulator) -> None:
    """RAM holds out-of-range values mid-write. A party_size of 255 must clamp
    to the six real slots rather than slicing beyond them."""
    _set_party(fake_emulator, size=255, levels=[20, 20, 20, 20, 20, 20])

    result = _build(fake_emulator)

    assert result[1:7].tolist() == pytest.approx([-0.6] * 6)
