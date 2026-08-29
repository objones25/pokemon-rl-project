import pytest

from pokemon_env import ram


def test_party_level_addresses_are_exactly_44_apart() -> None:
    """The party struct stride. A one-digit hex typo lands on a neighbouring
    field -- level sits immediately before maxHP -- and still reads plausible
    small integers rather than raising, so it presents as a bad reward. These
    six values are the reference implementation's verified list."""
    addresses = [ram.PARTY_LEVEL_BASE + ram.PARTY_STRIDE * i for i in range(ram.PARTY_SLOTS)]

    assert addresses == [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]


def test_party_hp_addresses_match_the_reference_list() -> None:
    addresses = [ram.PARTY_HP_BASE + ram.PARTY_STRIDE * i for i in range(ram.PARTY_SLOTS)]

    assert addresses == [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]


def test_opponent_level_addresses_match_the_reference_list() -> None:
    addresses = [ram.OPPONENT_LEVEL_BASE + ram.PARTY_STRIDE * i for i in range(ram.PARTY_SLOTS)]

    assert addresses == [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]


def test_read_u16_be_is_big_endian(fake_emulator) -> None:
    """Pokemon Red stores HP big-endian. Reading it little-endian turns 258 HP
    into 513 -- plausible, never raises, and quietly corrupts the heal reward."""
    fake_emulator.memory[0xD16C] = 0x01
    fake_emulator.memory[0xD16D] = 0x02

    assert ram.read_u16_be(fake_emulator, 0xD16C) == 258


def test_read_bit_is_lsb_first(fake_emulator) -> None:
    fake_emulator.memory[0xD754] = 0b0000_0101

    assert (ram.read_bit(fake_emulator, 0xD754, 0), ram.read_bit(fake_emulator, 0xD754, 1)) == (True, False)


def test_badge_count_is_a_popcount(fake_emulator) -> None:
    fake_emulator.memory[ram.BADGES_ADDR] = 0b1010_1010

    assert ram.badge_count(fake_emulator) == 4


def test_event_flag_count_spans_2488_flags(fake_emulator) -> None:
    """0xD747..0xD87E exclusive is 311 bytes = 2488 flags."""
    fake_emulator.memory[ram.EVENT_FLAGS_START : ram.EVENT_FLAGS_END] = b"\xff" * (
        ram.EVENT_FLAGS_END - ram.EVENT_FLAGS_START
    )

    assert ram.event_flag_count(fake_emulator) == 2488


def test_event_flag_boundaries_are_the_documented_addresses() -> None:
    """The width test above proves the range is 311 bytes wide, but it derives
    both endpoints from the constants under test -- a shift by the same
    amount in both would still pass. This pins the literal addresses."""
    assert (ram.EVENT_FLAGS_START, ram.EVENT_FLAGS_END) == (0xD747, 0xD87E)


def test_party_size_reads_the_party_count_address(fake_emulator) -> None:
    fake_emulator.memory[0xD163] = 4

    assert ram.party_size(fake_emulator) == 4


def test_party_levels_reads_every_slot_at_its_own_stride(fake_emulator) -> None:
    """Distinct values per slot are the point -- identical values would pass
    even if every slot read the same address."""
    fake_emulator.memory[ram.PARTY_LEVEL_BASE + ram.PARTY_STRIDE * 0] = 10
    fake_emulator.memory[ram.PARTY_LEVEL_BASE + ram.PARTY_STRIDE * 1] = 20
    fake_emulator.memory[ram.PARTY_LEVEL_BASE + ram.PARTY_STRIDE * 2] = 30
    fake_emulator.memory[ram.PARTY_LEVEL_BASE + ram.PARTY_STRIDE * 3] = 40
    fake_emulator.memory[ram.PARTY_LEVEL_BASE + ram.PARTY_STRIDE * 4] = 50
    fake_emulator.memory[ram.PARTY_LEVEL_BASE + ram.PARTY_STRIDE * 5] = 60

    assert ram.party_levels(fake_emulator) == [10, 20, 30, 40, 50, 60]


def test_opponent_levels_reads_every_slot_at_its_own_stride(fake_emulator) -> None:
    fake_emulator.memory[ram.OPPONENT_LEVEL_BASE + ram.PARTY_STRIDE * 0] = 10
    fake_emulator.memory[ram.OPPONENT_LEVEL_BASE + ram.PARTY_STRIDE * 1] = 20
    fake_emulator.memory[ram.OPPONENT_LEVEL_BASE + ram.PARTY_STRIDE * 2] = 30
    fake_emulator.memory[ram.OPPONENT_LEVEL_BASE + ram.PARTY_STRIDE * 3] = 40
    fake_emulator.memory[ram.OPPONENT_LEVEL_BASE + ram.PARTY_STRIDE * 4] = 50
    fake_emulator.memory[ram.OPPONENT_LEVEL_BASE + ram.PARTY_STRIDE * 5] = 60

    assert ram.opponent_levels(fake_emulator) == [10, 20, 30, 40, 50, 60]


def test_party_hp_returns_current_and_max_for_each_slot(fake_emulator) -> None:
    """Both fields are uint16 big-endian, so the low byte lands at addr + 1."""
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 30
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 60
    fake_emulator.memory[ram.PARTY_HP_BASE + ram.PARTY_STRIDE + 1] = 10
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + ram.PARTY_STRIDE + 1] = 40

    assert ram.party_hp(fake_emulator)[:2] == [(30, 60), (10, 40)]


def test_in_battle_is_false_in_the_overworld(fake_emulator) -> None:
    assert ram.in_battle(fake_emulator) is False


def test_in_battle_is_true_when_the_battle_flag_is_set(fake_emulator) -> None:
    fake_emulator.memory[0xD057] = 1

    assert ram.in_battle(fake_emulator) is True


def test_museum_ticket_set_reads_bit_zero_of_its_own_address(fake_emulator) -> None:
    fake_emulator.memory[0xD754] = 0b0000_0001

    assert ram.museum_ticket_set(fake_emulator) is True


def test_oak_parcel_set_reads_bit_one_of_0xd74e(fake_emulator) -> None:
    fake_emulator.memory[0xD74E] = 0b0000_0010

    assert ram.oak_parcel_set(fake_emulator) is True


def test_oak_parcel_set_is_false_when_bit_one_is_clear(fake_emulator) -> None:
    fake_emulator.memory[0xD74E] = 0b0000_0001

    assert ram.oak_parcel_set(fake_emulator) is False


def test_oak_pokedex_set_reads_bit_five_of_0xd74b(fake_emulator) -> None:
    fake_emulator.memory[0xD74B] = 0b0010_0000

    assert ram.oak_pokedex_set(fake_emulator) is True


def test_oak_pokedex_set_is_false_when_bit_five_is_clear(fake_emulator) -> None:
    fake_emulator.memory[0xD74B] = 0b0000_0000

    assert ram.oak_pokedex_set(fake_emulator) is False


def test_read_money_decodes_binary_coded_decimal(fake_emulator) -> None:
    """Three bytes, two decimal digits each. Read as plain hex, 0x12 0x34 0x56
    becomes 1193046 instead of 123456."""
    fake_emulator.memory[0xD347] = 0x12
    fake_emulator.memory[0xD348] = 0x34
    fake_emulator.memory[0xD349] = 0x56

    assert ram.read_money(fake_emulator) == 123456


def test_aggregate_hp_fraction_is_zero_when_max_hp_is_zero(fake_emulator) -> None:
    """All-zero memory is the pre-game state. Dividing by a zero max would
    produce nan, which propagates silently through the whole aux vector."""
    assert ram.aggregate_hp_fraction(fake_emulator) == pytest.approx(0.0)


def test_aggregate_hp_fraction_sums_across_the_party(fake_emulator) -> None:
    fake_emulator.memory[ram.PARTY_HP_BASE + 1] = 30
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + 1] = 60
    fake_emulator.memory[ram.PARTY_HP_BASE + ram.PARTY_STRIDE + 1] = 10
    fake_emulator.memory[ram.PARTY_MAX_HP_BASE + ram.PARTY_STRIDE + 1] = 40

    assert ram.aggregate_hp_fraction(fake_emulator) == pytest.approx(0.4)


def test_game_coords_returns_x_then_y_then_map(fake_emulator) -> None:
    """X is 0xD362 and Y is 0xD361 -- the higher address holds X. Swapping
    them produces a valid-looking coordinate that is simply the wrong tile."""
    fake_emulator.memory[0xD362] = 7
    fake_emulator.memory[0xD361] = 9
    fake_emulator.memory[0xD35E] = 12

    assert ram.game_coords(fake_emulator) == (7, 9, 12)


def test_level_score_subtracts_the_starter_baseline(fake_emulator) -> None:
    """Level 5 starter with min_level 2 and a 4-level starter allowance:
    max(5-2, 0) - 4 = -1, floored to 0. Without the floor the reward opens
    negative and the first level-up earns nothing."""
    fake_emulator.memory[ram.PARTY_LEVEL_BASE] = 5

    assert ram.level_score(fake_emulator) == 0


def test_level_score_counts_gains_above_the_baseline(fake_emulator) -> None:
    fake_emulator.memory[ram.PARTY_LEVEL_BASE] = 10

    assert ram.level_score(fake_emulator) == 4


def test_coord_key_is_injective_across_the_address_space() -> None:
    """Coordinates are packed into one int32 for weights_only-safe
    checkpointing. A collision would make two distinct tiles look like the
    same already-explored one."""
    assert ram.coord_key(255, 255, 0) != ram.coord_key(0, 0, 1)
