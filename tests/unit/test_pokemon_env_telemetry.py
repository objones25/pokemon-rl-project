import numpy as np
import pytest

from pokemon_env import ram
from pokemon_env.telemetry import contact_sheet, exploration_heatmap, rollout_metrics
from pokemon_env.vec_env import VecStep


def _vec_step(n_envs: int, reward: float = 0.0) -> VecStep:
    """Helper, not a test."""
    return VecStep(
        frames=np.zeros((n_envs, 1, 144, 160), dtype=np.uint8),
        aux=np.zeros((n_envs, 32), dtype=np.float32),
        reward=np.full(n_envs, reward, dtype=np.float32),
        done=np.zeros(n_envs, dtype=bool),
        episode_id=np.zeros(n_envs, dtype=np.int64),
    )


def _empty_stats(n_envs: int) -> list[dict]:
    """Helper, not a test: stats entries with no progress, for tests that only
    care about the reward/env-level fields rollout_metrics reports."""
    return [
        {
            "coord_keys": [],
            "badges": 0,
            "event_flags": 0,
            "step_count": 0,
            "episode_lengths": [],
        }
        for _ in range(n_envs)
    ]


def _vec_step_with_rewards(rewards: np.ndarray) -> VecStep:
    """Helper, not a test. Lets a test give each env a different reward so
    mean/max/sum are distinguishable."""
    n_envs = rewards.shape[0]
    return VecStep(
        frames=np.zeros((n_envs, 1, 144, 160), dtype=np.uint8),
        aux=np.zeros((n_envs, 32), dtype=np.float32),
        reward=rewards.astype(np.float32),
        done=np.zeros(n_envs, dtype=bool),
        episode_id=np.zeros(n_envs, dtype=np.int64),
    )


def test_contact_sheet_tiles_64_frames_into_an_8_by_8_grid() -> None:
    frames = np.zeros((64, 1, 144, 160), dtype=np.uint8)

    sheet = contact_sheet(frames)

    assert sheet.shape == (8 * 144, 8 * 160)


def test_contact_sheet_places_each_frame_at_its_row_major_grid_position() -> None:
    """A transposed tiling (row/column swapped) would put env 1 and env 3 at
    different pixel offsets than row-major placement does, which makes the
    sheet useless for spotting which env is stuck. With only env 0 marked, a
    3x3 grid still puts it top-left under either row-major or transposed
    indexing, so this checks env 1 and env 3 too -- their positions only
    agree with row-major placement."""
    frames = np.zeros((6, 1, 144, 160), dtype=np.uint8)
    frames[0] = 50
    frames[1] = 100
    frames[3] = 200

    sheet = contact_sheet(frames)

    assert int(sheet[0, 0]) == 50
    assert int(sheet[0, 160]) == 100
    assert int(sheet[144, 0]) == 200


def test_contact_sheet_pads_a_non_square_batch() -> None:
    frames = np.zeros((3, 1, 144, 160), dtype=np.uint8)

    sheet = contact_sheet(frames)

    assert sheet.shape == (2 * 144, 2 * 160)


def test_exploration_heatmap_marks_a_visited_coordinate() -> None:
    """Asserting only heatmap.sum() > 0 would pass for any unpacking of the
    coord_key bits, including x and y swapped. Pin down the exact cell that
    map=3, x=10, y=20 must land on given this module's own map-grid layout --
    a single map at a 64x64 image renders one tile at the origin, so the true
    (x, y) lands directly at (row=y, column=x) -- so a wrong shift/mask order
    fails this test instead of passing it."""
    heatmap = exploration_heatmap([ram.coord_key(x=10, y=20, map_id=3)], height=64, width=64)

    assert int(heatmap[20, 10]) == 1
    assert int(heatmap.sum()) == 1


def test_exploration_heatmap_is_empty_with_no_coordinates() -> None:
    heatmap = exploration_heatmap([], height=64, width=64)

    assert int(heatmap.sum()) == 0


def test_rollout_metrics_flattens_components_with_a_prefix() -> None:
    """W&B panels group on the prefix, so an unprefixed 'explore' would sit
    beside unrelated scalars."""
    metrics = rollout_metrics(
        _vec_step(4, reward=0.25),
        components={"explore": 0.3, "badges": 0.0},
        clip_fire_rate=0.0,
        respawns=0,
        stats=_empty_stats(4),
    )

    assert metrics["reward/explore"] == pytest.approx(0.3)


def test_rollout_metrics_reports_mean_reward() -> None:
    """With every env at the same reward, mean, max, and sum would all
    trivially agree (mean == max, and a mean-vs-sum swap could hide behind a
    later normalization). Differing per-env rewards make mean the only
    statistic that lands on this exact value."""
    metrics = rollout_metrics(
        _vec_step_with_rewards(np.array([0.1, 0.2, 0.3, 0.4])),
        components={},
        clip_fire_rate=0.0,
        respawns=0,
        stats=_empty_stats(4),
    )

    assert metrics["reward/mean"] == pytest.approx(0.25)


def test_rollout_metrics_surfaces_the_clip_fire_rate() -> None:
    """Above roughly 0.1% the weights are miscalibrated and achievement
    ordering is being flattened."""
    metrics = rollout_metrics(
        _vec_step(4), components={}, clip_fire_rate=0.02, respawns=0, stats=_empty_stats(4)
    )

    assert metrics["env/clip_fire_rate"] == pytest.approx(0.02)


def test_rollout_metrics_surfaces_worker_respawns() -> None:
    """A rising respawn rate is a leading indicator of memory pressure or a
    bad state, long before it shows in reward."""
    metrics = rollout_metrics(
        _vec_step(4), components={}, clip_fire_rate=0.0, respawns=3, stats=_empty_stats(4)
    )

    assert metrics["env/worker_respawns"] == pytest.approx(3.0)


def test_two_coordinates_one_tile_apart_do_not_collide_in_the_heatmap() -> None:
    """The old projection folded x and y mod 16, so (0, 0) and (0, 16) in the
    same map landed in the same pixel. That collision is why the artifact did
    not show what the env spec promised."""
    heatmap = exploration_heatmap([ram.coord_key(0, 0, 5), ram.coord_key(0, 16, 5)])

    assert int((heatmap > 0).sum()) == 2


def test_the_heatmap_counts_a_repeated_coordinate_once_per_occurrence() -> None:
    key = ram.coord_key(3, 4, 5)

    heatmap = exploration_heatmap([key, key])

    assert int(heatmap.max()) == 2


def test_two_distinct_maps_render_in_separate_tiles() -> None:
    """Map 1 and map 2 each contribute one unique coordinate, so the ranking's
    count-descending-then-map-id-ascending tiebreak puts map 1 at rank
    position 0 (tile origin (0, 0)) and map 2 at position 1. With the default
    256-wide image, maps_per_row = 256 // MAP_TILE(64) = 4, so position 1's
    origin is (row=0, column=64) -- one tile across, not one tile down.
    Position 0 is 0 under both '//' and '%', so only map 2's pixel, at
    column 64 + 5 = 69 rather than row 64 + 5 = 69, can catch those two
    operators being swapped in the origin computation."""
    heatmap = exploration_heatmap([ram.coord_key(5, 5, 1), ram.coord_key(5, 5, 2)])

    assert int(heatmap[5, 5]) == 1
    assert int(heatmap[5, 69]) == 1
    assert int((heatmap > 0).sum()) == 2


def _coord_keys_for_thirteen_maps_with_descending_unique_counts() -> list[int]:
    """Helper, not a test: 13 distinct maps (ids 0..12), where map id N owns
    (13 - N) unique coordinates -- map 0 the most, map 12 (the 13th distinct
    map) the fewest. Ranking by unique-coordinate count then puts map 12 in
    last place, one below MAPS_SHOWN=12."""
    return [
        ram.coord_key(x, 0, map_id)
        for map_id in range(13)
        for x in range(13 - map_id)
    ]


def test_a_map_ranked_below_the_top_twelve_is_dropped_from_the_heatmap() -> None:
    """Map 0 has the most unique coordinates (13), so it ranks first, at tile
    origin (0, 0); its first coordinate (x=0, y=0) renders at heatmap[0, 0].
    Map 12 has the fewest (1) and ranks last (position 12) -- one past
    MAPS_SHOWN=12, so it is dropped entirely. Were it kept, position 12 would
    land at origin_row=(12 // 4) * 64 = 192, origin_column=(12 % 4) * 64 = 0
    (maps_per_row=4 at the default 256-wide image); no map that survives the
    top-12 cut (positions 0..11, origin_row in {0, 64, 128}) ever reaches
    row 192, so that whole tile-sized region staying zero is specific
    evidence map 12 was cut, not incidental non-overlap."""
    heatmap = exploration_heatmap(_coord_keys_for_thirteen_maps_with_descending_unique_counts())

    assert int(heatmap[0, 0]) == 1
    assert int(heatmap[192:256, 0:64].sum()) == 0


def test_rollout_metrics_reports_badges_from_the_env_stats() -> None:
    step = _vec_step(n_envs=2)
    stats = [
        {"coord_keys": [1], "badges": 3, "event_flags": 10, "step_count": 5, "episode_lengths": []},
        {"coord_keys": [2], "badges": 1, "event_flags": 20, "step_count": 7, "episode_lengths": []},
    ]

    metrics = rollout_metrics(step, {}, 0.0, 0, stats)

    assert metrics["progress/badges_max"] == pytest.approx(3.0)


def test_rollout_metrics_counts_unique_coordinates_across_envs_without_double_counting() -> None:
    step = _vec_step(n_envs=2)
    stats = [
        {"coord_keys": [1, 2], "badges": 0, "event_flags": 0, "step_count": 0, "episode_lengths": []},
        {"coord_keys": [2, 3], "badges": 0, "event_flags": 0, "step_count": 0, "episode_lengths": []},
    ]

    metrics = rollout_metrics(step, {}, 0.0, 0, stats)

    assert metrics["explore/unique_coords_total"] == pytest.approx(3.0)


def test_rollout_metrics_reports_mean_completed_episode_length() -> None:
    step = _vec_step(n_envs=2)
    stats = [
        {"coord_keys": [], "badges": 0, "event_flags": 0, "step_count": 0, "episode_lengths": [10]},
        {"coord_keys": [], "badges": 0, "event_flags": 0, "step_count": 0, "episode_lengths": [20, 30]},
    ]

    metrics = rollout_metrics(step, {}, 0.0, 0, stats)

    assert metrics["episode/length_mean"] == pytest.approx(20.0)
