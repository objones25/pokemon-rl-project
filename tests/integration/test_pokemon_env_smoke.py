"""Opt-in integration tests against the real ROM and a real PyBoy.

Run with:
    uv run pytest -m slow tests/integration/test_pokemon_env_smoke.py -v

Auto-skipped when the ROM is absent, so a fresh checkout never fails. The ROM
is gitignored and must never be committed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pokemon_env.emulator import PyBoyEmulator
from pokemon_env.vec_env import VecStep

pytestmark = pytest.mark.slow

_ROM = Path("Pokemon Red.gb")

_needs_rom = pytest.mark.skipif(
    not _ROM.exists(), reason=f"{_ROM} not present; it is gitignored and must be supplied locally"
)


@_needs_rom
def test_save_state_is_small_enough_to_checkpoint_all_64_envs() -> None:
    """THE measurement the design spec makes task one. Measured: one state is
    167,677 B (~164 KiB) -- the earlier ~50 KB estimate was 3.3x low. 64 of
    them is 10.73 MB, 4.0% of the sequence model's 256 MiB KV cache, so
    "save both" holds: the cache-vs-emulator-state decision this measurement
    pivots on. The 256 KiB bound is only ~1.5x the measured size -- it is a
    regression guard against PyBoy's state growing, not a generous margin,
    so re-cost this before adding envs, frame history, or a second emulator
    per env."""
    emulator = PyBoyEmulator(str(_ROM))
    emulator.tick(60, False)

    state = emulator.save_state()
    emulator.close()

    assert len(state) < 256 * 1024


@_needs_rom
def test_screen_frame_matches_the_encoder_input_contract() -> None:
    """(144, 160) uint8. GrayscaleResNetEncoder rejects a transposed
    (N, 1, 160, 144), so getting this backwards is caught -- but only after
    the frame has already been through IPC."""
    emulator = PyBoyEmulator(str(_ROM))
    emulator.tick(60, True)

    frame = emulator.screen_frame()
    emulator.close()

    assert (frame.shape, frame.dtype.name) == ((144, 160), "uint8")


@_needs_rom
def test_screen_frame_returns_a_copy_not_a_live_view() -> None:
    """screen.ndarray references a buffer the next tick overwrites. A view
    would mean every frame in a rollout silently became the newest one."""
    emulator = PyBoyEmulator(str(_ROM))
    emulator.tick(60, True)
    first = emulator.screen_frame()
    before = first.copy()

    emulator.tick(600, True)
    emulator.close()

    assert (first == before).all()


@_needs_rom
def test_generated_init_state_reaches_oaks_lab_with_the_pokedex() -> None:
    """The script's frame counts are guesses until this runs. RAM values
    alone already proved insufficient once: the previous version of this
    test asserted only position/party/money/battle-state and passed even
    though the actual screen was still mid-dialogue ("My name is OAK!
    People call me") -- none of those four values can detect "is a text box
    currently open". Oak's parcel-delivery text ends in another multi-box
    dialogue sequence with the same shape, so this asserts an exact
    screenshot hash in addition to RAM state: PyBoy is deterministic given a
    fixed ROM and a fixed input sequence, so the final frame's exact bytes
    are reproducible run to run on the same ROM revision, the same property
    this test's RAM assertions already rely on.

    This test is deliberately ROM-revision sensitive, same as its
    predecessor: a different ROM (a different release, a hacked ROM, Blue
    instead of Red) would produce different RAM values and a different
    frame, and that SHOULD fail loudly here rather than silently changing
    what all 64 environments load every reset."""
    import hashlib

    from pokemon_env import ram
    from pokemon_env.init_state import INTRO_SCRIPT, generate_init_state

    emulator = PyBoyEmulator(str(_ROM))
    generate_init_state(emulator, INTRO_SCRIPT)
    party_size = ram.party_size(emulator)
    badges = ram.badge_count(emulator)
    oak_parcel = ram.oak_parcel_set(emulator)
    oak_pokedex = ram.oak_pokedex_set(emulator)
    frame_hash = hashlib.sha256(emulator.screen_frame().tobytes()).hexdigest()
    emulator.close()

    assert (party_size, badges, oak_parcel, oak_pokedex) == (1, 0, True, True)
    assert frame_hash == "46e2096b907947368d310929303a04005b39c4a278e3a7de2225c355b4522694"


_needs_init_state = pytest.mark.skipif(
    not Path("artifacts/init.state").exists(),
    reason="artifacts/init.state not generated; see src/pokemon_env/init_state.py",
)


def _seeded_action_generator(seed: int):
    """Helper, not a test: builds the seeded RNG that drives random actions.
    Lives at module level, not in the test body, matching the pattern
    already used for `np.random.default_rng(0)` calls elsewhere in this
    suite (e.g. tests/unit/test_contrastive_pretrain_dataset.py's
    `_grayscale_example`) -- the audit's UNSEEDED_RANDOM check flags any
    `np.random.*` call made directly inside a `test_*` function regardless
    of whether a seed is passed, since this suite has no global RNG-seeding
    conftest fixture."""
    import numpy as np

    return np.random.default_rng(seed)


def _drive_random_steps(vec_env, generator, steps: int) -> VecStep:
    """Helper, not a test: drives `steps` random actions and returns the last
    VecStep. Lives at module level so no loop sits in a test body. Each call
    to `generator.integers(0, 7, size=vec_env.n_envs)` draws one independent
    action per env -- not the same action broadcast to all of them -- which
    is what lets the four envs' game states genuinely diverge.

    The first step runs outside the loop so the return type is VecStep rather
    than `VecStep | None`. Seeding it with None instead would push an Optional
    into every caller, and a caller that silently accepted None would assert
    nothing at all."""
    if steps < 1:
        raise ValueError(f"steps={steps} must be at least 1 to return a VecStep")
    step = vec_env.step(generator.integers(0, 7, size=vec_env.n_envs))
    for _ in range(steps - 1):
        step = vec_env.step(generator.integers(0, 7, size=vec_env.n_envs))
    return step


def _distinct_frame_count(frames) -> int:
    """Module-level, not the test body: number of distinct frames in a
    (N, 1, 144, 160) batch, by hashing each env's raw bytes.

    On its own this is NOT sufficient to catch total collapse (every worker
    writing into the same slot): the unwritten slots keep shared memory's
    zero-initialised bytes, which are identical to each other but distinct
    from the one real frame, so a 4-env total collapse still reports 2. It
    must be paired with `_zero_frame_count` -- see the test below."""
    return len({frames[i, 0].tobytes() for i in range(frames.shape[0])})


def _zero_frame_count(frames) -> int:
    """Module-level, not the test body: number of frames in the batch that
    are entirely zero, i.e. a shared-memory slot no worker ever wrote into.
    Paired with `_distinct_frame_count` to catch total collapse: if every
    worker wrote slot 0, slots 1..3 stay all-zero, `_distinct_frame_count`
    alone reports 2 (one real frame + the shared zero value) and a bare
    `> 1` check would pass despite the catastrophe. Requiring
    `_zero_frame_count == 0` closes that gap."""
    return sum(1 for i in range(frames.shape[0]) if not frames[i, 0].any())


@_needs_rom
@_needs_init_state
def test_a_random_agent_drives_four_real_envs_end_to_end(tmp_path) -> None:
    """The acceptance gate for sub-project A: four real PyBoy processes, real
    frames through shared memory, real RAM reads, real rewards -- driven by a
    random policy, with no PPO anywhere. Four envs rather than 64 so the test
    is minutes, not hours; the vectorization logic is identical.

    Also covers *total* worker-to-slot collapse (every worker writing into
    the same shared-memory slot, or a slot nobody ever wrote into): the
    random policy already sends each env an independent action every step,
    so after enough steps the four envs' frames must be four distinct,
    non-zero values (see _distinct_frame_count / _zero_frame_count).

    This does NOT cover a pure index *swap* between two envs (e.g. env 0's
    worker writes into slot 1 and vice versa): both slots would still hold
    four distinct, non-zero real frames, so nothing at this layer can see a
    swap -- a review caught an earlier version of this test overclaiming
    that coverage. The direct worker-to-slot binding is covered at the unit
    level instead, in
    tests/unit/test_pokemon_env_subprocess_backend.py::test_worker_main_writes_only_into_its_own_slot,
    where an injected FakeEmulator lets worker_main itself run end-to-end --
    including its slot-index selection -- with a frame of known content and
    no spawned process.

    Also writes a contact sheet so a human can look at what the agents saw."""
    import numpy as np

    from pokemon_env.config import EnvConfig
    from pokemon_env.subprocess_backend import build_subprocess_vec_env
    from pokemon_env.telemetry import contact_sheet, rollout_metrics

    config = EnvConfig(n_envs=4, max_steps=64)
    vec_env, buffer = build_subprocess_vec_env(config)
    generator = _seeded_action_generator(0)
    try:
        vec_env.reset()
        step = _drive_random_steps(vec_env, generator, 32)
        metrics = rollout_metrics(
            step, vec_env.last_components, vec_env.clip_fire_rate, 0, vec_env.stats()
        )
        sheet = contact_sheet(step.frames)
        np.save(tmp_path / "contact_sheet.npy", sheet)
    finally:
        vec_env.close()
        buffer.close()
        buffer.unlink()

    assert (step.frames.shape, step.aux.shape) == ((4, 1, 144, 160), (4, 32))
    assert bool(((step.aux >= -1.0) & (step.aux <= 1.0)).all()) is True
    assert metrics["reward/mean"] >= 0.0
    assert (_distinct_frame_count(step.frames), _zero_frame_count(step.frames)) == (4, 0)
