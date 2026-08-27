# Pokemon Red Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 64-way vectorized Pokemon Red environment that emits
`[frame, aux_state, reward, done, episode_id]` and a frozen-encoder inference
path, verifiable end-to-end with a random agent before any PPO code exists.

**Architecture:** Subprocess workers each own one PyBoy emulator behind an
`Emulator` Protocol; raw frames cross into the parent through a single
`SharedMemory` block and are encoded in one batched GPU call. All game-specific
logic (RAM reads, the 32-d aux vector, the reward accumulator) is pure functions
over that Protocol, so it is unit-testable with no ROM and no PyBoy.

**Tech Stack:** Python 3.12, `uv`, PyBoy 2.x, PyTorch 2.13, numpy, W&B,
`multiprocessing` (spawn + `SharedMemory`), pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-08-26-pokemon-env-design.md`

## Global Constraints

- **No `gymnasium` and no `stable-baselines3`.** We write our own vectorization; gymnasium would supply `spaces.Discrete(7)` as documentation and an API we do not conform to.
- **Package management:** `uv` only. `uv add <pkg>`, `uv run <cmd>`. Never bare `pip`.
- **Target hardware is CUDA on RunPod.** MPS/CPU exist only to run tests locally. Never degrade the CUDA path for the dev machine.
- **Device:** `torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")`.
- **The ROM `Pokemon Red.gb` is gitignored and must never be committed.** Neither may any `.state` file — `artifacts/` is gitignored.
- **Frame contract:** `(N, 1, 144, 160)` uint8 in `[0, 255]`, cast to float but **never rescaled to `[0,1]`** — the encoder has Conv+BN fused, so no BatchNorm remains to absorb a different input scale and the features would be wrong with no error raised.
- **`AUX_STATE_DIM = 32`** — fixed by the merged `PolicyConfig.aux_state_dim`. Changing it changes the model.
- **`aux` components land in `[-1, 1]`; `reward` lands in `[0, 1]`.**
- **Encoder inference is `@torch.no_grad()`, never `@torch.inference_mode()`** — its output enters autograd at the PPO update.
- **Testing gates:** one behavior per test; no `if`/`for`/`while` in a test body; every test asserts an exact expected value; floats via `pytest.approx`; `pytest.raises` always names a specific exception and passes `match=`; `skip`/`xfail` carry `reason=`. Writes go to `tmp_path`. Anything needing the real ROM or PyBoy carries `@pytest.mark.slow`.
- **Prove each new test can fail.** Break the code it covers, confirm red, revert, and say which test you verified this way.
- **Branch coverage floor is 93** (`--cov-fail-under=93` in `pyproject.toml`). Do not lower it.
- After adding tests, run `uv run python ~/.claude/skills/pytest-expert/scripts/audit_tests.py tests/`. It currently reports **11 pre-existing findings** (`BRANCHING=6`, `UNSEEDED_RANDOM=5`), all triaged as tool false positives. Your work must not add to that count.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/pokemon_env/__init__.py` | package marker |
| `src/pokemon_env/config.py` | `EnvConfig` dataclass + YAML loader |
| `src/pokemon_env/emulator.py` | `Emulator` Protocol + `PyBoyEmulator` adapter |
| `src/pokemon_env/ram.py` | address constants + typed readers |
| `src/pokemon_env/aux_state.py` | the 32-d vector + `AUX_STATE_VERSION` |
| `src/pokemon_env/rewards.py` | reward components + max-historical accumulator |
| `src/pokemon_env/session.py` | one env's step/reset logic over one `Emulator` |
| `src/pokemon_env/vec_env.py` | `VecPokemonEnv`, `EnvBackend` Protocol, in-process backend |
| `src/pokemon_env/subprocess_backend.py` | spawn, `SharedMemory`, timeout, respawn |
| `src/pokemon_env/encoder.py` | frozen CNN inference + `latent_stats.json` |
| `src/pokemon_env/init_state.py` | button-script replay producing `init.state` |
| `src/pokemon_env/telemetry.py` | reward decomposition, contact sheet, exploration heatmap |
| `configs/pokemon_env.yaml` | the config values |
| `tests/unit/test_pokemon_env_*.py` | one test module per source module |
| `tests/integration/test_pokemon_env_smoke.py` | the one `@pytest.mark.slow` real-ROM test |

---

## Task 1: Dependency, config, Emulator Protocol, and the save-state measurement

The spec makes measuring the real `save_state` size task one, because the
cache-vs-emulator-state decision pivots on it.

**Files:**
- Create: `src/pokemon_env/__init__.py`, `src/pokemon_env/config.py`, `src/pokemon_env/emulator.py`, `configs/pokemon_env.yaml`
- Modify: `pyproject.toml` (add `src/pokemon_env` to `[tool.hatch.build.targets.wheel] packages`)
- Test: `tests/unit/test_pokemon_env_config.py`, `tests/unit/conftest.py` (add `FakeEmulator`), `tests/integration/test_pokemon_env_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EnvConfig` (frozen dataclass, fields below), `load_config(path) -> EnvConfig`, `Emulator` Protocol, `PyBoyEmulator`, and the `FakeEmulator` fixture every later task's tests use.

- [ ] **Step 1: Add the dependency and register the package**

```bash
uv add pyboy
```

In `pyproject.toml`, extend the packages list:

```toml
packages = ["src/data_collection", "src/observability", "src/contrastive_pretrain", "src/hf_storage", "src/sequence_model", "src/checkpointing", "src/pokemon_env"]
```

- [ ] **Step 2: Write the failing config tests**

`tests/unit/test_pokemon_env_config.py`:

```python
import pytest

from pokemon_env.config import EnvConfig, load_config


def test_load_config_reads_yaml_values(tmp_path) -> None:
    path = tmp_path / "env.yaml"
    path.write_text("n_envs: 8\naction_freq: 16\npress_frames: 4\n")

    config = load_config(path)

    assert (config.n_envs, config.action_freq, config.press_frames) == (8, 16, 4)


def test_load_config_rejects_an_unknown_field(tmp_path) -> None:
    """A typo'd key would otherwise be silently ignored and the run would use
    the default, which is indistinguishable from the setting having no effect."""
    path = tmp_path / "env.yaml"
    path.write_text("n_env: 8\n")

    with pytest.raises(ValueError, match=r"unknown config field\(s\): \['n_env'\]"):
        load_config(path)


def test_config_rejects_press_frames_that_leave_no_release_window() -> None:
    """The step is press -> tick(press_frames) -> release -> tick(action_freq -
    press_frames - 1) -> tick(1). With press_frames = action_freq - 1 the
    release tick count is 0, so the button is never released and every
    subsequent action is entered with it still held."""
    with pytest.raises(ValueError, match="press_frames=23 leaves 0 frames"):
        EnvConfig(action_freq=24, press_frames=23)


def test_config_defaults_match_the_reference_implementation() -> None:
    config = EnvConfig()

    assert (config.n_envs, config.action_freq, config.max_steps) == (64, 24, 163_840)
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_config.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env'`

- [ ] **Step 4: Implement `config.py`**

```python
"""Environment configuration, loaded from configs/pokemon_env.yaml.
Mirrors contrastive_pretrain.config's dataclass + yaml.safe_load pattern.

Reward weights are initial guesses from the design spec, chosen so a normal
step's reward lands well inside [0, 1] and the clip fires only on genuine
outliers. They are the parameters most likely to need tuning against the
first run's reward histogram."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EnvConfig:
    rom_path: str = "Pokemon Red.gb"
    init_state_path: str = "artifacts/init.state"
    frozen_encoder_repo_id: str = "objones25/pokemon-contrastive-encoder"
    n_envs: int = 64
    action_freq: int = 24
    press_frames: int = 8
    max_steps: int = 163_840
    worker_timeout_s: float = 60.0
    badge_weight: float = 1.00
    heal_weight: float = 0.50
    explore_weight: float = 0.30
    event_weight: float = 0.10
    level_weight: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        release_frames = self.action_freq - self.press_frames - 1
        if release_frames < 1:
            raise ValueError(
                f"press_frames={self.press_frames} leaves {release_frames} frames for the "
                f"release window at action_freq={self.action_freq}; the button would never "
                "be released and every later action would be entered with it still held"
            )
        if self.press_frames < 1:
            raise ValueError(f"press_frames={self.press_frames} must be at least 1")
        if self.n_envs < 1:
            raise ValueError(f"n_envs={self.n_envs} must be at least 1")

    @property
    def release_frames(self) -> int:
        """Frames to tick after releasing the button, before the final rendered
        frame. press + release + 1 == action_freq."""
        return self.action_freq - self.press_frames - 1


def load_config(path: str | Path) -> EnvConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    valid_fields = {f.name for f in fields(EnvConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")
    return EnvConfig(**data)
```

`configs/pokemon_env.yaml`:

```yaml
# Defaults live in EnvConfig; this file records the values this project runs
# with. action_freq, max_steps and n_envs match PWhiddy/PokemonRedExperiments
# v2 and the sequence-model spec's bound interface.
n_envs: 64
action_freq: 24
press_frames: 8
max_steps: 163840
badge_weight: 1.0
heal_weight: 0.5
explore_weight: 0.3
event_weight: 0.1
level_weight: 0.05
```

`src/pokemon_env/__init__.py`:

```python
"""Pokemon Red environment: PyBoy wrapper, RAM-derived observations, the
section-4 reward system, and 64-way vectorization.

Sub-project A of two. The PPO trainer that consumes this is sub-project B.
See docs/superpowers/specs/2026-08-26-pokemon-env-design.md.
"""
```

- [ ] **Step 5: Run to verify the config tests pass**

Run: `uv run pytest tests/unit/test_pokemon_env_config.py -q --no-cov`
Expected: PASS (4 tests)

- [ ] **Step 6: Implement `emulator.py`**

```python
"""The boundary that makes everything else testable.

PyBoy needs a real commercial ROM, which is gitignored and can never exist in
CI. Behind this Protocol, a FakeEmulator over a synthetic 64 KB bytearray
covers ram.py, aux_state.py, rewards.py and session.py -- where essentially
every game-specific bug will live -- with no ROM and no PyBoy.

Verified against PyBoy 2.x: window='null' (not 'headless' or 'dummy', both
removed in 2.0.0); screen.ndarray is (144, 160, 4) uint8 RGBA; tick(count,
render) renders only the LAST frame of the tick; button_press/button_release
take lowercase strings; load_state needs seek(0) first."""

from __future__ import annotations

import io
from typing import Protocol

import numpy as np

SCREEN_HEIGHT = 144
SCREEN_WIDTH = 160


class Emulator(Protocol):
    def tick(self, count: int, render: bool) -> bool: ...
    def button_press(self, button: str) -> None: ...
    def button_release(self, button: str) -> None: ...
    def read_memory(self, addr: int) -> int: ...
    def screen_frame(self) -> np.ndarray: ...
    def save_state(self) -> bytes: ...
    def load_state(self, state: bytes) -> None: ...
    def close(self) -> None: ...


class PyBoyEmulator:
    """Adapter over the real thing. Constructed only inside a worker process."""

    def __init__(self, rom_path: str) -> None:
        from pyboy import PyBoy

        self._pyboy = PyBoy(
            rom_path,
            window="null",
            sound_emulated=False,
            log_level="ERROR",
        )

    def tick(self, count: int, render: bool) -> bool:
        return self._pyboy.tick(count, render)

    def button_press(self, button: str) -> None:
        self._pyboy.button_press(button)

    def button_release(self, button: str) -> None:
        self._pyboy.button_release(button)

    def read_memory(self, addr: int) -> int:
        return self._pyboy.memory[addr]

    def screen_frame(self) -> np.ndarray:
        """(144, 160) uint8. The Game Boy is monochrome so channel 0 of the
        RGBA buffer is the grayscale image.

        ascontiguousarray is load-bearing twice: the channel slice is a
        non-contiguous view, and screen.ndarray references a live backing
        buffer that the next tick overwrites. Returning the view would hand
        callers a frame that silently changes underneath them."""
        return np.ascontiguousarray(self._pyboy.screen.ndarray[:, :, 0])

    def save_state(self) -> bytes:
        buffer = io.BytesIO()
        self._pyboy.save_state(buffer)
        return buffer.getvalue()

    def load_state(self, state: bytes) -> None:
        buffer = io.BytesIO(state)
        buffer.seek(0)  # PyBoy reads from the current position, not from 0
        self._pyboy.load_state(buffer)

    def close(self) -> None:
        self._pyboy.stop(save=False)
```

- [ ] **Step 7: Add `FakeEmulator` in `tests/unit/fakes.py` and a fixture for it**

It lives in its own module, not in `conftest.py`, because later tasks
construct several of them directly (one per env) and importing a `conftest`
module by name is fragile — pytest owns how conftest files are loaded.

Create `tests/unit/fakes.py`:

```python
"""Hand-written test doubles, importable by any test module.

Separate from conftest.py so tests can construct these directly rather than
only receiving them as fixtures -- the vectorized env tests need one fake per
env, which a single fixture cannot supply."""

from __future__ import annotations

import numpy as np


class FakeEmulator:
    """Hand-written fake typed against the Emulator Protocol, per CLAUDE.md's
    preference for fakes over mock.patch. Records every call so tests can
    assert the exact button/tick sequence."""

    def __init__(self, memory: bytearray | None = None, frame: np.ndarray | None = None) -> None:
        self.memory = bytearray(0x10000) if memory is None else memory
        self.frame = np.zeros((144, 160), dtype=np.uint8) if frame is None else frame
        self.calls: list[tuple] = []
        self.state = b"fake-emulator-state"
        self.closed = False

    def tick(self, count: int, render: bool) -> bool:
        self.calls.append(("tick", count, render))
        return True

    def button_press(self, button: str) -> None:
        self.calls.append(("press", button))

    def button_release(self, button: str) -> None:
        self.calls.append(("release", button))

    def read_memory(self, addr: int) -> int:
        return self.memory[addr]

    def screen_frame(self) -> np.ndarray:
        return self.frame.copy()

    def save_state(self) -> bytes:
        return self.state

    def load_state(self, state: bytes) -> None:
        self.state = state
        self.calls.append(("load_state", len(state)))

    def close(self) -> None:
        self.closed = True
```

Then append the fixture to `tests/unit/conftest.py`:

```python
from .fakes import FakeEmulator


@pytest.fixture
def fake_emulator() -> FakeEmulator:
    return FakeEmulator()
```

- [ ] **Step 8: Write the slow save-state measurement test**

`tests/integration/test_pokemon_env_smoke.py`:

```python
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

pytestmark = pytest.mark.slow

_ROM = Path("Pokemon Red.gb")

_needs_rom = pytest.mark.skipif(
    not _ROM.exists(), reason=f"{_ROM} not present; it is gitignored and must be supplied locally"
)


@_needs_rom
def test_save_state_is_small_enough_to_checkpoint_all_64_envs() -> None:
    """THE measurement the design spec makes task one. The whole
    cache-vs-emulator-state decision pivots on it: if 64 emulator states are
    negligible beside the sequence model's 256 MiB KV cache, both get saved
    together. The 256 KiB bound is 5x the ~50 KB estimate -- generous enough
    not to be brittle, tight enough that a wildly larger state fails loudly."""
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
```

- [ ] **Step 9: Run the slow tests and record the measurement**

Run: `uv run pytest -m slow tests/integration/test_pokemon_env_smoke.py -v`
Expected: 3 PASS (or 3 SKIP if the ROM is absent — in which case say so
explicitly in the task report; the measurement is a required deliverable).

Print the actual byte count and record it in the task report. If it exceeds
256 KiB, stop and report — the spec's "save both" conclusion depends on this
number and must be revisited, not silently overridden.

- [ ] **Step 10: Run the full suite and commit**

```bash
uv run pytest -q
git add -A
git commit -m "feat(env): config, Emulator Protocol, and the save-state measurement

Measures the real PyBoy save_state size, which the design spec makes task
one because the cache-vs-emulator-state decision pivots on it.

The Emulator Protocol is the boundary that makes every later task testable
without a ROM: PyBoy needs a real commercial cartridge image that is
gitignored and can never exist in CI."
```

---

## Task 2: RAM readers

**Files:**
- Create: `src/pokemon_env/ram.py`
- Test: `tests/unit/test_pokemon_env_ram.py`

**Interfaces:**
- Consumes: `Emulator` Protocol from Task 1 (only `read_memory`), `FakeEmulator` fixture.
- Produces: `PARTY_STRIDE`, `PARTY_SLOTS`, all address constants, and readers `read_u16_be(mem, addr) -> int`, `read_bit(mem, addr, bit) -> bool`, `party_size(mem) -> int`, `party_levels(mem) -> list[int]`, `party_hp(mem) -> list[tuple[int, int]]`, `aggregate_hp_fraction(mem) -> float`, `opponent_levels(mem) -> list[int]`, `badge_count(mem) -> int`, `event_flag_count(mem) -> int`, `museum_ticket_set(mem) -> bool`, `game_coords(mem) -> tuple[int, int, int]`, `in_battle(mem) -> bool`, `read_money(mem) -> int`, `level_score(mem) -> int`, `coord_key(x, y, map_id) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_pokemon_env_ram.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_ram.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env.ram'`

- [ ] **Step 3: Implement `ram.py`**

```python
"""Typed readers over Pokemon Red/Blue's RAM map.

Addresses and decoding are read from PWhiddy/PokemonRedExperiments' verified
readers, not inferred from constant names. Source of truth:
https://datacrystal.romhacking.net/wiki/Pokémon_Red/Blue:RAM_map

Everything here is a pure function over the Emulator Protocol's read_memory,
so all of it is testable against a synthetic bytearray."""

from __future__ import annotations

from pokemon_env.emulator import Emulator

PARTY_SLOTS = 6
PARTY_STRIDE = 44  # 0x2C

PARTY_SIZE_ADDR = 0xD163
PARTY_LEVEL_BASE = 0xD18C
PARTY_HP_BASE = 0xD16C
PARTY_MAX_HP_BASE = 0xD18D
OPPONENT_LEVEL_BASE = 0xD8C5

BADGES_ADDR = 0xD356
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E  # exclusive -- 311 bytes = 2488 flags
EVENT_FLAG_COUNT = (EVENT_FLAGS_END - EVENT_FLAGS_START) * 8
MUSEUM_TICKET_ADDR = 0xD754
MUSEUM_TICKET_BIT = 0

MAP_ID_ADDR = 0xD35E
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
IN_BATTLE_ADDR = 0xD057
MONEY_ADDRS = (0xD347, 0xD348, 0xD349)

MAX_MONEY = 999_999
MAX_BADGES = 8
MAX_LEVEL = 100
# A starter arrives at level 5; the reward should measure growth past that,
# not hand out 5 levels' worth of credit on the first step.
MIN_POKEMON_LEVEL = 2
STARTER_LEVEL_ALLOWANCE = 4


def read_u16_be(mem: Emulator, addr: int) -> int:
    """Pokemon Red stores 16-bit quantities big-endian."""
    return 256 * mem.read_memory(addr) + mem.read_memory(addr + 1)


def read_bit(mem: Emulator, addr: int, bit: int) -> bool:
    return bool((mem.read_memory(addr) >> bit) & 1)


def party_size(mem: Emulator) -> int:
    return mem.read_memory(PARTY_SIZE_ADDR)


def party_levels(mem: Emulator) -> list[int]:
    return [mem.read_memory(PARTY_LEVEL_BASE + PARTY_STRIDE * i) for i in range(PARTY_SLOTS)]


def opponent_levels(mem: Emulator) -> list[int]:
    return [mem.read_memory(OPPONENT_LEVEL_BASE + PARTY_STRIDE * i) for i in range(PARTY_SLOTS)]


def party_hp(mem: Emulator) -> list[tuple[int, int]]:
    """(current, max) per slot, both uint16 big-endian."""
    return [
        (
            read_u16_be(mem, PARTY_HP_BASE + PARTY_STRIDE * i),
            read_u16_be(mem, PARTY_MAX_HP_BASE + PARTY_STRIDE * i),
        )
        for i in range(PARTY_SLOTS)
    ]


def aggregate_hp_fraction(mem: Emulator) -> float:
    """Party-wide health in [0, 1]. Returns 0.0 rather than nan when total max
    HP is zero -- the pre-game state, where a nan would propagate silently
    through the entire aux vector."""
    slots = party_hp(mem)
    total_max = sum(maximum for _, maximum in slots)
    if total_max == 0:
        return 0.0
    return sum(current for current, _ in slots) / total_max


def badge_count(mem: Emulator) -> int:
    return bin(mem.read_memory(BADGES_ADDR)).count("1")


def event_flag_count(mem: Emulator) -> int:
    return sum(
        bin(mem.read_memory(addr)).count("1")
        for addr in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
    )


def museum_ticket_set(mem: Emulator) -> bool:
    return read_bit(mem, MUSEUM_TICKET_ADDR, MUSEUM_TICKET_BIT)


def game_coords(mem: Emulator) -> tuple[int, int, int]:
    """(x, y, map_id). X lives at the HIGHER address of the pair."""
    return (
        mem.read_memory(X_POS_ADDR),
        mem.read_memory(Y_POS_ADDR),
        mem.read_memory(MAP_ID_ADDR),
    )


def in_battle(mem: Emulator) -> bool:
    return mem.read_memory(IN_BATTLE_ADDR) != 0


def read_money(mem: Emulator) -> int:
    """Three bytes of binary-coded decimal, two digits each. Reading them as
    plain hex turns 123456 into 1193046."""
    total = 0
    for addr in MONEY_ADDRS:
        byte = mem.read_memory(addr)
        total = total * 100 + (byte >> 4) * 10 + (byte & 0x0F)
    return total


def level_score(mem: Emulator) -> int:
    """Total party levels above the starting baseline, floored at 0."""
    gained = sum(max(level - MIN_POKEMON_LEVEL, 0) for level in party_levels(mem))
    return max(gained - STARTER_LEVEL_ALLOWANCE, 0)


def coord_key(x: int, y: int, map_id: int) -> int:
    """Packs a coordinate into one int so the seen-set can be checkpointed as
    an int32 tensor -- torch.load(weights_only=True) will not restore a
    tuple-keyed dict. Each field is a uint8, so the packing is injective."""
    return (map_id << 16) | (x << 8) | y
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_pokemon_env_ram.py -q --no-cov`
Expected: PASS (14 tests)

- [ ] **Step 5: Prove a test can fail, then commit**

Change `read_u16_be` to `mem.read_memory(addr) + 256 * mem.read_memory(addr + 1)`,
confirm `test_read_u16_be_is_big_endian` goes red, revert. Name the test in
your report.

```bash
git add -A
git commit -m "feat(env): RAM readers with a stride guard

Addresses read from the reference implementation's verified readers rather
than inferred from constant names. The 44-byte stride test exists because
level sits immediately before maxHP, so a one-digit typo reads a neighbouring
field and returns plausible small integers instead of raising."
```

---

## Task 3: The 32-d aux state vector

**Files:**
- Create: `src/pokemon_env/aux_state.py`
- Test: `tests/unit/test_pokemon_env_aux_state.py`

**Interfaces:**
- Consumes: `ram` module from Task 2, `EnvConfig` from Task 1.
- Produces: `AUX_STATE_VERSION: int = 1`, `AUX_STATE_DIM: int = 32`, `RESERVED_SLOT: int = 31`, `ExplorationCounters` (frozen dataclass: `coords_seen: int`, `steps_since_new_coord: int`, `maps_visited: int`), and `build_aux_state(mem, step_count, exploration, max_steps) -> np.ndarray` returning `(32,)` float32.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_pokemon_env_aux_state.py`:

```python
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
    """Half-way through the episode: 0.5 raw -> 0.0 centered."""
    result = _build(fake_emulator, step_count=81_920)

    assert result[28].item() == pytest.approx(0.0)


def test_aux_state_version_is_recorded_for_checkpoint_validation() -> None:
    """A policy trained against layout v1 and fed v2 data is silently wrong in
    exactly the way a PolicyConfig mismatch is -- no crash, no shape error."""
    assert AUX_STATE_VERSION == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_aux_state.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env.aux_state'`

- [ ] **Step 3: Implement `aux_state.py`**

```python
"""The 32-d RAM-derived state vector that rides alongside each frame.

Width is fixed at 32 by the merged PolicyConfig.aux_state_dim -- changing it
changes the model, so the layout is versioned instead. Thirty real signals,
one reserved.

Every slot is normalized to [0, 1], clamped, then mapped 2x-1 into [-1, 1].
The centering is interface fit against the merged InputAdapter, whose proj is
nn.Linear(..., bias=False): a block of inputs with mean 0.5 becomes a fixed
offset vector the model must absorb with real capacity."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pokemon_env import ram
from pokemon_env.emulator import Emulator

AUX_STATE_VERSION = 1
AUX_STATE_DIM = 32
RESERVED_SLOT = 31

_COORD_SATURATION = 20_000
_STUCK_SATURATION = 1_000
_MAX_MAPS = 255


@dataclass(frozen=True)
class ExplorationCounters:
    """Env-side counters the RAM cannot supply."""

    coords_seen: int
    steps_since_new_coord: int
    maps_visited: int


def build_aux_state(
    mem: Emulator,
    step_count: int,
    exploration: ExplorationCounters,
    max_steps: int,
) -> np.ndarray:
    """(32,) float32 in [-1, 1]. See the design spec's slot table."""
    raw = np.zeros(AUX_STATE_DIM, dtype=np.float32)

    raw[0] = ram.party_size(mem) / ram.PARTY_SLOTS
    raw[1:7] = [level / ram.MAX_LEVEL for level in ram.party_levels(mem)]
    raw[7:13] = [
        (current / maximum) if maximum > 0 else 0.0 for current, maximum in ram.party_hp(mem)
    ]
    raw[13] = ram.badge_count(mem) / ram.MAX_BADGES
    raw[14] = ram.event_flag_count(mem) / ram.EVENT_FLAG_COUNT

    x, y, map_id = ram.game_coords(mem)
    raw[15] = map_id / 255.0
    raw[16] = x / 255.0
    raw[17] = y / 255.0
    raw[18] = 1.0 if ram.in_battle(mem) else 0.0
    raw[19] = ram.read_money(mem) / ram.MAX_MONEY
    raw[20:26] = [level / ram.MAX_LEVEL for level in ram.opponent_levels(mem)]
    raw[26] = ram.aggregate_hp_fraction(mem)

    raw[27] = math.log1p(exploration.coords_seen) / math.log1p(_COORD_SATURATION)
    raw[28] = step_count / max_steps
    raw[29] = min(exploration.steps_since_new_coord, _STUCK_SATURATION) / _STUCK_SATURATION
    raw[30] = exploration.maps_visited / _MAX_MAPS

    # Clamp BEFORE centering: RAM holds out-of-range values mid-write (a level
    # of 255), and an unclamped 2x-1 would inject a large outlier into a value
    # head the architecture plan warns is hypersensitive to input scale.
    np.clip(raw, 0.0, 1.0, out=raw)
    centered = raw * 2.0 - 1.0

    # The reserved slot is written as literal 0.0, not run through the
    # centering -- a "constant 0" centered becomes -1.0, which is a strong
    # constant signal rather than the absence of one.
    centered[RESERVED_SLOT] = 0.0
    return centered
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_pokemon_env_aux_state.py -q --no-cov`
Expected: PASS (8 tests)

- [ ] **Step 5: Prove a test can fail, then commit**

Delete the `centered[RESERVED_SLOT] = 0.0` line, confirm
`test_the_reserved_slot_is_exactly_zero_not_the_centered_value` goes red,
revert. Name it in your report.

```bash
git add -A
git commit -m "feat(env): the 32-d versioned aux state vector

Width is fixed by the merged PolicyConfig, so the layout is versioned rather
than resized. Clamped before centering because RAM holds out-of-range values
mid-write, and the reserved slot is exempt from centering because a centered
zero is -1.0."
```

---

## Task 4: The reward accumulator

**Files:**
- Create: `src/pokemon_env/rewards.py`
- Test: `tests/unit/test_pokemon_env_rewards.py`

**Interfaces:**
- Consumes: `ram` from Task 2, `EnvConfig` from Task 1.
- Produces: `RewardBreakdown` (frozen dataclass: `reward: float`, `clipped: bool`, `components: dict[str, float]`), and `RewardAccumulator` with `reset(mem) -> None`, `step(mem) -> RewardBreakdown`, `coords_seen: int` (property), `steps_since_new_coord: int` (property), `maps_visited: int` (property), `state_dict() -> dict`, `load_state_dict(state: dict) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_pokemon_env_rewards.py`:

```python
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
    fake_emulator.memory[ram.BADGES_ADDR] = 0b0000_0001

    result = accumulator.step(fake_emulator)

    assert result.reward == pytest.approx(1.0)


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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_rewards.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env.rewards'`

- [ ] **Step 3: Implement `rewards.py`**

```python
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


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    clipped: bool
    components: dict[str, float]


@dataclass
class _State:
    max_total: float = 0.0
    explore_sum: float = 0.0
    total_healing: float = 0.0
    base_event_flags: int = 0
    last_hp_fraction: float = 0.0
    last_party_size: int = 0
    steps_since_new_coord: int = 0
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

    @property
    def steps_since_new_coord(self) -> int:
        return self._state.steps_since_new_coord

    def reset(self, mem: Emulator) -> None:
        """Captures the event flags init.state already has set, so the agent
        is not paid on step one for progress it did not make."""
        self._state = _State(
            base_event_flags=ram.event_flag_count(mem),
            last_hp_fraction=ram.aggregate_hp_fraction(mem),
            last_party_size=ram.party_size(mem),
        )

    def step(self, mem: Emulator) -> RewardBreakdown:
        self._update_healing(mem)
        self._update_exploration(mem)

        components = {
            "badges": self._config.badge_weight * ram.badge_count(mem),
            "heal": self._config.heal_weight * self._state.total_healing,
            "explore": self._config.explore_weight * self._state.explore_sum,
            "events": self._config.event_weight * self._event_score(mem),
            "levels": self._config.level_weight * ram.level_score(mem),
        }
        total = sum(components.values())

        gain = max(0.0, total - self._state.max_total)
        self._state.max_total = max(self._state.max_total, total)
        return RewardBreakdown(reward=min(gain, 1.0), clipped=gain > 1.0, components=components)

    def _event_score(self, mem: Emulator) -> int:
        return max(
            ram.event_flag_count(mem)
            - self._state.base_event_flags
            - int(ram.museum_ticket_set(mem)),
            0,
        )

    def _update_healing(self, mem: Emulator) -> None:
        """Squared, so a full heal is worth far more than a trickle. Skipped
        when party size changed: HP fraction rises when a healthy Pokemon
        joins, and crediting that would pay for catching things twice."""
        current = ram.aggregate_hp_fraction(mem)
        size = ram.party_size(mem)
        if current > self._state.last_hp_fraction and size == self._state.last_party_size:
            delta = current - self._state.last_hp_fraction
            self._state.total_healing += delta * delta
        self._state.last_hp_fraction = current
        self._state.last_party_size = size

    def _update_exploration(self, mem: Emulator) -> None:
        """Only outside battle -- in battle the position bytes are stale, so
        recording them credits tiles never actually walked.

        The k-th newly discovered coordinate earns 1/sqrt(k), section 4's
        'decaying scalar reward'. The reference implementation's flat 0.1
        does not decay."""
        if ram.in_battle(mem):
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

    def state_dict(self) -> dict:
        """Coordinates leave as a sorted list of ints, not a set: the
        checkpoint is loaded with torch.load(weights_only=True), which will
        not restore a set or a tuple-keyed dict."""
        return {
            "max_total": self._state.max_total,
            "explore_sum": self._state.explore_sum,
            "total_healing": self._state.total_healing,
            "base_event_flags": self._state.base_event_flags,
            "last_hp_fraction": self._state.last_hp_fraction,
            "last_party_size": self._state.last_party_size,
            "steps_since_new_coord": self._state.steps_since_new_coord,
            "seen_coords": sorted(self._state.seen_coords),
            "seen_maps": sorted(self._state.seen_maps),
        }

    def load_state_dict(self, state: dict) -> None:
        self._state = _State(
            max_total=state["max_total"],
            explore_sum=state["explore_sum"],
            total_healing=state["total_healing"],
            base_event_flags=state["base_event_flags"],
            last_hp_fraction=state["last_hp_fraction"],
            last_party_size=state["last_party_size"],
            steps_since_new_coord=state["steps_since_new_coord"],
            seen_coords=set(state["seen_coords"]),
            seen_maps=set(state["seen_maps"]),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_pokemon_env_rewards.py -q --no-cov`
Expected: PASS (12 tests)

- [ ] **Step 5: Prove a test can fail, then commit**

Change `gain = max(0.0, total - self._state.max_total)` to
`gain = total - self._state.max_total`, confirm
`test_reward_is_never_negative_when_progress_is_lost` goes red, revert. Name
it in your report.

```bash
git add -A
git commit -m "feat(env): the section-4 max-historical reward accumulator

Delta-based and non-negative, so cycles pay nothing. Exploration decays as
1/sqrt(k), which section 4 requires and the reference implementation's flat
0.1 does not do. Clipping is an outlier guard rather than the normalizer, and
the clipped flag makes the clip-fire rate observable."
```

---

## Task 5: The per-env session

**Files:**
- Create: `src/pokemon_env/session.py`
- Test: `tests/unit/test_pokemon_env_session.py`

**Interfaces:**
- Consumes: `Emulator` (Task 1), `EnvConfig` (Task 1), `aux_state` (Task 3), `RewardAccumulator` (Task 4).
- Produces: `BUTTONS: tuple[str, ...]`, `ACTION_DIM: int = 7`, `StepResult` (frozen dataclass: `frame: np.ndarray`, `aux: np.ndarray`, `reward: float`, `done: bool`, `episode_id: int`, `components: dict[str, float]`, `clipped: bool`), and `EnvSession` with `reset() -> StepResult`, `step(action: int) -> StepResult`, `state_dict() -> dict`, `load_state_dict(state: dict) -> None`, `close() -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_pokemon_env_session.py`:

```python
import pytest

from pokemon_env.config import EnvConfig
from pokemon_env.session import ACTION_DIM, BUTTONS, EnvSession


@pytest.fixture
def session(fake_emulator) -> EnvSession:
    return EnvSession(fake_emulator, EnvConfig(max_steps=4), init_state=b"init")


def test_there_are_seven_actions_matching_the_policy_config() -> None:
    assert (len(BUTTONS), ACTION_DIM) == (7, 7)


def test_step_presses_holds_releases_and_renders_only_the_last_frame(
    fake_emulator, session
) -> None:
    """press -> tick(8, False) -> release -> tick(15, False) -> tick(1, True),
    totalling the 24 frames of frame-skip. Rendering only the final frame is
    PyBoy's documented performance guidance."""
    session.reset()
    fake_emulator.calls.clear()

    session.step(4)  # 'a'

    assert fake_emulator.calls == [
        ("press", "a"),
        ("tick", 8, False),
        ("release", "a"),
        ("tick", 15, False),
        ("tick", 1, True),
    ]


def test_reset_loads_the_init_state(fake_emulator, session) -> None:
    session.reset()

    assert fake_emulator.state == b"init"


def test_done_fires_exactly_at_the_step_budget(session) -> None:
    session.reset()
    session.step(0)
    session.step(0)
    session.step(0)

    final = session.step(0)

    assert (final.done, final.episode_id) == (True, 0)


def test_reset_increments_the_episode_id(session) -> None:
    session.reset()

    second = session.reset()

    assert second.episode_id == 1


def test_reset_returns_step_count_to_zero(session) -> None:
    session.reset()
    session.step(0)
    session.reset()

    result = session.step(0)

    assert result.done is False


def test_step_rejects_an_out_of_range_action(session) -> None:
    session.reset()

    with pytest.raises(ValueError, match="action=7 is outside"):
        session.step(7)


def test_frame_has_the_encoder_input_shape(session) -> None:
    result = session.reset()

    assert (result.frame.shape, result.frame.dtype.name) == ((144, 160), "uint8")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_session.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env.session'`

- [ ] **Step 3: Implement `session.py`**

```python
"""One environment: one emulator, one reward accumulator, one episode counter.

Deliberately knows nothing about processes or vectorization -- it is a plain
object driven by whichever backend owns it, which is what lets the whole
step/reset/reward path be tested against a FakeEmulator with no ROM.

Autoreset is NOT handled here. The session reports done and waits; VecPokemonEnv
owns the next-step autoreset that satisfies the sequence-model spec's
cache.reset(done) ordering contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokemon_env.aux_state import ExplorationCounters, build_aux_state
from pokemon_env.config import EnvConfig
from pokemon_env.emulator import Emulator
from pokemon_env.rewards import RewardAccumulator

# Index order is the action space. Must stay stable across checkpoints: a
# reordering silently remaps every action the policy has learned.
BUTTONS = ("down", "left", "right", "up", "a", "b", "start")
ACTION_DIM = len(BUTTONS)


@dataclass(frozen=True)
class StepResult:
    frame: np.ndarray  # (144, 160) uint8
    aux: np.ndarray  # (32,) float32
    reward: float
    done: bool
    episode_id: int
    components: dict[str, float]
    clipped: bool


class EnvSession:
    def __init__(self, emulator: Emulator, config: EnvConfig, init_state: bytes) -> None:
        self._emulator = emulator
        self._config = config
        self._init_state = init_state
        self._rewards = RewardAccumulator(config)
        self._step_count = 0
        self._episode_id = -1  # first reset() makes it 0

    def reset(self) -> StepResult:
        self._emulator.load_state(self._init_state)
        self._rewards.reset(self._emulator)
        self._step_count = 0
        self._episode_id += 1
        return self._observe(reward=0.0, clipped=False, components={})

    def step(self, action: int) -> StepResult:
        if not 0 <= action < ACTION_DIM:
            raise ValueError(f"action={action} is outside [0, {ACTION_DIM})")

        button = BUTTONS[action]
        self._emulator.button_press(button)
        self._emulator.tick(self._config.press_frames, False)
        self._emulator.button_release(button)
        self._emulator.tick(self._config.release_frames, False)
        self._emulator.tick(1, True)  # only the final frame is rendered

        self._step_count += 1
        breakdown = self._rewards.step(self._emulator)
        return self._observe(
            reward=breakdown.reward,
            clipped=breakdown.clipped,
            components=breakdown.components,
        )

    def _observe(self, reward: float, clipped: bool, components: dict[str, float]) -> StepResult:
        exploration = ExplorationCounters(
            coords_seen=self._rewards.coords_seen,
            steps_since_new_coord=self._rewards.steps_since_new_coord,
            maps_visited=self._rewards.maps_visited,
        )
        return StepResult(
            frame=self._emulator.screen_frame(),
            aux=build_aux_state(
                self._emulator, self._step_count, exploration, self._config.max_steps
            ),
            reward=reward,
            done=self._step_count >= self._config.max_steps,
            episode_id=self._episode_id,
            components=components,
            clipped=clipped,
        )

    def state_dict(self) -> dict:
        return {
            "emulator": self._emulator.save_state(),
            "rewards": self._rewards.state_dict(),
            "step_count": self._step_count,
            "episode_id": self._episode_id,
        }

    def load_state_dict(self, state: dict) -> None:
        self._emulator.load_state(state["emulator"])
        self._rewards.load_state_dict(state["rewards"])
        self._step_count = state["step_count"]
        self._episode_id = state["episode_id"]

    def close(self) -> None:
        self._emulator.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_pokemon_env_session.py -q --no-cov`
Expected: PASS (8 tests)

- [ ] **Step 5: Prove a test can fail, then commit**

Change `self._emulator.tick(1, True)` to `self._emulator.tick(1, False)`,
confirm `test_step_presses_holds_releases_and_renders_only_the_last_frame`
goes red, revert. Name it in your report.

```bash
git add -A
git commit -m "feat(env): the per-env session

One emulator, one reward accumulator, one episode counter, and no knowledge
of processes -- which is what lets the whole step/reset/reward path be tested
against a FakeEmulator with no ROM. Autoreset stays with VecPokemonEnv."
```

---

## Task 6: `VecPokemonEnv` and the in-process backend

**Files:**
- Create: `src/pokemon_env/vec_env.py`
- Test: `tests/unit/test_pokemon_env_vec_env.py`

**Interfaces:**
- Consumes: `EnvSession`, `StepResult` (Task 5), `EnvConfig` (Task 1).
- Produces: `EnvBackend` Protocol (`reset() -> StepResult`, `step(action: int) -> StepResult`, `state_dict() -> dict`, `load_state_dict(state: dict) -> None`, `close() -> None`), `InProcessBackend`, `VecStep` (frozen dataclass: `frames: np.ndarray` `(N,1,144,160)` uint8, `aux: np.ndarray` `(N,32)` float32, `reward: np.ndarray` `(N,)` float32, `done: np.ndarray` `(N,)` bool, `episode_id: np.ndarray` `(N,)` int64), and `VecPokemonEnv` with `reset() -> VecStep`, `step(actions) -> VecStep`, `state_dict()`, `load_state_dict(state)`, `close()`, `last_components: dict[str, float]`, `clip_fire_rate: float`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_pokemon_env_vec_env.py`:

```python
import numpy as np
import pytest

from pokemon_env.config import EnvConfig
from pokemon_env.session import EnvSession
from pokemon_env.vec_env import InProcessBackend, VecPokemonEnv

from .fakes import FakeEmulator


@pytest.fixture
def vec_env() -> VecPokemonEnv:
    config = EnvConfig(n_envs=3, max_steps=2)
    backends = [
        InProcessBackend(EnvSession(FakeEmulator(), config, init_state=b"init"))
        for _ in range(3)
    ]
    return VecPokemonEnv(backends, config)


def test_step_returns_batched_arrays_with_the_encoder_frame_shape(vec_env) -> None:
    vec_env.reset()

    result = vec_env.step(np.zeros(3, dtype=np.int64))

    assert (result.frames.shape, result.frames.dtype.name) == ((3, 1, 144, 160), "uint8")


def test_step_returns_one_aux_row_per_env(vec_env) -> None:
    vec_env.reset()

    result = vec_env.step(np.zeros(3, dtype=np.int64))

    assert (result.aux.shape, result.aux.dtype.name) == ((3, 32), "float32")


def test_done_is_reported_on_the_terminal_step_not_deferred(vec_env) -> None:
    """max_steps=2, so the second step is terminal. The sequence-model spec's
    cache.reset(done) contract requires reset to run AFTER the step whose
    transition ended the episode -- so done must be true on that step."""
    vec_env.reset()
    vec_env.step(np.zeros(3, dtype=np.int64))

    second = vec_env.step(np.zeros(3, dtype=np.int64))

    assert second.done.tolist() == [True, True, True]


def test_the_step_after_done_returns_the_reset_observation(vec_env) -> None:
    """Next-step autoreset: done at t returns the TERMINAL observation, and
    the reset observation arrives at t+1 with an incremented episode_id."""
    vec_env.reset()
    vec_env.step(np.zeros(3, dtype=np.int64))
    vec_env.step(np.zeros(3, dtype=np.int64))

    after = vec_env.step(np.zeros(3, dtype=np.int64))

    assert (after.episode_id.tolist(), after.done.tolist()) == ([1, 1, 1], [False, False, False])


def test_reset_starts_every_env_at_episode_zero(vec_env) -> None:
    result = vec_env.reset()

    assert result.episode_id.tolist() == [0, 0, 0]


def test_step_rejects_a_wrong_length_action_array(vec_env) -> None:
    vec_env.reset()

    with pytest.raises(ValueError, match="actions has length 2, expected 3"):
        vec_env.step(np.zeros(2, dtype=np.int64))


def test_clip_fire_rate_reports_the_fraction_of_clipped_steps(vec_env) -> None:
    """Above roughly 0.1% the reward weights are miscalibrated and the
    ordering between achievements is being flattened."""
    vec_env.reset()
    vec_env.step(np.zeros(3, dtype=np.int64))

    assert vec_env.clip_fire_rate == pytest.approx(0.0)


def test_state_dict_round_trips_the_per_env_step_counters(vec_env) -> None:
    """Asserts the per-env counters actually moved across, not just that a
    version constant matches -- the version would compare equal against a
    restore that loaded nothing."""
    vec_env.reset()
    vec_env.step(np.zeros(3, dtype=np.int64))
    state = vec_env.state_dict()

    config = EnvConfig(n_envs=3, max_steps=2)
    restored = VecPokemonEnv(
        [
            InProcessBackend(EnvSession(FakeEmulator(), config, init_state=b"init"))
            for _ in range(3)
        ],
        config,
    )
    restored.load_state_dict(state)

    assert [b["step_count"] for b in restored.state_dict()["backends"]] == [1, 1, 1]


def test_load_state_dict_rejects_a_stale_aux_state_version(vec_env) -> None:
    """A policy trained against layout v1 and fed v2 data reads different
    signals from the same indices, with no crash and no shape error."""
    vec_env.reset()
    state = vec_env.state_dict()
    state["aux_state_version"] = 99

    with pytest.raises(ValueError, match="AUX_STATE_VERSION=99"):
        vec_env.load_state_dict(state)


def test_load_state_dict_rejects_a_different_env_count(vec_env) -> None:
    vec_env.reset()
    state = vec_env.state_dict()
    state["backends"] = state["backends"][:2]

    with pytest.raises(ValueError, match="cannot be redistributed"):
        vec_env.load_state_dict(state)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_vec_env.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env.vec_env'`

- [ ] **Step 3: Implement `vec_env.py`**

```python
"""Parent-side vectorization over N env backends.

Autoreset is next-step, deliberately: done=True at step t returns the TERMINAL
observation, and the reset observation arrives at t+1. This exists to satisfy
the sequence-model spec's cache.reset(done) ordering contract -- reset must run
AFTER the step whose transition ended the episode, or the final transition of
every episode attends to a cleared cache. Making it structural here means the
trainer cannot get the ordering wrong.

The backend Protocol has two implementations: InProcessBackend (here, for
tests and debugging -- roughly 64x too slow for a real run) and
SubprocessBackend (subprocess_backend.py, what production uses)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pokemon_env.aux_state import AUX_STATE_DIM, AUX_STATE_VERSION
from pokemon_env.config import EnvConfig
from pokemon_env.emulator import SCREEN_HEIGHT, SCREEN_WIDTH
from pokemon_env.session import EnvSession, StepResult

VEC_ENV_SCHEMA_VERSION = 1


class EnvBackend(Protocol):
    def reset(self) -> StepResult: ...
    def step(self, action: int) -> StepResult: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def close(self) -> None: ...


class InProcessBackend:
    """Drives an EnvSession directly. Exists so vec_env logic -- autoreset
    ordering, episode_id monotonicity, batching -- is testable without
    spawning processes."""

    def __init__(self, session: EnvSession) -> None:
        self._session = session

    def reset(self) -> StepResult:
        return self._session.reset()

    def step(self, action: int) -> StepResult:
        return self._session.step(action)

    def state_dict(self) -> dict:
        return self._session.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._session.load_state_dict(state)

    def close(self) -> None:
        self._session.close()


@dataclass(frozen=True)
class VecStep:
    frames: np.ndarray  # (N, 1, 144, 160) uint8
    aux: np.ndarray  # (N, 32) float32
    reward: np.ndarray  # (N,) float32
    done: np.ndarray  # (N,) bool
    episode_id: np.ndarray  # (N,) int64


class VecPokemonEnv:
    def __init__(self, backends: list[EnvBackend], config: EnvConfig) -> None:
        if len(backends) != config.n_envs:
            raise ValueError(
                f"got {len(backends)} backends for n_envs={config.n_envs}; "
                "the batch dimension must match the configured env count"
            )
        self._backends = backends
        self._config = config
        self._needs_reset = np.zeros(config.n_envs, dtype=bool)
        self._last_components: dict[str, float] = {}
        self._clipped_steps = 0
        self._total_steps = 0

    @property
    def n_envs(self) -> int:
        return self._config.n_envs

    @property
    def last_components(self) -> dict[str, float]:
        """Mean reward per component over the most recent vector step."""
        return dict(self._last_components)

    @property
    def clip_fire_rate(self) -> float:
        if self._total_steps == 0:
            return 0.0
        return self._clipped_steps / self._total_steps

    def reset(self) -> VecStep:
        self._needs_reset[:] = False
        return self._collect([backend.reset() for backend in self._backends])

    def step(self, actions: np.ndarray) -> VecStep:
        if len(actions) != self._config.n_envs:
            raise ValueError(
                f"actions has length {len(actions)}, expected {self._config.n_envs}"
            )
        results = [
            backend.reset() if needs_reset else backend.step(int(action))
            for backend, action, needs_reset in zip(
                self._backends, actions, self._needs_reset, strict=True
            )
        ]
        self._needs_reset = np.array([result.done for result in results], dtype=bool)
        return self._collect(results)

    def _collect(self, results: list[StepResult]) -> VecStep:
        self._total_steps += len(results)
        self._clipped_steps += sum(1 for result in results if result.clipped)
        self._last_components = _mean_components(results)

        frames = np.empty(
            (len(results), 1, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8
        )
        aux = np.empty((len(results), AUX_STATE_DIM), dtype=np.float32)
        for i, result in enumerate(results):
            frames[i, 0] = result.frame
            aux[i] = result.aux

        return VecStep(
            frames=frames,
            aux=aux,
            reward=np.array([r.reward for r in results], dtype=np.float32),
            done=np.array([r.done for r in results], dtype=bool),
            episode_id=np.array([r.episode_id for r in results], dtype=np.int64),
        )

    def state_dict(self) -> dict:
        return {
            "schema_version": VEC_ENV_SCHEMA_VERSION,
            "aux_state_version": AUX_STATE_VERSION,
            "needs_reset": self._needs_reset.tolist(),
            "backends": [backend.state_dict() for backend in self._backends],
        }

    def load_state_dict(self, state: dict) -> None:
        if state["aux_state_version"] != AUX_STATE_VERSION:
            raise ValueError(
                f"checkpoint has AUX_STATE_VERSION={state['aux_state_version']}, "
                f"this build is {AUX_STATE_VERSION}. The aux vector's slot layout "
                "changed, so a policy trained against the old one would be silently "
                "reading different signals from the same indices."
            )
        if len(state["backends"]) != self._config.n_envs:
            raise ValueError(
                f"checkpoint holds {len(state['backends'])} envs but this run has "
                f"{self._config.n_envs}; per-env state cannot be redistributed"
            )
        self._needs_reset = np.array(state["needs_reset"], dtype=bool)
        for backend, backend_state in zip(self._backends, state["backends"], strict=True):
            backend.load_state_dict(backend_state)

    def close(self) -> None:
        for backend in self._backends:
            backend.close()


def _mean_components(results: list[StepResult]) -> dict[str, float]:
    """Helper: mean of each reward component across envs, for telemetry."""
    keys = {key for result in results for key in result.components}
    return {
        key: sum(result.components.get(key, 0.0) for result in results) / len(results)
        for key in sorted(keys)
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_pokemon_env_vec_env.py -q --no-cov`
Expected: PASS (11 tests)

- [ ] **Step 5: Prove a test can fail, then commit**

Change `step` so `self._needs_reset` is assigned *before* the results are
collected (i.e. reset immediately on done rather than next-step), confirm
`test_the_step_after_done_returns_the_reset_observation` goes red, revert.
Name it in your report.

```bash
git add -A
git commit -m "feat(env): VecPokemonEnv with structural next-step autoreset

done=True at step t returns the terminal observation; the reset observation
arrives at t+1. This makes the sequence-model spec's cache.reset(done)
ordering contract structural rather than a rule the trainer has to remember."
```

---

## Task 7: The subprocess backend

**Files:**
- Create: `src/pokemon_env/subprocess_backend.py`
- Test: `tests/unit/test_pokemon_env_subprocess_backend.py`

**Interfaces:**
- Consumes: `EnvBackend` Protocol, `StepResult` (Tasks 5–6), `EnvConfig` (Task 1).
- Produces: `Command` (str enum: `RESET`, `STEP`, `STATE_DICT`, `LOAD_STATE`, `CLOSE`), `FrameBuffer` (owns the `SharedMemory` block; `array` property, `close()`, `unlink()`), `worker_main(conn, shm_name, index, config, rom_path, init_state)`, `SubprocessBackend`, and `build_subprocess_vec_env(config) -> tuple[VecPokemonEnv, FrameBuffer]` (the caller keeps the buffer alive and calls `unlink()` after `close()`).

**Note on why frames use shared memory:** a frame is 23,040 B, so 64 of them
is 1.47 MB per vector step; at 1024 steps per rollout that is 1.5 GB of
pickling, roughly 19% of the 8.0 s rollout budget spent serializing. Shared
memory reduces it to a memcpy. No locking is needed — slices are disjoint and
the parent joins all responses before reading.

- [ ] **Step 1: Write the failing tests**

These test the parts that do not require spawning: the shared-memory buffer
and the command dispatch. Real process spawning is covered by the slow
integration test in Task 12.

`tests/unit/test_pokemon_env_subprocess_backend.py`:

```python
from collections.abc import Iterator

import numpy as np
import pytest

from pokemon_env.config import EnvConfig
from pokemon_env.session import EnvSession
from pokemon_env.subprocess_backend import Command, FrameBuffer, handle_command

from .fakes import FakeEmulator


@pytest.fixture
def frame_buffer() -> Iterator[FrameBuffer]:
    """Iterator, not FrameBuffer: a yield fixture's annotation describes what
    the function returns, and a type checker flags the mismatch."""
    buffer = FrameBuffer.create(n_envs=3)
    yield buffer
    buffer.close()
    buffer.unlink()


@pytest.fixture
def emulator() -> FakeEmulator:
    return FakeEmulator()


@pytest.fixture
def session(emulator: FakeEmulator) -> EnvSession:
    return EnvSession(emulator, EnvConfig(n_envs=3, max_steps=4), init_state=b"init")


def test_frame_buffer_has_one_slot_per_env(frame_buffer) -> None:
    assert (frame_buffer.array.shape, frame_buffer.array.dtype.name) == ((3, 144, 160), "uint8")


def test_frame_buffer_slots_are_independent(frame_buffer) -> None:
    """Workers write disjoint slices with no locking, so a shape or stride
    error that made them overlap would silently mix envs' observations."""
    frame_buffer.array[0] = 7
    frame_buffer.array[1] = 9

    assert (int(frame_buffer.array[0, 0, 0]), int(frame_buffer.array[1, 0, 0])) == (7, 9)


def test_frame_buffer_attaches_to_an_existing_block_by_name(frame_buffer) -> None:
    frame_buffer.array[2] = 5
    attached = FrameBuffer.attach(frame_buffer.name, n_envs=3)

    value = int(attached.array[2, 0, 0])
    attached.close()

    assert value == 5


def test_handle_reset_writes_the_frame_into_the_shared_slot(
    frame_buffer, emulator, session
) -> None:
    """The fake is injected as its own fixture rather than reached for through
    session._emulator -- a test that pokes a private attribute breaks on any
    refactor that renames it."""
    emulator.frame = np.full((144, 160), 3, dtype=np.uint8)

    handle_command(session, Command.RESET, None, frame_buffer.array[1])

    assert int(frame_buffer.array[1, 0, 0]) == 3


def test_handle_command_writes_only_its_own_slot(
    frame_buffer, emulator, session
) -> None:
    """Workers write disjoint slices with no locking. A slot-index error would
    silently overwrite a neighbouring env's observation."""
    emulator.frame = np.full((144, 160), 3, dtype=np.uint8)

    handle_command(session, Command.RESET, None, frame_buffer.array[1])

    assert (int(frame_buffer.array[0, 0, 0]), int(frame_buffer.array[2, 0, 0])) == (0, 0)


def test_handle_reset_returns_the_payload_without_the_frame(
    frame_buffer, session
) -> None:
    """The frame goes through shared memory; everything else is small enough
    to ride the pipe. Including the frame in the payload would reintroduce the
    1.5 GB-per-rollout pickling cost the shared block exists to remove."""
    payload = handle_command(session, Command.RESET, None, frame_buffer.array[0])

    assert "frame" not in payload


def test_handle_step_advances_the_episode(frame_buffer, session) -> None:
    handle_command(session, Command.RESET, None, frame_buffer.array[0])

    payload = handle_command(session, Command.STEP, 3, frame_buffer.array[0])

    assert (payload["episode_id"], payload["done"]) == (0, False)


def test_handle_command_rejects_an_unknown_command(frame_buffer, session) -> None:
    with pytest.raises(ValueError, match="unknown command"):
        handle_command(session, "NOT_A_COMMAND", None, frame_buffer.array[0])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_subprocess_backend.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env.subprocess_backend'`

- [ ] **Step 3: Implement `subprocess_backend.py`**

```python
"""64 subprocess workers, each owning one PyBoy emulator.

PyBoy is GIL-bound, so threads buy no parallelism. Frames cross into the
parent through one SharedMemory block -- a frame is 23,040 B, so 64 of them is
1.47 MB per vector step and 1.5 GB of pickling per 1024-step rollout, roughly
19% of the 8.0 s rollout budget. Shared memory makes it a memcpy. No locking:
slices are disjoint and the parent joins all responses before reading.

Spawn, not fork: fork would duplicate the parent's CUDA context into every
worker, which CUDA does not support."""

from __future__ import annotations

import multiprocessing as mp
from enum import StrEnum
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from pokemon_env.config import EnvConfig
from pokemon_env.emulator import SCREEN_HEIGHT, SCREEN_WIDTH, PyBoyEmulator
from pokemon_env.session import EnvSession, StepResult
from pokemon_env.vec_env import VecPokemonEnv


class Command(StrEnum):
    RESET = "RESET"
    STEP = "STEP"
    STATE_DICT = "STATE_DICT"
    LOAD_STATE = "LOAD_STATE"
    CLOSE = "CLOSE"


class FrameBuffer:
    """One (n_envs, 144, 160) uint8 block. `create` in the parent, `attach` by
    name in each worker."""

    def __init__(self, shm: SharedMemory, n_envs: int, owner: bool) -> None:
        self._shm = shm
        self._owner = owner
        self.array = np.ndarray(
            (n_envs, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8, buffer=shm.buf
        )

    @classmethod
    def create(cls, n_envs: int) -> FrameBuffer:
        shm = SharedMemory(create=True, size=n_envs * SCREEN_HEIGHT * SCREEN_WIDTH)
        return cls(shm, n_envs, owner=True)

    @classmethod
    def attach(cls, name: str, n_envs: int) -> FrameBuffer:
        return cls(SharedMemory(name=name), n_envs, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    def close(self) -> None:
        # Drop the ndarray's reference to the buffer first: SharedMemory.close()
        # raises BufferError while an exported memoryview is still alive.
        del self.array
        self._shm.close()

    def unlink(self) -> None:
        """Parent only. Releases the OS-level block."""
        self._shm.unlink()


def _payload(result: StepResult) -> dict:
    """Everything except the frame, which travels through shared memory."""
    return {
        "aux": result.aux,
        "reward": result.reward,
        "done": result.done,
        "episode_id": result.episode_id,
        "components": result.components,
        "clipped": result.clipped,
    }


def handle_command(
    session: EnvSession, command: str, argument: object, frame_slot: np.ndarray
) -> dict:
    """Pure dispatch, so the worker's behaviour is testable without spawning.
    Writes the frame into `frame_slot` and returns the pipe payload."""
    if command == Command.RESET:
        result = session.reset()
    elif command == Command.STEP:
        result = session.step(int(argument))  # type: ignore[arg-type]
    elif command == Command.STATE_DICT:
        return {"state": session.state_dict()}
    elif command == Command.LOAD_STATE:
        session.load_state_dict(argument)  # type: ignore[arg-type]
        return {"ok": True}
    else:
        raise ValueError(f"unknown command {command!r}")

    frame_slot[:] = result.frame
    return _payload(result)


def worker_main(
    conn: Connection,
    shm_name: str,
    index: int,
    config: EnvConfig,
    rom_path: str,
    init_state: bytes,
) -> None:
    """Worker entry point. Owns exactly one emulator for its lifetime."""
    buffer = FrameBuffer.attach(shm_name, config.n_envs)
    session = EnvSession(PyBoyEmulator(rom_path), config, init_state)
    try:
        while True:
            command, argument = conn.recv()
            if command == Command.CLOSE:
                break
            try:
                conn.send(("ok", handle_command(session, command, argument, buffer.array[index])))
            except Exception as error:  # noqa: BLE001 -- must reach the parent, not die silently
                conn.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        session.close()
        buffer.close()
        conn.close()


class SubprocessBackend:
    """Parent-side handle on one worker. Respawns it from init.state on death
    or timeout rather than taking the whole run down."""

    def __init__(
        self,
        index: int,
        shm_name: str,
        config: EnvConfig,
        rom_path: str,
        init_state: bytes,
        frame_slot: np.ndarray,
    ) -> None:
        self._index = index
        self._shm_name = shm_name
        self._config = config
        self._rom_path = rom_path
        self._init_state = init_state
        self._frame_slot = frame_slot
        self._respawns = 0
        self._spawn()

    @property
    def respawns(self) -> int:
        return self._respawns

    def _spawn(self) -> None:
        context = mp.get_context("spawn")
        self._conn, child_conn = context.Pipe()
        self._process = context.Process(
            target=worker_main,
            args=(
                child_conn,
                self._shm_name,
                self._index,
                self._config,
                self._rom_path,
                self._init_state,
            ),
            daemon=True,
        )
        self._process.start()
        child_conn.close()

    def _call(self, command: Command, argument: object = None) -> dict:
        self._conn.send((command, argument))
        if not self._conn.poll(self._config.worker_timeout_s):
            raise TimeoutError(
                f"env {self._index} did not answer {command} within "
                f"{self._config.worker_timeout_s}s; a 24-frame tick takes about 1ms, "
                "so this is a hang, not slowness"
            )
        status, payload = self._conn.recv()
        if status == "error":
            raise RuntimeError(f"env {self._index} worker failed: {payload}")
        return payload

    def _restart(self) -> StepResult:
        """Respawn from init.state, NOT from the last checkpoint's emulator
        state: that state pairs with a checkpoint-time reward baseline and
        coord set, so restoring it against a current-time accumulator would
        silently re-earn rewards for progress already banked."""
        self._respawns += 1
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=5)
        self._spawn()
        return self.reset()

    def _to_result(self, payload: dict) -> StepResult:
        return StepResult(
            frame=self._frame_slot,
            aux=payload["aux"],
            reward=payload["reward"],
            done=payload["done"],
            episode_id=payload["episode_id"],
            components=payload["components"],
            clipped=payload["clipped"],
        )

    def reset(self) -> StepResult:
        return self._to_result(self._call(Command.RESET))

    def step(self, action: int) -> StepResult:
        try:
            payload = self._call(Command.STEP, action)
        except (TimeoutError, RuntimeError, EOFError, BrokenPipeError):
            restarted = self._restart()
            # Force the episode boundary so the trainer resets its KV cache for
            # this env; a respawned worker shares no history with the old one.
            return StepResult(
                frame=restarted.frame,
                aux=restarted.aux,
                reward=0.0,
                done=True,
                episode_id=restarted.episode_id,
                components={},
                clipped=False,
            )
        return self._to_result(payload)

    def state_dict(self) -> dict:
        return self._call(Command.STATE_DICT)["state"]

    def load_state_dict(self, state: dict) -> None:
        self._call(Command.LOAD_STATE, state)

    def close(self) -> None:
        try:
            self._conn.send((Command.CLOSE, None))
        except (BrokenPipeError, OSError):
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()


def build_subprocess_vec_env(config: EnvConfig) -> tuple[VecPokemonEnv, FrameBuffer]:
    """Production constructor. The caller must keep the FrameBuffer alive for
    the env's lifetime and call unlink() after close()."""
    from pathlib import Path

    init_state = Path(config.init_state_path).read_bytes()
    buffer = FrameBuffer.create(config.n_envs)
    backends = [
        SubprocessBackend(
            index=i,
            shm_name=buffer.name,
            config=config,
            rom_path=config.rom_path,
            init_state=init_state,
            frame_slot=buffer.array[i],
        )
        for i in range(config.n_envs)
    ]
    return VecPokemonEnv(backends, config), buffer
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_pokemon_env_subprocess_backend.py -q --no-cov`
Expected: PASS (8 tests)

- [ ] **Step 5: Prove a test can fail, then commit**

Change `_payload` to include `"frame": result.frame`, confirm
`test_handle_reset_returns_the_payload_without_the_frame` goes red, revert.
Name it in your report.

```bash
git add -A
git commit -m "feat(env): subprocess backend with shared-memory frames

Spawn, not fork -- fork would duplicate the parent's CUDA context into every
worker. Frames go through one SharedMemory block because pickling them costs
1.5 GB per rollout, roughly 19% of the 8.0 s budget.

A dead or hung worker respawns from init.state and forces done=True rather
than taking the run down. Deliberately not from the last checkpoint's
emulator state: that pairs with a checkpoint-time reward baseline, so
restoring it against a current-time accumulator would re-earn banked
progress."
```

---

## Task 8: Frozen encoder and latent statistics

**Files:**
- Create: `src/pokemon_env/encoder.py`
- Test: `tests/unit/test_pokemon_env_encoder.py`

**Interfaces:**
- Consumes: `contrastive_pretrain.encoder_io.load_frozen_encoder(repo_id) -> nn.Module`, `hf_storage.client.HfClient` Protocol (`download_bytes(path) -> bytes | None`), `EnvConfig` (Task 1).
- Produces: `load_latent_stats(client) -> tuple[torch.Tensor, torch.Tensor]` and `LatentEncoder` with `encode(frames: np.ndarray) -> torch.Tensor`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_pokemon_env_encoder.py`:

```python
import json

import numpy as np
import pytest
import torch
from torch import nn

from pokemon_env.encoder import LatentEncoder, load_latent_stats


class FakeStatsClient:
    """Hand-written fake typed against hf_storage.client.HfClient."""

    def __init__(self, payload: bytes | None) -> None:
        self._payload = payload
        self.requested: list[str] = []

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        raise AssertionError("the encoder never uploads")

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        self.requested.append(path_in_repo)
        return self._payload


def _stats_payload(mean: list[float], std: list[float]) -> bytes:
    """Helper, not a test."""
    return json.dumps({"mean": mean, "std": std}).encode()


class TinyEncoder(nn.Module):
    """Stands in for the frozen ResNet: same contract, 400x smaller.

    Seeds itself so the suite has no unseeded randomness -- a flaky failure
    you cannot reproduce is a failure you cannot fix."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.head = nn.Linear(144 * 160, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.flatten(1))


def test_load_latent_stats_returns_mean_and_std_tensors() -> None:
    client = FakeStatsClient(_stats_payload([0.0] * 2048, [1.0] * 2048))

    mean, std = load_latent_stats(client)

    assert (tuple(mean.shape), tuple(std.shape)) == ((2048,), (2048,))


def test_load_latent_stats_requests_the_documented_filename() -> None:
    client = FakeStatsClient(_stats_payload([0.0] * 2048, [1.0] * 2048))

    load_latent_stats(client)

    assert client.requested == ["latent_stats.json"]


def test_load_latent_stats_rejects_a_zero_standard_deviation() -> None:
    """A dead encoder channel with std 0 divides by InputAdapter's 1e-6 floor
    and feeds ~1e6-scale inputs to a value head the architecture plan calls
    hypersensitive to input scale."""
    std = [1.0] * 2048
    std[7] = 0.0
    client = FakeStatsClient(_stats_payload([0.0] * 2048, std))

    with pytest.raises(ValueError, match="latent_std has 1 non-positive"):
        load_latent_stats(client)


def test_load_latent_stats_rejects_a_wrong_length_vector() -> None:
    client = FakeStatsClient(_stats_payload([0.0] * 512, [1.0] * 512))

    with pytest.raises(ValueError, match="expected 2048"):
        load_latent_stats(client)


def test_load_latent_stats_raises_when_the_file_is_missing() -> None:
    client = FakeStatsClient(None)

    with pytest.raises(FileNotFoundError, match="latent_stats.json"):
        load_latent_stats(client)


def test_encode_returns_one_latent_row_per_frame() -> None:
    encoder = LatentEncoder(TinyEncoder(), torch.device("cpu"))
    frames = np.zeros((3, 1, 144, 160), dtype=np.uint8)

    latents = encoder.encode(frames)

    assert tuple(latents.shape) == (3, 2048)


def test_encode_output_is_not_an_inference_tensor() -> None:
    """THE test. Latents recorded at rollout become inputs to forward_chunk at
    the PPO update, and a tensor made under inference_mode raises 'Inference
    tensors cannot be saved for backward' the moment the adapter tries to save
    it -- at the first update, on a paid GPU. requires_grad is False either
    way, so is_inference() is the only check that distinguishes them."""
    encoder = LatentEncoder(TinyEncoder(), torch.device("cpu"))
    frames = np.zeros((2, 1, 144, 160), dtype=np.uint8)

    latents = encoder.encode(frames)

    assert latents.is_inference() is False


def test_encode_rejects_a_transposed_frame_batch() -> None:
    """GrayscaleResNetEncoder rejects (N, 1, 160, 144), but only after the
    frame has already crossed IPC. Catching it here names the real problem."""
    encoder = LatentEncoder(TinyEncoder(), torch.device("cpu"))

    with pytest.raises(ValueError, match=r"expected \(N, 1, 144, 160\)"):
        encoder.encode(np.zeros((2, 1, 160, 144), dtype=np.uint8))


def test_encode_does_not_rescale_pixels_to_unit_range() -> None:
    """The published artifact has Conv+BN fused, so no BatchNorm remains to
    absorb a different input scale; rescaling to [0,1] produces wrong features
    with no error raised. A constant-255 frame must reach the encoder as 255."""
    encoder = LatentEncoder(TinyEncoder(), torch.device("cpu"))
    frames = np.full((1, 1, 144, 160), 255, dtype=np.uint8)

    latents = encoder.encode(frames)

    assert latents.abs().max().item() > 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_encoder.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env.encoder'`

- [ ] **Step 3: Implement `encoder.py`**

```python
"""Frozen-encoder inference and the latent statistics that normalize it.

Owns two things the sequence-model spec's handoff assigned to PPO, because
they belong wherever the frozen encoder lives: batched inference, and fetching
plus validating latent_stats.json. InputAdapter already raises on bad stats,
but it can only do so once someone hands it the values -- fetching them is
this module's job."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from contrastive_pretrain.model import EMBEDDING_DIM
from hf_storage.client import HfClient

LATENT_STATS_FILENAME = "latent_stats.json"
FRAME_HEIGHT = 144
FRAME_WIDTH = 160


def load_latent_stats(client: HfClient) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, std), each (2048,) float32, validated against InputAdapter's
    contract before anyone builds a policy with them."""
    payload = client.download_bytes(LATENT_STATS_FILENAME)
    if payload is None:
        raise FileNotFoundError(
            f"{LATENT_STATS_FILENAME} missing from the frozen-encoder repo; the policy "
            "cannot normalize latents without it, and unnormalized contrastive latents "
            "cause immediate PPO policy collapse"
        )
    stats = json.loads(payload)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)

    for name, tensor in (("latent_mean", mean), ("latent_std", std)):
        if tensor.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, expected ({EMBEDDING_DIM},)"
            )

    non_positive = int((std <= 0).sum())
    if non_positive:
        raise ValueError(
            f"latent_std has {non_positive} non-positive entries. A dead encoder channel "
            "divides by InputAdapter's 1e-6 floor and feeds ~1e6-scale inputs to the "
            "value head."
        )
    return mean, std


class LatentEncoder:
    """Batched frozen-CNN inference: (N, 1, 144, 160) uint8 -> (N, 2048)."""

    def __init__(self, encoder: nn.Module, device: torch.device) -> None:
        self._encoder = encoder.to(device).to(memory_format=torch.channels_last).eval()
        self._device = device

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """@torch.no_grad(), deliberately NOT @torch.inference_mode().

        Latents recorded during rollout become inputs to forward_chunk at the
        PPO update, and a tensor created under inference_mode raises "Inference
        tensors cannot be saved for backward" the moment the adapter tries to
        save it -- at the first update, on a paid GPU, not at rollout.
        Cloning is not a fix: inside an inference_mode context .clone()
        returns another inference tensor.

        Pixels are cast to float but NOT rescaled to [0, 1]: the published
        artifact has Conv+BN fused, so no BatchNorm remains to absorb a
        different input scale and the features would be wrong with no error."""
        if frames.ndim != 4 or frames.shape[1:] != (1, FRAME_HEIGHT, FRAME_WIDTH):
            raise ValueError(
                f"frames has shape {tuple(frames.shape)}, expected "
                f"(N, 1, {FRAME_HEIGHT}, {FRAME_WIDTH})"
            )
        batch = (
            torch.from_numpy(frames)
            .to(self._device, non_blocking=True)
            .float()
            .to(memory_format=torch.channels_last)
        )
        return self._encoder(batch)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_pokemon_env_encoder.py -q --no-cov`
Expected: PASS (9 tests)

- [ ] **Step 5: Prove a test can fail, then commit**

Change `@torch.no_grad()` to `@torch.inference_mode()`, confirm
`test_encode_output_is_not_an_inference_tensor` goes red, revert. Name it in
your report — this is the highest-value guard in the task.

```bash
git add -A
git commit -m "feat(env): frozen-encoder inference and latent-stats loading

encode() is @torch.no_grad(), deliberately not @torch.inference_mode(): its
output enters autograd at the PPO update, where an inference tensor raises
'Inference tensors cannot be saved for backward' on a paid GPU. Cloning does
not help -- inside inference_mode, clone returns another inference tensor.

Also claims the latent_stats.json fetch the sequence-model spec had assigned
to PPO; it belongs wherever the encoder lives."
```

---

## Task 9: `init.state` generation

**Files:**
- Create: `src/pokemon_env/init_state.py`, `artifacts/.gitignore`
- Modify: `.gitignore` (add `artifacts/`)
- Test: `tests/unit/test_pokemon_env_init_state.py`, add one case to `tests/integration/test_pokemon_env_smoke.py`

**Interfaces:**
- Consumes: `Emulator` Protocol (Task 1), `EnvConfig` (Task 1).
- Produces: `ButtonPress` (frozen dataclass: `button: str | None`, `frames: int`), `INTRO_SCRIPT: tuple[ButtonPress, ...]`, `generate_init_state(emulator, script) -> bytes`, `state_hash(state: bytes) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_pokemon_env_init_state.py`:

```python
import pytest

from pokemon_env.init_state import (
    INTRO_SCRIPT,
    ButtonPress,
    generate_init_state,
    state_hash,
)


def test_generate_init_state_returns_the_emulator_state(fake_emulator) -> None:
    fake_emulator.state = b"post-intro"

    result = generate_init_state(fake_emulator, (ButtonPress(button="a", frames=2),))

    assert result == b"post-intro"


def test_a_press_ticks_then_releases(fake_emulator) -> None:
    generate_init_state(fake_emulator, (ButtonPress(button="start", frames=3),))

    assert fake_emulator.calls == [
        ("press", "start"),
        ("tick", 3, False),
        ("release", "start"),
        ("tick", 1, False),
    ]


def test_a_wait_ticks_without_touching_any_button(fake_emulator) -> None:
    """A None button is a deliberate wait -- the intro has long unskippable
    animations, and pressing through them advances past the naming screen."""
    generate_init_state(fake_emulator, (ButtonPress(button=None, frames=60),))

    assert fake_emulator.calls == [("tick", 60, False)]


def test_the_intro_script_is_not_empty() -> None:
    """An empty script would produce a boot-screen state that loads fine and
    leaves all 64 agents stuck at the title."""
    assert len(INTRO_SCRIPT) > 0


def test_state_hash_is_stable_for_identical_bytes() -> None:
    assert state_hash(b"abc") == state_hash(b"abc")


def test_state_hash_differs_for_different_bytes() -> None:
    """Recorded in checkpoints so a resume detects that init.state changed
    underneath it -- a different starting state invalidates every reward
    baseline in the checkpoint."""
    assert state_hash(b"abc") != state_hash(b"abd")


def test_generate_init_state_rejects_a_non_positive_frame_count(fake_emulator) -> None:
    with pytest.raises(ValueError, match="frames=0 must be at least 1"):
        generate_init_state(fake_emulator, (ButtonPress(button="a", frames=0),))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_init_state.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env.init_state'`

- [ ] **Step 3: Implement `init_state.py`**

```python
"""Produces artifacts/init.state by replaying a committed button script.

Nothing ROM-derived enters git: the script is reviewable text, the ROM is the
one you already own, and the result lands in a gitignored directory. The
alternative of committing or downloading a third-party .state means shipping a
ROM memory dump with provenance we do not control; booting from scratch means
all 64 envs burn the intro, title and naming screens on every reset, learning
a button dance that has nothing to do with the task.

The script below advances past the title, intro cutscene and naming screens to
the first controllable overworld step. Frame counts are generous -- the intro
animations are long and unskippable, and overshooting a menu is far cheaper
than landing in one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pokemon_env.emulator import Emulator


@dataclass(frozen=True)
class ButtonPress:
    """`button=None` is a deliberate wait: the intro has unskippable
    animations, and pressing through them advances past the naming screen."""

    button: str | None
    frames: int


INTRO_SCRIPT: tuple[ButtonPress, ...] = (
    ButtonPress(None, 600),        # boot + Game Freak logo
    ButtonPress("start", 10),      # title screen
    ButtonPress(None, 180),
    ButtonPress("a", 10),          # NEW GAME
    ButtonPress(None, 120),
    ButtonPress("a", 10),          # through Oak's intro
    ButtonPress(None, 240),
    ButtonPress("a", 10),
    ButtonPress(None, 240),
    ButtonPress("a", 10),          # accept the default player name
    ButtonPress(None, 180),
    ButtonPress("a", 10),
    ButtonPress(None, 240),
    ButtonPress("a", 10),          # accept the default rival name
    ButtonPress(None, 600),        # the rest of the intro cutscene
    ButtonPress("a", 10),
    ButtonPress(None, 300),
)


def generate_init_state(emulator: Emulator, script: tuple[ButtonPress, ...]) -> bytes:
    """Replays `script` against a freshly booted emulator and returns the
    resulting save state."""
    for press in script:
        if press.frames < 1:
            raise ValueError(f"frames={press.frames} must be at least 1")
        if press.button is None:
            emulator.tick(press.frames, False)
            continue
        emulator.button_press(press.button)
        emulator.tick(press.frames, False)
        emulator.button_release(press.button)
        emulator.tick(1, False)
    return emulator.save_state()


def state_hash(state: bytes) -> str:
    """Recorded in checkpoints so a resume detects that init.state changed
    underneath it. A different starting state invalidates every reward
    baseline the checkpoint holds."""
    return hashlib.sha256(state).hexdigest()
```

- [ ] **Step 4: Add the `artifacts/` gitignore entry**

Append to `.gitignore`:

```
# Generated emulator save states. ROM-derived, so never committed -- see
# src/pokemon_env/init_state.py.
artifacts/
```

- [ ] **Step 5: Add the slow generation test**

Append to `tests/integration/test_pokemon_env_smoke.py`:

```python
@_needs_rom
def test_generated_init_state_leaves_the_player_in_the_overworld() -> None:
    """The script's frame counts are guesses until this runs. If the agent is
    still in a menu, every one of 64 envs starts every episode in that menu.
    party_size 0 and a non-zero map id is the post-intro, pre-starter state."""
    from pokemon_env import ram
    from pokemon_env.init_state import INTRO_SCRIPT, generate_init_state

    emulator = PyBoyEmulator(str(_ROM))
    state = generate_init_state(emulator, INTRO_SCRIPT)
    emulator.load_state(state)
    map_id = emulator.read_memory(ram.MAP_ID_ADDR)
    in_battle = ram.in_battle(emulator)
    emulator.close()

    assert (map_id != 0, in_battle) == (True, False)
```

- [ ] **Step 6: Generate the real state and run everything**

```bash
mkdir -p artifacts
uv run python -c "
from pokemon_env.emulator import PyBoyEmulator
from pokemon_env.init_state import INTRO_SCRIPT, generate_init_state, state_hash
from pathlib import Path
e = PyBoyEmulator('Pokemon Red.gb')
s = generate_init_state(e, INTRO_SCRIPT)
e.close()
Path('artifacts/init.state').write_bytes(s)
print('bytes:', len(s), 'sha256:', state_hash(s)[:16])
"
uv run pytest tests/unit/test_pokemon_env_init_state.py -q --no-cov
uv run pytest -m slow tests/integration/test_pokemon_env_smoke.py -q
```

Expected: unit tests PASS (7). If
`test_generated_init_state_leaves_the_player_in_the_overworld` fails, the
frame counts need adjusting — tune `INTRO_SCRIPT` until it passes and report
the final script. **Do not weaken the assertion to make it pass.**

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(env): generate init.state from a committed button script

Nothing ROM-derived enters git: the script is reviewable text, the ROM is the
one you already own, and the output lands in a gitignored artifacts/ dir.
The state's hash is recorded in checkpoints so a resume detects that the
starting state changed underneath it, which would invalidate every reward
baseline the checkpoint holds."
```

---

## Task 10: Telemetry

**Files:**
- Create: `src/pokemon_env/telemetry.py`
- Test: `tests/unit/test_pokemon_env_telemetry.py`

**Interfaces:**
- Consumes: `VecStep` (Task 6), `ram.coord_key` (Task 2).
- Produces: `contact_sheet(frames: np.ndarray) -> np.ndarray`, `exploration_heatmap(coord_keys, height=256, width=256) -> np.ndarray`, `rollout_metrics(step, components, clip_fire_rate, respawns) -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_pokemon_env_telemetry.py`:

```python
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


def test_contact_sheet_tiles_64_frames_into_an_8_by_8_grid() -> None:
    frames = np.zeros((64, 1, 144, 160), dtype=np.uint8)

    sheet = contact_sheet(frames)

    assert sheet.shape == (8 * 144, 8 * 160)


def test_contact_sheet_places_the_first_frame_top_left() -> None:
    """A transposed tiling would put env 0 somewhere else, which makes the
    sheet useless for spotting which env is stuck."""
    frames = np.zeros((4, 1, 144, 160), dtype=np.uint8)
    frames[0] = 200

    sheet = contact_sheet(frames)

    assert int(sheet[0, 0]) == 200


def test_contact_sheet_pads_a_non_square_batch() -> None:
    frames = np.zeros((3, 1, 144, 160), dtype=np.uint8)

    sheet = contact_sheet(frames)

    assert sheet.shape == (2 * 144, 2 * 160)


def test_exploration_heatmap_marks_a_visited_coordinate() -> None:
    heatmap = exploration_heatmap([ram.coord_key(x=10, y=20, map_id=3)], height=64, width=64)

    assert int(heatmap.sum()) > 0


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
    )

    assert metrics["reward/explore"] == pytest.approx(0.3)


def test_rollout_metrics_reports_mean_reward() -> None:
    metrics = rollout_metrics(
        _vec_step(4, reward=0.25), components={}, clip_fire_rate=0.0, respawns=0
    )

    assert metrics["reward/mean"] == pytest.approx(0.25)


def test_rollout_metrics_surfaces_the_clip_fire_rate() -> None:
    """Above roughly 0.1% the weights are miscalibrated and achievement
    ordering is being flattened."""
    metrics = rollout_metrics(
        _vec_step(4), components={}, clip_fire_rate=0.02, respawns=0
    )

    assert metrics["env/clip_fire_rate"] == pytest.approx(0.02)


def test_rollout_metrics_surfaces_worker_respawns() -> None:
    """A rising respawn rate is a leading indicator of memory pressure or a
    bad state, long before it shows in reward."""
    metrics = rollout_metrics(
        _vec_step(4), components={}, clip_fire_rate=0.0, respawns=3
    )

    assert metrics["env/worker_respawns"] == pytest.approx(3.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_telemetry.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env.telemetry'`

- [ ] **Step 3: Implement `telemetry.py`**

```python
"""Structured metrics and the two visual artifacts a human can sanity-check
without reading logs.

Per CLAUDE.md: every pipeline component emits JSON-lines progress and a live
W&B run, and anything that discards data says why -- here that is the
clip-fire rate, since clipping is the one place this component drops signal."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from pokemon_env.vec_env import VecStep


def contact_sheet(frames: np.ndarray) -> np.ndarray:
    """(N, 1, 144, 160) uint8 -> one tiled grayscale image.

    The fastest way to see that all 64 agents are stuck in the same menu.
    Grid is the smallest square that fits N; unused tiles stay black."""
    n_envs, _, height, width = frames.shape
    side = math.ceil(math.sqrt(n_envs))
    sheet = np.zeros((side * height, side * width), dtype=np.uint8)
    for i in range(n_envs):
        row, column = divmod(i, side)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = frames[
            i, 0
        ]
    return sheet


def exploration_heatmap(
    coord_keys: Iterable[int], height: int = 256, width: int = 256
) -> np.ndarray:
    """Visit counts over all envs, projected into one image.

    coord_key packs (map_id << 16) | (x << 8) | y. Maps are laid out on a grid
    by id rather than by true world position -- the reference implementation's
    global_map.py has the real projection, and swapping it in later changes
    only this function."""
    heatmap = np.zeros((height, width), dtype=np.uint32)
    maps_per_row = max(width // 16, 1)
    for key in coord_keys:
        map_id = (key >> 16) & 0xFF
        x = (key >> 8) & 0xFF
        y = key & 0xFF
        origin_row = (map_id // maps_per_row) * 16
        origin_column = (map_id % maps_per_row) * 16
        row = (origin_row + y % 16) % height
        column = (origin_column + x % 16) % width
        heatmap[row, column] += 1
    return heatmap


def rollout_metrics(
    step: VecStep,
    components: dict[str, float],
    clip_fire_rate: float,
    respawns: int,
) -> dict[str, float]:
    """Flat scalar dict, ready for wandb.log and for a JSON-lines record."""
    metrics = {
        "reward/mean": float(step.reward.mean()),
        "reward/max": float(step.reward.max()),
        "env/clip_fire_rate": float(clip_fire_rate),
        "env/worker_respawns": float(respawns),
        "env/episodes_finished": float(step.done.sum()),
    }
    for name, value in components.items():
        metrics[f"reward/{name}"] = float(value)
    return metrics
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_pokemon_env_telemetry.py -q --no-cov`
Expected: PASS (9 tests)

- [ ] **Step 5: Prove a test can fail, then commit**

Change `side = math.ceil(math.sqrt(n_envs))` to `side = n_envs`, confirm
`test_contact_sheet_tiles_64_frames_into_an_8_by_8_grid` and
`test_contact_sheet_pads_a_non_square_batch` both go red, revert. Name them in
your report.

```bash
git add -A
git commit -m "feat(env): rollout metrics, contact sheet, exploration heatmap

The heatmap is the single most informative artifact this project produces --
it answers 'is it actually playing?' at a glance. Map placement is a grid by
id for now; swapping in the reference implementation's true world projection
changes only exploration_heatmap."
```

---

## Task 11: End-to-end random-agent integration test

This is the deliverable that makes Sub-project A verifiable on its own, which
is the entire reason the spec split it from the PPO trainer.

**Files:**
- Modify: `tests/integration/test_pokemon_env_smoke.py`

**Interfaces:**
- Consumes: everything from Tasks 1–10.
- Produces: nothing — this is the acceptance gate.

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_pokemon_env_smoke.py`:

```python
_needs_init_state = pytest.mark.skipif(
    not Path("artifacts/init.state").exists(),
    reason="artifacts/init.state not generated; see src/pokemon_env/init_state.py",
)


@_needs_rom
@_needs_init_state
def test_a_random_agent_drives_four_real_envs_end_to_end(tmp_path) -> None:
    """The acceptance gate for sub-project A: four real PyBoy processes, real
    frames through shared memory, real RAM reads, real rewards -- driven by a
    random policy, with no PPO anywhere. Four envs rather than 64 so the test
    is minutes, not hours; the vectorization logic is identical.

    Also writes a contact sheet so a human can look at what the agents saw."""
    import numpy as np

    from pokemon_env.config import EnvConfig
    from pokemon_env.subprocess_backend import build_subprocess_vec_env
    from pokemon_env.telemetry import contact_sheet, rollout_metrics

    config = EnvConfig(n_envs=4, max_steps=64)
    vec_env, buffer = build_subprocess_vec_env(config)
    generator = np.random.default_rng(0)
    try:
        vec_env.reset()
        step = vec_env.step(generator.integers(0, 7, size=4))
        for _ in range(31):
            step = vec_env.step(generator.integers(0, 7, size=4))
        metrics = rollout_metrics(step, vec_env.last_components, vec_env.clip_fire_rate, 0)
        sheet = contact_sheet(step.frames)
        np.save(tmp_path / "contact_sheet.npy", sheet)
    finally:
        vec_env.close()
        buffer.close()
        buffer.unlink()

    assert (step.frames.shape, step.aux.shape) == ((4, 1, 144, 160), (4, 32))
    assert bool(((step.aux >= -1.0) & (step.aux <= 1.0)).all()) is True
    assert metrics["reward/mean"] >= 0.0
```

Note: this test body contains a `for` loop, which the audit's `BRANCHING`
check flags. Extract it into a module-level helper before running the audit:

```python
def _drive_random_steps(vec_env, generator, steps: int):
    """Helper, not a test: drives `steps` random actions and returns the last
    VecStep. Lives at module level so no loop sits in a test body."""
    step = None
    for _ in range(steps):
        step = vec_env.step(generator.integers(0, 7, size=vec_env.n_envs))
    return step
```

and call `step = _drive_random_steps(vec_env, generator, 32)`.

- [ ] **Step 2: Run it**

Run: `uv run pytest -m slow tests/integration/test_pokemon_env_smoke.py -v`
Expected: all slow tests PASS. This exercises spawn, shared memory, the full
observation path and the reward accumulator against the real game.

If it hangs, the 60 s worker timeout will surface it as a `TimeoutError`
naming the env index — that is the failure mode the timeout exists for.

- [ ] **Step 3: Run the full suite, the audit, and commit**

```bash
uv run pytest -q
uv run python ~/.claude/skills/pytest-expert/scripts/audit_tests.py tests/
```

Expected: full suite PASS with coverage at or above 93; audit reports **no
more than the 11 pre-existing findings**.

```bash
git add -A
git commit -m "test(env): random-agent end-to-end acceptance gate

Four real PyBoy processes, real frames through shared memory, real RAM reads
and rewards, driven by a random policy with no PPO anywhere. This is the
deliverable that makes sub-project A verifiable on its own, which is the
reason the spec split it from the trainer."
```

---

## Task 12: Wire the environment checkpoint into `checkpointing.io`

**Files:**
- Create: `src/pokemon_env/checkpoint.py`
- Test: `tests/unit/test_pokemon_env_checkpoint.py`

**Interfaces:**
- Consumes: `checkpointing.io.save_checkpoint/load_checkpoint` (existing), `VecPokemonEnv.state_dict()` (Task 6), `init_state.state_hash` (Task 9).
- Produces: `ENV_CHECKPOINT_PATTERN: str = "env_update*.pt"`, `build_env_checkpoint_state(update, vec_env, init_state_hash) -> dict`, `restore_env_checkpoint(vec_env, state, init_state_hash) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_pokemon_env_checkpoint.py`:

```python
import pytest

from checkpointing.io import find_latest_checkpoint, load_checkpoint, save_checkpoint
from pokemon_env.checkpoint import (
    ENV_CHECKPOINT_PATTERN,
    build_env_checkpoint_state,
    restore_env_checkpoint,
)
from pokemon_env.config import EnvConfig
from pokemon_env.session import EnvSession
from pokemon_env.vec_env import InProcessBackend, VecPokemonEnv

from .fakes import FakeEmulator


def _vec_env() -> VecPokemonEnv:
    """Helper, not a test."""
    config = EnvConfig(n_envs=2, max_steps=8)
    return VecPokemonEnv(
        [
            InProcessBackend(EnvSession(FakeEmulator(), config, init_state=b"init"))
            for _ in range(2)
        ],
        config,
    )


def test_state_records_the_update_number() -> None:
    vec_env = _vec_env()
    vec_env.reset()

    state = build_env_checkpoint_state(update=5, vec_env=vec_env, init_state_hash="abc")

    assert state["update"] == 5


def test_restore_rejects_a_changed_init_state() -> None:
    """A different starting state invalidates every reward baseline in the
    checkpoint -- the max_historical values describe progress from a game
    position the new state does not share."""
    vec_env = _vec_env()
    vec_env.reset()
    state = build_env_checkpoint_state(update=1, vec_env=vec_env, init_state_hash="abc")

    with pytest.raises(ValueError, match="init.state changed"):
        restore_env_checkpoint(_vec_env(), state, init_state_hash="def")


def test_restore_round_trips_through_the_shared_checkpoint_io(tmp_path) -> None:
    vec_env = _vec_env()
    vec_env.reset()
    state = build_env_checkpoint_state(update=3, vec_env=vec_env, init_state_hash="abc")
    path = tmp_path / "env_update00000003.pt"

    save_checkpoint(path, state)
    restored = _vec_env()
    restore_env_checkpoint(restored, load_checkpoint(path), init_state_hash="abc")

    assert restored.state_dict()["schema_version"] == state["env"]["schema_version"]


def test_the_checkpoint_pattern_does_not_collide_with_the_other_runs(tmp_path) -> None:
    """The PPO run, the policy checkpoints and the pretraining run may share
    one network volume. A pattern that globbed the others would prune their
    resume points."""
    (tmp_path / "env_update00000100.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000900.pt").write_bytes(b"")

    latest = find_latest_checkpoint(tmp_path, pattern=ENV_CHECKPOINT_PATTERN)

    assert latest == tmp_path / "env_update00000100.pt"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_pokemon_env_checkpoint.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pokemon_env.checkpoint'`

- [ ] **Step 3: Implement `checkpoint.py`**

```python
"""Environment-side checkpoint schema.

Owns what a resumable env needs; the file I/O underneath (atomic write,
discovery, retention) is checkpointing.io, shared with contrastive_pretrain
and sequence_model.

The filename pattern is distinct so a PPO run, its policy checkpoints and the
pretraining run can share one network volume without pruning each other's
resume points -- which is why checkpointing.io takes the glob as a parameter."""

from __future__ import annotations

from pokemon_env.aux_state import AUX_STATE_VERSION
from pokemon_env.vec_env import VecPokemonEnv

ENV_CHECKPOINT_PATTERN = "env_update*.pt"


def build_env_checkpoint_state(
    update: int, vec_env: VecPokemonEnv, init_state_hash: str
) -> dict:
    return {
        "update": update,
        "aux_state_version": AUX_STATE_VERSION,
        "init_state_hash": init_state_hash,
        "env": vec_env.state_dict(),
    }


def restore_env_checkpoint(
    vec_env: VecPokemonEnv, state: dict, init_state_hash: str
) -> None:
    """`vec_env.load_state_dict` already rejects an AUX_STATE_VERSION or
    env-count mismatch. The one thing only this layer can see is the starting
    state: a changed init.state invalidates every max_historical baseline in
    the checkpoint, because those describe progress from a game position the
    new state does not share."""
    if state["init_state_hash"] != init_state_hash:
        raise ValueError(
            f"init.state changed since this checkpoint was written "
            f"({state['init_state_hash'][:16]} -> {init_state_hash[:16]}). Every reward "
            "baseline in it describes progress from a starting position this run does "
            "not share; regenerate the checkpoint or restore the original init.state."
        )
    vec_env.load_state_dict(state["env"])
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/test_pokemon_env_checkpoint.py -q --no-cov`
Expected: PASS (4 tests)

- [ ] **Step 5: Prove a test can fail, run everything, commit**

Delete the `init_state_hash` comparison, confirm
`test_restore_rejects_a_changed_init_state` goes red, revert. Name it in your
report.

```bash
uv run pytest -q
uv run python ~/.claude/skills/pytest-expert/scripts/audit_tests.py tests/
git add -A
git commit -m "feat(env): environment checkpoint schema over the shared IO

Reuses checkpointing.io for atomic writes, discovery and retention, with a
distinct filename glob so this run does not prune the policy or pretraining
checkpoints sharing the volume.

Validates the init.state hash, which is the one thing no lower layer can see:
a changed starting state invalidates every max_historical baseline in the
checkpoint."
```

---

## Definition of done

- [ ] `uv run pytest -q` passes with branch coverage at or above 93.
- [ ] `uv run pytest -m slow tests/integration/test_pokemon_env_smoke.py` passes with the real ROM.
- [ ] `audit_tests.py tests/` reports no more than the 11 pre-existing findings.
- [ ] The measured PyBoy `save_state` size is recorded in the Task 1 report.
- [ ] For every task, the report names at least one test verified to fail when its code is broken.
- [ ] No `.gb` or `.state` file is tracked by git: `git ls-files | grep -Ei '\.(gb|gbc|state)$'` returns nothing.

## Handoff to Sub-project B (the PPO trainer)

Carried forward verbatim from the spec, because these are the requirements
most likely to be lost at the seam:

1. **Recompute `π_old` and `V_old`** with one `no_grad` `forward_chunk` pass at
   update start, as the importance-ratio denominator and the GAE baseline —
   never the rollout-recorded values. The KV cache is carried across update
   boundaries, so the behaviour policy is not exactly `π_θ_old`.
2. **Log `max|ratio − 1|` at epoch 1 as a hard invariant** — exactly 0 after
   the above; anything else is a real bug.
3. **Save the KV cache and the emulator state together**, per the size
   measurement in Task 1. Neither is coherent alone.
4. `n_steps`, `n_epochs`, `γ`, `ent_coef`, clip range, GAE `λ`. Note
   `n_epochs=1` makes the epoch-1 ratio the *only* ratio, raising the stakes on
   item 1 rather than lowering them.
5. **The CUDA SDPA backend measurement** — a gate before the first paid GPU hour.
