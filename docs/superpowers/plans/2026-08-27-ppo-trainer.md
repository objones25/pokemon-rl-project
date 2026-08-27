# PPO Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/ppo/`, a PPO trainer that drives the merged
`sequence_model` policy against the merged `pokemon_env` on a CUDA pod, with
resume, telemetry, and four measured pre-flight gates.

**Architecture:** A rolling GPU-resident buffer holds `burn_in + n_steps + 1`
frame latents per env. Each update recomputes `π_old` and `V_old` with one
`no_grad` `forward_chunk` sweep, runs GAE off `V_old`, then takes one optimizer
step per env-minibatch across `n_epochs`. Policy and env checkpoints are written
as a pair committed by a manifest file.

**Tech Stack:** PyTorch 2.13 (hand-rolled PPO — `stable-baselines3` is *not* a
dependency), PyBoy behind a Protocol, Hugging Face Hub for the frozen encoder,
W&B plus JSON-lines for observability, pytest 9.1.1, `uv`.

**Spec:** `docs/superpowers/specs/2026-08-27-ppo-trainer-design.md`

## Global Constraints

Copied from the spec and `CLAUDE.md`. Every task's requirements include these.

- **Package management is `uv` only.** `uv add <pkg>`, `uv run <cmd>`. Never bare `pip` or `venv`.
- **Branch coverage floor is 93%** and is ratcheted upward, never lowered. No `omit`, no `# pragma: no cover`.
- `uv run python scripts/audit_tests.py tests/` (pytest-expert skill) must stay **at or below the 11-finding pre-existing baseline**.
- `audit_observability.py src/` must reach **9 findings** (down from 12) by the end of Task 5, and never rise.
- `uv run ruff check` must be clean.
- **Prove every new test can fail.** Break the code it covers, confirm red, revert, and name the verified test in the task report. A test that passes against broken code is decorative and does not count.
- **CUDA on RunPod is the optimization target.** MPS/CPU paths exist only so tests run locally; never degrade the CUDA path for the dev machine.
- **Tests must never load `.env`.** Live-tier tests read the ambient credential through `requires_hf_credentials` in `tests/conftest.py`.
- Never commit ROMs (`*.gb`, `*.gbc`), `artifacts/` save states, or `.env`. The GitHub remote is public.
- **Heads emit raw logits.** No softmax/sigmoid before a loss that fuses it.
- **Step order is `zero_grad` → forward → loss → backward → step.**
- Checkpoints save `state_dict`, never module objects, and load with `weights_only=True`.
- `LatentEncoder.encode` and `RecurrentTransformerPolicy.step` use `@torch.no_grad()`, **never** `@torch.inference_mode()` — their outputs enter an autograd graph at the update.
- `max_grad_norm = 0.5` (SB3's PPO default), deliberately not `CLAUDE.md`'s transformer default of 1.0.
- **Test doubles are hand-written fakes** typed against the Protocol the consumer uses (`tests/unit/fakes.py`), not `mock.patch` stubs.
- **Test hard gates:** one behavior per test; long declarative names; no `if`/`for`/`while` in a test body (use `parametrize`); every test asserts an exact expected value; floats via `pytest.approx`; `pytest.raises` always names a specific exception and passes `match=`; `skip`/`xfail` always carry `reason=`.
- `src/pokemon_env/` must never know about PPO, advantages, or losses.
- `src/sequence_model/` must never contain a training loop.

---

## File Structure

**Created — `src/ppo/`:**

| File | Responsibility |
|---|---|
| `__init__.py` | Empty package marker |
| `config.py` | `PPOConfig` frozen dataclass, `load_config` |
| `buffer.py` | `RolloutBuffer` — rolling GPU storage, chunk assembly |
| `normalizer.py` | `ReturnScaler` — running std of discounted returns |
| `gae.py` | `compute_gae` — pure tensor function |
| `losses.py` | `ppo_losses` — pure; clipped policy, value, entropy |
| `rollout.py` | `collect_rollout` — env → encoder → policy → buffer |
| `update.py` | `run_update` — `π_old` pass, GAE, minibatch loop |
| `checkpoint.py` | Manifest-committed paired save/load/prune/resume |
| `telemetry.py` | Per-update scalar aggregation and artifacts |
| `trainer.py` | `PPODeps`, `run_training` — outer loop and failure handling |
| `preflight.py` | The four gates |
| `cli.py` | `pokemon-ppo` entrypoint |

**Modified:**

| File | Change |
|---|---|
| `src/pokemon_env/config.py` | Delete `seed`, `frozen_encoder_repo_id` |
| `src/pokemon_env/rewards.py` | Add `coord_keys()` |
| `src/pokemon_env/session.py` | Episode-length history, `stats()` |
| `src/pokemon_env/subprocess_backend.py` | `Command.STATS`, `logger.exception`, obs markers |
| `src/pokemon_env/vec_env.py` | `EnvBackend.stats()`, `VecPokemonEnv.stats()`, schema bump |
| `src/pokemon_env/telemetry.py` | Heatmap reprojection, `rollout_metrics` extension |
| `src/observability/tracking.py` | `WandbRun` config/x-axes/resume/context-manager |
| `src/sequence_model/policy.py` | Add `diagnostics()` |
| `pyproject.toml` | `src/ppo` package, `pokemon-ppo` script |
| `configs/ppo.yaml` | New config file |

**Tests created:** one `tests/unit/test_ppo_<module>.py` per new module, plus
`tests/integration/test_ppo_smoke.py` for the slow tier.

---

### Task 1: `PPOConfig`, and delete the two dead `EnvConfig` fields

**Files:**
- Create: `src/ppo/__init__.py`, `src/ppo/config.py`, `configs/ppo.yaml`
- Modify: `src/pokemon_env/config.py`, `pyproject.toml`
- Test: `tests/unit/test_ppo_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PPOConfig` (frozen dataclass) and `load_config(path) -> PPOConfig`. Every later task imports `from ppo.config import PPOConfig`. `PPOConfig.burn_in(context_len)` and `PPOConfig.buffer_capacity(context_len)` are the derived-shape helpers Tasks 6 and 11 rely on.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ppo_config.py`:

```python
"""PPOConfig loading and validation."""

from __future__ import annotations

import pytest

from ppo.config import PPOConfig, load_config


def test_load_config_reads_every_field_from_yaml(tmp_path) -> None:
    path = tmp_path / "ppo.yaml"
    path.write_text("n_steps: 512\nn_epochs: 2\nfrozen_encoder_revision: abc123\n")

    config = load_config(path)

    assert (config.n_steps, config.n_epochs, config.frozen_encoder_revision) == (512, 2, "abc123")


def test_load_config_rejects_an_unknown_field(tmp_path) -> None:
    path = tmp_path / "ppo.yaml"
    path.write_text("frozen_encoder_revision: abc123\nnot_a_field: 3\n")

    with pytest.raises(ValueError, match="unknown config field"):
        load_config(path)


def test_config_rejects_a_missing_frozen_encoder_revision() -> None:
    with pytest.raises(ValueError, match="frozen_encoder_revision must be pinned"):
        PPOConfig()


def test_config_rejects_an_n_envs_not_divisible_by_minibatch_envs() -> None:
    with pytest.raises(ValueError, match="minibatch_envs=7 does not divide"):
        PPOConfig(frozen_encoder_revision="abc123", minibatch_envs=7).validate_against_n_envs(64)


def test_burn_in_is_one_less_than_the_context_length() -> None:
    config = PPOConfig(frozen_encoder_revision="abc123")

    assert config.burn_in(context_len=1024) == 1023


def test_buffer_capacity_is_burn_in_plus_n_steps_plus_one_bootstrap_slot() -> None:
    config = PPOConfig(frozen_encoder_revision="abc123", n_steps=1024)

    assert config.buffer_capacity(context_len=1024) == 2048
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppo'`.

- [ ] **Step 3: Create the package and the config module**

Create `src/ppo/__init__.py` as an empty file.

Create `src/ppo/config.py`:

```python
"""PPO trainer configuration, loaded from configs/ppo.yaml.

Mirrors pokemon_env.config's dataclass + yaml.safe_load pattern: frozen
dataclass, unknown-field rejection, validation in __post_init__.

Shape helpers live here rather than as constants because a later
curriculum stage raises context_len and n_steps together; nothing in the
trainer may hard-code 1024 or 2048."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PPOConfig:
    n_steps: int = 1024
    n_epochs: int = 3
    minibatch_envs: int = 8
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    # SB3's PPO default, deliberately not CLAUDE.md's transformer default of
    # 1.0 -- that value is for language-model pretraining.
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    warmup_steps: int = 100
    abort_approx_kl: float = 0.5
    max_nan_minibatches_per_update: int = 3
    seed: int = 0
    frozen_encoder_repo_id: str = "objones25/pokemon-contrastive-encoder"
    frozen_encoder_revision: str | None = None
    checkpoint_dir: str = "/workspace/checkpoints"
    keep_last_n: int = 3
    checkpoint_every_updates: int = 25
    artifact_every_updates: int = 25
    hub_snapshot_every_updates: int = 75
    diagnostics_layer: int = -1

    def __post_init__(self) -> None:
        if self.frozen_encoder_revision is None:
            raise ValueError(
                "frozen_encoder_revision must be pinned to a resolved commit. An "
                "unpinned revision lets a mid-run push to the encoder repo change "
                "the features underneath a running agent, with nothing raised."
            )
        if self.n_steps < 1:
            raise ValueError(f"n_steps={self.n_steps} must be at least 1")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs={self.n_epochs} must be at least 1")
        if self.minibatch_envs < 1:
            raise ValueError(f"minibatch_envs={self.minibatch_envs} must be at least 1")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma={self.gamma} must lie in (0, 1)")

    def validate_against_n_envs(self, n_envs: int) -> None:
        """n_envs lives on EnvConfig, so the divisibility check cannot run in
        __post_init__. The trainer calls this once at startup."""
        if n_envs % self.minibatch_envs:
            raise ValueError(
                f"minibatch_envs={self.minibatch_envs} does not divide n_envs={n_envs}; "
                "a ragged final minibatch would change the effective batch size of one "
                "optimizer step per epoch"
            )

    def burn_in(self, context_len: int) -> int:
        """The minimum prefix giving every trained position a full context
        window. Any smaller and the first trained positions see less context
        than the model was sized for."""
        return context_len - 1

    def buffer_capacity(self, context_len: int) -> int:
        """burn-in + trained region + one bootstrap slot for V(s_T)."""
        return self.burn_in(context_len) + self.n_steps + 1


def load_config(path: str | Path) -> PPOConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    valid_fields = {f.name for f in fields(PPOConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")
    return PPOConfig(**data)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_config.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Delete the two dead `EnvConfig` fields**

In `src/pokemon_env/config.py`, delete these two lines from the `EnvConfig`
dataclass body:

```python
    frozen_encoder_repo_id: str = "objones25/pokemon-contrastive-encoder"
    seed: int = 0
```

Both moved to `PPOConfig`: the env encodes nothing, and PyBoy is deterministic
given `init.state` plus an action sequence, so the only randomness is action
sampling, which belongs to PPO.

- [ ] **Step 6: Confirm no config file or test referenced them**

Run: `uv run grep -rn "frozen_encoder_repo_id\|EnvConfig(.*seed" configs/ tests/ src/pokemon_env/`
Expected: no hits in `configs/pokemon_env.yaml` (verified — it sets neither), and
no hits in `src/pokemon_env/`. If a test constructs `EnvConfig(seed=...)`, delete
that argument.

- [ ] **Step 7: Write the config file**

Create `configs/ppo.yaml`:

```yaml
# Defaults live in PPOConfig; this file records the values this project runs
# with. gamma/ent_coef match PWhiddy/PokemonRedExperiments v2; gae_lambda,
# clip_range, vf_coef and max_grad_norm are stable-baselines3 PPO defaults.
#
# checkpoint_every_updates and hub_snapshot_every_updates are provisional:
# pre-flight gate 2 measures iteration time and replaces them. The binding
# requirement is at most 20 minutes of work lost.
n_steps: 1024
n_epochs: 3
minibatch_envs: 8
gamma: 0.997
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
lr: 0.0003
warmup_steps: 100
seed: 0
frozen_encoder_revision: main
checkpoint_dir: /workspace/checkpoints
keep_last_n: 3
checkpoint_every_updates: 25
artifact_every_updates: 25
hub_snapshot_every_updates: 75
```

- [ ] **Step 8: Register the package**

In `pyproject.toml`, add `"src/ppo"` to
`[tool.hatch.build.targets.wheel].packages`.

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest -q && uv run ruff check`
Expected: PASS, coverage at or above 93%.

- [ ] **Step 10: Commit**

```bash
git add src/ppo configs/ppo.yaml pyproject.toml tests/unit/test_ppo_config.py src/pokemon_env/config.py
git commit -m "feat(ppo): PPOConfig, and delete the two dead EnvConfig fields"
```

---

### Task 2: Env-side exploration and progress stats

**Files:**
- Modify: `src/pokemon_env/rewards.py`, `src/pokemon_env/session.py`
- Test: `tests/unit/test_pokemon_env_rewards.py`, `tests/unit/test_pokemon_env_session.py`

**Interfaces:**
- Consumes: `PPOConfig` from Task 1 (not directly used here).
- Produces: `RewardAccumulator.coord_keys() -> list[int]` and
  `EnvSession.stats() -> dict` with keys
  `{"coord_keys": list[int], "badges": int, "event_flags": int, "step_count": int, "episode_lengths": list[int]}`.
  Task 3 ships this dict across the worker pipe; Task 4 turns it into scalars.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_pokemon_env_rewards.py`:

```python
def test_coord_keys_returns_every_visited_coordinate_key() -> None:
    accumulator = RewardAccumulator(EnvConfig())
    emulator = FakeEmulator()
    accumulator.reset(emulator)
    emulator.write(ram.X_POS_ADDR, 3)
    emulator.write(ram.Y_POS_ADDR, 4)
    emulator.write(ram.MAP_ID_ADDR, 38)
    accumulator.step(emulator)

    assert accumulator.coord_keys() == [ram.coord_key(3, 4, 38)]
```

Append to `tests/unit/test_pokemon_env_session.py`:

```python
def test_stats_reports_the_coordinate_keys_the_accumulator_has_seen() -> None:
    session = EnvSession(FakeEmulator(), EnvConfig(), init_state=b"")
    session.reset()
    session.step(0)

    assert session.stats()["coord_keys"] == [ram.coord_key(0, 0, 0)]


def test_stats_reports_the_length_of_a_completed_episode() -> None:
    session = EnvSession(FakeEmulator(), EnvConfig(max_steps=2), init_state=b"")
    session.reset()
    session.step(0)
    session.step(0)
    session.reset()

    assert session.stats()["episode_lengths"] == [2]


def test_stats_drains_the_episode_length_history_so_lengths_are_not_double_counted() -> None:
    session = EnvSession(FakeEmulator(), EnvConfig(max_steps=2), init_state=b"")
    session.reset()
    session.step(0)
    session.step(0)
    session.reset()
    session.stats()

    assert session.stats()["episode_lengths"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_pokemon_env_rewards.py tests/unit/test_pokemon_env_session.py -v`
Expected: FAIL with `AttributeError: 'RewardAccumulator' object has no attribute 'coord_keys'`
and `AttributeError: 'EnvSession' object has no attribute 'stats'`.

- [ ] **Step 3: Add `coord_keys` to `RewardAccumulator`**

In `src/pokemon_env/rewards.py`, beside the existing `coords_seen` property:

```python
    def coord_keys(self) -> list[int]:
        """The packed coordinate keys themselves, not just the count.

        Sorted so the value is stable across runs -- the set's iteration
        order is not, and an unstable order would make the exploration
        heatmap artifact differ between two runs over identical states.
        A method rather than a property because it materializes a list
        that can reach tens of thousands of entries."""
        return sorted(self._state.seen_coords)
```

- [ ] **Step 4: Add episode-length tracking and `stats` to `EnvSession`**

In `src/pokemon_env/session.py`, add to `__init__`:

```python
        self._episode_lengths: list[int] = []
```

In `reset`, record the length of the episode that just ended, before the
counter is cleared:

```python
    def reset(self) -> StepResult:
        # Recorded before _step_count is cleared. The first reset has nothing
        # to record: _episode_id is still -1, meaning no episode has run.
        if self._episode_id >= 0:
            self._episode_lengths.append(self._step_count)
        self._emulator.load_state(self._init_state)
        self._rewards.reset(self._emulator)
        self._step_count = 0
        self._episode_id += 1
        return self._observe(reward=0.0, clipped=False, components={})
```

Add the accessor:

```python
    def stats(self) -> dict:
        """Telemetry the parent process cannot compute for itself, gathered in
        one call so PPO makes one round trip per update rather than per step.

        Draining `episode_lengths` is deliberate: the caller aggregates across
        updates, and returning the full history every call would double-count
        every episode in every later update."""
        lengths = self._episode_lengths
        self._episode_lengths = []
        return {
            "coord_keys": self._rewards.coord_keys(),
            "badges": ram.badge_count(self._emulator),
            "event_flags": ram.event_flag_count(self._emulator),
            "step_count": self._step_count,
            "episode_lengths": lengths,
        }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_pokemon_env_rewards.py tests/unit/test_pokemon_env_session.py -v`
Expected: PASS.

- [ ] **Step 6: Prove the drain test can fail**

Change `stats` to `return {... "episode_lengths": self._episode_lengths}` without
clearing. Run the tests; `test_stats_drains_the_episode_length_history_so_lengths_are_not_double_counted`
must go red. Revert.

- [ ] **Step 7: Commit**

```bash
git add src/pokemon_env/rewards.py src/pokemon_env/session.py tests/unit/test_pokemon_env_rewards.py tests/unit/test_pokemon_env_session.py
git commit -m "feat(env): expose coordinate keys, episode lengths, and progress stats"
```

---

### Task 3: `Command.STATS` through the worker protocol

**Files:**
- Modify: `src/pokemon_env/subprocess_backend.py`, `src/pokemon_env/vec_env.py`, `tests/unit/fakes.py`
- Test: `tests/unit/test_pokemon_env_subprocess_backend.py`, `tests/unit/test_pokemon_env_vec_env.py`

**Interfaces:**
- Consumes: `EnvSession.stats()` from Task 2.
- Produces: `EnvBackend.stats() -> dict` on the Protocol, `SubprocessBackend.stats()`, `InProcessBackend.stats()`, and `VecPokemonEnv.stats() -> list[dict]` (one dict per env, in env order). Task 4 consumes the list.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_pokemon_env_subprocess_backend.py`:

```python
def test_handle_command_returns_session_stats_for_the_stats_command() -> None:
    session = EnvSession(FakeEmulator(), EnvConfig(), init_state=b"")
    session.reset()
    frame_slot = np.zeros((144, 160), dtype=np.uint8)

    payload = handle_command(session, Command.STATS, None, frame_slot)

    assert payload["stats"]["step_count"] == 0


def test_handle_command_does_not_overwrite_the_frame_slot_for_stats() -> None:
    session = EnvSession(FakeEmulator(), EnvConfig(), init_state=b"")
    session.reset()
    frame_slot = np.full((144, 160), 7, dtype=np.uint8)

    handle_command(session, Command.STATS, None, frame_slot)

    assert frame_slot[0, 0] == 7
```

Append to `tests/unit/test_pokemon_env_vec_env.py`:

```python
def test_stats_returns_one_entry_per_env_in_env_order() -> None:
    backends = [FakeBackend(step_count=1), FakeBackend(step_count=2)]
    vec_env = VecPokemonEnv(backends, EnvConfig(n_envs=2))

    assert [entry["step_count"] for entry in vec_env.stats()] == [1, 2]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_pokemon_env_subprocess_backend.py tests/unit/test_pokemon_env_vec_env.py -v`
Expected: FAIL with `AttributeError: STATS`.

- [ ] **Step 3: Add the command and its handler**

In `src/pokemon_env/subprocess_backend.py`, add to `Command`:

```python
    STATS = "STATS"
```

Add a branch to `handle_command`, before the `else` that raises. It returns
early, like `STATE_DICT`, because there is no `StepResult` and therefore no
frame to write:

```python
    elif command == Command.STATS:
        return {"stats": session.stats()}
```

Add the backend method beside `state_dict`:

```python
    def stats(self) -> dict:
        """One round trip per update, not per step. The payload is ~1.3 KB per
        env, against the 168 KB a STATE_DICT round trip ships to extract the
        same coordinates."""
        return self._call(Command.STATS)["stats"]
```

- [ ] **Step 4: Add `stats` to the Protocol and both backends**

In `src/pokemon_env/vec_env.py`, add to the `EnvBackend` Protocol:

```python
    def stats(self) -> dict: ...
```

Add to `InProcessBackend`:

```python
    def stats(self) -> dict:
        return self._session.stats()
```

Add to `VecPokemonEnv`:

```python
    def stats(self) -> list[dict]:
        """One dict per env, in env order. Called once per PPO update."""
        return [backend.stats() for backend in self._backends]
```

- [ ] **Step 5: Update the test fake**

In `tests/unit/fakes.py`, add `stats` to whatever fake implements `EnvBackend`.
The Protocol now requires it, so a fake without it stops satisfying the type:

```python
    def stats(self) -> dict:
        return {
            "coord_keys": list(self._coord_keys),
            "badges": self._badges,
            "event_flags": self._event_flags,
            "step_count": self._step_count,
            "episode_lengths": [],
        }
```

Add `coord_keys`, `badges`, `event_flags`, and `step_count` as constructor
keyword arguments defaulting to `()`, `0`, `0`, and `0`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_pokemon_env_subprocess_backend.py tests/unit/test_pokemon_env_vec_env.py -v`
Expected: PASS.

- [ ] **Step 7: Stop the worker's traceback dying in the worker**

At `src/pokemon_env/subprocess_backend.py` line 157, the worker forwards
`"TypeName: msg"` to the parent but the traceback never leaves the worker. Add a
log line before the send:

```python
            except Exception as error:  # noqa: BLE001 -- must reach the parent, not die silently
                logger.exception("worker_command_failed", extra={"command": str(command)})
                conn.send(("error", f"{type(error).__name__}: {error}"))
```

Add `logger = logging.getLogger(__name__)` at module level if it is not already
present, with `import logging`.

- [ ] **Step 8: Mark the two benign shutdown excepts**

The `except OSError: pass` in `_close_connection` and the
`except (BrokenPipeError, OSError): pass` in `close` are closing an
already-closed pipe and talking to an already-dead worker. Both are deliberate.
Add the audit's explicit allow marker so they stop reading as oversights:

```python
        except OSError:  # obs: allow LOG007 -- the pipe is already closed; nothing to report
            pass
```

```python
        except (BrokenPipeError, OSError):  # obs: allow LOG007 -- worker already gone at shutdown
            pass
```

- [ ] **Step 9: Run the observability audit**

Run: `uv run python ~/.claude/skills/observability-expert/scripts/audit_observability.py src/`
Expected: 10 findings (`LOG004:5, LOG006:2, LOG007:3` — the three
`subprocess_backend` entries are resolved; `tracking.py`'s two remain until
Task 5).

- [ ] **Step 10: Run the full suite and commit**

```bash
uv run pytest -q && uv run ruff check
git add src/pokemon_env/subprocess_backend.py src/pokemon_env/vec_env.py tests/
git commit -m "feat(env): STATS worker command, and stop losing the worker's traceback"
```

---

### Task 4: Heatmap reprojection and the missing rollout scalars

**Files:**
- Modify: `src/pokemon_env/telemetry.py`
- Test: `tests/unit/test_pokemon_env_telemetry.py`

**Interfaces:**
- Consumes: `VecPokemonEnv.stats()` from Task 3.
- Produces: `exploration_heatmap(coord_keys, height=256, width=256) -> np.ndarray` (signature unchanged, projection changed) and `rollout_metrics(step, components, clip_fire_rate, respawns, stats) -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_pokemon_env_telemetry.py`:

```python
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
```

Add the module-level helper (not in a test body — the audit flags loops in
tests, and this keeps the arrange step to one line):

```python
def _vec_step(n_envs: int) -> VecStep:
    """Helper, not a test: a minimal VecStep whose reward fields are what
    rollout_metrics reads. Frame and aux contents are irrelevant here."""
    return VecStep(
        frames=np.zeros((n_envs, 1, 144, 160), dtype=np.uint8),
        aux=np.zeros((n_envs, 32), dtype=np.float32),
        reward=np.zeros(n_envs, dtype=np.float32),
        done=np.zeros(n_envs, dtype=bool),
        episode_id=np.zeros(n_envs, dtype=np.int64),
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_pokemon_env_telemetry.py -v`
Expected: FAIL — the collision test fails because both coordinates land in one
pixel, and the `rollout_metrics` tests fail with a `TypeError` on the extra
argument.

- [ ] **Step 3: Reproject the heatmap**

Replace the body of `exploration_heatmap` in `src/pokemon_env/telemetry.py`:

```python
MAP_TILE = 64
MAPS_SHOWN = 12


def exploration_heatmap(
    coord_keys: Iterable[int], height: int = 256, width: int = 256
) -> np.ndarray:
    """Visit counts over all envs, projected into one image.

    ram.coord_key packs (map_id << 16) | (x << 8) | y, each field a uint8, so
    unpacking here mirrors that exactly.

    Coordinates render at their true (x, y) inside a per-map tile. The earlier
    version folded x and y mod 16, which collided most distinct coordinates in
    the same map -- the artifact looked plausible and showed almost nothing.
    Only the top MAPS_SHOWN maps by unique-coordinate count get a tile: a
    64x64 tile holds Pokemon Red's largest map, and a 256x256 image holds
    twelve of them at 4 per row with room for the labels a caller may draw."""
    counts: dict[int, int] = {}
    for key in coord_keys:
        counts[key] = counts.get(key, 0) + 1

    unique_per_map: dict[int, int] = {}
    for key in counts:
        map_id = (key >> 16) & 0xFF
        unique_per_map[map_id] = unique_per_map.get(map_id, 0) + 1

    ranked = sorted(unique_per_map, key=lambda m: (-unique_per_map[m], m))[:MAPS_SHOWN]
    tile_index = {map_id: position for position, map_id in enumerate(ranked)}
    maps_per_row = max(width // MAP_TILE, 1)

    heatmap = np.zeros((height, width), dtype=np.uint32)
    for key, count in counts.items():
        map_id = (key >> 16) & 0xFF
        position = tile_index.get(map_id)
        if position is None:
            continue
        origin_row = (position // maps_per_row) * MAP_TILE
        origin_column = (position % maps_per_row) * MAP_TILE
        row = origin_row + min((key & 0xFF), MAP_TILE - 1)
        column = origin_column + min(((key >> 8) & 0xFF), MAP_TILE - 1)
        if row < height and column < width:
            heatmap[row, column] += count
    return heatmap
```

- [ ] **Step 4: Extend `rollout_metrics`**

Replace the function in `src/pokemon_env/telemetry.py`:

```python
def rollout_metrics(
    step: VecStep,
    components: dict[str, float],
    clip_fire_rate: float,
    respawns: int,
    stats: list[dict],
) -> dict[str, float]:
    """Flat scalar dict, ready for wandb.log and for a JSON-lines record.

    `stats` is VecPokemonEnv.stats() -- one entry per env. Unique coordinates
    are counted across the union of all envs, not summed per env: two envs that
    walked the same route have explored one route, and summing would report
    twice the exploration that happened."""
    unique_coords = {key for entry in stats for key in entry["coord_keys"]}
    unique_maps = {(key >> 16) & 0xFF for key in unique_coords}
    lengths = [length for entry in stats for length in entry["episode_lengths"]]

    metrics = {
        "reward/mean": float(step.reward.mean()),
        "reward/max": float(step.reward.max()),
        "env/clip_fire_rate": float(clip_fire_rate),
        "env/worker_respawns": float(respawns),
        "env/episodes_finished": float(step.done.sum()),
        "progress/badges_max": float(max(entry["badges"] for entry in stats)),
        "progress/badges_mean": float(
            sum(entry["badges"] for entry in stats) / len(stats)
        ),
        "progress/event_flags_max": float(max(entry["event_flags"] for entry in stats)),
        "explore/unique_coords_total": float(len(unique_coords)),
        "explore/unique_maps": float(len(unique_maps)),
        "episode/length_mean": float(sum(lengths) / len(lengths)) if lengths else 0.0,
    }
    for name, value in components.items():
        metrics[f"reward/{name}"] = float(value)
    return metrics
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_pokemon_env_telemetry.py -v`
Expected: PASS.

- [ ] **Step 6: Prove the union test can fail**

Change `unique_coords` to `sum(len(entry["coord_keys"]) for entry in stats)`.
`test_rollout_metrics_counts_unique_coordinates_across_envs_without_double_counting`
must go red (reporting 4.0 instead of 3.0). Revert.

- [ ] **Step 7: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/pokemon_env/telemetry.py tests/unit/test_pokemon_env_telemetry.py
git commit -m "fix(env): render true coordinates in the heatmap, emit the missing rollout scalars"
```

---

### Task 5: `WandbRun` — config, x-axes, resume, context manager

**Files:**
- Modify: `src/observability/tracking.py`
- Test: `tests/unit/test_observability_tracking.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `WandbRun(wandb_module, project, name, config=None, step_metrics=None, run_id=None)` with `.log`, `.finish`, `.__enter__`, `.__exit__`, and `.run_id`. `ExperimentRunLike` gains `__enter__`/`__exit__`. Task 14 and Task 15 use all of it.

- [ ] **Step 1: Write the failing tests**

Create or append to `tests/unit/test_observability_tracking.py`:

```python
class FakeWandbRun:
    """Hand-written fake for the Run object wandb.init returns."""

    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.defined: list[tuple[str, str]] = []
        self.finished_with: object = "not-finished"
        self.id = "fake-run-id"

    def log(self, metrics: dict) -> None:
        self.logged.append(metrics)

    def define_metric(self, name: str, step_metric: str) -> None:
        self.defined.append((name, step_metric))

    def finish(self, exit_code: int = 0) -> None:
        self.finished_with = exit_code


class FakeWandbModule:
    def __init__(self) -> None:
        self.run = FakeWandbRun()
        self.init_kwargs: dict = {}

    def init(self, **kwargs) -> FakeWandbRun:
        self.init_kwargs = kwargs
        return self.run


def test_the_config_reaches_wandb_init() -> None:
    module = FakeWandbModule()

    WandbRun(module, project="p", name="n", config={"lr": 0.1})

    assert module.init_kwargs["config"] == {"lr": 0.1}


def test_a_run_id_is_passed_with_resume_allow_so_a_preempted_run_continues() -> None:
    module = FakeWandbModule()

    WandbRun(module, project="p", name="n", run_id="abc")

    assert (module.init_kwargs["id"], module.init_kwargs["resume"]) == ("abc", "allow")


def test_no_resume_arguments_are_sent_when_no_run_id_is_given() -> None:
    module = FakeWandbModule()

    WandbRun(module, project="p", name="n")

    assert "resume" not in module.init_kwargs


def test_step_metrics_are_declared_as_x_axes() -> None:
    module = FakeWandbModule()

    WandbRun(module, project="p", name="n", step_metrics={"loss/*": "train/update"})

    assert module.run.defined == [("loss/*", "train/update")]


def test_log_never_passes_a_step_argument() -> None:
    """wandb drops a log whose step is below the current one, with no
    exception. The x-axis is declared instead."""
    module = FakeWandbModule()
    run = WandbRun(module, project="p", name="n")

    run.log({"loss": 1.0, "train/update": 3})

    assert module.run.logged == [{"loss": 1.0, "train/update": 3}]


def test_leaving_the_context_normally_finishes_with_exit_code_zero() -> None:
    module = FakeWandbModule()

    with WandbRun(module, project="p", name="n"):
        pass

    assert module.run.finished_with == 0


def test_an_exception_inside_the_context_marks_the_run_failed() -> None:
    module = FakeWandbModule()

    with pytest.raises(RuntimeError, match="boom"), WandbRun(module, project="p", name="n"):
        raise RuntimeError("boom")

    assert module.run.finished_with == 1


def test_null_experiment_run_is_usable_as_a_context_manager() -> None:
    """*Deps default to NullExperimentRun, so every call site that wraps the
    run in a `with` must work without a tracker configured."""
    with NullExperimentRun() as run:
        run.log({"a": 1.0})

    assert isinstance(run, NullExperimentRun)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_observability_tracking.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'config'`.

- [ ] **Step 3: Extend `WandbRun`**

Replace the class in `src/observability/tracking.py`:

```python
class ExperimentRunLike(Protocol):
    def log(self, metrics: dict) -> None: ...
    def finish(self) -> None: ...
    def __enter__(self) -> ExperimentRunLike: ...
    def __exit__(self, exc_type, exc, tb) -> None: ...


class WandbRun:
    def __init__(
        self,
        wandb_module,
        project: str,
        name: str,
        config: dict | None = None,
        step_metrics: dict[str, str] | None = None,
        run_id: str | None = None,
    ) -> None:
        """`run_id` with resume="allow" is what keeps a preempted multi-day run
        as ONE dashboard run. Without it every pod preemption starts a fresh
        run and a 48-hour curve arrives as several disconnected fragments.

        `step_metrics` maps a metric glob to its x-axis, declared once via
        define_metric, so `log` never passes `step=` -- a step below the
        current one is dropped silently."""
        kwargs: dict = {"project": project, "name": name, "config": config or {}}
        if run_id is not None:
            kwargs["id"] = run_id
            kwargs["resume"] = "allow"
        self._run = wandb_module.init(**kwargs)
        for pattern, axis in (step_metrics or {}).items():
            self._run.define_metric(pattern, step_metric=axis)

    @property
    def run_id(self) -> str:
        return str(self._run.id)

    def log(self, metrics: dict) -> None:
        try:
            self._run.log(metrics)
        except Exception:  # noqa: BLE001 -- must swallow any wandb failure, whatever its type
            logger.warning("wandb_log_failed", exc_info=True)

    def finish(self, exit_code: int = 0) -> None:
        try:
            self._run.finish(exit_code=exit_code)
        except Exception:  # noqa: BLE001 -- must swallow any wandb failure, whatever its type
            logger.warning("wandb_finish_failed", exc_info=True)

    def __enter__(self) -> WandbRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(exit_code=1 if exc_type is not None else 0)


class NullExperimentRun:
    """No-op ExperimentRunLike, used as *Deps.wandb_run's default so
    callers never need to special-case "no experiment tracker configured"."""

    def log(self, metrics: dict) -> None:
        pass

    def finish(self, exit_code: int = 0) -> None:
        pass

    def __enter__(self) -> NullExperimentRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_observability_tracking.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm the existing call sites still work**

Run: `uv run pytest tests/unit/test_pipeline.py tests/unit/test_cli.py tests/unit/test_contrastive_pretrain_cli.py -v`
Expected: PASS. Every new `__init__` argument has a default and `finish`'s
`exit_code` is keyword-with-default, so `data_collection` and
`contrastive_pretrain` are unaffected.

- [ ] **Step 6: Prove the resume test can fail**

Delete the `kwargs["resume"] = "allow"` line.
`test_a_run_id_is_passed_with_resume_allow_so_a_preempted_run_continues` must go
red. Revert.

- [ ] **Step 7: Run the observability audit**

Run: `uv run python ~/.claude/skills/observability-expert/scripts/audit_observability.py src/`
Expected: **9 findings** (`LOG004:5, LOG007:4` — wait, the two `tracking.py`
LOG006 entries are now resolved and the three `subprocess_backend` LOG007
entries were resolved in Task 3, so the remainder is `LOG004:5` in
`data_collection/curation.py` plus `LOG007:4` minus the three marked ones).
The exact expected line is `9 finding(s) [LOG004:5, LOG007:4]` reduced to the
`data_collection` findings only; record the literal output in the task report
and confirm it is 9.

- [ ] **Step 8: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/observability/tracking.py tests/unit/test_observability_tracking.py
git commit -m "feat(observability): W&B config, x-axes, resume, and context-manager semantics"
```

---

### Task 6: `RecurrentTransformerPolicy.diagnostics`

**Files:**
- Modify: `src/sequence_model/policy.py`
- Test: `tests/unit/test_sequence_model_policy.py`

**Interfaces:**
- Consumes: `GroupedQueryAttention.attention_diagnostics(x, cos, sin, mask) -> (q, k, probabilities)`, `sequence_model.telemetry.{attention_logit_max, attention_distance_mass, residual_norm}`.
- Produces: `RecurrentTransformerPolicy.diagnostics(latent, aux_state, prev_action, prev_reward, abs_pos, episode_id, layer=-1) -> dict[str, float]` with keys `attn/logit_max`, `model/residual_norm`, and `attn/dist_<bucket>` for each `DISTANCE_BUCKETS` label. Task 14 calls it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_sequence_model_policy.py`:

```python
def test_diagnostics_reports_one_mass_entry_per_distance_bucket() -> None:
    policy, inputs = _tiny_policy_and_chunk_inputs()

    metrics = policy.diagnostics(**inputs)

    assert sum(1 for key in metrics if key.startswith("attn/dist_")) == len(DISTANCE_BUCKETS)


def test_diagnostics_attention_mass_sums_to_one() -> None:
    policy, inputs = _tiny_policy_and_chunk_inputs()

    metrics = policy.diagnostics(**inputs)
    total = sum(value for key, value in metrics.items() if key.startswith("attn/dist_"))

    assert total == pytest.approx(1.0, abs=1e-5)


def test_diagnostics_reports_a_positive_residual_norm() -> None:
    policy, inputs = _tiny_policy_and_chunk_inputs()

    metrics = policy.diagnostics(**inputs)

    assert metrics["model/residual_norm"] > 0.0


def test_diagnostics_does_not_build_a_gradient_graph() -> None:
    """It runs on a sampled minibatch beside the update; if it kept a graph it
    would hold the full attention matrix alive across the optimizer step."""
    policy, inputs = _tiny_policy_and_chunk_inputs()
    inputs["latent"].requires_grad_(True)

    policy.diagnostics(**inputs)

    assert inputs["latent"].grad is None
```

Add the module-level helper:

```python
def _tiny_policy_and_chunk_inputs() -> tuple[RecurrentTransformerPolicy, dict]:
    """Helper, not a test: a CPU policy small enough to materialize the full
    attention matrix, plus one seeded chunk of inputs."""
    torch.manual_seed(0)
    config = PolicyConfig(
        d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
    )
    policy = RecurrentTransformerPolicy(config, latent_mean=torch.zeros(16), latent_std=torch.ones(16))
    batch, length = 2, 6
    inputs = {
        "latent": torch.randn(batch, length, 16),
        "aux_state": torch.randn(batch, length, 4),
        "prev_action": torch.zeros(batch, length, dtype=torch.int64),
        "prev_reward": torch.zeros(batch, length),
        "abs_pos": torch.arange(length).expand(batch, length).contiguous(),
        "episode_id": torch.zeros(batch, length, dtype=torch.int64),
    }
    return policy, inputs
```

Adjust `RecurrentTransformerPolicy(...)`'s construction to match its real
signature if it differs — read `src/sequence_model/policy.py:57` first.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_policy.py -k diagnostics -v`
Expected: FAIL with `AttributeError: 'RecurrentTransformerPolicy' object has no attribute 'diagnostics'`.

- [ ] **Step 3: Add the method**

In `src/sequence_model/policy.py`, add to `RecurrentTransformerPolicy`, after
`forward_chunk`:

```python
    @torch.no_grad()
    def diagnostics(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        abs_pos: torch.Tensor,
        episode_id: torch.Tensor,
        layer: int = -1,
    ) -> dict[str, float]:
        """Leading indicators, computed on a sampled minibatch every N updates.

        Never on the hot path: attention_diagnostics materializes the full
        attention matrix that SDPA deliberately never forms.

        This lives on the policy rather than in the PPO package because
        attention_diagnostics' `x` is the post-attn_norm input to one block --
        not reconstructible from outside without duplicating the stack -- and
        residual_norm needs the final hidden state, which ChunkOutput does not
        expose."""
        x = self.adapter(latent, aux_state, prev_action, prev_reward)
        cos, sin = rope_tables(abs_pos, self.config.head_dim, self.config.rope_theta)
        mask = build_chunk_mask(abs_pos, episode_id, self.config.context_len)

        blocks = cast("list[TransformerBlock]", list(self.blocks))
        tapped = blocks[layer]
        probabilities: torch.Tensor | None = None
        logit_max = 0.0
        for block in blocks:
            if block is tapped:
                normed = block.attn_norm(x)
                q, k, probabilities = block.attention.attention_diagnostics(normed, cos, sin, mask)
                logit_max = attention_logit_max(q, k, mask)
            x = block.forward_chunk(x, cos, sin, mask)

        assert probabilities is not None  # noqa: S101 -- `tapped` is drawn from `blocks`
        hidden = self.final_norm(x)
        metrics = {
            "attn/logit_max": logit_max,
            "model/residual_norm": residual_norm(hidden),
        }
        for label, mass in attention_distance_mass(probabilities).items():
            metrics[f"attn/dist_{label}"] = mass
        return metrics
```

Add the import at the top of `policy.py`:

```python
from sequence_model.telemetry import (
    attention_distance_mass,
    attention_logit_max,
    residual_norm,
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_policy.py -k diagnostics -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Prove the mass test can fail**

Change `attention_distance_mass(probabilities)` to
`attention_distance_mass(probabilities * 0.5)`.
`test_diagnostics_attention_mass_sums_to_one` must stay green (the function
normalizes by its own total) — if it does, instead drop the `"0"` bucket from
`DISTANCE_BUCKETS` temporarily and confirm the sum falls below 1.0. Revert
whichever mutation you used and record it in the report.

- [ ] **Step 6: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/sequence_model/policy.py tests/unit/test_sequence_model_policy.py
git commit -m "feat(sequence-model): policy-level diagnostics for the PPO telemetry consumers"
```

---

### Task 7: `RolloutBuffer`

**Files:**
- Create: `src/ppo/buffer.py`
- Test: `tests/unit/test_ppo_buffer.py`

**Interfaces:**
- Consumes: `PPOConfig.burn_in`, `PPOConfig.buffer_capacity` from Task 1.
- Produces: `RolloutBuffer(config, policy_config, n_envs, device)` with `.write(slot, latent, aux, action, prev_action, prev_reward, reward, done, episode_id, abs_pos, logprob, value)`, `.shift()`, `.chunk(env_indices) -> ChunkInputs`, `.trained_slice`, and `.write_cursor`. Tasks 11 and 12 consume it.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ppo_buffer.py`:

```python
"""RolloutBuffer index arithmetic. Every silent bug in this trainer that is
not a ratio bug is an index bug."""

from __future__ import annotations

import pytest
import torch

from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from sequence_model.config import PolicyConfig


def _tiny_buffer() -> RolloutBuffer:
    """Helper, not a test: context_len 4 -> burn_in 3, n_steps 4, capacity 8."""
    return RolloutBuffer(
        config=PPOConfig(frozen_encoder_revision="x", n_steps=4),
        policy_config=PolicyConfig(
            d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1,
            d_ff=64, context_len=4, latent_dim=8, aux_state_dim=4,
        ),
        n_envs=2,
        device=torch.device("cpu"),
    )


def test_capacity_is_burn_in_plus_n_steps_plus_the_bootstrap_slot() -> None:
    assert _tiny_buffer().capacity == 8


def test_the_trained_slice_starts_after_the_burn_in_and_is_n_steps_long() -> None:
    buffer = _tiny_buffer()

    assert (buffer.trained_slice.start, buffer.trained_slice.stop) == (3, 7)


def test_chunk_returns_burn_in_plus_n_steps_plus_bootstrap_positions() -> None:
    buffer = _tiny_buffer()

    chunk = buffer.chunk(torch.tensor([0, 1]))

    assert chunk.latent.shape == (2, 8, 8)


def test_a_written_latent_is_readable_at_the_slot_it_was_written_to() -> None:
    buffer = _tiny_buffer()
    latent = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    buffer.write(slot=5, latent=latent, aux=torch.zeros(2, 4), action=torch.zeros(2, dtype=torch.int64),
                 prev_action=torch.zeros(2, dtype=torch.int64), prev_reward=torch.zeros(2),
                 reward=torch.zeros(2), done=torch.zeros(2, dtype=torch.bool),
                 episode_id=torch.zeros(2, dtype=torch.int64), abs_pos=torch.zeros(2, dtype=torch.int64),
                 logprob=torch.zeros(2), value=torch.zeros(2))

    assert buffer.chunk(torch.tensor([0, 1])).latent[1, 5].tolist() == pytest.approx(
        latent[1].to(torch.float16).float().tolist()
    )


def test_shift_moves_the_last_capacity_minus_n_steps_slots_to_the_front() -> None:
    """After a shift, the previous update's bootstrap observation must land at
    the first trained slot -- that is what makes every observation trained
    exactly once."""
    buffer = _tiny_buffer()
    marker = torch.full((2, 8), 9.0)
    buffer.write(slot=7, latent=marker, aux=torch.zeros(2, 4), action=torch.zeros(2, dtype=torch.int64),
                 prev_action=torch.zeros(2, dtype=torch.int64), prev_reward=torch.zeros(2),
                 reward=torch.zeros(2), done=torch.zeros(2, dtype=torch.bool),
                 episode_id=torch.zeros(2, dtype=torch.int64), abs_pos=torch.zeros(2, dtype=torch.int64),
                 logprob=torch.zeros(2), value=torch.zeros(2))

    buffer.shift()

    assert buffer.chunk(torch.tensor([0, 1])).latent[0, 3, 0].item() == pytest.approx(9.0)


def test_shift_sets_the_write_cursor_to_the_first_trained_slot_plus_one() -> None:
    buffer = _tiny_buffer()

    buffer.shift()

    assert buffer.write_cursor == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppo.buffer'`.

- [ ] **Step 3: Implement the buffer**

Create `src/ppo/buffer.py`:

```python
"""Rolling, GPU-resident storage for one PPO update's worth of transitions.

Stores fp16 latents rather than frames: 4,096 B/step against 23,040 B/step,
and the frozen CNN makes latents deterministic so there is nothing to
recompute. At 64 envs x 2048 slots x 2048 dims that is 537 MB.

Layout per env, in slot indices:

    [0, burn_in)                    burn-in prefix, no loss
    [burn_in, burn_in + n_steps)    trained region
    burn_in + n_steps               bootstrap slot, supplies V(s_T) only

shift() advances by exactly n_steps, so the previous update's bootstrap
observation becomes the new region's first trained slot and every observation
is trained exactly once."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ppo.config import PPOConfig
from sequence_model.config import PolicyConfig


@dataclass(frozen=True)
class ChunkInputs:
    """Exactly the arguments forward_chunk takes, plus the fields the loss and
    GAE need. Grouped so the update never assembles them by hand twice."""

    latent: torch.Tensor  # (B, L, latent_dim) float32
    aux_state: torch.Tensor  # (B, L, aux_state_dim)
    prev_action: torch.Tensor  # (B, L) int64
    prev_reward: torch.Tensor  # (B, L)
    abs_pos: torch.Tensor  # (B, L) int64
    episode_id: torch.Tensor  # (B, L) int64
    action: torch.Tensor  # (B, L) int64
    reward: torch.Tensor  # (B, L)
    done: torch.Tensor  # (B, L) bool
    rollout_logprob: torch.Tensor  # (B, L) -- diagnostic only, never the ratio
    rollout_value: torch.Tensor  # (B, L) -- diagnostic only, never the baseline


class RolloutBuffer:
    def __init__(
        self,
        config: PPOConfig,
        policy_config: PolicyConfig,
        n_envs: int,
        device: torch.device,
    ) -> None:
        self._config = config
        self.burn_in = config.burn_in(policy_config.context_len)
        self.capacity = config.buffer_capacity(policy_config.context_len)
        self.n_envs = n_envs

        shape = (n_envs, self.capacity)
        # fp16, not bf16: latents are bounded encoder outputs and fp16's extra
        # mantissa costs nothing here, while the buffer is the single largest
        # allocation in the trainer.
        self._latent = torch.zeros(*shape, policy_config.latent_dim, dtype=torch.float16, device=device)
        self._aux = torch.zeros(*shape, policy_config.aux_state_dim, dtype=torch.float32, device=device)
        self._action = torch.zeros(*shape, dtype=torch.int64, device=device)
        self._prev_action = torch.zeros(*shape, dtype=torch.int64, device=device)
        self._prev_reward = torch.zeros(*shape, dtype=torch.float32, device=device)
        self._reward = torch.zeros(*shape, dtype=torch.float32, device=device)
        self._done = torch.zeros(*shape, dtype=torch.bool, device=device)
        self._episode_id = torch.zeros(*shape, dtype=torch.int64, device=device)
        self._abs_pos = torch.zeros(*shape, dtype=torch.int64, device=device)
        self._logprob = torch.zeros(*shape, dtype=torch.float32, device=device)
        self._value = torch.zeros(*shape, dtype=torch.float32, device=device)

        self.write_cursor = 0

    @property
    def trained_slice(self) -> slice:
        return slice(self.burn_in, self.burn_in + self._config.n_steps)

    def write(
        self,
        slot: int,
        latent: torch.Tensor,
        aux: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        episode_id: torch.Tensor,
        abs_pos: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._latent[:, slot] = latent.to(torch.float16)
        self._aux[:, slot] = aux
        self._action[:, slot] = action
        self._prev_action[:, slot] = prev_action
        self._prev_reward[:, slot] = prev_reward
        self._reward[:, slot] = reward
        self._done[:, slot] = done
        self._episode_id[:, slot] = episode_id
        self._abs_pos[:, slot] = abs_pos
        self._logprob[:, slot] = logprob
        self._value[:, slot] = value
        self.write_cursor = slot + 1

    def shift(self) -> None:
        """Drop the oldest n_steps slots. What remains is the new burn-in plus
        the observation that becomes the new region's first trained slot, so
        the next rollout collects exactly n_steps observations."""
        keep = self.capacity - self._config.n_steps
        for tensor in self._tensors():
            tensor[:, :keep] = tensor[:, self._config.n_steps :].clone()
        self.write_cursor = keep

    def chunk(self, env_indices: torch.Tensor) -> ChunkInputs:
        """One minibatch: a subset of ENVS at full length. Never a slice of
        time -- the burn-in binds the time axis."""
        return ChunkInputs(
            latent=self._latent[env_indices].float(),
            aux_state=self._aux[env_indices],
            prev_action=self._prev_action[env_indices],
            prev_reward=self._prev_reward[env_indices],
            abs_pos=self._abs_pos[env_indices],
            episode_id=self._episode_id[env_indices],
            action=self._action[env_indices],
            reward=self._reward[env_indices],
            done=self._done[env_indices],
            rollout_logprob=self._logprob[env_indices],
            rollout_value=self._value[env_indices],
        )

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self._latent, self._aux, self._action, self._prev_action,
            self._prev_reward, self._reward, self._done, self._episode_id,
            self._abs_pos, self._logprob, self._value,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_buffer.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Prove the shift test can fail**

Change `shift`'s `keep` to `self.capacity - self._config.n_steps - 1`.
`test_shift_moves_the_last_capacity_minus_n_steps_slots_to_the_front` and
`test_shift_sets_the_write_cursor_to_the_first_trained_slot_plus_one` must both
go red. Revert.

- [ ] **Step 6: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/ppo/buffer.py tests/unit/test_ppo_buffer.py
git commit -m "feat(ppo): rolling rollout buffer with the shift-by-n_steps contract"
```

---

### Task 8: GAE and the return scaler

**Files:**
- Create: `src/ppo/gae.py`, `src/ppo/normalizer.py`
- Test: `tests/unit/test_ppo_gae.py`, `tests/unit/test_ppo_normalizer.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `compute_gae(reward, value, episode_id, gamma, gae_lambda) -> tuple[advantages, returns]`, both `(B, T)`; and `ReturnScaler(gamma)` with `.update(returns) -> None`, `.scale -> float`, `.state_dict()`, `.load_state_dict(state)`. Task 12 uses both.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ppo_gae.py`:

```python
"""GAE, checked against arithmetic done by hand rather than against itself."""

from __future__ import annotations

import pytest
import torch

from ppo.gae import compute_gae


def test_gae_matches_a_hand_computed_three_step_trajectory() -> None:
    """gamma=0.5, lambda=1.0, one episode, rewards [1, 1, 1],
    values [0, 0, 0, 0]. With lambda=1 the advantage is the discounted
    return: A2 = 1, A1 = 1 + 0.5*1 = 1.5, A0 = 1 + 0.5*1.5 = 1.75."""
    reward = torch.tensor([[1.0, 1.0, 1.0]])
    value = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    episode_id = torch.zeros(1, 4, dtype=torch.int64)

    advantages, _ = compute_gae(reward, value, episode_id, gamma=0.5, gae_lambda=1.0)

    assert advantages[0].tolist() == pytest.approx([1.75, 1.5, 1.0])


def test_returns_are_advantages_plus_the_baseline_value() -> None:
    reward = torch.tensor([[1.0, 1.0, 1.0]])
    value = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    episode_id = torch.zeros(1, 4, dtype=torch.int64)

    advantages, returns = compute_gae(reward, value, episode_id, gamma=0.5, gae_lambda=1.0)

    assert returns[0].tolist() == pytest.approx((advantages[0] + value[0, :3]).tolist())


def test_gae_does_not_bootstrap_across_an_episode_boundary() -> None:
    """Step 1 ends an episode. Its advantage must be reward - value with no
    contribution from step 2, which belongs to a different episode."""
    reward = torch.tensor([[0.0, 1.0, 100.0]])
    value = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    episode_id = torch.tensor([[0, 0, 1, 1]])

    advantages, _ = compute_gae(reward, value, episode_id, gamma=0.5, gae_lambda=1.0)

    assert advantages[0, 1].item() == pytest.approx(1.0)


def test_gae_uses_the_bootstrap_slot_for_the_final_transition() -> None:
    """The last trained transition's next-state value comes from the extra
    slot, not from a zero."""
    reward = torch.tensor([[0.0]])
    value = torch.tensor([[0.0, 8.0]])
    episode_id = torch.zeros(1, 2, dtype=torch.int64)

    advantages, _ = compute_gae(reward, value, episode_id, gamma=0.5, gae_lambda=1.0)

    assert advantages[0, 0].item() == pytest.approx(4.0)
```

Create `tests/unit/test_ppo_normalizer.py`:

```python
"""ReturnScaler: divides value targets by a running std, never shifts the mean."""

from __future__ import annotations

import pytest
import torch

from ppo.normalizer import ReturnScaler


def test_scale_starts_at_one_so_the_first_update_is_unscaled() -> None:
    assert ReturnScaler(gamma=0.99).scale == pytest.approx(1.0)


def test_scale_approaches_the_standard_deviation_of_the_returns_it_has_seen() -> None:
    scaler = ReturnScaler(gamma=0.99)
    returns = torch.tensor([[-10.0, 10.0, -10.0, 10.0]])

    scaler.update(returns)

    assert scaler.scale == pytest.approx(10.0, rel=0.05)


def test_the_scaler_never_shifts_the_mean_so_advantage_signs_survive() -> None:
    scaler = ReturnScaler(gamma=0.99)
    scaler.update(torch.tensor([[100.0, 102.0, 104.0]]))

    assert scaler.normalize(torch.tensor([-5.0]))[0].item() < 0.0


def test_state_round_trips_through_a_checkpoint() -> None:
    scaler = ReturnScaler(gamma=0.99)
    scaler.update(torch.tensor([[-10.0, 10.0]]))
    restored = ReturnScaler(gamma=0.99)

    restored.load_state_dict(scaler.state_dict())

    assert restored.scale == pytest.approx(scaler.scale)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_gae.py tests/unit/test_ppo_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement GAE**

Create `src/ppo/gae.py`:

```python
"""Generalized Advantage Estimation over one update's trained region.

`value` carries T+1 entries: the T trained positions plus the bootstrap slot.
Episode boundaries are read from `episode_id`, not from `done` alone -- a
respawned worker also produces a discontinuity, and it arrives as a new
episode id rather than as a done flag on the previous step."""

from __future__ import annotations

import torch


def compute_gae(
    reward: torch.Tensor,
    value: torch.Tensor,
    episode_id: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`reward` (B, T), `value` (B, T+1), `episode_id` (B, T+1).

    Returns (advantages, returns), both (B, T). `returns` is
    `advantages + value[:, :T]`, which is the standard value target."""
    steps = reward.shape[1]
    # continues[t] is 0 where step t+1 belongs to a different episode, which
    # is exactly where the bootstrap must not cross.
    continues = (episode_id[:, 1:] == episode_id[:, :-1]).to(reward.dtype)

    advantages = torch.zeros_like(reward)
    running = torch.zeros_like(reward[:, 0])
    for t in range(steps - 1, -1, -1):
        delta = reward[:, t] + gamma * value[:, t + 1] * continues[:, t] - value[:, t]
        running = delta + gamma * gae_lambda * continues[:, t] * running
        advantages[:, t] = running
    return advantages, advantages + value[:, :steps]
```

- [ ] **Step 4: Implement the return scaler**

Create `src/ppo/normalizer.py`:

```python
"""Running scale for value targets.

Rewards are clipped to [0, 1] per step and gamma is 0.997, so an undiscounted
horizon puts value targets in the tens -- against a critic the architecture
plan calls hypersensitive to input scale. Dividing by a running std of the
return keeps the target near unit scale for the whole run.

No mean shift, deliberately: subtracting a mean would change the sign of an
advantage computed against the unshifted value, and sign is the only part of
the advantage the policy gradient cannot recover from being wrong."""

from __future__ import annotations

import torch

EPSILON = 1e-8


class ReturnScaler:
    def __init__(self, gamma: float) -> None:
        self._gamma = gamma
        self._count = 0.0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def scale(self) -> float:
        """Starts at 1.0 so the first update is unscaled rather than divided by
        a variance estimated from nothing."""
        if self._count < 2:
            return 1.0
        return float(max((self._m2 / self._count) ** 0.5, EPSILON))

    def update(self, returns: torch.Tensor) -> None:
        """Chan et al.'s parallel variance update, so a whole update's returns
        fold in with one pass and no history is retained."""
        batch = returns.detach().flatten().float()
        batch_count = float(batch.numel())
        if batch_count == 0:
            return
        batch_mean = float(batch.mean())
        batch_m2 = float(((batch - batch_mean) ** 2).sum())

        delta = batch_mean - self._mean
        total = self._count + batch_count
        self._m2 += batch_m2 + delta * delta * self._count * batch_count / total
        self._mean += delta * batch_count / total
        self._count = total

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        return returns / self.scale

    def state_dict(self) -> dict:
        return {"count": self._count, "mean": self._mean, "m2": self._m2}

    def load_state_dict(self, state: dict) -> None:
        self._count = float(state["count"])
        self._mean = float(state["mean"])
        self._m2 = float(state["m2"])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_gae.py tests/unit/test_ppo_normalizer.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 6: Prove the episode-boundary test can fail**

Replace `continues` with `torch.ones_like(reward)`.
`test_gae_does_not_bootstrap_across_an_episode_boundary` must go red (reporting
51.0 instead of 1.0). Revert.

- [ ] **Step 7: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/ppo/gae.py src/ppo/normalizer.py tests/unit/test_ppo_gae.py tests/unit/test_ppo_normalizer.py
git commit -m "feat(ppo): GAE with episode-aware bootstrapping, and the return scaler"
```

---

### Task 9: PPO losses

**Files:**
- Create: `src/ppo/losses.py`
- Test: `tests/unit/test_ppo_losses.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ppo_losses(logits, value, action, logprob_old, advantage, value_target, config) -> LossOutput` where `LossOutput` carries `policy`, `value`, `entropy`, `total`, `clip_fraction`, `approx_kl`, `max_abs_ratio_dev`. Task 12 consumes it.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ppo_losses.py`:

```python
"""The clipped surrogate, checked at the points where clipping changes it."""

from __future__ import annotations

import pytest
import torch

from ppo.config import PPOConfig
from ppo.losses import ppo_losses


def _config() -> PPOConfig:
    return PPOConfig(frozen_encoder_revision="x", ent_coef=0.0, vf_coef=0.0)


def test_the_ratio_is_exactly_one_when_logprob_old_came_from_these_logits() -> None:
    """The invariant the whole design turns on, at the level of one function."""
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 7)
    action = torch.zeros(1, 3, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

    output = ppo_losses(
        logits, torch.zeros(1, 3), action, logprob_old,
        advantage=torch.ones(1, 3), value_target=torch.zeros(1, 3), config=_config(),
    )

    assert output.max_abs_ratio_dev == pytest.approx(0.0, abs=1e-6)


def test_the_policy_loss_is_the_negative_advantage_when_the_ratio_is_one() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 7)
    action = torch.zeros(1, 3, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

    output = ppo_losses(
        logits, torch.zeros(1, 3), action, logprob_old,
        advantage=torch.full((1, 3), 2.0), value_target=torch.zeros(1, 3), config=_config(),
    )

    assert output.policy.item() == pytest.approx(-2.0)


def test_a_positive_advantage_is_clipped_at_one_plus_the_clip_range() -> None:
    """ratio = e^1 = 2.718, clip_range 0.2 -> the surrogate caps at 1.2 * A."""
    logits = torch.zeros(1, 1, 2)
    action = torch.zeros(1, 1, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1)[..., 0] - 1.0

    output = ppo_losses(
        logits, torch.zeros(1, 1), action, logprob_old,
        advantage=torch.ones(1, 1), value_target=torch.zeros(1, 1), config=_config(),
    )

    assert output.policy.item() == pytest.approx(-1.2)


def test_the_clip_fraction_counts_the_positions_where_clipping_bound() -> None:
    logits = torch.zeros(1, 2, 2)
    action = torch.zeros(1, 2, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1)[..., 0] - torch.tensor([[1.0, 0.0]])

    output = ppo_losses(
        logits, torch.zeros(1, 2), action, logprob_old,
        advantage=torch.ones(1, 2), value_target=torch.zeros(1, 2), config=_config(),
    )

    assert output.clip_fraction == pytest.approx(0.5)


def test_the_value_loss_is_the_mean_squared_error_against_the_target() -> None:
    config = PPOConfig(frozen_encoder_revision="x", ent_coef=0.0, vf_coef=1.0)
    logits = torch.zeros(1, 2, 2)
    action = torch.zeros(1, 2, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1)[..., 0]

    output = ppo_losses(
        logits, torch.zeros(1, 2), action, logprob_old,
        advantage=torch.zeros(1, 2), value_target=torch.full((1, 2), 3.0), config=config,
    )

    assert output.value.item() == pytest.approx(9.0)


def test_the_entropy_of_a_uniform_two_way_logit_is_log_two() -> None:
    config = PPOConfig(frozen_encoder_revision="x", ent_coef=1.0, vf_coef=0.0)
    logits = torch.zeros(1, 1, 2)
    action = torch.zeros(1, 1, dtype=torch.int64)
    logprob_old = torch.log_softmax(logits, dim=-1)[..., 0]

    output = ppo_losses(
        logits, torch.zeros(1, 1), action, logprob_old,
        advantage=torch.zeros(1, 1), value_target=torch.zeros(1, 1), config=config,
    )

    assert output.entropy.item() == pytest.approx(0.6931472, abs=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppo.losses'`.

- [ ] **Step 3: Implement the losses**

Create `src/ppo/losses.py`:

```python
"""The clipped surrogate, the value loss, and the entropy bonus.

`logits` are raw -- log_softmax is applied here, and the caller must never
apply a softmax before handing them over.

Value clipping is off (clip_range_vf defaults to None), matching SB3.
SB3's own docstring says that clipping "depends on the reward scaling", and
the reward scale here is set by ReturnScaler at runtime rather than known in
advance."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ppo.config import PPOConfig


@dataclass(frozen=True)
class LossOutput:
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor
    clip_fraction: float
    approx_kl: float
    max_abs_ratio_dev: float


def ppo_losses(
    logits: torch.Tensor,
    value: torch.Tensor,
    action: torch.Tensor,
    logprob_old: torch.Tensor,
    advantage: torch.Tensor,
    value_target: torch.Tensor,
    config: PPOConfig,
) -> LossOutput:
    """All tensors are (B, T) except `logits`, which is (B, T, action_dim)."""
    log_probabilities = F.log_softmax(logits, dim=-1)
    logprob = log_probabilities.gather(-1, action.unsqueeze(-1)).squeeze(-1)

    log_ratio = logprob - logprob_old
    ratio = log_ratio.exp()

    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range) * advantage
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = F.mse_loss(value, value_target)
    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1).mean()

    total = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

    with torch.no_grad():
        # Schulman's low-variance estimator, the same one SB3 reports.
        approx_kl = float(((ratio - 1.0) - log_ratio).mean())
        clip_fraction = float(((ratio - 1.0).abs() > config.clip_range).float().mean())
        max_abs_ratio_dev = float((ratio - 1.0).abs().max())

    return LossOutput(
        policy=policy_loss,
        value=value_loss,
        entropy=entropy,
        total=total,
        clip_fraction=clip_fraction,
        approx_kl=approx_kl,
        max_abs_ratio_dev=max_abs_ratio_dev,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_losses.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Prove the clipping test can fail**

Replace `policy_loss = -torch.min(unclipped, clipped).mean()` with
`-unclipped.mean()`. `test_a_positive_advantage_is_clipped_at_one_plus_the_clip_range`
must go red (reporting -2.718 instead of -1.2). Revert.

- [ ] **Step 6: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/ppo/losses.py tests/unit/test_ppo_losses.py
git commit -m "feat(ppo): clipped surrogate, value loss, entropy bonus"
```

---

### Task 10: `collect_rollout`

**Files:**
- Create: `src/ppo/rollout.py`
- Test: `tests/unit/test_ppo_rollout.py`, `tests/unit/fakes.py`

**Interfaces:**
- Consumes: `RolloutBuffer` (Task 7), `VecPokemonEnv.step`, `LatentEncoder.encode`, `policy.step`, `cache.reset`.
- Produces: `collect_rollout(vec_env, encoder, policy, cache, buffer, state, n_steps, generator, device) -> RolloutState` and `RolloutState(prev_action, prev_reward, last_step)`. Task 15 drives it.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ppo_rollout.py`:

```python
"""The rollout's ordering contract. Each of these guards a bug that produces
correctly-shaped tensors and a silently wrong model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ppo.rollout import RolloutState, collect_rollout


def test_the_cache_is_reset_after_the_terminal_step_not_before_the_next_one() -> None:
    """If reset ran first, the terminal observation would attend to a cleared
    cache and the last transition of every episode would be trained on the
    wrong context."""
    harness = _rollout_harness(done_at_step=1)

    collect_rollout(**harness.kwargs(n_steps=3))

    assert harness.cache.reset_calls_after_step == [1]


def test_the_previous_action_becomes_episode_start_after_a_done() -> None:
    """Autoreset is next-step: the action taken at the terminal step is
    meaningless as context for the fresh episode that arrives at t+1."""
    harness = _rollout_harness(done_at_step=1)

    collect_rollout(**harness.kwargs(n_steps=3))

    assert harness.policy.prev_actions_seen[2].tolist() == [7, 7]


def test_the_previous_reward_is_zeroed_after_a_done() -> None:
    harness = _rollout_harness(done_at_step=1, reward=0.5)

    collect_rollout(**harness.kwargs(n_steps=3))

    assert harness.policy.prev_rewards_seen[2].tolist() == pytest.approx([0.0, 0.0])


def test_the_recorded_absolute_position_is_the_one_the_policy_used() -> None:
    """cache.abs_pos advances inside policy.step, so a snapshot taken after
    the call records the NEXT position and RoPE in the update no longer
    matches RoPE in the rollout."""
    harness = _rollout_harness(done_at_step=None)

    collect_rollout(**harness.kwargs(n_steps=3))
    chunk = harness.buffer.chunk(torch.tensor([0, 1]))

    assert chunk.abs_pos[0, harness.buffer.burn_in : harness.buffer.burn_in + 3].tolist() == [0, 1, 2]


def test_the_rollout_writes_exactly_n_steps_slots() -> None:
    harness = _rollout_harness(done_at_step=None)
    start = harness.buffer.write_cursor

    collect_rollout(**harness.kwargs(n_steps=3))

    assert harness.buffer.write_cursor - start == 3
```

Add `_rollout_harness` and its fakes to `tests/unit/fakes.py`: a `FakeVecEnv`
returning a scripted `VecStep` per call with `done` set at the requested step; a
`RecordingCache` wrapping a real `RolloutCache` that appends the step index to
`reset_calls_after_step` whenever `reset` is called with any `done` set; and a
`RecordingPolicy` that records `prev_action` and `prev_reward` per call and
returns uniform logits and zero values. Type all three against the Protocols
their consumers use.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_rollout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppo.rollout'`.

- [ ] **Step 3: Implement the rollout**

Create `src/ppo/rollout.py`:

```python
"""One rollout segment: env -> frozen encoder -> policy -> buffer.

Ordering here is load-bearing three times over, and every one of the three
produces correctly-shaped tensors when wrong:

  1. cache.reset(done) runs AFTER the step whose transition ended the episode.
  2. prev_action becomes EPISODE_START and prev_reward zero after a done,
     because autoreset is next-step.
  3. abs_pos is snapshotted BEFORE policy.step, which advances the cache."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ppo.buffer import RolloutBuffer


@dataclass
class RolloutState:
    """What carries across a rollout boundary. `prev_action`/`prev_reward` are
    the policy's inputs for the next step; `last_step` is the VecStep whose
    observation has been written but whose action has not yet been applied."""

    prev_action: torch.Tensor
    prev_reward: torch.Tensor


def collect_rollout(
    vec_env,
    encoder,
    policy,
    cache,
    buffer: RolloutBuffer,
    state: RolloutState,
    n_steps: int,
    generator: torch.Generator,
    device: torch.device,
    episode_start_action: int,
    autocast_dtype: torch.dtype,
) -> RolloutState:
    """Advances the env `n_steps` times, writing one buffer slot per step."""
    for _ in range(n_steps):
        actions = _sample_placeholder(policy, generator)  # replaced below; see Step 4
        raise NotImplementedError
    return state
```

The loop body is written in Step 4 — this step only establishes the module and
its imports so the failure message changes from `ModuleNotFoundError` to
`NotImplementedError`.

- [ ] **Step 4: Write the loop body**

Replace `collect_rollout`'s body:

```python
    for _ in range(n_steps):
        step = vec_env.step(state.prev_action.detach().cpu().numpy().astype(np.int64))

        latent = encoder.encode(step.frames)
        aux = torch.from_numpy(step.aux).to(device)
        reward = torch.from_numpy(step.reward).to(device)
        done = torch.from_numpy(step.done).to(device)
        episode_id = torch.from_numpy(step.episode_id).to(device)

        # Snapshotted BEFORE policy.step: step() calls cache.advance(), so
        # reading abs_pos afterwards records the position of the NEXT token and
        # RoPE in forward_chunk would no longer match RoPE in the rollout.
        abs_pos = cache.abs_pos.clone()

        with torch.autocast(device.type, dtype=autocast_dtype):
            output = policy.step(latent, aux, state.prev_action, state.prev_reward, cache)

        log_probabilities = torch.log_softmax(output.logits.float(), dim=-1)
        action = torch.multinomial(
            log_probabilities.exp(), num_samples=1, generator=generator
        ).squeeze(-1)
        logprob = log_probabilities.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        buffer.write(
            slot=buffer.write_cursor,
            latent=latent,
            aux=aux,
            action=action,
            prev_action=state.prev_action,
            prev_reward=state.prev_reward,
            reward=reward,
            done=done,
            episode_id=episode_id,
            abs_pos=abs_pos,
            logprob=logprob,
            value=output.value.float(),
        )

        # AFTER the step whose transition ended the episode, never before the
        # next one -- the terminal observation must attend to its own episode.
        cache.reset(done)

        state.prev_action = torch.where(
            done, torch.full_like(action, episode_start_action), action
        )
        state.prev_reward = torch.where(done, torch.zeros_like(reward), reward)
    return state
```

Delete the `_sample_placeholder` line and the `raise NotImplementedError`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_rollout.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 6: Prove the abs_pos test can fail**

Move `abs_pos = cache.abs_pos.clone()` to after the `policy.step` call.
`test_the_recorded_absolute_position_is_the_one_the_policy_used` must go red
(reporting `[1, 2, 3]`). Revert.

- [ ] **Step 7: Prove the reset-ordering test can fail**

Move `cache.reset(done)` to the top of the loop body.
`test_the_cache_is_reset_after_the_terminal_step_not_before_the_next_one` must
go red. Revert.

- [ ] **Step 8: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/ppo/rollout.py tests/unit/test_ppo_rollout.py tests/unit/fakes.py
git commit -m "feat(ppo): rollout loop with the reset-after-step and abs_pos snapshot contracts"
```

---

### Task 11: `run_update`

**Files:**
- Create: `src/ppo/update.py`
- Test: `tests/unit/test_ppo_update.py`

**Interfaces:**
- Consumes: `RolloutBuffer.chunk` (Task 7), `compute_gae` and `ReturnScaler` (Task 8), `ppo_losses` (Task 9).
- Produces: `run_update(policy, optimizer, scheduler, buffer, scaler, config, policy_config, n_envs, device, autocast_dtype) -> UpdateStats`, where `UpdateStats` carries `losses`, `clip_fraction`, `approx_kl`, `max_abs_ratio_dev_epoch1_mb1`, `max_abs_ratio_dev`, `explained_variance`, `staleness_logprob_l1`, `skipped_minibatches`, `grad_norm`. Task 15 calls it.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ppo_update.py`:

```python
"""The update, and the invariant the whole design turns on."""

from __future__ import annotations

import pytest
import torch

from ppo.update import run_update


def test_the_first_minibatch_of_epoch_one_has_a_ratio_of_exactly_one() -> None:
    """THE load-bearing test. pi_old is recomputed by a no_grad forward_chunk
    at update start, so before any optimizer step the ratio must be exactly 1.
    Using the rollout-recorded logprobs instead makes this fail."""
    harness = _update_harness()

    stats = run_update(**harness.kwargs())

    assert stats.max_abs_ratio_dev_epoch1_mb1 == pytest.approx(0.0, abs=1e-6)


def test_the_update_takes_one_optimizer_step_per_minibatch_per_epoch() -> None:
    harness = _update_harness(n_envs=4, minibatch_envs=2, n_epochs=3)

    run_update(**harness.kwargs())

    assert harness.optimizer.step_calls == 6


def test_the_recomputed_logprobs_differ_from_the_rollout_recorded_ones() -> None:
    """The staleness diagnostic that justified recomputing pi_old. It is
    reported, not asserted to be zero -- a nonzero value is the expected
    steady state, because the KV cache is carried across update boundaries."""
    harness = _update_harness(rollout_logprob_offset=0.25)

    stats = run_update(**harness.kwargs())

    assert stats.staleness_logprob_l1 == pytest.approx(0.25, abs=1e-5)


def test_a_non_finite_loss_skips_the_minibatch_without_stepping() -> None:
    harness = _update_harness(n_envs=2, minibatch_envs=2, n_epochs=1, nan_advantage=True)

    stats = run_update(**harness.kwargs())

    assert (stats.skipped_minibatches, harness.optimizer.step_calls) == (1, 0)


def test_too_many_non_finite_minibatches_abort_the_update() -> None:
    harness = _update_harness(
        n_envs=2, minibatch_envs=2, n_epochs=4, nan_advantage=True, max_nan=3
    )

    with pytest.raises(RuntimeError, match="non-finite loss in 3 minibatches"):
        run_update(**harness.kwargs())


def test_advantages_are_normalized_once_over_the_whole_update_batch() -> None:
    """Per-minibatch normalization would make each minibatch's targets depend
    on which envs happened to land in it."""
    harness = _update_harness(n_envs=4, minibatch_envs=2)

    run_update(**harness.kwargs())

    assert harness.advantage_std_seen == pytest.approx(1.0, abs=1e-3)
```

Add `_update_harness` to the test module: it builds a tiny `PolicyConfig`
(`d_model=32, n_layers=2, n_heads=2, head_dim=16, n_kv_heads=1, d_ff=64,
context_len=4, latent_dim=8, aux_state_dim=4`), a real
`RecurrentTransformerPolicy`, a `RolloutBuffer` filled with seeded synthetic
data, a `CountingOptimizer` wrapping a real `torch.optim.AdamW` that increments
`step_calls`, and a `ReturnScaler`. It records the std of the advantages passed
into the first minibatch in `advantage_std_seen`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_update.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppo.update'`.

- [ ] **Step 3: Implement the update**

Create `src/ppo/update.py`:

```python
"""One PPO update over the buffer's trained region.

Structure, in order:

  1. Recompute pi_old and V_old with a no_grad forward_chunk sweep over ALL
     envs, under the same autocast context as the training step.
  2. GAE off V_old, advantages normalized once over the whole batch.
  3. n_epochs x (n_envs / minibatch_envs) minibatches, one optimizer step each.

Step 1 is deliberately NOT fused into epoch 1. Fusing is only valid if epoch 1
takes no optimizer step until every minibatch has been seen, which forces
gradient accumulation and drops the update from 24 optimizer steps to 4. The
~1.7% a separate pass costs buys back the gradient-step count and an invariant
that holds by construction: at (epoch 1, minibatch 1) the policy has not
changed, so max|ratio - 1| is exactly 0."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from ppo.gae import compute_gae
from ppo.losses import ppo_losses
from ppo.normalizer import ReturnScaler
from sequence_model.config import PolicyConfig


@dataclass(frozen=True)
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    clip_fraction: float
    approx_kl: float
    max_abs_ratio_dev_epoch1_mb1: float
    max_abs_ratio_dev: float
    explained_variance: float
    staleness_logprob_l1: float
    skipped_minibatches: int
    grad_norm: float


def run_update(
    policy,
    optimizer,
    scheduler,
    buffer: RolloutBuffer,
    scaler: ReturnScaler,
    config: PPOConfig,
    policy_config: PolicyConfig,
    n_envs: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> UpdateStats:
    burn_in = buffer.burn_in
    trained = buffer.trained_slice
    env_order = torch.arange(n_envs, device=device)
    minibatches = env_order.split(config.minibatch_envs)

    logprob_old, value_old, staleness = _recompute_old(
        policy, buffer, minibatches, burn_in, trained, autocast_dtype, device
    )

    episode_id = _gather(buffer, minibatches, "episode_id")[:, trained.start : trained.stop + 1]
    reward = _gather(buffer, minibatches, "reward")[:, trained]
    advantage, returns = compute_gae(
        reward, value_old, episode_id, config.gamma, config.gae_lambda
    )

    # Once, over the whole update batch -- not per minibatch.
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    scaler.update(returns)
    value_target = scaler.normalize(returns)

    explained = _explained_variance(value_old[:, :-1], value_target)

    first_dev: float | None = None
    last = None
    skipped = 0
    grad_norm = 0.0
    for epoch in range(config.n_epochs):
        for index, envs in enumerate(minibatches):
            chunk = buffer.chunk(envs)
            rows = envs
            with torch.autocast(device.type, dtype=autocast_dtype):
                output = policy.forward_chunk(
                    chunk.latent, chunk.aux_state, chunk.prev_action, chunk.prev_reward,
                    chunk.abs_pos, chunk.episode_id, burn_in,
                )
                loss = ppo_losses(
                    output.logits[:, : config.n_steps],
                    output.value[:, : config.n_steps],
                    chunk.action[:, trained],
                    logprob_old[rows],
                    advantage[rows],
                    value_target[rows],
                    config,
                )

            if epoch == 0 and index == 0:
                first_dev = loss.max_abs_ratio_dev

            if not torch.isfinite(loss.total):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if skipped >= config.max_nan_minibatches_per_update:
                    raise RuntimeError(
                        f"non-finite loss in {skipped} minibatches of one update; "
                        "aborting rather than stepping on corrupt gradients"
                    )
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            last = loss

    assert first_dev is not None and last is not None  # noqa: S101 -- n_epochs >= 1
    return UpdateStats(
        policy_loss=float(last.policy),
        value_loss=float(last.value),
        entropy=float(last.entropy),
        total_loss=float(last.total),
        clip_fraction=last.clip_fraction,
        approx_kl=last.approx_kl,
        max_abs_ratio_dev_epoch1_mb1=first_dev,
        max_abs_ratio_dev=last.max_abs_ratio_dev,
        explained_variance=explained,
        staleness_logprob_l1=staleness,
        skipped_minibatches=skipped,
        grad_norm=grad_norm,
    )


@torch.no_grad()
def _recompute_old(
    policy, buffer, minibatches, burn_in, trained, autocast_dtype, device
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Returns (logprob_old (N, T), value_old (N, T+1), staleness_l1).

    value_old carries T+1 entries: the trained region plus the bootstrap slot
    forward_chunk emits beyond it."""
    logprobs, values, staleness = [], [], []
    for envs in minibatches:
        chunk = buffer.chunk(envs)
        with torch.autocast(device.type, dtype=autocast_dtype):
            output = policy.forward_chunk(
                chunk.latent, chunk.aux_state, chunk.prev_action, chunk.prev_reward,
                chunk.abs_pos, chunk.episode_id, burn_in,
            )
        log_probabilities = torch.log_softmax(output.logits.float(), dim=-1)
        action = chunk.action[:, burn_in:]
        gathered = log_probabilities.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        logprobs.append(gathered[:, : trained.stop - trained.start])
        values.append(output.value.float())
        staleness.append(
            (gathered[:, : trained.stop - trained.start] - chunk.rollout_logprob[:, trained])
            .abs()
            .mean()
        )
    return (
        torch.cat(logprobs, dim=0),
        torch.cat(values, dim=0),
        float(torch.stack(staleness).mean()),
    )


def _gather(buffer: RolloutBuffer, minibatches, field: str) -> torch.Tensor:
    """Reassembles one buffer field in minibatch order, so its rows line up
    with the concatenated pi_old tensors."""
    return torch.cat([getattr(buffer.chunk(envs), field) for envs in minibatches], dim=0)


def _explained_variance(value: torch.Tensor, target: torch.Tensor) -> float:
    """1 - Var(target - value) / Var(target). Zero means the critic is no
    better than predicting the mean; negative means worse."""
    variance = target.var()
    if float(variance) == 0.0:
        return 0.0
    return float(1.0 - (target - value).var() / variance)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_update.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Prove the load-bearing test can fail**

In `run_update`, change `logprob_old[rows]` to `chunk.rollout_logprob[:, trained]`
— using the rollout-recorded logprobs, which is exactly the bug requirement 1
exists to prevent. `test_the_first_minibatch_of_epoch_one_has_a_ratio_of_exactly_one`
must go red. Revert. **Name this specific verification in the task report;** a
version of the test that stays green under this mutation is decorative and the
task is not done.

- [ ] **Step 6: Prove the normalization test can fail**

Move the advantage normalization inside the minibatch loop, normalizing
`advantage[rows]` per minibatch.
`test_advantages_are_normalized_once_over_the_whole_update_batch` must go red.
Revert.

- [ ] **Step 7: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/ppo/update.py tests/unit/test_ppo_update.py
git commit -m "feat(ppo): update pass with recomputed pi_old and the epoch-1 ratio invariant"
```

---

### Task 12: Checkpoint orchestration

**Files:**
- Create: `src/ppo/checkpoint.py`
- Test: `tests/unit/test_ppo_checkpoint.py`

**Interfaces:**
- Consumes: `checkpointing.io.{save_checkpoint, load_checkpoint, prune_checkpoints}`, `sequence_model.checkpoint.{build_policy_checkpoint_state, restore_policy_checkpoint, rebuild_cache, capture_rng_state, restore_rng_state}`, `pokemon_env.checkpoint.{build_env_checkpoint_state, restore_env_checkpoint}`, `ReturnScaler.state_dict` (Task 8).
- Produces: `write_checkpoint(directory, update, global_step, policy, optimizer, scheduler, cache, vec_env, scaler, config, init_state_hash, wandb_run_id) -> Path` and `resume(directory, policy, optimizer, scheduler, vec_env, scaler, policy_config, config, init_state_hash) -> ResumeResult | None`, plus `POLICY_PATTERN`, `MANIFEST_PATTERN`. Task 15 calls both.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ppo_checkpoint.py`:

```python
"""Checkpoint pairing. save_checkpoint is already atomic per file; the failure
it cannot see is one of the two files landing."""

from __future__ import annotations

import json

import pytest
import torch

from ppo.checkpoint import MANIFEST_PATTERN, resume, write_checkpoint


def test_the_manifest_names_both_files_and_their_sizes(tmp_path) -> None:
    harness = _checkpoint_harness(tmp_path)

    write_checkpoint(**harness.kwargs(update=3))
    manifest = json.loads((tmp_path / "manifest_update3.json").read_text())

    assert set(manifest) >= {"update", "global_step", "policy_file", "env_file", "sizes"}


def test_resume_returns_none_when_the_directory_is_empty(tmp_path) -> None:
    harness = _checkpoint_harness(tmp_path)

    assert resume(**harness.resume_kwargs()) is None


def test_resume_skips_a_checkpoint_whose_manifest_was_never_written(tmp_path) -> None:
    """A crash between the two .pt writes and the manifest write leaves an
    incoherent pair. Taking the newest .pt file regardless would resume a
    policy against an env from a different update."""
    harness = _checkpoint_harness(tmp_path)
    write_checkpoint(**harness.kwargs(update=1))
    write_checkpoint(**harness.kwargs(update=2))
    (tmp_path / "manifest_update2.json").unlink()

    result = resume(**harness.resume_kwargs())

    assert result.update == 1


def test_resume_skips_a_checkpoint_whose_env_file_is_truncated(tmp_path) -> None:
    harness = _checkpoint_harness(tmp_path)
    write_checkpoint(**harness.kwargs(update=1))
    write_checkpoint(**harness.kwargs(update=2))
    (tmp_path / "env_update2.pt").write_bytes(b"short")

    result = resume(**harness.resume_kwargs())

    assert result.update == 1


def test_resume_restores_the_return_scaler_state(tmp_path) -> None:
    harness = _checkpoint_harness(tmp_path)
    harness.scaler.update(torch.tensor([[-10.0, 10.0]]))
    write_checkpoint(**harness.kwargs(update=1))
    harness.scaler.load_state_dict({"count": 0.0, "mean": 0.0, "m2": 0.0})

    resume(**harness.resume_kwargs())

    assert harness.scaler.scale == pytest.approx(10.0, rel=0.05)


def test_resume_drops_the_cache_when_the_context_length_changed(tmp_path) -> None:
    """A curriculum stage that raises context_len cannot reuse the ring
    buffer. That is reported and the run warms up again, rather than raising."""
    harness = _checkpoint_harness(tmp_path)
    write_checkpoint(**harness.kwargs(update=1))

    result = resume(**harness.resume_kwargs(context_len=8))

    assert result.cache is None
```

Add `_checkpoint_harness`, building a tiny policy, an `AdamW`, a real
`RolloutCache`, a `FakeVecEnv` whose `state_dict`/`load_state_dict` round-trip a
dict, and a `ReturnScaler`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppo.checkpoint'`.

- [ ] **Step 3: Implement the orchestration**

Create `src/ppo/checkpoint.py`:

```python
"""Paired policy + env checkpoints, committed by a manifest.

checkpointing.io.save_checkpoint is already atomic per file (.tmp + replace).
The failure it cannot see is ONE OF THE TWO landing: a crash between the policy
write and the env write leaves a policy that believes in a game position the
emulator no longer occupies. The manifest is written last and is the commit
point; a resume that finds no manifest, or files whose sizes disagree with it,
falls back to the previous update.

Filename globs are distinct from contrastive_pretrain's and from the env's so
one network volume can hold every run's checkpoints without any of them
pruning another's resume point."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from checkpointing.io import load_checkpoint, prune_checkpoints, save_checkpoint
from pokemon_env.checkpoint import build_env_checkpoint_state, restore_env_checkpoint
from ppo.config import PPOConfig
from ppo.normalizer import ReturnScaler
from sequence_model.checkpoint import (
    build_policy_checkpoint_state,
    capture_rng_state,
    rebuild_cache,
    restore_policy_checkpoint,
    restore_rng_state,
)
from sequence_model.config import PolicyConfig

logger = logging.getLogger(__name__)

POLICY_PATTERN = "policy_update*.pt"
ENV_PATTERN = "env_update*.pt"
MANIFEST_PATTERN = "manifest_update*.json"


@dataclass(frozen=True)
class ResumeResult:
    update: int
    global_step: int
    cache: object | None
    wandb_run_id: str | None


def write_checkpoint(
    directory: Path,
    update: int,
    global_step: int,
    policy,
    optimizer,
    scheduler,
    cache,
    vec_env,
    scaler: ReturnScaler,
    config: PPOConfig,
    init_state_hash: str,
    wandb_run_id: str | None,
    git_commit: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    policy_file = directory / f"policy_update{update}.pt"
    env_file = directory / f"env_update{update}.pt"

    save_checkpoint(
        policy_file,
        build_policy_checkpoint_state(
            update, global_step, policy, optimizer, scheduler, cache, capture_rng_state()
        ),
    )
    save_checkpoint(env_file, build_env_checkpoint_state(update, vec_env, init_state_hash))

    manifest = {
        "update": update,
        "global_step": global_step,
        "policy_file": policy_file.name,
        "env_file": env_file.name,
        "sizes": {policy_file.name: policy_file.stat().st_size, env_file.name: env_file.stat().st_size},
        "return_scaler": scaler.state_dict(),
        "torch_version": torch.__version__,
        "git_commit": git_commit,
        "frozen_encoder_revision": config.frozen_encoder_revision,
        "wandb_run_id": wandb_run_id,
    }
    manifest_file = directory / f"manifest_update{update}.json"
    # Written LAST. Everything above may exist without this; nothing resumes
    # from a checkpoint this file does not name.
    manifest_file.write_text(json.dumps(manifest, indent=2))

    for pattern in (POLICY_PATTERN, ENV_PATTERN, MANIFEST_PATTERN):
        prune_checkpoints(directory, config.keep_last_n, pattern)
    logger.info("checkpoint_written", extra={"update": update, "path": str(manifest_file)})
    return manifest_file


def resume(
    directory: Path,
    policy,
    optimizer,
    scheduler,
    vec_env,
    scaler: ReturnScaler,
    policy_config: PolicyConfig,
    config: PPOConfig,
    init_state_hash: str,
) -> ResumeResult | None:
    for manifest_file in sorted(directory.glob(MANIFEST_PATTERN), reverse=True):
        manifest = json.loads(manifest_file.read_text())
        if not _files_intact(directory, manifest):
            logger.warning(
                "checkpoint_incomplete_skipped", extra={"manifest": str(manifest_file)}
            )
            continue

        policy_state = load_checkpoint(directory / manifest["policy_file"])
        restore_policy_checkpoint(policy, optimizer, scheduler, policy_state)
        restore_env_checkpoint(
            vec_env, load_checkpoint(directory / manifest["env_file"]), init_state_hash
        )
        scaler.load_state_dict(manifest["return_scaler"])
        restore_rng_state(policy_state.get("rng_state"))

        cache = _restore_cache(policy_state, policy_config)
        logger.info(
            "resumed_from_checkpoint",
            extra={"update": manifest["update"], "global_step": manifest["global_step"]},
        )
        return ResumeResult(
            update=int(manifest["update"]),
            global_step=int(manifest["global_step"]),
            cache=cache,
            wandb_run_id=manifest.get("wandb_run_id"),
        )
    return None


def _files_intact(directory: Path, manifest: dict) -> bool:
    for name, size in manifest["sizes"].items():
        path = directory / name
        if not path.exists() or path.stat().st_size != size:
            return False
    return True


def _restore_cache(policy_state: dict, policy_config: PolicyConfig):
    """A context_len change at a curriculum boundary invalidates the ring
    buffer. Compared BEFORE calling rebuild_cache rather than catching its
    ValueError, which is raised for several distinct reasons."""
    cache_state = policy_state.get("cache")
    if cache_state is None:
        return None
    saved_context = int(cache_state["k"].shape[3])
    if saved_context != policy_config.context_len:
        logger.warning(
            "cache_dropped_context_changed",
            extra={"saved": saved_context, "live": policy_config.context_len},
        )
        return None
    return rebuild_cache(cache_state, policy_config)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_checkpoint.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Prove the manifest-commit test can fail**

Replace `resume`'s manifest scan with a scan of `POLICY_PATTERN` that takes the
newest policy file regardless.
`test_resume_skips_a_checkpoint_whose_manifest_was_never_written` must go red
(returning update 2). Revert.

- [ ] **Step 6: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/ppo/checkpoint.py tests/unit/test_ppo_checkpoint.py
git commit -m "feat(ppo): manifest-committed paired policy and env checkpoints"
```

---

### Task 13: PPO telemetry

**Files:**
- Create: `src/ppo/telemetry.py`
- Test: `tests/unit/test_ppo_telemetry.py`

**Interfaces:**
- Consumes: `UpdateStats` (Task 11), `pokemon_env.telemetry.rollout_metrics` (Task 4), `policy.diagnostics` (Task 6).
- Produces: `STEP_METRICS` (the `define_metric` mapping), `update_metrics(stats, env_metrics, update, global_step, iteration_s, lr, peak_vram_gb) -> dict[str, float]`, and `wandb_config(ppo_config, env_config, policy_config, gate_results, git_commit, gpu_name) -> dict`. Task 15 uses all three.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ppo_telemetry.py`:

```python
"""Metric assembly. Instrumentation bugs are invisible in every test that does
not look at the output."""

from __future__ import annotations

import pytest

from ppo.telemetry import STEP_METRICS, update_metrics, wandb_config


def test_every_metric_family_declares_an_x_axis() -> None:
    """Without define_metric, wandb uses its own internal step counter and a
    resumed run's points land at the wrong x."""
    assert set(STEP_METRICS.values()) == {"train/update"}


def test_update_metrics_carries_the_x_axis_value_as_a_field() -> None:
    """log() must never pass step=; the axis travels as a normal metric."""
    metrics = update_metrics(**_stats_kwargs(update=7))

    assert metrics["train/update"] == pytest.approx(7.0)


def test_update_metrics_reports_the_epoch_one_ratio_deviation() -> None:
    metrics = update_metrics(**_stats_kwargs(max_abs_ratio_dev_epoch1_mb1=0.0))

    assert metrics["ratio/max_abs_dev_epoch1_mb1"] == pytest.approx(0.0)


def test_update_metrics_merges_the_env_metrics_unchanged() -> None:
    metrics = update_metrics(**_stats_kwargs(env_metrics={"reward/mean": 0.25}))

    assert metrics["reward/mean"] == pytest.approx(0.25)


def test_no_secret_shaped_key_reaches_the_wandb_config() -> None:
    """A W&B config is readable by everyone with project access."""
    config = wandb_config(**_config_kwargs())
    suspicious = [key for key in config if any(word in key.lower() for word in ("token", "key", "secret", "password"))]

    assert suspicious == []


def test_the_wandb_config_records_the_pinned_encoder_revision() -> None:
    config = wandb_config(**_config_kwargs())

    assert config["ppo/frozen_encoder_revision"] == "abc123"
```

Add `_stats_kwargs` and `_config_kwargs` as module-level helpers building a
default `UpdateStats` and the three dataclasses.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppo.telemetry'`.

- [ ] **Step 3: Implement the telemetry**

Create `src/ppo/telemetry.py`:

```python
"""Per-update scalars and the W&B config.

Numbers moving over time go to W&B; events with a cause go to the JSON-lines
log. Nothing is emitted per env step -- 65,536 of those per update would be a
self-inflicted outage."""

from __future__ import annotations

from dataclasses import asdict

from pokemon_env.config import EnvConfig
from ppo.config import PPOConfig
from ppo.update import UpdateStats
from sequence_model.config import PolicyConfig

# Every metric this trainer logs is a per-update scalar, so one axis covers
# them all. Declared via define_metric so log() never passes step=.
STEP_METRICS: dict[str, str] = {
    "loss/*": "train/update",
    "ratio/*": "train/update",
    "value/*": "train/update",
    "train/*": "train/update",
    "staleness/*": "train/update",
    "reward/*": "train/update",
    "env/*": "train/update",
    "progress/*": "train/update",
    "explore/*": "train/update",
    "episode/*": "train/update",
    "attn/*": "train/update",
    "model/*": "train/update",
    "perf/*": "train/update",
    "system/*": "train/update",
}


def update_metrics(
    stats: UpdateStats,
    env_metrics: dict[str, float],
    update: int,
    global_step: int,
    iteration_s: float,
    env_steps_per_sec: float,
    lr: float,
    peak_vram_gb: float,
) -> dict[str, float]:
    metrics = {
        "train/update": float(update),
        "train/env_step": float(global_step),
        "train/lr": float(lr),
        "train/grad_norm": stats.grad_norm,
        "train/skipped_minibatches": float(stats.skipped_minibatches),
        "loss/policy": stats.policy_loss,
        "loss/value": stats.value_loss,
        "loss/entropy": stats.entropy,
        "loss/total": stats.total_loss,
        "ratio/max_abs_dev_epoch1_mb1": stats.max_abs_ratio_dev_epoch1_mb1,
        "ratio/max_abs_dev": stats.max_abs_ratio_dev,
        "ratio/clip_fraction": stats.clip_fraction,
        "ratio/approx_kl": stats.approx_kl,
        "staleness/logprob_l1": stats.staleness_logprob_l1,
        "value/explained_variance": stats.explained_variance,
        "perf/iteration_s": float(iteration_s),
        "perf/env_steps_per_sec": float(env_steps_per_sec),
        "system/peak_vram_gb": float(peak_vram_gb),
    }
    metrics.update(env_metrics)
    return metrics


def wandb_config(
    ppo_config: PPOConfig,
    env_config: EnvConfig,
    policy_config: PolicyConfig,
    gate_results: dict,
    git_commit: str,
    gpu_name: str,
    torch_version: str,
) -> dict:
    """The three dataclasses plus provenance. Gate results are included so the
    chosen SDPA backend and the measured throughput are part of the run record
    rather than a number in someone's terminal scrollback.

    Nothing here reads the environment: a W&B config is readable by everyone
    with project access, and a credential in it is a credential published."""
    config: dict = {}
    for prefix, dataclass_instance in (
        ("ppo", ppo_config), ("env", env_config), ("policy", policy_config)
    ):
        for key, value in asdict(dataclass_instance).items():
            config[f"{prefix}/{key}"] = value
    for key, value in gate_results.items():
        config[f"gate/{key}"] = value
    config["run/git_commit"] = git_commit
    config["run/gpu"] = gpu_name
    config["run/torch_version"] = torch_version
    return config
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_telemetry.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Prove the secret test can fail**

Add `config["hf_token"] = "x"` to `wandb_config`.
`test_no_secret_shaped_key_reaches_the_wandb_config` must go red. Revert.

- [ ] **Step 6: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/ppo/telemetry.py tests/unit/test_ppo_telemetry.py
git commit -m "feat(ppo): per-update metrics, declared x-axes, and a credential-free W&B config"
```

---

### Task 14: The trainer loop

**Files:**
- Create: `src/ppo/trainer.py`
- Test: `tests/unit/test_ppo_trainer.py`

**Interfaces:**
- Consumes: everything from Tasks 7–13.
- Produces: `PPODeps` (dataclass) and `run_training(deps, max_updates=None) -> None`. Task 16's CLI calls it.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ppo_trainer.py`:

```python
"""The outer loop: cadence, resume, and the abort guards."""

from __future__ import annotations

import pytest

from ppo.trainer import PPODeps, run_training


def test_the_warmup_runs_before_the_first_update(tmp_path) -> None:
    """burn_in observations must exist before update 0, or L varies and
    torch.compile recompiles on the second update."""
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=1)

    assert harness.vec_env.step_calls == harness.burn_in + harness.n_steps


def test_a_checkpoint_is_written_at_the_configured_cadence(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, checkpoint_every_updates=2)

    run_training(harness.deps, max_updates=4)

    assert len(list(tmp_path.glob("manifest_update*.json"))) == 2


def test_an_approx_kl_above_the_threshold_aborts_the_run(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, forced_approx_kl=1.0)

    with pytest.raises(RuntimeError, match="approx_kl"):
        run_training(harness.deps, max_updates=2)


def test_the_epoch_one_ratio_invariant_is_asserted_not_merely_logged(tmp_path) -> None:
    harness = _trainer_harness(tmp_path, forced_epoch1_dev=0.01)

    with pytest.raises(AssertionError, match="epoch-1 minibatch-1 ratio"):
        run_training(harness.deps, max_updates=1)


def test_metrics_are_logged_once_per_update(tmp_path) -> None:
    harness = _trainer_harness(tmp_path)

    run_training(harness.deps, max_updates=3)

    assert len(harness.wandb_run.logged) == 3
```

`_trainer_harness` builds `PPODeps` with a `FakeVecEnv`, a fake encoder
returning zeros, a tiny real policy, a `FakeExperimentRun` recording `logged`,
and a `PPOConfig` with tiny shapes. `forced_approx_kl` and `forced_epoch1_dev`
inject a stub `run_update` through `PPODeps.run_update`, which defaults to the
real one — dependency injection rather than `mock.patch`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppo.trainer'`.

- [ ] **Step 3: Implement the trainer**

Create `src/ppo/trainer.py`:

```python
"""The outer PPO loop.

Order per iteration: rollout -> update -> telemetry -> cadence work
(checkpoint, artifacts, hub snapshot) -> buffer shift.

The buffer shift happens LAST, so a checkpoint written mid-iteration describes
a buffer state the next resume can reproduce by collecting n_steps fresh
observations."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from observability.tracking import ExperimentRunLike, NullExperimentRun
from pokemon_env.config import EnvConfig
from pokemon_env.telemetry import rollout_metrics
from ppo import checkpoint as ppo_checkpoint
from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from ppo.normalizer import ReturnScaler
from ppo.rollout import RolloutState, collect_rollout
from ppo.telemetry import update_metrics
from ppo.update import run_update
from sequence_model.config import PolicyConfig

logger = logging.getLogger(__name__)


@dataclass
class PPODeps:
    config: PPOConfig
    env_config: EnvConfig
    policy_config: PolicyConfig
    vec_env: object
    encoder: object
    policy: object
    optimizer: object
    scheduler: object | None
    device: torch.device
    autocast_dtype: torch.dtype
    init_state_hash: str
    git_commit: str
    wandb_run: ExperimentRunLike = field(default_factory=NullExperimentRun)
    run_update: Callable = run_update


def run_training(deps: PPODeps, max_updates: int | None = None) -> None:
    config = deps.config
    config.validate_against_n_envs(deps.env_config.n_envs)

    directory = Path(config.checkpoint_dir)
    scaler = ReturnScaler(config.gamma)
    buffer = RolloutBuffer(
        config, deps.policy_config, deps.env_config.n_envs, deps.device
    )
    cache = deps.policy.new_cache(
        deps.env_config.n_envs, deps.device, dtype=deps.autocast_dtype
    )
    generator = torch.Generator(device=deps.device).manual_seed(config.seed)

    resumed = ppo_checkpoint.resume(
        directory, deps.policy, deps.optimizer, deps.scheduler, deps.vec_env,
        scaler, deps.policy_config, config, deps.init_state_hash,
    )
    update = 0
    global_step = 0
    warmup_needed = True
    if resumed is not None:
        update, global_step = resumed.update + 1, resumed.global_step
        if resumed.cache is not None:
            cache, warmup_needed = resumed.cache, False
        logger.info(
            "resumed", extra={"update": update, "global_step": global_step}
        )

    deps.vec_env.reset()
    state = RolloutState(
        prev_action=torch.full(
            (deps.env_config.n_envs,),
            deps.policy_config.episode_start_action,
            dtype=torch.int64,
            device=deps.device,
        ),
        prev_reward=torch.zeros(deps.env_config.n_envs, device=deps.device),
    )

    rollout_kwargs = {
        "vec_env": deps.vec_env, "encoder": deps.encoder, "policy": deps.policy,
        "cache": cache, "buffer": buffer, "generator": generator,
        "device": deps.device,
        "episode_start_action": deps.policy_config.episode_start_action,
        "autocast_dtype": deps.autocast_dtype,
    }

    if warmup_needed:
        # Burn-in for update 0 and a gradient target for nothing. Without it L
        # varies on the first update and torch.compile recompiles.
        state = collect_rollout(state=state, n_steps=buffer.burn_in, **rollout_kwargs)
        # The bootstrap slot needs one observation beyond the trained region;
        # every later rollout inherits it from the previous update's shift.
        state = collect_rollout(state=state, n_steps=1, **rollout_kwargs)
        logger.info("warmup_complete", extra={"steps": buffer.burn_in + 1})

    while max_updates is None or update < (resumed.update + 1 if resumed else 0) + max_updates:
        started = time.monotonic()
        if deps.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        state = collect_rollout(state=state, n_steps=config.n_steps, **rollout_kwargs)
        global_step += config.n_steps * deps.env_config.n_envs

        stats = deps.run_update(
            deps.policy, deps.optimizer, deps.scheduler, buffer, scaler, config,
            deps.policy_config, deps.env_config.n_envs, deps.device, deps.autocast_dtype,
        )

        # An assert, not a metric: after recomputing pi_old the policy has not
        # changed at (epoch 1, minibatch 1), so any deviation is a real bug.
        assert stats.max_abs_ratio_dev_epoch1_mb1 < 1e-5, (  # noqa: S101
            f"epoch-1 minibatch-1 ratio deviated by "
            f"{stats.max_abs_ratio_dev_epoch1_mb1}; pi_old was not recomputed from "
            "the current weights"
        )
        if abs(stats.approx_kl) > config.abort_approx_kl:
            raise RuntimeError(
                f"approx_kl {stats.approx_kl} exceeded {config.abort_approx_kl}; "
                "aborting with the checkpoint intact"
            )

        elapsed = time.monotonic() - started
        env_metrics = rollout_metrics(
            deps.vec_env.last_step, deps.vec_env.last_components,
            deps.vec_env.clip_fire_rate, _respawns(deps.vec_env), deps.vec_env.stats(),
        )
        deps.wandb_run.log(
            update_metrics(
                stats, env_metrics, update, global_step, elapsed,
                config.n_steps * deps.env_config.n_envs / max(elapsed, 1e-9),
                _current_lr(deps.optimizer), _peak_vram_gb(deps.device),
            )
        )

        if update % config.checkpoint_every_updates == 0:
            ppo_checkpoint.write_checkpoint(
                directory, update, global_step, deps.policy, deps.optimizer,
                deps.scheduler, cache, deps.vec_env, scaler, config,
                deps.init_state_hash, _run_id(deps.wandb_run), deps.git_commit,
            )

        buffer.shift()
        update += 1

    logger.info("training_finished", extra={"update": update, "global_step": global_step})


def _respawns(vec_env) -> int:
    return sum(getattr(backend, "respawns", 0) for backend in getattr(vec_env, "_backends", []))


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _peak_vram_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _run_id(run: ExperimentRunLike) -> str | None:
    return getattr(run, "run_id", None)
```

`VecPokemonEnv` must expose `last_step`; add a `self._last_step` assignment in
`_collect` and a `last_step` property beside `last_components` if it is not
already there.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_trainer.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Prove the invariant assert can fail**

Change the assert's threshold to `< 1.0`.
`test_the_epoch_one_ratio_invariant_is_asserted_not_merely_logged` must go red.
Revert.

- [ ] **Step 6: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/ppo/trainer.py src/pokemon_env/vec_env.py tests/unit/test_ppo_trainer.py
git commit -m "feat(ppo): trainer loop with warmup, cadence, resume, and the abort guards"
```

---

### Task 15: Pre-flight gates

**Files:**
- Create: `src/ppo/preflight.py`
- Test: `tests/unit/test_ppo_preflight.py`

**Interfaces:**
- Consumes: `PPOConfig`, `PolicyConfig`.
- Produces: `sdpa_backend_report(policy_config, minibatch_envs, seq_len, device) -> dict`, `throughput_report(build_env, n_envs_candidates, steps) -> dict`, and `run_gates(...) -> dict` — the `gate_results` Task 13's `wandb_config` records.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ppo_preflight.py`:

```python
"""The gates themselves are CUDA measurements; what is unit-testable is that
they ask the right question with the right shapes."""

from __future__ import annotations

import pytest
import torch

from ppo.preflight import sdpa_params_for, sdpa_backend_report
from sequence_model.config import PolicyConfig


def _policy_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=8, head_dim=16, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=8, aux_state_dim=4,
    )


def test_the_query_has_query_head_width_and_the_key_has_kv_head_width() -> None:
    """attention.py calls SDPA with enable_gqa=True, so k and v are NOT
    expanded. A gate run with symmetric head counts measures a call the model
    never makes."""
    params = sdpa_params_for(_policy_config(), minibatch_envs=4, seq_len=16, device=torch.device("cpu"))

    assert (params.query.shape[1], params.key.shape[1]) == (8, 2)


def test_enable_gqa_is_set_on_the_params() -> None:
    """enable_gqa is itself an SDPAParams field and can disqualify a backend
    on its own."""
    params = sdpa_params_for(_policy_config(), minibatch_envs=4, seq_len=16, device=torch.device("cpu"))

    assert params.enable_gqa is True


def test_the_mask_is_a_bool_tensor_broadcastable_over_heads() -> None:
    params = sdpa_params_for(_policy_config(), minibatch_envs=4, seq_len=16, device=torch.device("cpu"))

    assert (params.attn_mask.dtype, params.attn_mask.shape) == (torch.bool, (4, 1, 16, 16))


def test_the_report_names_every_candidate_backend() -> None:
    report = sdpa_backend_report(_policy_config(), minibatch_envs=4, seq_len=16, device=torch.device("cpu"))

    assert set(report) == {"flash", "efficient", "shapes"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_preflight.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppo.preflight'`.

- [ ] **Step 3: Implement the gates**

Create `src/ppo/preflight.py`:

```python
"""Measurements that must pass before a paid run starts.

Gate 1 asks torch directly which SDPA backends can serve the model's real
call, rather than reading kernel names out of a profile. Verified against
torch 2.13: torch.nn.attention exposes can_use_flash_attention(params,
debug=True) and can_use_efficient_attention(params, debug=True), and
SDPAParams takes seven positional arguments --
(query, key, value, attn_mask, dropout, is_causal, enable_gqa)."""

from __future__ import annotations

import logging
import time

import torch
from torch.backends.cuda import SDPAParams
from torch.nn.attention import can_use_efficient_attention, can_use_flash_attention

from sequence_model.config import PolicyConfig

logger = logging.getLogger(__name__)


def sdpa_params_for(
    policy_config: PolicyConfig, minibatch_envs: int, seq_len: int, device: torch.device
) -> SDPAParams:
    """The model's real call shape. attention.py passes enable_gqa=True, so k
    and v keep n_kv_heads width and are never expanded to query-head width."""
    query = torch.zeros(
        minibatch_envs, policy_config.n_heads, seq_len, policy_config.head_dim,
        dtype=torch.bfloat16, device=device,
    )
    key = torch.zeros(
        minibatch_envs, policy_config.n_kv_heads, seq_len, policy_config.head_dim,
        dtype=torch.bfloat16, device=device,
    )
    mask = torch.ones(minibatch_envs, 1, seq_len, seq_len, dtype=torch.bool, device=device)
    return SDPAParams(query, key, key.clone(), mask, 0.0, False, True)


def sdpa_backend_report(
    policy_config: PolicyConfig, minibatch_envs: int, seq_len: int, device: torch.device
) -> dict:
    """Gate 1. A materialized bool mask rules out FlashAttention; MATH would
    materialize roughly 537 MB of scores at (8, 8, 2048) in bf16. If neither
    alternative is usable, the restructure decision surfaces here rather than
    after the money is spent."""
    params = sdpa_params_for(policy_config, minibatch_envs, seq_len, device)
    report = {
        "flash": bool(can_use_flash_attention(params, debug=True)),
        "efficient": bool(can_use_efficient_attention(params, debug=True)),
        "shapes": {
            "query": list(params.query.shape),
            "key": list(params.key.shape),
            "enable_gqa": True,
        },
    }
    logger.info("sdpa_backend_report", extra=report)
    return report


def throughput_report(build_env, n_envs_candidates: list[int], steps: int) -> dict:
    """Gate 2. Answers the env spec's open question -- whether 64 envs is right
    for our per-step cost -- with a number, and its measured iteration time
    sets checkpoint_every_updates.

    64 PyBoy workers are 64 processes, so vCPU is the binding constraint here,
    not VRAM."""
    results: dict[str, float] = {}
    for n_envs in n_envs_candidates:
        vec_env, buffer = build_env(n_envs)
        try:
            vec_env.reset()
            started = time.monotonic()
            actions = torch.zeros(n_envs, dtype=torch.int64).numpy()
            for _ in range(steps):
                vec_env.step(actions)
            elapsed = time.monotonic() - started
        finally:
            vec_env.close()
            buffer.close()
            buffer.unlink()
        results[f"env_steps_per_sec_at_{n_envs}"] = steps * n_envs / elapsed
    logger.info("throughput_report", extra=results)
    return results
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_preflight.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Prove the GQA shape test can fail**

Change `key`'s head dimension to `policy_config.n_heads`.
`test_the_query_has_query_head_width_and_the_key_has_kv_head_width` must go red.
Revert.

- [ ] **Step 6: Commit**

```bash
uv run pytest -q && uv run ruff check
git add src/ppo/preflight.py tests/unit/test_ppo_preflight.py
git commit -m "feat(ppo): SDPA-backend and throughput pre-flight gates"
```

---

### Task 16: CLI, and the slow-tier acceptance test

**Files:**
- Create: `src/ppo/cli.py`, `tests/integration/test_ppo_smoke.py`
- Modify: `pyproject.toml`, `CLAUDE.md`
- Test: `tests/unit/test_ppo_cli.py`

**Interfaces:**
- Consumes: everything.
- Produces: `pokemon-ppo train` and `pokemon-ppo preflight` entrypoints.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/test_ppo_cli.py`:

```python
"""CLI wiring. The heavy paths are exercised by the slow tier."""

from __future__ import annotations

import pytest

from ppo.cli import build_parser


def test_the_train_subcommand_defaults_to_resuming() -> None:
    args = build_parser().parse_args(["train"])

    assert args.fresh is False


def test_the_fresh_flag_opts_out_of_resuming() -> None:
    args = build_parser().parse_args(["train", "--fresh"])

    assert args.fresh is True


def test_the_preflight_subcommand_takes_the_env_counts_to_measure() -> None:
    args = build_parser().parse_args(["preflight", "--n-envs", "16", "32"])

    assert args.n_envs == [16, 32]


def test_an_unknown_subcommand_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nonsense"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_ppo_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppo.cli'`.

- [ ] **Step 3: Implement the CLI**

Create `src/ppo/cli.py` following `src/contrastive_pretrain/cli.py`'s structure:
`build_parser()` returning an `argparse.ArgumentParser` with `train` and
`preflight` subcommands, and `main()` that calls `configure()` from
`observability.logging_config`, loads `configs/ppo.yaml` and
`configs/pokemon_env.yaml` and `configs/sequence_model.yaml`, resolves the
frozen encoder with `load_frozen_encoder(repo_id, revision)` and
`load_latent_stats`, builds the vec env with `build_subprocess_vec_env`,
constructs `PPODeps`, and runs `run_training` inside
`with WandbRun(wandb, project="pokemon-ppo", name=..., config=wandb_config(...), step_metrics=STEP_METRICS, run_id=...) as run:`.

The `run_id` is read from `<checkpoint_dir>/wandb_run_id.txt` if present and
written there on first start, so a preempted run resumes into the same W&B run
rather than fragmenting the curve.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_cli.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Register the entrypoint**

In `pyproject.toml`, add to `[project.scripts]`:

```toml
pokemon-ppo = "ppo.cli:main"
```

- [ ] **Step 6: Write the slow-tier acceptance test**

Create `tests/integration/test_ppo_smoke.py`:

```python
"""Opt-in acceptance test against the real ROM and real PyBoy workers.

Run with:
    uv run pytest -m slow tests/integration/test_ppo_smoke.py -v

Auto-skips when the ROM or init.state is absent, so a fresh checkout never
fails. Four envs and three updates rather than 64 and thousands, so this is
minutes rather than days; the loop is identical."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_ROM = Path("Pokemon Red.gb")
_INIT = Path("artifacts/init.state")

_needs_rom = pytest.mark.skipif(
    not _ROM.exists(), reason=f"{_ROM} not present; it is gitignored and must be supplied locally"
)
_needs_init_state = pytest.mark.skipif(
    not _INIT.exists(), reason="artifacts/init.state not generated; see src/pokemon_env/init_state.py"
)


@_needs_rom
@_needs_init_state
def test_three_real_updates_hold_the_epoch_one_ratio_invariant(tmp_path) -> None:
    """The sub-project's acceptance gate, minus the pod. Four real PyBoy
    processes, real frames, a real frozen encoder, and a real PPO update --
    with the invariant asserted inside run_training on every update."""
    harness = _real_harness(tmp_path, n_envs=4, n_steps=8)

    run_training(harness.deps, max_updates=3)

    assert harness.wandb_run.logged[-1]["ratio/max_abs_dev_epoch1_mb1"] == pytest.approx(0.0, abs=1e-5)


@_needs_rom
@_needs_init_state
def test_a_run_resumes_from_its_checkpoint_without_a_loss_discontinuity(tmp_path) -> None:
    """Resume is state-faithful, not bit-reproducible: PyBoy across respawned
    subprocesses does not reproduce a byte-identical step ordering. This
    asserts the checkpoint round-trips and training continues, never bitwise
    equality."""
    harness = _real_harness(tmp_path, n_envs=4, n_steps=8, checkpoint_every_updates=1)
    run_training(harness.deps, max_updates=2)

    resumed = _real_harness(tmp_path, n_envs=4, n_steps=8, checkpoint_every_updates=1)
    run_training(resumed.deps, max_updates=1)

    assert resumed.wandb_run.logged[0]["train/update"] == pytest.approx(2.0)
```

- [ ] **Step 7: Run the slow tier**

Run: `uv run pytest -m slow tests/integration/test_ppo_smoke.py -v`
Expected: PASS if the ROM and `artifacts/init.state` are present; two skips with
their stated reasons otherwise. Report which happened.

- [ ] **Step 8: Update `CLAUDE.md`**

In the "Codebase map" table add a row:

```
| `src/ppo/` | Rollout, GAE, clipped losses, the update pass, checkpoint orchestration, telemetry, pre-flight gates | Knows about RAM addresses or emulator internals |
```

In the entry-points paragraph, add `pokemon-ppo {train,preflight}`. In the
opening section, replace "**PPO is the only stage not yet built**" with a line
recording that `src/ppo/` now exists and that the four gates in the spec's §8
are what stand between it and the first paid run.

- [ ] **Step 9: Run every gate**

```bash
uv run pytest -q
uv run ruff check
uv run python scripts/audit_tests.py tests/
uv run python ~/.claude/skills/observability-expert/scripts/audit_observability.py src/
```

Expected: suite green with branch coverage at or above 93%; ruff clean;
`audit_tests.py` at or below the 11-finding baseline; `audit_observability.py`
at 9 findings. Report each number.

- [ ] **Step 10: Commit**

```bash
git add src/ppo/cli.py pyproject.toml tests/unit/test_ppo_cli.py tests/integration/test_ppo_smoke.py CLAUDE.md
git commit -m "feat(ppo): pokemon-ppo CLI, slow-tier acceptance test, and docs"
```

---

## Self-Review

**Spec coverage.** §1 framework → Task 15 (verified APIs) and Global
Constraints. §2 package layout → all tasks. §3 config → Task 1. §4 rollout →
Task 10. §5 update, indexing, `π_old`, GAE, return scaler, invariant → Tasks 7,
8, 9, 11. §6 checkpoint and failure handling → Tasks 12 and 14. §7
observability, both env gaps, `WandbRun` → Tasks 3, 4, 5, 13. §8 gates →
Task 15 for gates 1–2, Task 14's memory behaviour plus Task 16's slow tier for
gates 3–4; **gate 3 (the memory probe) and gate 4 (the 50-update pod run) are
executed on a pod, not in the suite** — Task 16 ships the code path and the
report records the measured numbers. §9 curriculum room → Task 1's derived
shape helpers and Task 12's `_restore_cache`. §10 testing → every task's test
steps. §11 required changes → Tasks 1–6, one task per package.

**Placeholder scan.** No "TBD", no "add appropriate error handling", no "similar
to Task N". Three places name a file to read before writing (`policy.py:57`'s
constructor signature in Task 6, `contrastive_pretrain/cli.py`'s structure in
Task 16, and the `EnvBackend` fake in Task 3) — these are pointers to existing
code, not deferred decisions.

**Type consistency.** `stats()` returns `dict` at the session and backend level
and `list[dict]` at the vec-env level, consistently in Tasks 2, 3, 4, and 14.
`rollout_metrics` gains its fifth parameter in Task 4 and is called with five
arguments in Task 14. `UpdateStats` field names in Task 11 match every read in
Tasks 13 and 14. `ChunkInputs` field names in Task 7 match every use in
Task 11. `ReturnScaler.state_dict` in Task 8 matches the manifest key in
Task 12.

**One gap found and closed inline:** Task 14 requires `VecPokemonEnv.last_step`,
which does not exist today — the modification is now named in Task 14's Step 3
and its file list.
