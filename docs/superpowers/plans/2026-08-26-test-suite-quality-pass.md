# Test Suite Quality Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the whole test suite (25 unit files + 1 integration file, currently 199 passing / 13 deselected, 83% branch coverage) to the `pytest-expert` skill's hard-gate bar — strict config, no float `==`, no assertion-free tests, no unregistered/ambiguous markers, no duplicated test-double/fixture code — and close the coverage/DRY gaps four parallel review passes found, while resolving the two deferred follow-up items from `docs/superpowers/plans/2026-08-25-contrastive-pretrain-resize-cache-followups.md`.

**Architecture:** No production-code behavior changes — this is a test-suite-only pass. One task per file (or tightly-coupled file group) so each is independently reviewable: install the strict config gate first (so every later task is validated against it), consolidate duplicated test doubles/fixtures into `tests/conftest.py` (project-wide) or a new `tests/unit/conftest.py` (contrastive_pretrain-scoped), then work through correctness fixes (float comparisons, loose `pytest.raises`, assertion-free tests, branching-that-should-be-parametrize) and new coverage-gap tests file by file.

**Tech Stack:** pytest 9.1.1, pytest-cov, `uv`, PyTorch 2.13, `datasets`, `huggingface_hub` 1.28.0.

**Spec:** This plan's "spec" is the aggregated findings below — the project's own `pytest-expert` skill's audit script (`scripts/audit_tests.py`) run against `tests/`, `ruff check --config assets/ruff-pytest.toml` run against `tests/`, and four parallel subagent reviews (one per package: contrastive_pretrain data/model, contrastive_pretrain training/ops, data_collection, hf_storage/observability/integration), cross-checked against the actual source files. No separate spec doc exists; this plan **is** the spec-plus-plan for this initiative, per this project's "own spec and implementation plan before code is written" convention.

## Global Constraints

- Package management: `uv` only — `uv add --group dev <pkg>`, `uv run <cmd>`. No bare `pip`/`venv`.
- Every new/changed test must independently satisfy `python /Users/theelusivegerbilfish/.claude/skills/pytest-expert/scripts/audit_tests.py tests/` (zero findings) once this plan is complete — run it against the whole `tests/` directory, never a single file (the script's `Path.rglob` silently returns nothing when pointed at a file, not a directory — a real bug worth reporting upstream, not this plan's to fix).
- No production code in `src/` changes in this plan. If any new/strengthened test fails against current `src/` behavior, STOP — that's a real bug the review surfaced, not a plan-execution error — and report it instead of "fixing" the test to match broken behavior.
- Every task's new/changed tests must pass under the strict config Task 1 installs (this makes Task 1 a hard dependency of every later task — do it first).
- Keep diffs surgical: don't reformat or restructure code the task doesn't touch, per this project's Karpathy-guidelines convention.

---

### Task 1: Install the strict pytest config gate, add pytest-cov, fix the two warnings it surfaces

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/unit/test_contrastive_pretrain_checkpoint.py:174-178`

**Interfaces:**
- Produces: the `slow` / `expensive` marker split every later task (9, and any test needing to select/deselect them) relies on; the `--cov-fail-under=80` gate every later task's coverage-adding work is measured against.

**Context:** Baseline measurements taken directly against this repo before writing this plan:
- `uv run pytest -q -m "not slow"` → 199 passed, 13 deselected, **2 real warning sites** (4 total warning instances): (a) `test_contrastive_pretrain_checkpoint.py:178` calls `scheduler.step()` before `optimizer.step()` ever runs, triggering PyTorch's `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()`; (b) `torchdata/stateful_dataloader/stateful_dataloader.py:379` calls the deprecated `torch.set_vital(...)` internally (third-party, not our code — hit by 3 of the dataloader tests).
- `uv run --with pytest-cov pytest -q -m "not slow" --cov=src --cov-branch --cov-report=term-missing:skip-covered` → **83% total branch coverage**. This is the correct baseline for a `-m "not slow and not expensive"` selection too, since today ALL of `test_contrastive_pretrain_train.py`'s `run_training`-driving tests are tagged `slow` (Task 9 splits them into `slow`/`expensive`, but that split doesn't change what today's "not slow" run already excludes).
- `pytest-cov` is not currently a dependency (only `pytest>=9.1.1` is, in `[dependency-groups] dev`) — must be added.
- Project has no `[tool.ruff]` config and no `ruff` dev dependency. This plan deliberately does NOT add ruff as a project dependency/gate (that's a separate infra decision outside "evaluate the test suite" scope) — `ruff`'s PT findings were used only as one-time input to this plan's fixes (Tasks 4-7), not adopted as an ongoing gate.

- [ ] **Step 1: Add pytest-cov as a dev dependency**

```bash
uv add --group dev pytest-cov
```

- [ ] **Step 2: Fix the real warning — reorder `optimizer.step()` before `scheduler.step()`**

In `tests/unit/test_contrastive_pretrain_checkpoint.py`, the test builds a scheduler and steps it 10 times with no corresponding optimizer step, which is what triggers the warning. Since the test only cares about the scheduler's LR schedule (not making real optimizer updates), add a no-op `optimizer.step()` before each `scheduler.step()`:

```python
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
```

- [ ] **Step 3: Replace `[tool.pytest.ini_options]` in `pyproject.toml`**

`pytest>=9.1.1` is a floating (not pinned) dependency, so per the pytest-expert skill's own config template, use the four individual `strict_*` keys rather than the blanket `strict = true` (which auto-adopts future strictness options — only safe with a pinned pytest). Replace the existing block:

```toml
[tool.pytest.ini_options]
minversion = "9.0"
testpaths = ["tests"]
strict_config = true
strict_markers = true
strict_xfail = true
strict_parametrization_ids = true
addopts = [
    "-ra",
    "--tb=short",
    "--cov=src",
    "--cov-branch",
    "--cov-report=term-missing:skip-covered",
    "--cov-fail-under=80",
    "-m", "not slow and not expensive",
]
markers = [
    "slow: requires real network/ffmpeg/HF credentials -- e.g. downloading real Hub shards or the real pretrained ImageNet ResNet-50 weights (deselect with -m 'not slow')",
    "expensive: CPU-bound and slow but needs no network/credentials -- e.g. anything driving run_training, which trains a real (uncached) torch.compile of a ResNet-50 on CPU at ~60s per call (deselect with -m 'not expensive')",
]
filterwarnings = [
    "error",
    # torchdata's StatefulDataLoader calls the deprecated torch.set_vital()
    # internally; nothing in this project's code path can avoid it. Third-party,
    # not ours to fix -- revisit if a torchdata upgrade removes the call.
    "ignore:'set_vital' is deprecated, please do not call:UserWarning:torchdata.stateful_dataloader.stateful_dataloader",
]

[tool.coverage.run]
branch = true
source = ["src"]
omit = ["*/__main__.py"]

[tool.coverage.report]
show_missing = true
skip_covered = true
fail_under = 80
exclude_also = [
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "@(abc\\.)?abstractmethod",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
]
```

Note: `-m "not slow and not expensive"` references the `expensive` marker before Task 9 retags any test with it — this is fine, `strict_markers` only rejects *used-but-unregistered* markers, and an addopts `-m` selector referencing a marker no test yet carries is a no-op filter, not an error.

- [ ] **Step 4: Run the full suite and confirm it's green under the new strict config**

```bash
uv run pytest -q
```

Expected: same 199 passed (13 still deselected — they're `slow`, correctly excluded), zero warnings, coverage report prints with total ≥80%. If `--cov=src` errors because `pytest-cov` isn't picked up, re-run `uv sync` first. If any previously-passing test now fails under `filterwarnings = ["error"]`, that's a **new** warning this baseline run didn't hit (nondeterministic import order, etc.) — add a narrowly-scoped `ignore:` line for it with a comment, following the same pattern as the torchdata line above; do not blanket-ignore.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/unit/test_contrastive_pretrain_checkpoint.py
git commit -m "test: install strict pytest config gate, coverage floor, and slow/expensive marker split"
```

---

### Task 2: Consolidate the three duplicated `FakeHfClient` definitions

**Files:**
- Modify: `tests/unit/test_hf_uploader.py:1-16, 81, 93, 152`
- Modify: `tests/unit/test_pipeline.py:1-22, 60, 108, 269`
- Modify: `tests/unit/test_contrastive_pretrain_encoder_io.py:14, 157, 177, 257, 278`

**Interfaces:**
- Consumes: `tests/conftest.py`'s existing `FakeHfClient` class (`files: dict[str, bytes]`, `upload_calls: list[str]`, `commits: list[dict]`, methods `upload_bytes`/`upload_many_bytes`/`download_bytes`) and its `fake_hf_client` fixture — both already exist, unchanged by this task.

**Context:** `test_hf_uploader.py` and `test_pipeline.py` each define a byte-for-byte identical local `FakeHfClient` (a strict subset of the shared one — same `upload_bytes`/`download_bytes`, missing only `upload_many_bytes`, which `HfUploader` never calls) but store uploaded bytes under `self.uploads`, while the shared conftest class stores them under `self.files`. `test_contrastive_pretrain_encoder_io.py` already imports the shared class (`from tests.conftest import FakeHfClient as _FakeHfClient`) but bypasses the `fake_hf_client` fixture that exists for exactly this, instantiating it manually 4 times instead.

- [ ] **Step 1: `test_hf_uploader.py` — delete the local class, import the shared one, rename `.uploads` → `.files`**

Delete lines 8-16 (the local `class FakeHfClient` block). Add to the top-of-file imports:

```python
from tests.conftest import FakeHfClient
```

Rename every `client.uploads[...]` to `client.files[...]` — lines 81, 93, 152 (each is `assert client.uploads["..."] == b"..."` → `assert client.files["..."] == b"..."`, no other change). `_FlakyThenSucceedsClient(FakeHfClient)` and `_AlwaysRateLimitedClient(FakeHfClient)` need no changes — both only override `upload_bytes`, and the shared class's `upload_bytes` has the identical signature.

- [ ] **Step 2: `test_pipeline.py` — same treatment**

Delete lines 14-22 (the local `class FakeHfClient` block). Add:

```python
from tests.conftest import FakeHfClient
```

Rename `client.uploads` → `client.files` at lines 60, 108, 269 (each is `assert any(path.startswith("...") for path in client.uploads)` → same with `client.files`).

- [ ] **Step 3: `test_contrastive_pretrain_encoder_io.py` — use the `fake_hf_client` fixture instead of manual construction**

Remove the `from tests.conftest import FakeHfClient as _FakeHfClient` import (line 14) — no longer needed. Add `fake_hf_client` as a parameter to each of the four test functions that currently write `client = _FakeHfClient()`, and use `fake_hf_client` in place of `client` for the rest of that test body:

- `test_push_frozen_encoder_uploads_three_files` (was line 157) → add `fake_hf_client` param, delete the `client = _FakeHfClient()` line, use `fake_hf_client` in the `push_frozen_encoder(...)` call and the two asserts below it.
- `test_push_frozen_encoder_publishes_all_three_files_as_one_atomic_commit` (was line 177) → same treatment.
- `test_load_frozen_encoder_from_client_matches_exported_weights` (was line 257) → same treatment.
- `test_load_frozen_encoder_from_client_rejects_mismatched_config` (was line 278) → same treatment.

- [ ] **Step 4: Run the affected files and confirm green**

```bash
uv run pytest -q tests/unit/test_hf_uploader.py tests/unit/test_pipeline.py tests/unit/test_contrastive_pretrain_encoder_io.py
```

Expected: all pass, same counts as before.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_hf_uploader.py tests/unit/test_pipeline.py tests/unit/test_contrastive_pretrain_encoder_io.py
git commit -m "test: consolidate three duplicated FakeHfClient doubles onto the shared conftest one"
```

---

### Task 3: New `tests/unit/conftest.py` — shared `encoder` fixture, refactor its 17 call sites

**Files:**
- Create: `tests/unit/conftest.py`
- Modify: `tests/unit/test_contrastive_pretrain_model.py` (6 call sites)
- Modify: `tests/unit/test_contrastive_pretrain_encoder_io.py` (11 call sites)

**Interfaces:**
- Produces: `encoder_and_dim` fixture (`tuple[GrayscaleResNetEncoder, int]`, function-scoped) and `encoder` fixture (`GrayscaleResNetEncoder`, derived from `encoder_and_dim`) — importable by any file under `tests/unit/` with no import statement (pytest auto-discovers `conftest.py`).

**Context:** `build_encoder(pretrained=False)` (from `contrastive_pretrain.model`) is called 6 times in `test_contrastive_pretrain_model.py` and 11 times in `test_contrastive_pretrain_encoder_io.py` — 17 near-identical construction sites across just these two files, both scoped entirely to `contrastive_pretrain` (no other package constructs a vision encoder), so this belongs in a new `tests/unit/conftest.py`, not the project-wide root `tests/conftest.py`.

- [ ] **Step 1: Create `tests/unit/conftest.py`**

```python
"""Fixtures shared across contrastive_pretrain's test files under tests/unit/.
Scoped here (not the root tests/conftest.py) because build_encoder is
specific to contrastive_pretrain -- no other package's tests need it.
"""

import pytest
import torch
from torch import nn

from contrastive_pretrain.model import build_encoder


@pytest.fixture
def encoder_and_dim() -> tuple[nn.Module, int]:
    """Fresh, untrained GrayscaleResNetEncoder + its embedding dim -- shared
    by every contrastive_pretrain test that needs a real encoder instance
    without downloading pretrained ImageNet weights."""
    return build_encoder(pretrained=False)


@pytest.fixture
def encoder(encoder_and_dim: tuple[nn.Module, int]) -> nn.Module:
    return encoder_and_dim[0]
```

- [ ] **Step 2: Refactor `test_contrastive_pretrain_model.py`'s 6 call sites**

At each of lines 14, 19, 28, 34, 46, 54, replace `encoder, _ = build_encoder(pretrained=False)` with an `encoder` parameter on the enclosing test function, and delete the now-redundant local line. Where a test also needs `dim` (check each call site for `encoder, dim = build_encoder(...)` usage further in the body — use the `encoder_and_dim` fixture there instead of `encoder`). Remove the `from contrastive_pretrain.model import build_encoder` import if no call site in the file still needs it directly.

- [ ] **Step 3: Refactor `test_contrastive_pretrain_encoder_io.py`'s 11 call sites**

Same treatment at lines 48, 89, 131, 145, 156, 176, 189, 229, 252, 277, 289 — replace `encoder, _ = build_encoder(pretrained=False)` (or `encoder, dim = ...` where `dim` is used, e.g. `test_compute_latent_stats_shapes`) with the `encoder` or `encoder_and_dim` fixture parameter. Remove the `build_encoder` import only if no remaining call site needs it directly (it's still imported for use inside `_load_frozen_encoder_from_client`-adjacent tests if any construct a second, independent encoder for comparison — check each site before removing the import).

- [ ] **Step 4: Run both files and confirm green**

```bash
uv run pytest -q tests/unit/test_contrastive_pretrain_model.py tests/unit/test_contrastive_pretrain_encoder_io.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/conftest.py tests/unit/test_contrastive_pretrain_model.py tests/unit/test_contrastive_pretrain_encoder_io.py
git commit -m "test: add shared encoder fixture to tests/unit/conftest.py, dedupe 17 call sites"
```

---

### Task 4: Fix every `FLOAT_EQ` finding — swap `==` for `pytest.approx(...)`

**Files:**
- Modify: `tests/unit/test_augmentation.py:22-26`
- Modify: `tests/unit/test_contrastive_pretrain_config.py:11,13,14,27`
- Modify: `tests/unit/test_batcher.py:64`
- Modify: `tests/unit/test_hf_storage_retry.py:74,80,81`
- Modify: `tests/unit/test_pipeline.py:88`

**Interfaces:** None — pure test-file edits, no shared fixtures involved.

- [ ] **Step 1: `test_augmentation.py`**

```python
    assert config.crop_min_area_fraction == pytest.approx(0.93)
    assert config.brightness_range == pytest.approx(0.15)
    assert config.contrast_range == pytest.approx(0.15)
    assert config.noise_sigma_max == pytest.approx(8.0)
    assert config.blur_sigma_max == pytest.approx(0.8)
```

(`max_translate_px`, `blur_kernel_size`, `jpeg_quality_min`, `jpeg_quality_max` are ints — leave those `==` comparisons unchanged.) Add `import pytest` at the top if not already present (it's already imported at line 1).

- [ ] **Step 2: `test_contrastive_pretrain_config.py`**

Add `import pytest` at the top of the file (currently only imports from `contrastive_pretrain.config`). Then:

```python
    assert config.learning_rate == pytest.approx(3e-4)
    assert config.weight_decay == pytest.approx(1e-6)
    assert config.temperature == pytest.approx(0.1)
```

(lines 11, 13, 14 respectively — `batch_size`, `warmup_steps`, `max_epochs`, `checkpoint_interval_steps`, `shuffle_buffer_size` stay `==`, they're ints) and at line 27:

```python
    assert config.learning_rate == pytest.approx(0.001)
```

- [ ] **Step 3: `test_batcher.py:64`**

Add `import pytest` if not already present, then:

```python
    assert reloaded[2]["timestamp_s"] == pytest.approx(2.0)
```

- [ ] **Step 4: `test_hf_storage_retry.py:74,80,81`**

`import pytest` is already present (line 1). Change:

```python
    assert delay == pytest.approx(120.0)
```

and

```python
    assert backoff(1, RuntimeError("connection reset")) == pytest.approx(1.0)
    assert backoff(2, RuntimeError("connection reset")) == pytest.approx(2.0)
```

- [ ] **Step 5: `test_pipeline.py:88`**

Add `import pytest` if not already present, then change the float assertion at line 88 (`assert progress_records[0].video_time_s == 100.0`) to:

```python
    assert progress_records[0].video_time_s == pytest.approx(100.0)
```

- [ ] **Step 6: Run all five files and confirm green**

```bash
uv run pytest -q tests/unit/test_augmentation.py tests/unit/test_contrastive_pretrain_config.py tests/unit/test_batcher.py tests/unit/test_hf_storage_retry.py tests/unit/test_pipeline.py
```

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_augmentation.py tests/unit/test_contrastive_pretrain_config.py tests/unit/test_batcher.py tests/unit/test_hf_storage_retry.py tests/unit/test_pipeline.py
git commit -m "test: replace float == comparisons with pytest.approx across 5 files"
```

---

### Task 5: Add `match=` to every under-specified `pytest.raises`

**Files:**
- Modify: `tests/unit/test_contrastive_pretrain_train.py:31`
- Modify: `tests/unit/test_registry.py:95`
- Modify: `tests/unit/test_contrastive_pretrain_encoder_io.py:304`

**Interfaces:** None.

- [ ] **Step 1: `test_contrastive_pretrain_train.py:31`**

In `test_run_memory_probe_does_not_swallow_other_errors`, the raised error is `ValueError("something else")` (line 29). Change:

```python
    with pytest.raises(ValueError, match="something else"):
```

- [ ] **Step 2: `test_registry.py:95`**

The test builds a registry entry missing `crop_w`/`crop_h` and expects `load_registry` to reject it. Read `src/data_collection/registry.py`'s validation error message for the missing-fields case first (the file's other `pytest.raises(ValueError, match="game")` test two tests above, at a similar call site, shows the established pattern — mirror it using the actual missing-fields message text from the source, not a guess). Change:

```python
    with pytest.raises(ValueError, match="missing"):
        load_registry(path)
```

(Verify `"missing"` actually appears in the real exception message by reading `registry.py`'s validation code before finalizing — adjust the `match=` string to the real substring if it differs.)

- [ ] **Step 3: `test_contrastive_pretrain_encoder_io.py:304`** (line number pre-Task-3-refactor; locate by test name if it shifted)

`test_load_frozen_encoder_raises_on_revision_parameter` — the source (`encoder_io.py:176-178`) raises `NotImplementedError("revision pinning is not yet supported — hf_storage.client.HfClient has no revision parameter")`. Change:

```python
    with pytest.raises(NotImplementedError, match="revision"):
        load_frozen_encoder("objones25/test-repo", revision="v1")
```

- [ ] **Step 4: Run all three files and confirm green**

```bash
uv run pytest -q tests/unit/test_contrastive_pretrain_train.py tests/unit/test_registry.py tests/unit/test_contrastive_pretrain_encoder_io.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_contrastive_pretrain_train.py tests/unit/test_registry.py tests/unit/test_contrastive_pretrain_encoder_io.py
git commit -m "test: add match= to under-specified pytest.raises assertions"
```

---

### Task 6: Fix genuine `BRANCHING` violations; document the legitimate exceptions

**Files:**
- Modify: `tests/unit/test_pipeline.py:181-191, 245-269`
- Modify: `tests/unit/test_contrastive_pretrain_resize_cache.py:34-65, 107-125`
- Modify: `tests/unit/test_extract.py:8-18`
- Modify: `tests/unit/test_contrastive_pretrain_train.py` (comment only, near line 601)
- Modify: `tests/unit/test_hf_storage_retry.py:10-23, 33-53` (comment only)

**Interfaces:** None.

**Context:** The audit's AST walk can't distinguish "branching that hides a test case" (a real violation — fix it) from "branching inside a fake/spy/state-machine double that simulates per-call behavior" (a legitimate, already-documented exception to rule 4 — leave the code, add a one-line comment citing the exception so a future pass doesn't re-litigate it). This task does both, file by file.

- [ ] **Step 1: `test_pipeline.py:186` — replace the `if` with dict dispatch**

`test_run_pipeline_processes_next_video_after_one_fails`'s `frame_source` stub currently does:

```python
    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        if video_source.video_id == "bad_video":
            raise RuntimeError("network blip")
            yield  # pragma: no cover - unreachable, makes this a generator
        for _ in range(3):
            yield _frame()
```

Replace with dict dispatch (the direct generator analogue of a mock's `side_effect` list):

```python
    def _raising_source(video_source: VideoSource, resume_seconds: float = 0.0):
        raise RuntimeError("network blip")
        yield  # pragma: no cover - unreachable, makes this a generator

    def _yielding_source(video_source: VideoSource, resume_seconds: float = 0.0):
        for _ in range(3):
            yield _frame()

    frame_generators = {"bad_video": _raising_source, "good_video": _yielding_source}

    def frame_source(video_source: VideoSource, resume_seconds: float = 0.0):
        return frame_generators[video_source.video_id](video_source, resume_seconds)
```

- [ ] **Step 2: `test_pipeline.py:265` — replace the `for` loop with `all()`**

```python
    result = run_pipeline(sources, deps)

    assert result == PipelineResult(completed=5, failed=0)
    manifest = uploader.load_manifest()
    # Every video's completion must actually land in the manifest -- this is
    # the thing a manifest-save race between threads would lose.
    assert all(manifest.is_complete(s.video_id) for s in sources)
    assert all(
        any(path.startswith(f"shards/{s.video_id}/") for path in client.files) for s in sources
    )
```

(Uses `client.files`, matching Task 2's rename.)

- [ ] **Step 3: `test_pipeline.py:222` — leave the `if`, add an exception comment**

`test_run_pipeline_resumes_from_last_checkpoint_after_failure`'s `if len(call_resume_seconds) == 1: raise ...` drives the *same video's* two sequential retry attempts, not two independent input cases — this is the documented state-machine exception to rule 4. Add a comment directly above it:

```python
        # rule 4 exception: sequences this one video's two retry attempts
        # (fail on the first call, succeed on the second), not independent
        # input cases -- see pytest-expert skill's rules.md.
        if len(call_resume_seconds) == 1:
            raise RuntimeError("network blip after first pass")
```

- [ ] **Step 4: `test_contrastive_pretrain_resize_cache.py:57` — unroll the `for` loop**

`test_build_local_resize_cache_writes_resized_shard_for_each_listed_path`'s closing loop asserts the same properties on both shards; unrolling preserves "one call handles both shards together" while removing the loop:

```python
    def _assert_resized_shard_matches(output_path, expected_pixels) -> None:
        assert output_path.exists()
        reloaded = datasets.Dataset.from_parquet(str(output_path))
        assert reloaded.num_rows == 1
        assert reloaded[0]["image"].size == (160, 144)  # PIL size is (width, height)
        assert reloaded[0]["image"].mode == "L"
        assert np.array_equal(np.array(reloaded[0]["image"]), expected_pixels)

    _assert_resized_shard_matches(tmp_path / "shards/vidA/00000.parquet", expected_pixels)
    _assert_resized_shard_matches(tmp_path / "shards/vidB/00000.parquet", expected_pixels)
```

(Replaces the `for shard_path in shard_bytes: ...` block; keep everything above it, including the `expected`/`expected_pixels` computation, unchanged.)

- [ ] **Step 5: `test_contrastive_pretrain_resize_cache.py:112` — leave the `if`, add an exception comment**

Inside `failing_download`'s `if path == failing_shard: raise ...` (in `test_build_local_resize_cache_leaves_no_partial_file_on_failure_and_resumes_correctly`) — this branches on which shard a fake download function was called with, simulating "one shard fails, the other doesn't," not test-case branching. Add:

```python
    def failing_download(path: str) -> bytes:
        # rule 4 exception: this is the fake's simulated per-call behavior
        # (one shard fails, the other doesn't), not case-selection.
        if path == failing_shard:
            raise RuntimeError("simulated network failure")
        return _native_res_shard_bytes("vidA")
```

Also add the same style comment at the second occurrence in this file (the `_FlakyApi.list_repo_files`'s `if len(attempts) < 3:` around line 285 — locate by content, add `# rule 4 exception: simulates two transient failures then success across repeated calls, not independent cases.` directly above it).

- [ ] **Step 6: `test_extract.py:15` — split the compound assertion (ruff PT018)**

```python
    assert "-i" in cmd
    assert cmd[cmd.index("-i") + 1] == "https://example.com/stream.m3u8"
```

(Replaces the single `assert "-i" in cmd and cmd[cmd.index("-i") + 1] == "..."` line.)

- [ ] **Step 7: `test_contrastive_pretrain_train.py` — exception comment near line 601**

Locate the `_spy_info`-style logger-wrapping fake's `if event == "data_wait":` branch (used to filter which log calls get recorded). Add directly above it: `# rule 4 exception: this filters which of the spy's captured calls to keep, not test-case branching.`

- [ ] **Step 8: `test_hf_storage_retry.py:16,42` — exception comments**

Above each `if attempts["count"] < 3:` inside the nested `flaky()` helpers (lines 16 and 42), add: `# rule 4 exception: models "still failing" state across repeated calls within one retry_with_backoff() run, not an input case.`

- [ ] **Step 9: Run all affected files**

```bash
uv run pytest -q tests/unit/test_pipeline.py tests/unit/test_contrastive_pretrain_resize_cache.py tests/unit/test_extract.py tests/unit/test_contrastive_pretrain_train.py tests/unit/test_hf_storage_retry.py
```

- [ ] **Step 10: Commit**

```bash
git add tests/unit/test_pipeline.py tests/unit/test_contrastive_pretrain_resize_cache.py tests/unit/test_extract.py tests/unit/test_contrastive_pretrain_train.py tests/unit/test_hf_storage_retry.py
git commit -m "test: fix genuine branching-as-cases violations, document legitimate fake-double branching"
```

---

### Task 7: Fix the two real `NO_ASSERT` findings

**Files:**
- Modify: `tests/unit/test_tracking.py:112-117`
- Modify: `tests/unit/test_contrastive_pretrain_train.py` (near line 629/652)

**Interfaces:** None.

**Context:** A third audit hit, `test_contrastive_pretrain_train.py:45`'s `test_check_finite_loss_passes_for_finite_value`, is already compliant (it has the required `# must not raise` comment per rule 6) — no change needed there, it's a tooling false positive from the audit script not parsing comment intent.

- [ ] **Step 1: `test_tracking.py` — assert the documented no-op return values**

`NullExperimentRun.log`/`.finish` are documented (`tracking.py:63-67`) to always return `None`. Change:

```python
def test_null_experiment_run_is_a_no_op() -> None:
    run = NullExperimentRun()

    assert run.log({"anything": 1}) is None
    assert run.finish() is None
```

- [ ] **Step 2: `test_contrastive_pretrain_train.py` — assert a checkpoint was actually written**

`test_run_training_completes_a_few_steps_without_nan` currently calls `run_training(deps)` with no assertion, relying entirely on "didn't raise" (which only implicitly proves `check_finite_loss` never fired). Read the test's full body first to find its existing `tmp_path`/checkpoint-dir setup (mirror the pattern the sibling test at line ~183-184 already uses for its own checkpoint-glob assertion), then add after the `run_training(deps)` call:

```python
    checkpoints = list((tmp_path / "checkpoints").glob("checkpoint_step*.pt"))
    assert checkpoints
```

(Adjust the glob's base path to match whatever `network_volume_checkpoint_dir` this specific test's `TrainingConfig` actually points at — check the test's own config construction rather than assuming `tmp_path / "checkpoints"` verbatim.)

- [ ] **Step 3: Run both files**

```bash
uv run pytest -q tests/unit/test_tracking.py tests/unit/test_contrastive_pretrain_train.py -m "not slow and not expensive"
```

Note: `test_run_training_completes_a_few_steps_without_nan` is marked `slow` (stays `slow` after Task 9 — it needs real Hub credentials) and won't run in this default invocation; run it explicitly once with credentials available to confirm the new assertion passes: `uv run pytest -q tests/unit/test_contrastive_pretrain_train.py -m slow -k completes_a_few_steps_without_nan`.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_tracking.py tests/unit/test_contrastive_pretrain_train.py
git commit -m "test: replace assertion-free smoke tests with real return-value/output assertions"
```

---

### Task 8: Deferred item 1 — regression test for `output_tmp_path.unlink`'s crash-after-temp-files-exist scenario

**Files:**
- Modify: `tests/unit/test_contrastive_pretrain_resize_cache.py`

**Interfaces:**
- Consumes: `contrastive_pretrain.resize_cache.build_local_resize_cache`, `contrastive_pretrain.resize_cache.os` (module-level `os` import, monkeypatched for this one test).

**Context:** Resolves item 1 of `docs/superpowers/plans/2026-08-25-contrastive-pretrain-resize-cache-followups.md`. The existing failure-path test (`test_build_local_resize_cache_leaves_no_partial_file_on_failure_and_resumes_correctly`) raises inside `download_shard`, *before* `raw_tmp_path`/`output_tmp_path` exist — it never reaches the `finally` block's `output_tmp_path.unlink` line (`resize_cache.py:104`). To actually hit that line, the crash must happen after `resized_dataset.to_parquet(str(output_tmp_path))` produces a real file but before `os.replace(output_tmp_path, output_path)` completes (`resize_cache.py:96-97`). Monkeypatching `os.replace` lands precisely in that window.

- [ ] **Step 1: Add the regression test**

Append to `tests/unit/test_contrastive_pretrain_resize_cache.py`:

```python
def test_build_local_resize_cache_removes_orphaned_temp_files_when_os_replace_crashes(
    tmp_path, monkeypatch
) -> None:
    """Regression test for the finally block's output_tmp_path.unlink line:
    forces a crash AFTER to_parquet() has produced a real output_tmp_path
    (and raw_tmp_path still exists too) but BEFORE os.replace() promotes it
    into place -- the exact ordering that line exists to clean up after."""
    shard_path = "shards/vidA/00000.parquet"
    output_path = tmp_path / shard_path
    raw_tmp_path = output_path.parent / f"{output_path.name}.raw.tmp"
    output_tmp_path = output_path.parent / f"{output_path.name}.tmp"

    def _crash(src, dst) -> None:
        raise RuntimeError("simulated crash before rename")

    monkeypatch.setattr(contrastive_pretrain.resize_cache.os, "replace", _crash)

    with pytest.raises(RuntimeError, match="simulated crash before rename"):
        build_local_resize_cache(
            list_shard_paths=lambda: [shard_path],
            download_shard=lambda path: _native_res_shard_bytes("vidA"),
            local_cache_dir=tmp_path,
        )

    assert not raw_tmp_path.exists()
    assert not output_tmp_path.exists()
    assert not output_path.exists()  # the rename never happened
```

`import contrastive_pretrain.resize_cache` is already at the top of this file (line 9) — no new import needed.

- [ ] **Step 2: Run it and confirm it passes against current (already-correct) code**

```bash
uv run pytest -q tests/unit/test_contrastive_pretrain_resize_cache.py -k removes_orphaned_temp_files
```

Expected: PASS. If it fails, the `finally` block has a real bug — stop and report rather than adjusting the test.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_contrastive_pretrain_resize_cache.py
git commit -m "test: add regression test for output_tmp_path cleanup on a crash before os.replace"
```

---

### Task 9: Deferred item 3, corrected — eliminate the CPU-slow tax instead of just relocating it

**Superseded design, kept for the record:** this task originally proposed retagging 7 `run_training`-driving tests from `slow` to a new `expensive` marker, moving their ~60s-per-test cost to a separate opt-in tier rather than the default suite. **A user directly rejected that approach during execution**, on this principle: *"Unless you are actually testing whether torch.compile works, you should monkeypatch it — there is absolutely no reason a test should run for 9 minutes unless the thing being tested takes 9 minutes to run."* That's correct and the original design didn't meet the bar — it treated the slowness as inherent and routed around it instead of asking whether it was necessary. Verified empirically before rewriting this task: `test_run_training_completes_and_checkpoints_at_epoch_boundary` measured at **72.78s** unmodified; monkeypatching `torch.compile` to an identity passthrough alone brought it to **36.73s** (confirmed the other ~36s was NOT pytest-cov overhead — measured `--no-cov` too, same number); additionally monkeypatching `build_encoder` to a tiny fake `nn.Module` (same `(N,1,144,160) -> (N, EMBEDDING_DIM)` interface, no real ResNet-50) brought it to **2.52s** — a 29x reduction, fast enough to belong in the default suite with no marker at all. Verified safe: grepped the 7 tests for any dependency on `GrayscaleResNetEncoder`-specific internals (`.backbone`, isinstance checks) — none found; the 7 tests check orchestration (checkpointing, resuming, logging, publish-gating), not encoder architecture or `torch.compile`'s own correctness (which is PyTorch's job to verify, not this project's — pytest-expert's own "not worth testing" list names third-party library behavior explicitly). Confirmed `push_frozen_encoder`'s `fuse_conv_bn_modules` step and `.to(memory_format=torch.channels_last)` are both safe no-ops against a `nn.Linear`-based fake (no Conv/BN pairs to fuse; `.to(memory_format=...)` silently no-ops on non-4D tensors, verified directly).

**Consequence:** the `expensive` marker Task 1 registered is no longer used by anything after this task — Task 9 now also removes it (a marker registered-but-unused is dead config, and this project's conventions favor no speculative abstraction). `test_run_training_completes_a_few_steps_without_nan` (the 8th, credentialed test) is deliberately EXCLUDED from this fixture — it is the one test whose purpose plausibly benefits from exercising the real encoder and real `torch.compile` end-to-end (it already pays real network/credential cost for other reasons), so it stays exactly as-is: `@pytest.mark.slow`, real weights, real compilation.

**Files:**
- Modify: `tests/unit/test_contrastive_pretrain_train.py`
- Modify: `pyproject.toml` (remove the now-unused `expensive` marker and its `addopts` reference)

**Interfaces:**
- Produces: a `fast_run_training` pytest fixture (local to this file) that the 7 retargeted tests consume as a parameter.

- [ ] **Step 1: Add a `_FakeEncoder` class and `fast_run_training` fixture**

Place this near `_FakeStreamingDataset`'s definition (locate by content, not line number — earlier tasks have shifted this file). Import `EMBEDDING_DIM` from `contrastive_pretrain.model` alongside the file's existing `build_encoder`/`build_projector` import.

```python
class _FakeEncoder(nn.Module):
    """Tiny stand-in for GrayscaleResNetEncoder -- same (N,1,144,160) ->
    (N, EMBEDDING_DIM) interface, but cheap enough that these orchestration
    tests (checkpointing/resuming/logging around run_training, not the
    encoder's own architecture) don't pay a real ResNet-50's eager-mode CPU
    cost on every call. Combined with the torch.compile bypass below, this
    took test_run_training_completes_and_checkpoints_at_epoch_boundary from
    ~73s to ~2.5s (measured directly, 29x) -- neither the real encoder nor
    real compilation is what any of these tests check; PyTorch owns
    verifying torch.compile's own correctness, and none of these tests
    inspect GrayscaleResNetEncoder-specific internals."""

    def __init__(self) -> None:
        super().__init__()
        self._linear = nn.Linear(144 * 160, EMBEDDING_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x.flatten(1).float())


@pytest.fixture
def fast_run_training(monkeypatch):
    """Bypasses torch.compile's uncached JIT compilation (see
    tests/conftest.py's TORCHINDUCTOR_FX_GRAPH_CACHE=0) and the real
    ResNet-50 backbone, for run_training-driving tests that check
    orchestration, not the encoder's architecture or torch.compile's own
    correctness. Do NOT use this fixture on a test that specifically needs
    to exercise the real encoder or real compilation end-to-end (there is
    exactly one such test in this file, and it deliberately does not use
    this fixture -- see its own docstring)."""
    monkeypatch.setattr("contrastive_pretrain.train.torch.compile", lambda model, **kwargs: model)
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_encoder", lambda pretrained: (_FakeEncoder(), EMBEDDING_DIM)
    )
```

- [ ] **Step 2: Remove `@pytest.mark.slow` and add the fixture to the 7 CPU-only tests**

At each of these 7 test functions (identify by name, not line number: `test_run_training_completes_and_checkpoints_at_epoch_boundary`, `test_run_training_resumes_projector_and_makes_progress`, `test_run_training_skips_dataloader_state_when_local_cache_dir_changed`, `test_run_training_restores_dataloader_state_when_local_cache_dir_matches`, `test_run_training_skips_publish_when_val_loss_does_not_improve`, `test_run_training_logs_contact_sheet_exactly_once_per_epoch`, `test_data_wait_metric_excludes_epoch_boundary_overhead`): delete the `@pytest.mark.slow` decorator entirely (no replacement marker — they now belong in the default fast suite), and add `fast_run_training` as a parameter to the test function's signature (alongside its existing `tmp_path, monkeypatch` params — order doesn't matter).

Leave `test_run_training_completes_a_few_steps_without_nan` completely untouched — no fixture, `@pytest.mark.slow` stays.

- [ ] **Step 3: Clean up the mid-file import block (E402)**

This file has `import contrastive_pretrain.train` and 3 related imports sitting after the first ~9 tests instead of at the top, with no side-effecting statement between them requiring that placement — an artifact of incremental growth. Move those 4 import lines up to join the file's top-of-file import block, in their original relative order.

- [ ] **Step 4: Remove the now-unused `expensive` marker from `pyproject.toml`**

In `[tool.pytest.ini_options]`: delete the `expensive` line from `markers = [...]`, and change `addopts`'s `"-m", "not slow and not expensive",` back to `"-m", "not slow",`.

- [ ] **Step 5: Run the file and the full suite**

```bash
uv run pytest -q tests/unit/test_contrastive_pretrain_train.py -v
```

Expected: all tests in this file now pass with only 1 deselected (the credentialed one) — the 7 formerly-slow tests run as part of the default selection, each in a few seconds, not ~60-70s.

```bash
uv run pytest -q
```

Expected: total passing count is now 7 higher than before this task (the 7 tests moved from deselected-slow into the default run), coverage still ≥80%.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_contrastive_pretrain_train.py pyproject.toml
git commit -m "test: eliminate run_training tests' CPU-slow tax via fake encoder + torch.compile bypass, not a slow/expensive split"
```

---

### Task 10: `checkpoint.py` — local fixture + two coverage-gap tests

**Files:**
- Modify: `tests/unit/test_contrastive_pretrain_checkpoint.py`

**Interfaces:**
- Produces: `make_training_components` fixture (local to this file — single-file DRY concern, not shared elsewhere), factory returning `tuple[nn.Linear, nn.Linear, AdamW, CosineAnnealingLR]`.

**Context:** All 6 tests in this file independently rebuild `model = nn.Linear(2, 2)`, `projector = nn.Linear(2, 4)`, `optimizer = torch.optim.AdamW(...)`, `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(...)`. Two real coverage gaps: `find_latest_checkpoint`'s "directory doesn't exist at all" branch (`checkpoint.py:79`) is never hit — the existing empty-result test passes an *existing* empty `tmp_path`, not a missing path — and `load_checkpoint` has no test for a missing file at all.

- [ ] **Step 1: Add the `make_training_components` fixture near the top of the file**

```python
@pytest.fixture
def make_training_components():
    def _make():
        model = nn.Linear(2, 2)
        projector = nn.Linear(2, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return model, projector, optimizer, scheduler

    return _make
```

- [ ] **Step 2: Refactor all 6 tests to use it**

Each test currently has its own 4-line block building `model`/`projector`/`optimizer`/`scheduler` (or a subset). Replace each with `model, projector, optimizer, scheduler = make_training_components()` (adding `make_training_components` as a test parameter), keeping any per-test post-construction customization (e.g. the 10x `scheduler.step()` loop in the LR-restore test from Task 1) unchanged below the new one-liner.

- [ ] **Step 3: Add the missing-directory coverage-gap test**

```python
def test_find_latest_checkpoint_returns_none_when_directory_does_not_exist(tmp_path) -> None:
    assert find_latest_checkpoint(tmp_path / "nonexistent") is None
```

- [ ] **Step 4: Add the missing-file `load_checkpoint` test**

```python
def test_load_checkpoint_raises_on_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pt")
```

(`torch.load` raises `FileNotFoundError` for a missing path — verify this by running the test in Step 5 rather than assuming; if `torch.load` actually raises a different exception type on the installed torch version, use that type instead.)

- [ ] **Step 5: Run the file and confirm green**

```bash
uv run pytest -q tests/unit/test_contrastive_pretrain_checkpoint.py
```

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_contrastive_pretrain_checkpoint.py
git commit -m "test: dedupe checkpoint test setup via a factory fixture, close two coverage gaps"
```

---

### Task 11: `encoder_io.py` — three coverage-gap tests

**Files:**
- Modify: `tests/unit/test_contrastive_pretrain_encoder_io.py`

**Interfaces:**
- Consumes: `encoder`/`encoder_and_dim` fixtures from Task 3's `tests/unit/conftest.py`.

- [ ] **Step 1: `_load_frozen_encoder_from_client`'s `FileNotFoundError` branch**

`encoder_io.py:133-134` raises when either `config.json` or `model.safetensors` is missing. Add:

```python
def test_load_frozen_encoder_from_client_raises_when_files_missing(fake_hf_client) -> None:
    from contrastive_pretrain.encoder_io import _load_frozen_encoder_from_client

    with pytest.raises(FileNotFoundError, match="model.safetensors or config.json"):
        _load_frozen_encoder_from_client(fake_hf_client)
```

(`fake_hf_client` from the root `tests/conftest.py` — a client nothing has ever been pushed to, so both downloads return `None`.)

- [ ] **Step 2: `compute_latent_stats`'s truncation branch (`i >= max_examples: break`)**

`encoder_io.py:198-200` is never hit because the existing shapes test passes exactly `max_examples` rows. Add a case with more rows than `max_examples`:

```python
def test_compute_latent_stats_truncates_at_max_examples(encoder_and_dim) -> None:
    encoder, dim = encoder_and_dim
    rows = [
        {"original": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8)}
        for _ in range(10)
    ]

    mean, std = compute_latent_stats(encoder, rows, device=torch.device("cpu"), max_examples=3)

    assert mean.shape == (dim,)
    assert std.shape == (dim,)
```

(This proves truncation doesn't crash/misshape the output; `compute_latent_stats` doesn't expose how many rows it actually consumed, so shape correctness under a hard cap is the strongest black-box assertion available here — consistent with rule 10's guidance to assert what's actually knowable.)

- [ ] **Step 3: `compute_latent_stats`'s train/eval mode restoration**

The docstring claims `encoder.train(was_training)` restores the original mode on exit (`encoder_io.py:203`) — untested. Add:

```python
def test_compute_latent_stats_restores_original_training_mode(encoder) -> None:
    encoder.train()
    rows = [{"original": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8)} for _ in range(2)]

    compute_latent_stats(encoder, rows, device=torch.device("cpu"), max_examples=2)

    assert encoder.training is True
```

- [ ] **Step 4: Run the file**

```bash
uv run pytest -q tests/unit/test_contrastive_pretrain_encoder_io.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_contrastive_pretrain_encoder_io.py
git commit -m "test: close three encoder_io.py coverage gaps (missing-files, truncation, mode restore)"
```

---

### Task 12: `config.py` — unknown-field error path test

**Files:**
- Modify: `tests/unit/test_contrastive_pretrain_config.py`

**Interfaces:** None.

- [ ] **Step 1: Add the test**

`config.py:58-60` raises `ValueError(f"unknown config field(s): {sorted(unknown)}")` when a YAML config has a field `TrainingConfig` doesn't declare — the one real error path in this module, currently untested.

```python
def test_load_config_rejects_unknown_field(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("not_a_real_field: 123\n")

    with pytest.raises(ValueError, match="unknown config field"):
        load_config(path)
```

(Add `import pytest` at the top if not already added by Task 4, Step 2.)

- [ ] **Step 2: Run the file**

```bash
uv run pytest -q tests/unit/test_contrastive_pretrain_config.py
```

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_contrastive_pretrain_config.py
git commit -m "test: add coverage for load_config's unknown-field error path"
```

---

### Task 13: `dataset.py` — DRY the repeated schema/monkeypatch/stub patterns, close the `build_val_dataset` asymmetry

**Files:**
- Modify: `tests/unit/test_contrastive_pretrain_dataset.py`

**Interfaces:** None (fixtures here stay local to this file — the `Features` schema and `_load_base_stream` monkeypatch pattern are specific to this file's synthetic-stream tests).

**Context:** `test_contrastive_pretrain_dataset.py` redefines the same `datasets.Features({"image": ..., "video_id": ..., "timestamp_s": ..., "game": ...})` schema independently 3 times, repeats a `monkeypatch.setattr("contrastive_pretrain.dataset._load_base_stream", lambda config: _synthetic_frame_stream([...]))` pattern across ~7 tests, and defines an identical inline `_CountingResize` stub class in 3 separate tests. `build_train_dataset` has a test for "still applies the resize map when `local_cache_dir` is unset" but `build_val_dataset`'s mirror-image branch (`dataset.py:169-170`) has no equivalent test — only its "skips the map when cache dir is set" branch is covered.

- [ ] **Step 1: Extract the repeated `Features` schema to a module-level constant**

Near the top of the file (after imports), add:

```python
_ROW_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "video_id": datasets.Value("string"),
        "timestamp_s": datasets.Value("float64"),
        "game": datasets.Value("string"),
    }
)
```

Replace the 3 inline redefinitions with `_ROW_FEATURES` (the one variant missing `"game"` — verify by reading that call site whether `"game"` is actually needed there; if not, keep that one occurrence separate rather than forcing an ill-fitting shared constant on it).

- [ ] **Step 2: Extract a `patch_load_base_stream` local fixture for the repeated monkeypatch pattern**

```python
@pytest.fixture
def patch_load_base_stream(monkeypatch):
    def _apply(video_ids: list[str]):
        stream = _synthetic_frame_stream(video_ids)
        monkeypatch.setattr("contrastive_pretrain.dataset._load_base_stream", lambda config: stream)
        return stream

    return _apply
```

(Confirm `_synthetic_frame_stream`'s actual signature by reading its definition in this file before finalizing — adjust the fixture's `_apply` signature to match if it takes different/additional arguments.) Refactor each of the ~7 tests currently doing `monkeypatch.setattr("contrastive_pretrain.dataset._load_base_stream", lambda config: _synthetic_frame_stream([...]))` to instead take `patch_load_base_stream` as a parameter and call `patch_load_base_stream([...])`.

- [ ] **Step 3: Hoist `_CountingResize` to module level**

The 3 inline redefinitions of `_CountingResize` (an instantiation-counting stub for `_ResizeToCanonicalWithProgress`) become one module-level class definition near the top of the file; delete the 3 inline copies and reference the module-level one from each test.

- [ ] **Step 4: Add the missing `build_val_dataset` mirror test**

Read the existing `test_build_train_dataset_still_uses_resize_map_when_local_cache_dir_is_unset` test in full first, then write its `build_val_dataset` equivalent following the same structure (same `patch_load_base_stream` usage, same assertion style — that `_ResizeToCanonicalWithProgress`/the resize map is still applied when `config.local_cache_dir` is `None`):

```python
def test_build_val_dataset_still_uses_resize_map_when_local_cache_dir_is_unset(
    patch_load_base_stream, monkeypatch
) -> None:
    patch_load_base_stream(["D1SrSFZrV7A"])
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.ensure_local_cache", lambda config: None
    )
    config = TrainingConfig(val_video_ids=("D1SrSFZrV7A",), local_cache_dir=None)

    ds = build_val_dataset(config)

    row = next(iter(ds))
    assert "original" in row and "view_a" in row and "view_b" in row
```

(Match this test's exact assertions to whatever the sibling `build_train_dataset` version actually asserts — read it first; this sketch shows the shape, adjust to mirror the real one exactly, including its `_CountingResize`-based instantiation-count check if the train-side version uses one.)

- [ ] **Step 5: Run the file**

```bash
uv run pytest -q tests/unit/test_contrastive_pretrain_dataset.py
```

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_contrastive_pretrain_dataset.py
git commit -m "test: dedupe dataset.py test schema/monkeypatch/stub patterns, add build_val_dataset mirror test"
```

---

### Task 14: `contrastive_pretrain/cli.py` — `limit==1` branch test, strengthen the deps-building assertion

**Files:**
- Modify: `tests/unit/test_contrastive_pretrain_cli.py`

**Interfaces:** None.

- [ ] **Step 1: Add the `limit == 1` coverage-gap test**

`cli.py:69`'s `[round(...) for i in range(limit)] if limit > 1 else [0]` ternary's `limit == 1` branch is never hit (existing tests use the default 12 or `--limit 2`). Add:

```python
def test_preview_command_with_limit_one_selects_a_single_frame(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(5):
        frame = np.full((144, 160), 50 + i, dtype=np.uint8)
        Image.fromarray(frame).save(frames_dir / f"frame_{i}.png")

    out_path = tmp_path / "preview.png"
    runner = CliRunner()
    result = runner.invoke(
        main, ["preview", "--frames-dir", str(frames_dir), "--out", str(out_path), "--limit", "1"]
    )

    assert result.exit_code == 0, result.output
    saved = np.array(Image.open(out_path))
    assert saved.shape == (144, 480)  # one row, one (original|view_a|view_b) triple wide
```

- [ ] **Step 2: Strengthen `test_train_command_builds_deps_from_config_and_calls_run_training`**

Currently only asserts `captured["deps"].config.batch_size == 8` despite the test name promising "builds deps from config" broadly. `RealHfClient` and `WandbRun` are already monkeypatched to identifiable stand-ins (`lambda *a, **k: object()` for `RealHfClient`) — make that stand-in identifiable and assert on it, mirroring what the sibling export-command test already does:

```python
    monkeypatch.setattr("contrastive_pretrain.cli.run_training", lambda deps: captured.update(deps=deps))
    monkeypatch.setattr("contrastive_pretrain.cli.RealHfClient", lambda *_a, **_k: "the-frozen-encoder-client")
    monkeypatch.setattr("contrastive_pretrain.cli.HfApi", lambda: _FakeHfApi())
    monkeypatch.setattr("contrastive_pretrain.cli.get_token", lambda: "fake-token")
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setattr("contrastive_pretrain.cli.wandb", _FakeWandbModule())

    runner = CliRunner()
    result = runner.invoke(main, ["train", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert captured["deps"].config.batch_size == 8
    assert captured["deps"].frozen_encoder_client == "the-frozen-encoder-client"
```

(Replaces the old `lambda *a, **k: object()` and its single assertion — keep the rest of the test unchanged. Verify `TrainingDeps` actually exposes a `.frozen_encoder_client` attribute by reading `contrastive_pretrain/train.py`'s `TrainingDeps` definition before finalizing.)

- [ ] **Step 3: Rename unused lambda params (ruff ARG005 cleanup, cosmetic)**

At the `lambda *a, **k: object()` / `lambda *a, **k: "the-client"` / `lambda config: [...]` sites in this file, prefix unused params with `_` (`lambda *_a, **_k: ...`, `lambda _config: [...]`) — no behavior change.

- [ ] **Step 4: Run the file**

```bash
uv run pytest -q tests/unit/test_contrastive_pretrain_cli.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_contrastive_pretrain_cli.py
git commit -m "test: cover cli.py's limit==1 branch, strengthen the deps-building assertion"
```

---

### Task 15: `data_collection/cli.py` — success-path and option-wiring coverage

**Files:**
- Modify: `tests/unit/test_cli.py`

**Interfaces:** None.

**Context:** Every existing `run` test in this file drives a failure branch (missing HF token, missing W&B key, or a faked `PipelineResult(failed=1)`); nothing proves the command exits 0 on success. Similarly `curate`'s only test is the missing-argument failure — its success path (delegating to `curation.run_curation`) is untested. `--batch-size`/`--checkpoint-interval`/`--max-concurrent-videos` are declared options but no test confirms they reach `PipelineDeps`.

- [ ] **Step 1: `run` command success path**

```python
def test_run_command_exits_zero_when_pipeline_reports_no_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("data_collection.cli.get_token", lambda: "fake-token")
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setattr("data_collection.cli.wandb", _FakeWandbModule())
    monkeypatch.setattr(
        "data_collection.cli.pipeline.run_pipeline",
        lambda sources, deps: pipeline.PipelineResult(completed=0, failed=0),
    )
    registry_path = tmp_path / "video_sources.yaml"
    registry_path.write_text("videos: []\n")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--repo-id", "me/pokemon-frames", "--registry", str(registry_path)]
    )

    assert result.exit_code == 0, result.output
```

- [ ] **Step 2: `run` command option wiring**

```python
def test_run_command_wires_batch_and_checkpoint_options_onto_deps(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("data_collection.cli.get_token", lambda: "fake-token")
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setattr("data_collection.cli.wandb", _FakeWandbModule())
    captured = {}
    monkeypatch.setattr(
        "data_collection.cli.pipeline.run_pipeline",
        lambda sources, deps: captured.update(deps=deps) or pipeline.PipelineResult(completed=0, failed=0),
    )
    registry_path = tmp_path / "video_sources.yaml"
    registry_path.write_text("videos: []\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--repo-id", "me/pokemon-frames",
            "--registry", str(registry_path),
            "--batch-size", "50",
            "--checkpoint-interval", "999",
            "--max-concurrent-videos", "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["deps"].batch_size == 50
    assert captured["deps"].checkpoint_interval_samples == 999
    assert captured["deps"].max_concurrent_videos == 3
```

(Verify `PipelineDeps`'s exact field names — `batch_size`, `checkpoint_interval_samples`, `max_concurrent_videos` — against `data_collection/pipeline.py`'s dataclass before finalizing; the `run` command in `cli.py:101-108` passes them through with these names, but confirm.)

- [ ] **Step 3: `curate` command success path**

```python
def test_curate_command_delegates_to_run_curation(tmp_path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "data_collection.cli.curation.run_curation",
        lambda **kwargs: captured.update(kwargs),
    )
    registry_path = tmp_path / "video_sources.yaml"
    approved_dir = tmp_path / "approved"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "curate", "https://youtube.com/watch?v=abc123",
            "--game", "red",
            "--registry", str(registry_path),
            "--approved-dir", str(approved_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["video_url"] == "https://youtube.com/watch?v=abc123"
    assert captured["game"] == "red"
    assert captured["registry_path"] == registry_path
    assert captured["approved_dir"] == approved_dir
```

(`curate`'s body at `cli.py:39-44` calls `curation.run_curation(video_url=url, approved_dir=approved_dir, registry_path=registry, game=game)` — matches these kwarg names exactly.)

- [ ] **Step 4: Run the file**

```bash
uv run pytest -q tests/unit/test_cli.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_cli.py
git commit -m "test: cover data_collection cli.py's success paths and option wiring"
```

---

### Task 16: `dedup.py` — Hamming-distance boundary test

**Files:**
- Modify: `tests/unit/test_dedup.py`

**Interfaces:** None.

- [ ] **Step 1: Read the existing file first**

Read `tests/unit/test_dedup.py` in full to see how existing tests construct two frames at a known perceptual-hash Hamming distance (likely via `imagehash.phash` on two synthetic images, or a documented helper) — reuse that exact pattern rather than inventing a new one.

- [ ] **Step 2: Add a boundary test at a non-default, exact threshold**

```python
def test_is_duplicate_treats_distance_equal_to_threshold_as_duplicate() -> None:
    deduper = PerceptualHashDeduper(hamming_threshold=0)
    frame = np.zeros((144, 160), dtype=np.uint8)

    assert deduper.is_duplicate(frame) is False  # first frame: nothing to compare against yet
    assert deduper.is_duplicate(frame) is True  # identical frame -> distance 0 == threshold 0
```

(Uses `hamming_threshold=0` and an identical repeated frame, which guarantees `distance == 0 == threshold`, directly exercising the `distance <= self._hamming_threshold` boundary at `dedup.py:20` without needing to hand-construct a specific non-zero Hamming distance.)

- [ ] **Step 3: Run the file**

```bash
uv run pytest -q tests/unit/test_dedup.py
```

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_dedup.py
git commit -m "test: add PerceptualHashDeduper hamming_threshold boundary coverage"
```

---

### Task 17: `hf_storage/client.py` — download-call tracking + non-`EntryNotFoundError` passthrough

**Files:**
- Modify: `tests/unit/test_hf_storage_client.py`

**Interfaces:** None.

**Context:** `_FakeHfApi.hf_hub_download`'s `repo_id`/`repo_type` params are accepted (to match `RealHfClient.download_bytes`'s real call shape) but silently discarded — no test can currently catch a regression where the wrong `repo_id`/`repo_type` gets forwarded. `download_bytes` only catches `EntryNotFoundError` (`client.py:71`); nothing proves other exceptions propagate unmodified.

- [ ] **Step 1: Read the file first**

Read `tests/unit/test_hf_storage_client.py` in full — its `_FakeHfApi` class and existing test structure — before making edits, to match its established style exactly.

- [ ] **Step 2: Add call tracking to `_FakeHfApi.hf_hub_download`**

```python
    def __init__(self) -> None:
        self.commits: list[dict] = []
        self.download_calls: list[tuple[str, str, str]] = []

    def hf_hub_download(self, repo_id: str, filename: str, repo_type: str) -> str:
        self.download_calls.append((repo_id, filename, repo_type))
        dest = self._tmp_path / filename.replace("/", "_")
        if not dest.exists():
            ...
```

(Merge this into the existing `__init__`/`hf_hub_download` bodies rather than replacing them wholesale — add the one new list and the one new `.append()` call, keep everything else in both methods unchanged.)

- [ ] **Step 3: Add a test asserting on it**

```python
def test_download_bytes_forwards_repo_id_and_repo_type(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)  # match this file's actual constructor signature
    client = RealHfClient(api, "me/repo", repo_type="model")
    api._tmp_path.joinpath("file.txt").write_bytes(b"hi")

    client.download_bytes("file.txt")

    assert api.download_calls == [("me/repo", "file.txt", "model")]
```

(Verify `_FakeHfApi`'s real constructor signature and how existing tests seed a file into it before finalizing — match the established pattern exactly rather than guessing.)

- [ ] **Step 4: Add the non-`EntryNotFoundError` passthrough test**

```python
def test_download_bytes_does_not_swallow_unrelated_errors() -> None:
    class _BrokenApi:
        def hf_hub_download(self, repo_id: str, filename: str, repo_type: str) -> str:
            raise RuntimeError("connection reset")

    client = RealHfClient(_BrokenApi(), "me/repo")

    with pytest.raises(RuntimeError, match="connection reset"):
        client.download_bytes("file.txt")
```

- [ ] **Step 5: Run the file**

```bash
uv run pytest -q tests/unit/test_hf_storage_client.py
```

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_hf_storage_client.py
git commit -m "test: track hf_hub_download call args, cover non-EntryNotFoundError passthrough"
```

---

### Task 18: `visualization.py` — strengthen the grid-dimensions test with real pixel content

**Files:**
- Modify: `tests/unit/test_visualization.py`

**Interfaces:** None.

**Context:** `test_build_contact_sheet_grid_dimensions` currently only checks `sheet.shape`, never pixel content — a broken placement loop that still produces the right output shape would pass undetected. There's also no coverage of the incomplete-last-row zero-padding case.

- [ ] **Step 1: Read the existing test first**

Read `tests/unit/test_visualization.py:10-16`'s current test in full to match its exact frame-construction style.

- [ ] **Step 2: Strengthen it with exact cell-content and zero-padding assertions**

Extend the existing test (or add a new one immediately after it, whichever better matches the file's convention) with 10 frames at `cols=4` (2 full rows of 4, one row with 2 frames + 2 empty cells) using distinct per-frame pixel values so each cell is individually checkable:

```python
def test_build_contact_sheet_places_each_frame_in_its_own_cell_and_zero_pads_the_rest() -> None:
    frames = [np.full((144, 160), value, dtype=np.uint8) for value in range(10)]

    sheet = build_contact_sheet(frames, cols=4)

    assert sheet.shape == (144 * 3, 160 * 4)  # 10 frames at 4 cols -> 3 rows
    # Spot-check placement: frame 0 in the top-left cell, frame 9 (last) in
    # row 2, col 1 -- divmod(9, 4) == (2, 1).
    assert np.all(sheet[:144, :160] == 0)
    assert np.all(sheet[288:, 160:320] == 9)
    # The last row's remaining two cells (cols 2 and 3) were never written --
    # must stay at the zero-fill default, not leak stale/garbage data.
    assert np.all(sheet[288:, 320:480] == 0)
    assert np.all(sheet[288:, 480:] == 0)
```

- [ ] **Step 3: Run the file**

```bash
uv run pytest -q tests/unit/test_visualization.py
```

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_visualization.py
git commit -m "test: strengthen build_contact_sheet test with exact cell content and zero-padding checks"
```

---

### Task 19: `logging_config.py` — `exc_info` branch coverage + fix the root-logger global-state leak

**Files:**
- Modify: `tests/unit/test_logging_config.py`

**Interfaces:** None.

**Context:** `JSONFormatter.format`'s `if record.exc_info is not None:` branch (`logging_config.py:32-33`) has no test. Separately, `test_configure_logging_writes_json_to_the_given_stream` calls `configure_logging(stream=stream)`, which runs `logging.config.dictConfig(...)` and replaces the **root logger's** handlers for the rest of the pytest process — nothing restores them afterward, and no autouse fixture in `tests/conftest.py` does either. This is real cross-test global state, just mutated via a function call rather than a module-level literal.

- [ ] **Step 1: Read the existing test file first**

Read `tests/unit/test_logging_config.py` in full to match its exact style before editing.

- [ ] **Step 2: Add the `exc_info` coverage test**

```python
def test_json_formatter_includes_exc_info_when_present(caplog) -> None:
    logger = logging.getLogger("test_logging_config_exc_info")
    formatter = JSONFormatter()

    try:
        raise ValueError("boom")
    except ValueError:
        record = logger.makeRecord(
            logger.name, logging.ERROR, __file__, 0, "failed", (), sys.exc_info()
        )

    payload = json.loads(formatter.format(record))

    assert "exc_info" in payload
    assert "ValueError: boom" in payload["exc_info"]
```

(Match the exact import style/helper pattern the rest of this file already uses for constructing a `LogRecord`/invoking the formatter directly — read Step 1's output first and adjust this sketch to match rather than introducing a new construction style. Add `import sys` if not already present.)

- [ ] **Step 3: Fix the root-logger leak in `test_configure_logging_writes_json_to_the_given_stream`**

Snapshot and restore the root logger's handlers/level around the `configure_logging(...)` call:

```python
def test_configure_logging_writes_json_to_the_given_stream() -> None:
    original_handlers = list(logging.root.handlers)
    original_level = logging.root.level
    try:
        stream = io.StringIO()
        configure_logging(stream=stream)
        logging.getLogger("somewhere").info("hello")

        payload = json.loads(stream.getvalue())
        assert payload["message"] == "hello"
    finally:
        logging.root.handlers = original_handlers
        logging.root.setLevel(original_level)
```

(Merge this `try`/`finally` wrapping into the existing test body exactly as written today — only add the snapshot/restore, don't change any existing assertions. Add `import logging` at the top if not already present; `io` is presumably already imported given the existing `stream=` usage — verify.)

- [ ] **Step 4: Run the file, then the whole suite to confirm the leak fix doesn't change other tests' behavior**

```bash
uv run pytest -q tests/unit/test_logging_config.py
uv run pytest -q
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_logging_config.py
git commit -m "test: cover JSONFormatter's exc_info branch, stop configure_logging leaking root-logger state across tests"
```

---

### Task 20: `test_augmentation.py` — extract the 13x-duplicated marker-frame block, consolidate imports

**Files:**
- Modify: `tests/unit/test_augmentation.py`

**Interfaces:** None (stays local to this file — the marker-block idiom is specific to augmentation testing).

**Context:** The exact two-line block `frame = torch.zeros((1, 144, 160), dtype=torch.uint8); frame[0, 70:74, 78:82] = 255` (a 4x4 marker block, distinct from the existing single-pixel `_marker_frame()` helper at line 32) is repeated verbatim 13 times across the file. Separately, 6 import blocks are scattered mid-file instead of consolidated at the top — an artifact of incremental growth, not a deliberate ordering requirement.

- [ ] **Step 1: Add a second helper next to the existing `_marker_frame`**

```python
def _marker_block_frame(
    size: tuple[int, int] = (144, 160), block: tuple[slice, slice] = (slice(70, 74), slice(78, 82))
) -> torch.Tensor:
    frame = torch.zeros((1, *size), dtype=torch.uint8)
    frame[0, block[0], block[1]] = 255
    return frame
```

- [ ] **Step 2: Replace all 13 inline occurrences**

At each of the 13 sites (lines ~218-219, 228-229, 256-257, 266-267, 276-277, 291-292, 303-304, 316-317, 363-364, 387-388, 405-406, 417-418, 447-448 — locate by the exact two-line pattern, since exact line numbers will have shifted from earlier tasks' edits), replace the two-line block with `frame = _marker_block_frame()`. If a given call site uses a differently-positioned or differently-sized marker, pass explicit `size=`/`block=` args instead of hardcoding a new inline block.

- [ ] **Step 3: Consolidate the 6 scattered import blocks into the top-of-file import**

Merge every mid-file `from contrastive_pretrain.augmentation import ...` block (currently duplicated at roughly lines 141-145, 187-192, 239-243, 287, 381-383, on top of the existing top-of-file block) into the single import block at the top of the file, de-duplicating any repeated names. Delete the mid-file copies entirely.

- [ ] **Step 4: Run the file**

```bash
uv run pytest -q tests/unit/test_augmentation.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_augmentation.py
git commit -m "test: dedupe augmentation.py's 13x-repeated marker-block construction, consolidate scattered imports"
```

---

### Task 21: Final verification — full suite, audit script, marker selections, coverage

**Files:** None (verification only).

**Interfaces:** None.

- [ ] **Step 1: Run the full default suite**

```bash
uv run pytest -q
```

Expected: all tests pass (should be 200 (post-Task-8) + 7 (Task 9's de-slowed tests rejoining the default run) + however many new tests Tasks 10-19 added, minus none removed), zero warnings, coverage report shows total ≥80%, only `slow` tests deselected (no `expensive` marker exists anymore as of Task 9's correction).

- [ ] **Step 2: (superseded — no `expensive` marker to check)** Task 9's original design registered an `expensive` marker for a separate opt-in tier; that design was rejected during execution and replaced with monkeypatching `torch.compile`/`build_encoder` so the 7 affected tests run fast in the default suite instead (see Task 9's "Superseded design" note). There is no `expensive` marker left to verify here — skip straight to Step 3.

- [ ] **Step 3: Run the `slow` subset (only if real HF/W&B credentials are available in this environment)**

```bash
uv run pytest -q -m slow
```

Expected: the credentialed tests pass, including Task 7's strengthened `test_run_training_completes_a_few_steps_without_nan`. If credentials aren't available here, skip this step and note it explicitly rather than claiming it passed.

- [ ] **Step 4: Run the pytest-expert audit script against the whole tree**

```bash
uv run python /Users/theelusivegerbilfish/.claude/skills/pytest-expert/scripts/audit_tests.py tests/
```

Expected: zero findings (the original 32 are all resolved by Tasks 4-7; any newly-added test that trips a finding must be fixed before this task is done, not excused).

- [ ] **Step 5: Prove at least one new test can actually fail**

Pick one new test from Tasks 8-19 (e.g. Task 8's `output_tmp_path` regression test) and temporarily break the corresponding source behavior (e.g. comment out the `output_tmp_path.unlink(missing_ok=True)` line in `resize_cache.py`), confirm the test goes red, then revert. Report in the final summary which test was used for this check.

- [ ] **Step 6: Final commit (if Step 5's revert left anything uncommitted, verify `git status` is clean first)**

```bash
git status
```

No commit needed for this task — it's verification-only. If `git status` shows anything unexpected, investigate before ending the plan.

---

## Deferred / Not Included In This Pass

Recorded here (per this project's own convention in the `2026-08-25-contrastive-pretrain-resize-cache-followups.md` plan) so nothing found during review gets silently dropped, even though none of these block the tasks above.

- **The audit script's 5 `UNSEEDED_RANDOM` findings are false positives, not defects.** Every flagged `np.random.default_rng(...)` call (`test_contrastive_pretrain_dataset.py` x4, `test_pipeline.py` x1) already passes an explicit literal or call-count-derived seed — the audit's `suite_is_seeded()` check just can't see that because no `conftest.py` in the tree centrally seeds an RNG, so it flags every call site regardless of arguments. No task above changes this; flagging the check's blind spot is enough. Worth reporting upstream to the pytest-expert skill maintainer, not fixing here.
- **`torch`/`numpy` RNG seeding inconsistency in `test_contrastive_pretrain_train.py`.** `_FakeStreamingDataset.__iter__` and `compute_val_loss`'s inline batch generator produce unseeded random tensors, unlike sibling files (`test_augmentation.py`, `test_contrastive_pretrain_losses.py`, `test_contrastive_pretrain_encoder_io.py`) which call `torch.manual_seed(0)` first. Today's assertions in that file are structural (checkpoint counts, step counts), not numeric-outcome-sensitive, so this isn't causing flakiness — but if a future test in that file asserts on a loss value or NaN-freeness, seed it first. Deferred rather than fixed now because it touches no currently-failing or currently-ambiguous assertion.
- **Adopting `ruff` as a project-wide lint dependency/gate was considered and rejected for this plan.** The project currently has no `[tool.ruff]` config and no `ruff` dev dependency; `ruff check --config <pytest-expert's ruff-pytest.toml>` was used only as one-time input (surfacing PT011/PT018/E402, folded into Tasks 5/6/9/20) — not adopted as an ongoing gate. That's a separate infra decision (would also need a line-length convention, since the project has none today) outside "evaluate the test suite" scope; revisit separately if wanted.
- **Minor coverage gaps noted by the review but not worth a dedicated task**, each low-value relative to its cost: `TrainingConfig`'s `pretrained`/`num_workers`/`seed`/`network_volume_checkpoint_dir` defaults are asserted nowhere (inert config values); `build_encoder(pretrained=True)` (downloads real ImageNet weights) has no test at any marker tier; `run_training`'s `if device.type == "cuda":` branch is untested by construction (CPU-only CI) — would need a `gpu` marker and a GPU runner to close, neither of which exist yet; `dataset.py`'s `_load_base_stream` non-cache branch is only exercised by an existing `@pytest.mark.slow` network test, never by a fast mocked one; `checkpoint.py`'s `save_checkpoint` `.tmp`-file-doesn't-survive-a-successful-save case (atomicity itself isn't really at risk — `torch.save` + `Path.replace` is a thin wrapper); `data_collection/pipeline.py`'s `_sample_for_contact_sheet` downsampling branch (`len(images) > 64`) is untested since every test's `batch_size` stays under that. Pick any of these up in a future pass if `--cov-fail-under` needs raising past 80%.
