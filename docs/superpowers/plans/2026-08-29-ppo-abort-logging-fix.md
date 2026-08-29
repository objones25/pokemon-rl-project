# PPO Abort Logging Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gap where a NaN-storm abort from `run_update` propagates out of
`run_training` without the structured `training_aborted` log record the other two
abort paths already get.

**Architecture:** Widen the existing `try` block in `run_training`'s main loop
(`src/ppo/trainer.py`) so `deps.run_update(...)` is called *inside* it, alongside
`_check_abort_conditions(stats, config)`. The single existing `except
(AssertionError, RuntimeError)` handler then covers all three abort conditions
instead of two. No new exception types, no new control flow, no signature changes.

**Tech Stack:** Python 3, pytest 9.1.1 (strict config, branch coverage floor 93%),
PyTorch 2.13 (CPU-only for this test — the fix and its tests touch no PyTorch API).

**Spec:** `docs/superpowers/specs/2026-08-29-ppo-abort-logging-fix-design.md`

## Global Constraints

- Every behavior change needs a failing test written first (TDD; project convention).
- Test doubles are hand-written fakes wired through `PPODeps.run_update`, never
  `mock.patch` — this file already establishes that pattern via `_stub_run_update`.
- `pytest.raises` always names a specific exception and passes `match=`.
- `caplog`-based assertions check the specific log record (`r.message ==
  "training_aborted"` and `r.exc_info[0]`), not just that the exception propagated —
  a test that only asserts `pytest.raises` would pass against the current buggy code
  too, since the exception already propagates; the log record is what's missing.
- No signature changes to `run_training`, `PPODeps`, `run_update`, or
  `UpdateStats` — this is a pure control-flow fix inside `_run_training`.
- Branch coverage floor is 93% and must not drop; do not lower it to land this change.
- Verified against the actual installed code, not the spec's paraphrase: read
  `src/ppo/trainer.py` (current lines 240–256), `src/ppo/update.py` (current lines
  158–179, the `RuntimeError` message format at line 176), `src/ppo/config.py`
  (`max_nan_minibatches_per_update: int = 3`), and
  `tests/unit/test_ppo_trainer.py` (current `_stub_run_update` at lines 70–88,
  `_trainer_harness` at lines 91–150, and the two sibling
  `training_aborted`-logging tests at lines 419–443) before writing this plan —
  every line number and code excerpt below reflects the real files, not the spec.

---

## Context the implementer needs

**Why this file, this function:** `run_training` (`src/ppo/trainer.py`) is the outer
PPO loop. Each iteration calls `deps.run_update(...)` (returns `UpdateStats`), then
`_check_abort_conditions(stats, config)` (raises `AssertionError` for the epoch-1
ratio invariant or `RuntimeError` for an `approx_kl` breach). Today only the second
call is wrapped in `try`/`except (AssertionError, RuntimeError):
logger.exception("training_aborted", ...)`. But `run_update` itself
(`src/ppo/update.py:158-179`) has a third abort condition — too many non-finite
(NaN/Inf) minibatch losses in one update — that also raises `RuntimeError`, and it
propagates out of `deps.run_update(...)` at trainer.py before the `try` block even
starts. That abort is real but currently invisible to the log.

**The exact current code** (`src/ppo/trainer.py`, inside `_run_training`'s `while`
loop):

```python
        stats = deps.run_update(
            deps.policy, deps.optimizer, deps.scheduler, buffer, scaler, config,
            deps.policy_config, deps.env_config.n_envs, deps.device, deps.autocast_dtype,
        )

        # Both abort paths log with exc_info before propagating: this is a
        # 48-hour unattended run, and the difference between "died at hour 30
        # on the ratio invariant" and "died at hour 30 on KL" is the whole
        # diagnosis. The exception itself still propagates untouched.
        try:
            _check_abort_conditions(stats, config)
        except (AssertionError, RuntimeError):
            logger.exception(
                "training_aborted",
                extra={"update": update, "global_step": global_step},
            )
            raise
```

**The fix:** move the `stats = deps.run_update(...)` call inside the `try`, above
`_check_abort_conditions(stats, config)`, and update the comment to reflect three
abort paths instead of two. That's the entire production-code change.

**How the existing test harness works** (`tests/unit/test_ppo_trainer.py`), so the
new test follows the same pattern instead of inventing a new one:

- `_trainer_harness(tmp_path, *, checkpoint_every_updates=25,
  artifact_every_updates=25, forced_approx_kl=None, forced_epoch1_dev=None)`
  builds a `_TrainerHarness` (a tiny real `RecurrentTransformerPolicy`, a
  `FakeVecEnv`, a `FakeLatentEncoder`, a `FakeExperimentRun`, and a `PPODeps`).
  When `forced_approx_kl` or `forced_epoch1_dev` is not `None`, it swaps
  `deps.run_update` for `_stub_run_update(...)` instead of the real
  `ppo.update.run_update`.
- `_stub_run_update(*, approx_kl: float, epoch1_dev: float)` returns a closure
  matching `run_update`'s exact positional signature (`policy, optimizer,
  scheduler, buffer, scaler, config, policy_config, n_envs, device,
  autocast_dtype`) that returns a hand-built `UpdateStats` with those two fields
  set and everything else zeroed, instead of doing the real forward/backward pass.
- The two existing abort-logging tests (`test_an_abort_logs_an_error_carrying_the_
  traceback` and `test_the_epoch_one_ratio_abort_also_logs_an_error_carrying_the_
  traceback`) each: build a harness forcing one condition, run under `caplog.at_
  level(logging.ERROR, logger="ppo.trainer")` plus `pytest.raises(<the exception
  type>)`, filter `caplog.records` for `r.message == "training_aborted"`, and
  assert `[r.exc_info[0] for r in aborted] == [<the exception type>]`.

The new test needs `_stub_run_update` and `_trainer_harness` to also support
*raising* `RuntimeError` directly from the stubbed `run_update` call — mirroring
`run_update`'s real NaN-storm abort — since neither currently has a way to do that
(both only ever return a crafted `UpdateStats`).

**`run_update`'s real NaN-storm abort message** (`src/ppo/update.py:174-178`),
which the stub must reproduce so the test's `match=` pattern is meaningful:

```python
                if skipped >= config.max_nan_minibatches_per_update:
                    raise RuntimeError(
                        f"non-finite loss in {skipped} minibatches of one update; "
                        "aborting rather than stepping on corrupt gradients"
                    )
```

`PPOConfig.max_nan_minibatches_per_update` defaults to `3` (`src/ppo/config.py:43`),
so the real message at the default threshold reads `"non-finite loss in 3
minibatches of one update; aborting rather than stepping on corrupt gradients"`.
The stub does not need to compute a real `skipped` count — it raises directly, so
hard-coding the message at the default threshold (matching the format the real code
produces) is enough for the test's `match="non-finite loss"` to mean something.

---

## Task 1: Regression test proving the NaN-storm abort is (or isn't) logged, then the fix

**Files:**
- Modify: `tests/unit/test_ppo_trainer.py` — extend `_stub_run_update`, extend
  `_trainer_harness`, add one new test.
- Modify: `src/ppo/trainer.py` — widen the `try` block in `_run_training`, update
  its comment.

**Interfaces:**
- Consumes: `ppo.trainer.PPODeps.run_update: Callable` (already exists,
  `src/ppo/trainer.py:153`) — the injection point every stub already uses.
  `ppo.update.UpdateStats` (already exists, `src/ppo/update.py:34-53`) — the
  dataclass `_stub_run_update` constructs.
- Produces: nothing new for later tasks — this is the only task in this plan.

- [ ] **Step 1: Extend `_stub_run_update` to optionally raise instead of return**

In `tests/unit/test_ppo_trainer.py`, replace the existing `_stub_run_update`
function (current lines 70–88) with:

```python
def _stub_run_update(*, approx_kl: float, epoch1_dev: float, forced_nan_abort: bool = False):
    """A stand-in for `run_update` that skips the real forward/backward
    pass entirely, so the KL-abort and epoch-1-invariant tests can force an
    exact value onto exactly the two fields the trainer inspects -- wired in
    through `PPODeps.run_update`, never `mock.patch`. `forced_nan_abort`
    instead raises the same `RuntimeError` `run_update`'s own NaN-storm abort
    raises, at the default `max_nan_minibatches_per_update` threshold, so a
    test can prove that abort path reaches `run_training`'s log handler too."""

    def _run_update(
        policy, optimizer, scheduler, buffer, scaler, config, policy_config,
        n_envs, device, autocast_dtype,
    ) -> UpdateStats:
        if forced_nan_abort:
            raise RuntimeError(
                "non-finite loss in 3 minibatches of one update; aborting "
                "rather than stepping on corrupt gradients"
            )
        return UpdateStats(
            policy_loss=0.0, value_loss=0.0, entropy=0.0, total_loss=0.0,
            clip_fraction=0.0, approx_kl=approx_kl,
            max_abs_ratio_dev_epoch1_mb1=epoch1_dev, max_abs_ratio_dev=epoch1_dev,
            explained_variance=0.0, staleness_logprob_l1=0.0, skipped_minibatches=0,
            grad_norm=0.0, policy_grad_norm=0.0, value_grad_norm=0.0,
        )

    return _run_update
```

- [ ] **Step 2: Wire `forced_nan_abort` through `_trainer_harness`**

In `tests/unit/test_ppo_trainer.py`, change the `_trainer_harness` signature
(current lines 91–98) from:

```python
def _trainer_harness(
    tmp_path: Path,
    *,
    checkpoint_every_updates: int = 25,
    artifact_every_updates: int = 25,
    forced_approx_kl: float | None = None,
    forced_epoch1_dev: float | None = None,
) -> _TrainerHarness:
```

to:

```python
def _trainer_harness(
    tmp_path: Path,
    *,
    checkpoint_every_updates: int = 25,
    artifact_every_updates: int = 25,
    forced_approx_kl: float | None = None,
    forced_epoch1_dev: float | None = None,
    forced_nan_abort: bool = False,
) -> _TrainerHarness:
```

And update its docstring's last sentence to also mention `forced_nan_abort`:

```python
    """A tiny real policy (matching the other ppo/ harnesses' shapes) plus a
    `FakeVecEnv`/`FakeLatentEncoder` and a `FakeExperimentRun`. `run_update`
    defaults to the real one; `forced_approx_kl`/`forced_epoch1_dev`/
    `forced_nan_abort` swap in `_stub_run_update` instead, through
    `PPODeps.run_update`."""
```

Then change the stub-selection block (current lines 124–129) from:

```python
    run_update_dep = run_update
    if forced_approx_kl is not None or forced_epoch1_dev is not None:
        run_update_dep = _stub_run_update(
            approx_kl=0.0 if forced_approx_kl is None else forced_approx_kl,
            epoch1_dev=0.0 if forced_epoch1_dev is None else forced_epoch1_dev,
        )
```

to:

```python
    run_update_dep = run_update
    if forced_approx_kl is not None or forced_epoch1_dev is not None or forced_nan_abort:
        run_update_dep = _stub_run_update(
            approx_kl=0.0 if forced_approx_kl is None else forced_approx_kl,
            epoch1_dev=0.0 if forced_epoch1_dev is None else forced_epoch1_dev,
            forced_nan_abort=forced_nan_abort,
        )
```

- [ ] **Step 3: Write the failing test**

In `tests/unit/test_ppo_trainer.py`, add this test immediately after
`test_the_epoch_one_ratio_abort_also_logs_an_error_carrying_the_traceback`
(current lines 431–443), so it sits next to its two siblings:

```python
def test_a_nan_storm_abort_also_logs_an_error_carrying_the_traceback(
    tmp_path, caplog
) -> None:
    """The third abort path -- run_update's own NaN-storm abort -- raises
    RuntimeError from OUTSIDE the try block that wraps _check_abort_conditions,
    at src/ppo/trainer.py's `stats = deps.run_update(...)` call. Before the fix
    this propagates with no training_aborted record; the assertion on the log
    record (not just pytest.raises) is what would have caught that, since the
    exception already reaches the caller either way."""
    harness = _trainer_harness(tmp_path, forced_nan_abort=True)

    with (
        caplog.at_level(logging.ERROR, logger="ppo.trainer"),
        pytest.raises(RuntimeError, match="non-finite loss"),
    ):
        run_training(harness.deps, max_updates=1)
    aborted = [r for r in caplog.records if r.message == "training_aborted"]

    assert [r.exc_info[0] for r in aborted] == [RuntimeError]
```

- [ ] **Step 4: Run the new test and confirm it fails for the right reason**

Run: `uv run pytest tests/unit/test_ppo_trainer.py::test_a_nan_storm_abort_also_logs_an_error_carrying_the_traceback -v`

Expected: FAIL. `pytest.raises(RuntimeError, match="non-finite loss")` passes (the
exception already propagates today), but the final `assert` fails because `aborted
== []` — the `training_aborted` record was never written, since
`deps.run_update(...)` raises before the `try` block in `_run_training` is
reached. Confirm the failure is specifically on the `aborted` assertion, not on
`pytest.raises` — if `pytest.raises` itself fails, something else is wrong (e.g.
the stub's message doesn't match, or `forced_nan_abort` isn't wired through) and
must be fixed before proceeding.

- [ ] **Step 5: Apply the fix in `src/ppo/trainer.py`**

In `_run_training`'s main loop (current lines 240–256), replace:

```python
        stats = deps.run_update(
            deps.policy, deps.optimizer, deps.scheduler, buffer, scaler, config,
            deps.policy_config, deps.env_config.n_envs, deps.device, deps.autocast_dtype,
        )

        # Both abort paths log with exc_info before propagating: this is a
        # 48-hour unattended run, and the difference between "died at hour 30
        # on the ratio invariant" and "died at hour 30 on KL" is the whole
        # diagnosis. The exception itself still propagates untouched.
        try:
            _check_abort_conditions(stats, config)
        except (AssertionError, RuntimeError):
            logger.exception(
                "training_aborted",
                extra={"update": update, "global_step": global_step},
            )
            raise
```

with:

```python
        # All three abort paths -- the epoch-1 ratio invariant, the approx_kl
        # threshold, and run_update's own NaN-storm abort -- log with
        # exc_info before propagating: this is a 48-hour unattended run, and
        # the difference between "died at hour 30 on the ratio invariant" and
        # "died at hour 30 on KL" is the whole diagnosis. The exception
        # itself still propagates untouched.
        try:
            stats = deps.run_update(
                deps.policy, deps.optimizer, deps.scheduler, buffer, scaler, config,
                deps.policy_config, deps.env_config.n_envs, deps.device, deps.autocast_dtype,
            )
            _check_abort_conditions(stats, config)
        except (AssertionError, RuntimeError):
            logger.exception(
                "training_aborted",
                extra={"update": update, "global_step": global_step},
            )
            raise
```

Nothing between the old `stats = deps.run_update(...)` line and the `try` block
does anything with `stats` that needs to run outside a `try` — the assignment
moves in whole, unchanged.

- [ ] **Step 6: Run the new test again and confirm it passes**

Run: `uv run pytest tests/unit/test_ppo_trainer.py::test_a_nan_storm_abort_also_logs_an_error_carrying_the_traceback -v`

Expected: PASS.

- [ ] **Step 7: Run the full `test_ppo_trainer.py` and `test_ppo_update.py` files to confirm no regression**

Run: `uv run pytest tests/unit/test_ppo_trainer.py tests/unit/test_ppo_update.py -v`

Expected: all tests PASS, including the two pre-existing abort-logging tests
(`test_an_abort_logs_an_error_carrying_the_traceback`,
`test_the_epoch_one_ratio_abort_also_logs_an_error_carrying_the_traceback`) and
`tests/unit/test_ppo_update.py::test_too_many_non_finite_minibatches_abort_the_
update` (this task does not touch `update.py`, so this test's own pass/fail is
unaffected — confirm it still passes as a sanity check on the untouched file).

- [ ] **Step 8: Run the full suite with coverage to confirm the branch-coverage floor still holds**

Run: `uv run pytest`

Expected: all tests pass (697+ tests, since this adds one), and the branch
coverage gate (93% floor, configured in `pyproject.toml`) still passes — the
widened `try` block does not remove any covered branch, and the new test adds
coverage rather than removing it.

- [ ] **Step 9: Run the pytest-expert audit script**

Run: `python /Users/theelusivegerbilfish/.claude/skills/pytest-expert/scripts/audit_tests.py tests/unit/test_ppo_trainer.py`

Expected: no findings for the new test (`test_a_nan_storm_abort_also_logs_an_
error_carrying_the_traceback`) or the extended `_stub_run_update`/
`_trainer_harness` — no `if`/`for`/`while` in the test body, `pytest.raises`
names `RuntimeError` with `match=`, the test asserts an exact expected value
(`[RuntimeError]`), no unseeded randomness or network calls introduced.

- [ ] **Step 10: Commit**

```bash
git add src/ppo/trainer.py tests/unit/test_ppo_trainer.py
git commit -m "fix(ppo): log training_aborted on run_update's own NaN-storm abort

The main loop's try block only wrapped _check_abort_conditions, so
run_update's NaN-storm RuntimeError propagated out of run_training with
no structured log record -- the one abort path most likely to fire on
a 48-hour unattended run was also the one least likely to leave a
diagnosable trace. Widen the try to also cover deps.run_update(...)."
```

---

## Self-review notes (from the plan-writing pass)

- **Spec coverage:** the spec's three sections — "Fix" (move the call inside
  `try`, update the comment), "Testing" (extend `_stub_run_update`, add the
  mirrored test), and the two "Why not..." rationale notes (consolidate into one
  `except`, no new exception types) — are all covered by Task 1's steps 1–5. The
  spec's "Non-goals" section requires no task: no change to
  `max_nan_minibatches_per_update`, no retry-and-continue behavior, no touching
  the sibling findings (latent-stats sampling bug has its own spec/plan; the
  dead `clip_range_vf` field, untyped collaborators, and manifest atomicity gap
  are un-specced and out of scope here).
- **Placeholder scan:** no TBD/TODO, no "add appropriate error handling," no
  "similar to Task N" — every code block above is the literal text to write.
- **Type consistency:** `_stub_run_update`'s closure signature
  (`policy, optimizer, scheduler, buffer, scaler, config, policy_config, n_envs,
  device, autocast_dtype`) matches `run_update`'s real positional signature
  exactly (verified via `src/ppo/update.py:57-68`), and matches how
  `deps.run_update(...)` is called at `src/ppo/trainer.py:240-243` (all
  positional, same order) — the stub is a drop-in replacement either way.
- **PyTorch relevance:** confirmed via the pytorch skill that this fix touches no
  PyTorch API — no tensor ops, no autograd, no device/dtype handling change. The
  fix is pure Python control flow (a `try` block's scope) plus a test double.
  Nothing in `references/` applies beyond the general non-negotiables, which
  this change does not violate (it doesn't touch a model, a loss, or a step).
