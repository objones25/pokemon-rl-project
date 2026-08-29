# PPO Abort Logging Gap — Design Spec

**Amendment to `docs/superpowers/specs/2026-08-27-ppo-trainer-design.md`.**
That spec (and `src/ppo/trainer.py`'s own inline comment) states that "both
abort paths log with exc_info before propagating." There are actually three
abort paths, and the third — a NaN-storm abort — bypasses the logging
guarantee entirely. This spec closes that gap.

Found during a full-codebase review requested ahead of the first paid PPO run,
not from a production incident — no paid run has happened yet.

## The bug

`run_training`'s main loop (`src/ppo/trainer.py:240-256`):

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

`deps.run_update(...)` is called **outside** the `try` block. `_check_abort_conditions`
(`trainer.py:306-320`) covers exactly two conditions: the epoch-1/minibatch-1
ratio invariant (`AssertionError`) and the `approx_kl` threshold
(`RuntimeError`). But `run_update` — specifically the inner loop in
`src/ppo/update.py:158-179` — has a **third** abort condition of its own:

```python
if not torch.isfinite(loss.total):
    skipped += 1
    optimizer.zero_grad(set_to_none=True)
    logger.warning("nan_minibatch_skipped", extra={...})
    if skipped >= config.max_nan_minibatches_per_update:
        raise RuntimeError(
            f"non-finite loss in {skipped} minibatches of one update; "
            "aborting rather than stepping on corrupt gradients"
        )
    continue
```

When `skipped >= config.max_nan_minibatches_per_update`, this `RuntimeError`
propagates directly out of `deps.run_update(...)` at line 240 — before the
`try` block even begins. It is never caught by the `except (AssertionError,
RuntimeError)` clause, `logger.exception("training_aborted", ...)` never
fires, and the run dies with only Python's default unhandled-exception
traceback on stdout/stderr — not the structured JSON-lines record with
`exc_info=True` this project's CLAUDE.md requires for every long-running
component ("Observability first... every pipeline/training component logs
structured (JSON-lines) progress"), and not the record `WandbRun` or any
downstream log aggregation on a RunPod pod would be watching for.

**This is exactly the failure mode the existing comment says it already
handles.** A NaN storm — corrupt gradients from a bad batch, a numerical
instability, a hardware fault producing garbage floats — is arguably the
single most likely uncontrolled failure on a 48-hour unattended GPU run, more
likely than either of the two conditions that *are* covered. The one abort
path most likely to actually fire is also the one least likely to leave a
diagnosable trace.

### Confirmed via the existing tests, not just reading the code

- `tests/unit/test_ppo_update.py::test_too_many_non_finite_minibatches_abort_the_update`
  proves `run_update` raises `RuntimeError` in isolation — it does not, and
  cannot, prove anything about `trainer.py`'s logging, since it calls
  `run_update` directly.
- `tests/unit/test_ppo_trainer.py::test_an_abort_logs_an_error_carrying_the_traceback`
  and `::test_the_epoch_one_ratio_abort_also_logs_an_error_carrying_the_traceback`
  both force their abort condition through `_stub_run_update`
  (`test_ppo_trainer.py:70-88`), a fake that returns a crafted `UpdateStats` —
  it has no path to simulate `run_update` *raising*, so there is no test
  anywhere in the suite that exercises a NaN-abort reaching `run_training`
  and checks whether it gets logged. The gap is real and currently invisible
  to the test suite, not merely theoretical.

## Fix

Move `deps.run_update(...)` inside the existing `try` block, so the one
handler that already exists covers all three abort paths instead of two:

```python
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

This is the entire fix — a four-line move, no new exception types, no new
control flow. It works because `run_update`'s NaN-abort already raises the
same `RuntimeError` type the handler already catches; the handler was simply
scoped one call too narrow. Nothing between the old `stats = deps.run_update(...)`
line and the `try` block does anything with `stats` that needs to happen
outside a `try` — the assignment moves in whole.

**Why not add a separate `except` clause instead of widening the existing
one**: a second, NaN-specific try/except around just `deps.run_update(...)`
would duplicate the exact same `logger.exception("training_aborted", ...)`
call this project already has, for no behavioral difference — the log record,
the re-raise, and the semantics are identical for all three conditions.
Consolidating into one `try` is both the smaller diff and removes a place
where the two blocks could drift out of sync (e.g., a future editor updating
one handler's `extra={...}` fields and forgetting the other).

**Why this doesn't risk mis-labeling an unrelated failure as "training
aborted"**: `run_update`'s call graph (forward/backward pass, optimizer step,
gradient clipping) does not raise `RuntimeError` or `AssertionError` for
reasons other than the three abort conditions and the NaN check under normal
operation — a genuinely unrelated failure (e.g., a `MemoryError` from a real
OOM, a `KeyboardInterrupt`) is not an instance of either caught type and
propagates through this block exactly as before, untouched. The widened
`try` does not change what gets caught, only which call the existing catch
now surrounds.

## What changes

- `src/ppo/trainer.py`: the `run_training` main loop, moving the
  `deps.run_update(...)` call inside the existing `try` block that wraps
  `_check_abort_conditions`. No signature changes anywhere.
- The module's own inline comment ("Both abort paths log with exc_info...")
  should be updated to say "all three abort paths" (or better, drop the count
  and say "every RuntimeError/AssertionError this loop can raise"), so it
  doesn't quietly become inaccurate again the next time an abort condition is
  added to either `run_update` or `_check_abort_conditions`.

## Testing

- Extend `_stub_run_update` (`tests/unit/test_ppo_trainer.py:70-88`) with a
  `forced_nan_abort: bool = False` parameter that, when set, raises
  `RuntimeError("non-finite loss in N minibatches of one update; aborting "
  "rather than stepping on corrupt gradients")` instead of returning an
  `UpdateStats` — the same fake-function pattern already used for
  `forced_approx_kl`/`forced_epoch1_dev`, not `mock.patch`.
- Add `test_a_nan_storm_abort_also_logs_an_error_carrying_the_traceback`,
  mirroring the two existing `training_aborted`-logging tests exactly: run
  `run_training` with the new stub, assert `pytest.raises(RuntimeError)`, and
  assert a `caplog` record named `"training_aborted"` exists with
  `r.exc_info[0] == RuntimeError`. This is the test that would have caught
  the bug this spec fixes — write it first, confirm it fails against the
  current code (the exception still propagates and the test's `pytest.raises`
  still passes, but the `aborted` list comes back empty because the log
  record was never written — assert on the log record specifically, not just
  that the exception propagated, or the test would pass against the buggy
  code too), then apply the fix and confirm it passes.
- No change needed to `tests/unit/test_ppo_update.py`'s existing
  `test_too_many_non_finite_minibatches_abort_the_update` — that test's job
  (proving `run_update` raises in isolation) is unaffected by where the
  caller catches it.

## Non-goals

- Does not change `max_nan_minibatches_per_update`'s default or semantics —
  purely a logging-visibility fix for the abort path that already exists.
- Does not add retry-and-continue behavior for a NaN storm — the existing
  design deliberately aborts rather than continuing on corrupt gradients, and
  that decision is unchanged here.
- Does not address any of the other findings from the same review pass (the
  latent-stats sampling bug has its own sibling spec,
  `docs/superpowers/specs/2026-08-29-latent-stats-sampling-fix-design.md`; the
  dead `clip_range_vf` config field, the untyped `run_update`/`checkpoint.py`
  collaborators, and the PPO manifest atomicity gap are separate, un-specced
  findings from the same review).
