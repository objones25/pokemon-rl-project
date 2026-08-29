# VecPokemonEnv Step Concurrency — Design Spec

**Amendment to `docs/superpowers/specs/2026-08-26-pokemon-env-design.md`.** That
spec's Vectorization section costs the *serialization* path (frame pickling,
solved by shared memory) but never costed the *dispatch* path across the 64
subprocess workers. This spec closes that gap: `VecPokemonEnv.step()` and
`.reset()` currently dispatch to the 64 `SubprocessBackend`s **sequentially**,
which throws away the parallelism 64 separate OS processes exist to provide.

Found while investigating "is there a serious inefficiency in the rollout path
we're missing" ahead of the first paid PPO run — not from a production
incident, since no paid run has happened yet.

## The bug

`VecPokemonEnv.step()` (`src/pokemon_env/vec_env.py:132-137`):

```python
results = [
    backend.reset() if needs_reset else backend.step(int(action))
    for backend, action, needs_reset in zip(
        self._backends, actions, self._needs_reset, strict=True
    )
]
```

and `.reset()` (`vec_env.py:125`) do the same thing: a plain Python list
comprehension calling one backend's `step`/`reset` at a time. Each call reaches
`SubprocessBackend._call()` (`src/pokemon_env/subprocess_backend.py:239-250`):

```python
def _call(self, command: Command, argument: object = None) -> dict:
    self._conn.send((command, argument))
    if not self._conn.poll(self._config.worker_timeout_s):
        raise TimeoutError(...)
    status, payload = self._conn.recv()
    ...
```

This `send` → `poll` → `recv` is a full blocking round trip to **one** worker.
Because the list comprehension calls it once per backend, worker *i+1* is not
even sent its command until worker *i*'s entire round trip — including that
worker's real emulator tick — has finished. Sixty-four independent processes
that could all be executing their 24-frame tick at the same instant are
instead executed back to back.

### Why this matters at this project's scale

`subprocess_backend.py`'s own module docstring and the env design spec's
Failure handling section both state the assumption a 24-frame tick takes
**about 1 ms** (`src/pokemon_env/subprocess_backend.py:244`,
`docs/superpowers/specs/2026-08-26-pokemon-env-design.md:427`). Take that
figure as given — it is this project's own documented assumption, not
re-measured here.

Under correct overlap, one vector step's wall time is bounded by roughly the
*slowest* of the 64 workers: ≈ 1 ms, plus whatever IPC/gather overhead the
fix below adds. Under the current sequential dispatch, one vector step's wall
time is the **sum** across all 64 workers: 64 × 1 ms ≈ 64 ms, purely from the
loop structure — this bound does not depend on how large the true per-worker
IPC overhead turns out to be, only on the tick-time figure already documented
in this codebase. That is a **~64×** structural penalty on the dispatch loop
alone.

The PPO trainer spec's `n_steps = 1024` means every update calls
`VecPokemonEnv.step()` 1,024 times
(`docs/superpowers/specs/2026-08-27-ppo-trainer-design.md:97`,
§5). At the sequential rate that is ≈ 1024 × 64 ms ≈ **65 s per rollout**
from dispatch serialization alone, against properly-overlapped dispatch's
≈ 1024 × 1 ms ≈ **1 s**. The env design spec sized its 1.5 GB/rollout pickling
cost against an assumed **8.0 s rollout budget**
(`docs/superpowers/specs/2026-08-26-pokemon-env-design.md`, Vectorization
section) — the dispatch serialization alone is large enough to blow that
budget by roughly 8×, independent of and in addition to whatever the pickling
cost turns out to be.

**This is a projection from a documented per-tick assumption, not a fresh
measurement.** Real IPC syscall overhead, the parent process's own GIL while
issuing 64 sequential `send`/`recv` calls, and shared-memory write costs could
shift the exact constant in either direction. The qualitative conclusion —
that a sequential loop over 64 independent processes serializes work that
should overlap — does not depend on the exact constant. **Gate 2 (rollout
throughput, already in the PPO trainer spec) is the place to measure this
fix's actual before/after wall-clock effect**, not this document.

This is a **different** inefficiency from the one already investigated and
declined this session (overlapping rollout *k+1* with gradient update *k*,
which was rejected on on-policy-staleness grounds). That question was about
overlapping two different *phases* of PPO across update boundaries. This one
is about overlapping 64 *workers* within a single step, entirely inside the
rollout phase, and carries none of the staleness cost — the 64 workers are
already independent and already produce the same results in any completion
order.

## Fix

Split each backend's `step`/`reset` into two phases — *send* and *recv* — so
`VecPokemonEnv` can fire all 64 sends before waiting on any reply.

### Interface change

`EnvBackend` (`vec_env.py:34-40`) gains two-phase methods in place of the
current synchronous `step`/`reset`:

```python
class EnvBackend(Protocol):
    def send_reset(self) -> None: ...
    def send_step(self, action: int) -> None: ...
    def recv(self) -> StepResult: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def stats(self) -> dict: ...
    def close(self) -> None: ...
```

`VecPokemonEnv.step()` becomes:

```python
for backend, action, needs_reset in zip(self._backends, actions, self._needs_reset, strict=True):
    if needs_reset:
        backend.send_reset()
    else:
        backend.send_step(int(action))
results = [backend.recv() for backend in self._backends]
```

Same shape for `.reset()`. This is the **entire** change at the `VecPokemonEnv`
level — one Protocol, one code path, both backends implement it. That keeps
the SOLID boundary this codebase already uses everywhere else (Protocol +
hand-written fake) instead of forking a second "batched" code path that only
`SubprocessBackend` would exercise, which is what a `BatchStepBackend`-style
alternative design would require and is explicitly rejected here as
speculative — nothing today needs `VecPokemonEnv` to know a backend is
subprocess-backed.

### `InProcessBackend`

Has no concurrency to exploit — it drives one `EnvSession` in the parent
process. `send_step`/`send_reset` just run the work eagerly and stash the
`StepResult`; `recv()` returns and clears it:

```python
def send_step(self, action: int) -> None:
    self._pending = self._session.step(action)

def recv(self) -> StepResult:
    result, self._pending = self._pending, None
    return result
```

Cost is identical to today — still "roughly 64× too slow for a real run" as
already documented, unaffected by this change, because that backend exists
for tests and debugging, not the real run.

### `SubprocessBackend`

```python
def send_step(self, action: int) -> None:
    self._conn.send((Command.STEP, action))

def send_reset(self) -> None:
    self._conn.send((Command.RESET, None))

def recv(self) -> StepResult:
    if not self._conn.poll(self._config.worker_timeout_s):
        raise TimeoutError(...)
    status, payload = self._conn.recv()
    ...
```

`send()` on a small payload (a `Command` enum plus an int action) does not
block in practice — the OS pipe buffer is far larger than one action's pickled
size — so firing 64 sends in the parent's single thread before any `recv()` is
cheap and non-blocking. By the time the parent starts `recv()`-ing, every
worker that is going to answer quickly already has its command and may already
be replying, so the 64 sequential `recv()` calls are now bounded by each
worker's *already-in-flight* work rather than restarting that work one worker
at a time.

### Respawn stays per-backend

The existing `step()`/`reset()` wrapper's try/except-and-respawn
(`subprocess_backend.py:298-323`) moves to wrap `send_step`+`recv` (or
`send_reset`+`recv`) as a pair, still owned by `SubprocessBackend`, still
invisible to `VecPokemonEnv`. A `BrokenPipeError` on `send_step` — the worker
is already dead — and a `TimeoutError`/`EOFError`/`BrokenPipeError` on `recv`
both route to the same `_restart()` path that exists today. One backend
failing during the gather phase must not prevent `VecPokemonEnv` from calling
`recv()` on the other 63 — `VecPokemonEnv`'s `results = [backend.recv() for
backend in self._backends]` list comprehension already gives that for free,
since each backend catches its own exception internally exactly as it does
today; nothing in `VecPokemonEnv` needs to change to get this.

### Adjacent bug found during review — decide it here, since this refactor touches the exact code

A dedicated review of `src/pokemon_env/` (independent of this spec) surfaced a
real defect sitting in the exact method this refactor rewrites, and flagged
that a plan built from this spec would otherwise carry it forward unexamined.

`handle_command` (`subprocess_backend.py:121-141`) catches bare `Exception`
inside the worker and sends back `("error", f"{type(error).__name__}:
{error}")` for **any** exception raised by `session.step()`/`session.reset()`
— not just process-level failures. Today's `_call()` (and, unchanged in
substance, tomorrow's `recv()`) turns that into a `RuntimeError`, and
`SubprocessBackend.step()`/`.reset()`'s except tuple —
`(TimeoutError, RuntimeError, EOFError, BrokenPipeError)` — routes a
`RuntimeError` from an explicit `("error", ...)` reply through `_restart()`
identically to a hung or crashed process. A genuine, deterministic bug in
`rewards.py` or `aux_state.py` — this project's own code, not the emulator or
the OS — would therefore reproduce on every episode after the respawned
worker reloads `init.state`, and be silently discarded as an elevated
`respawns` count on a telemetry dashboard, never as a crash. This is the
inverse of CLAUDE.md's stated line between programmer errors (should crash
loud) and operating errors (should be retried) — a real bug in this project's
own reward/observation code is currently indistinguishable, from the parent's
side, from a segfaulted PyBoy process.

Because this refactor already rewrites `SubprocessBackend`'s exception
handling wholesale (moving it from one `_call()` into `send_step`+`recv`), the
fix is nearly free to fold in here, and leaving it for a separate change means
touching this exact code twice for two unrelated reasons. Distinguish the two
cases in `recv()`:

- `status == "error"` (the worker explicitly caught and reported an exception
  from `session.step()`/`.reset()`) → this is a **software bug**, not a
  process failure. Re-raise it as-is rather than routing it into `_restart()`.
  A worker that can still send is not a dead or hung worker; the process is
  fine, the code is wrong, and CLAUDE.md's own convention says that crashes.
- `TimeoutError` (no reply within `worker_timeout_s`), `EOFError`, or
  `BrokenPipeError` (the pipe itself failed) → these remain **process
  failures** exactly as today, and keep going through `_restart()` unchanged.

This changes behavior: a bug in `rewards.py`/`aux_state.py` now takes down the
run with a clear traceback instead of silently degrading it — which is the
intended effect. Implementing it requires one boolean check at the top of
`recv()` (was the payload an explicit error reply, or did the call itself
fail?) and does not otherwise change the shape of the refactor above. Add a
test that a `("error", ...)` reply propagates as a raised exception from
`recv()` rather than triggering a respawn, alongside the existing respawn
tests for the three process-failure exception types — `tests/unit/test_pokemon_env_subprocess_backend.py:218-222`
currently proves an error payload raises through the *unrelated* bare
`_call()`/`state_dict()` path, which does not go through `step()`/`reset()`'s
recovery logic at all, so this specific distinction is not yet under test
anywhere.

### Escalation path, if Gate 2 shows this isn't enough

`multiprocessing.connection.wait(object_list, timeout=None)` — verified via
`inspect.signature` against the installed 3.12 stdlib — blocks until any
connection in the list is readable and returns the ready subset. If
sequential `recv()` still leaves meaningful serialization (e.g. because one
slow worker's `poll(timeout)` still blocks the parent from reading the 63
already-ready replies behind it in loop order), the gather loop can be
rewritten as:

```python
pending = list(self._backends)          # or a dict keyed by connection
ready_results: dict[int, StepResult] = {}
deadline = time.monotonic() + self._config.worker_timeout_s
while pending:
    remaining = deadline - time.monotonic()
    ready = wait([b._conn for b in pending], timeout=max(remaining, 0))
    if not ready:
        break  # whatever is left in `pending` timed out
    for conn in ready:
        ...
```

This is **not** proposed as the first implementation. It requires reaching
into `SubprocessBackend`'s private connection from `VecPokemonEnv`, which
breaks the current encapsulation (`VecPokemonEnv` only ever calls the
`EnvBackend` Protocol's public methods) and needs its own respawn-during-gather
design. Per CLAUDE.md's Karpathy guideline — minimal surgical diffs, verify one
stage before building the next — implement the simple send/recv split first,
measure it against Gate 2, and only reach for `wait()`-based gathering if the
simple split's measured throughput still falls short of the rollout budget.

## What changes

- `src/pokemon_env/vec_env.py`: `EnvBackend` Protocol, `VecPokemonEnv.step()`,
  `VecPokemonEnv.reset()`.
- `src/pokemon_env/subprocess_backend.py`: `SubprocessBackend` gains
  `send_step`/`send_reset`/`recv`, replacing `step`/`reset`/`_call`. The
  respawn logic in `_restart`, `_to_result`, `_reset_once` is unchanged in
  substance, just re-homed onto the new methods — **except** `recv()` must
  distinguish an explicit `("error", ...)` reply (re-raise, do not respawn)
  from `TimeoutError`/`EOFError`/`BrokenPipeError` (respawn, as today); see
  "Adjacent bug found during review" above.
- `InProcessBackend` (`vec_env.py`) gains the eager-stash
  `send_step`/`send_reset`/`recv` shown above.
- Tests referencing `EnvBackend.step`/`.reset()` directly (the
  `FakeVecEnv`/backend fakes this project's PPO and env test suites already
  use) need their fakes updated to the two-phase interface. Grep
  `tests/unit/` and `tests/integration/` for `.step(` and `.reset(` calls
  against a backend (not `VecPokemonEnv` or `VecStep`) before starting.
- No change to `EnvBackend.state_dict`, `.load_state_dict`, `.stats`, or
  `.close` — those run once per PPO update (or at shutdown), not once per
  rollout step, so their sequential cost is three orders of magnitude below
  the per-step dispatch loop and not worth the same treatment.

## Testing

- A hand-written fake backend (extending the existing `FakeVecEnv` pattern in
  `tests/unit/fakes.py`) that records call order across `send_step`/`recv`
  proves `VecPokemonEnv` issues every `send_*` before any `recv()` — the
  property this whole fix exists to establish. Assert the fake's call log is
  `[send, send, ..., send, recv, recv, ..., recv]`, not interleaved.
- A test that one backend's `send_step` raising doesn't stop `recv()` from
  being called on the rest — mirrors the existing per-backend respawn tests
  but exercises the interleaved-failure case that only exists once send/recv
  are split.
- A test that a worker's explicit `("error", ...)` reply propagates as a
  raised exception from `recv()` rather than triggering `_restart()` — the
  fix from "Adjacent bug found during review" above. Parametrize alongside a
  second case confirming `TimeoutError`/`EOFError`/`BrokenPipeError` still do
  respawn, so the two paths are pinned as deliberately different outcomes,
  not merged back together by a future edit.
- No unit test can prove the wall-clock win — that needs real subprocesses and
  real timing, which is exactly what Gate 2 already measures. Rerun Gate 2's
  `pokemon-ppo preflight --n-envs 16 32 64` before and after this change and
  record both numbers in the run log; this spec's ~64× dispatch-loop
  projection is a claim to be confirmed there, not asserted as fact here.
- Existing tests exercising `SubprocessBackend`'s respawn-on-timeout and
  respawn-on-broken-pipe behavior must still pass unmodified in substance —
  only the method names they call against change.

## Non-goals

- Does not touch `EnvSession`, `ram.py`, `aux_state.py`, or reward logic — this
  is purely the parent-side dispatch loop.
- Does not revisit the rollout/gradient-update overlap question — already
  analyzed and declined this session on staleness grounds, unrelated to this
  fix.
- Does not change `n_envs`, frame-skip, or any other PPO trainer spec
  hyperparameter. Gate 2 may reveal that a different `n_envs` is now optimal
  once dispatch no longer serializes, but that is a measurement to make after
  this fix lands, not a decision to pre-commit here.
