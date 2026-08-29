# VecPokemonEnv Step Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `VecPokemonEnv.step()`/`.reset()` and their backends' dispatch into two phases (send, then recv) so the 64 subprocess workers overlap instead of being called one full round trip at a time, and fix an adjacent bug where a real bug in this project's own reward/observation code is silently swallowed as a worker respawn.

**Architecture:** `EnvBackend` gains `send_reset()`/`send_step()`/`recv()` in place of `reset()`/`step()`. `VecPokemonEnv` fires every backend's `send_*` in one loop, then collects every `recv()` in a second loop — no code path change beyond that at the `VecPokemonEnv` level. `InProcessBackend` fakes the split by running eagerly on `send_*` and returning the stashed result on `recv()`. `SubprocessBackend` actually overlaps: `send_*` writes to the pipe and returns; `recv()` polls and reads. A `BrokenPipeError`/`OSError` on send is swallowed internally (never propagated out of the dispatch loop) and re-surfaces from `recv()` on the existing `_restart()` respawn path. `recv()` also fixes the adjacent bug: an explicit `("error", ...)` reply is re-raised as `RuntimeError` and must never be caught by the respawn path, while `TimeoutError`/`EOFError`/`BrokenPipeError` still respawn exactly as today.

**Tech Stack:** Python 3.12, `multiprocessing` (spawn context, `Pipe`, `SharedMemory`), pytest 9.1.1 with hand-written fakes (no `mock.patch`). No PyTorch or asyncio in this change — the concurrency here is OS-process/pipe-level, not asyncio; async-expert's structured-concurrency principles (bound the fan-out, give every failure an explicit policy, never let one failure orphan the rest, never swallow a signal that should propagate) apply conceptually and are cited per-task below, but no `async def` code is introduced.

**Spec:** `docs/superpowers/specs/2026-08-29-vec-env-step-concurrency-design.md`

## Global Constraints

- `EnvBackend` Protocol (`src/pokemon_env/vec_env.py`) becomes exactly: `send_reset() -> None`, `send_step(action: int) -> None`, `recv() -> StepResult`, `state_dict() -> dict`, `load_state_dict(state: dict) -> None`, `stats() -> dict`, `close() -> None`. `reset()`/`step()` are removed, not deprecated — no backwards-compatibility shim.
- `VecPokemonEnv.step()`/`.reset()` must issue every backend's `send_*` before calling `recv()` on any backend. This ordering is the entire point of the change and is directly asserted by a test.
- `SubprocessBackend.send_step()`/`send_reset()` must never let a `BrokenPipeError`/`OSError` from `self._conn.send()` propagate to the caller — `VecPokemonEnv`'s dispatch loop has no try/except, so a raised exception there would abort dispatch to every backend after it, which is strictly worse than today's sequential loop. The failure is swallowed and surfaces from the paired `recv()` instead.
- `recv()` distinguishes a worker's explicit `("error", ...)` reply (a software bug in `session.step()`/`.reset()`, i.e. this project's own `rewards.py`/`aux_state.py` code) from a `TimeoutError`/`EOFError`/`BrokenPipeError` (a process failure). The former re-raises as `RuntimeError` and must **never** reach `_restart()`. The latter still respawns exactly as today, including forcing `done=True`/`reward=0.0` when the failed dispatch was a `send_step` (not a `send_reset`).
- `state_dict()`, `load_state_dict()`, `stats()`, `close()` are unchanged — they stay synchronous single-round-trip calls via the existing `_call()` helper. Per spec: they run once per PPO update or at shutdown, three orders of magnitude less often than the per-step dispatch loop, so they are explicitly out of scope for the two-phase treatment.
- No change to `n_envs`, frame-skip, or any other PPO trainer hyperparameter (spec Non-goals).
- Every new/changed test follows this repo's pytest-expert conventions: one behavior per test, no `if`/`for`/`while` in a test body, exact expected values (not loose ranges), `pytest.raises(SpecificException, match=...)`.
- Negative-space guard: send/recv call order becomes a real invariant for the first time in this codebase (it was implicit and unbreakable when `step()`/`reset()` were one synchronous call). `SubprocessBackend` and `InProcessBackend` both raise `RuntimeError` (not a bare `assert`, matching this codebase's existing `FrameBuffer.unlink()` precedent for "always-on" misuse guards) on `recv()` with no prior matching `send_*`, and on a second `send_*` before the first is `recv()`'d.

---

## Task 1: `EnvBackend` Protocol + `InProcessBackend` two-phase dispatch

**Files:**
- Modify: `src/pokemon_env/vec_env.py:34-40` (`EnvBackend` Protocol), `src/pokemon_env/vec_env.py:43-75` (`InProcessBackend`)
- Test: `tests/unit/test_pokemon_env_vec_env.py`

**Interfaces:**
- Consumes: `pokemon_env.session.EnvSession.step(action: int) -> StepResult`, `.reset() -> StepResult` (unchanged, already exist).
- Produces: `EnvBackend.send_reset() -> None`, `EnvBackend.send_step(action: int) -> None`, `EnvBackend.recv() -> StepResult` — the Protocol Task 2 and Task 3 implement against.

This task proves the two-phase contract works end-to-end using the cheapest backend (no subprocess, no shared memory) before touching the concurrency-bearing code in Task 3.

- [ ] **Step 1: Write the failing tests for `InProcessBackend`'s two-phase contract**

Add to `tests/unit/test_pokemon_env_vec_env.py` (near the top, after the existing fixtures/imports — `InProcessBackend`, `EnvSession`, `EnvConfig` are already imported in this file):

```python
def test_inprocess_backend_recv_returns_the_result_from_send_step() -> None:
    config = EnvConfig(n_envs=1, max_steps=2)
    backend = InProcessBackend(EnvSession(FakeEmulator(), config, init_state=b"init"))
    backend.send_reset()
    backend.recv()

    backend.send_step(0)
    result = backend.recv()

    assert (result.episode_id, result.done) == (0, False)


def test_inprocess_backend_recv_without_a_prior_send_raises() -> None:
    """The two-phase split makes call order a real invariant for the first
    time -- a stray recv() with nothing dispatched must be caught here, not
    return a stale or default result."""
    config = EnvConfig(n_envs=1, max_steps=2)
    backend = InProcessBackend(EnvSession(FakeEmulator(), config, init_state=b"init"))

    with pytest.raises(RuntimeError, match="no matching send_step/send_reset"):
        backend.recv()


def test_inprocess_backend_send_step_while_a_previous_dispatch_is_unread_raises() -> None:
    """Guards the exact bug class this refactor introduces: a future edit to
    VecPokemonEnv that dispatches twice before recv() would otherwise
    silently overwrite which command recv() answers for."""
    config = EnvConfig(n_envs=1, max_steps=2)
    backend = InProcessBackend(EnvSession(FakeEmulator(), config, init_state=b"init"))
    backend.send_reset()

    with pytest.raises(RuntimeError, match="previous dispatch has not been recv"):
        backend.send_step(0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_pokemon_env_vec_env.py -k inprocess_backend -v`
Expected: FAIL with `AttributeError: 'InProcessBackend' object has no attribute 'send_reset'` (or `send_step`) for all three.

- [ ] **Step 3: Rewrite `EnvBackend` and `InProcessBackend`**

In `src/pokemon_env/vec_env.py`, replace the `EnvBackend` Protocol:

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

Replace `InProcessBackend`:

```python
class InProcessBackend:
    """Drives an EnvSession directly. Exists so vec_env logic -- autoreset
    ordering, episode_id monotonicity, batching -- is testable without
    spawning processes.

    Has no concurrency to exploit -- it drives one EnvSession in the parent
    process -- so send_step/send_reset just run the work eagerly and stash
    the StepResult; recv() returns and clears it. The stash also enforces
    the send/recv call-order invariant SubprocessBackend depends on for real
    concurrency: a second send before a recv, or a recv with nothing
    pending, is a programmer error in the dispatch loop, not something that
    should silently succeed with a stale value."""

    def __init__(self, session: EnvSession) -> None:
        self._session = session
        self._pending: StepResult | None = None

    def _dispatch(self, result: StepResult) -> None:
        if self._pending is not None:
            raise RuntimeError(
                "InProcessBackend: send_step/send_reset called while a previous "
                "dispatch has not been recv()'d yet -- sends and recvs must "
                "alternate one-for-one"
            )
        self._pending = result

    def send_reset(self) -> None:
        self._dispatch(self._session.reset())

    def send_step(self, action: int) -> None:
        self._dispatch(self._session.step(action))

    def recv(self) -> StepResult:
        if self._pending is None:
            raise RuntimeError(
                "InProcessBackend: recv() called with no matching "
                "send_step/send_reset"
            )
        result, self._pending = self._pending, None
        return result

    def state_dict(self) -> dict:
        """Same envelope as SubprocessBackend's, so a checkpoint is portable
        between the two backends. The parent-side counters are constant here:
        an in-process backend has no worker to lose and never respawns."""
        return {
            "session": self._session.state_dict(),
            "respawns": 0,
            "episode_offset": 0,
            "last_episode_id": -1,
        }

    def load_state_dict(self, state: dict) -> None:
        self._session.load_state_dict(state["session"])

    def stats(self) -> dict:
        return self._session.stats()

    def close(self) -> None:
        self._session.close()
```

Note: `_dispatch` runs the real `session.reset()`/`.step()` call *eagerly* (inside `send_reset`/`send_step`, not `recv`) — this mirrors the docstring's claim that `InProcessBackend` has no concurrency to exploit, and keeps its behavior (exceptions raised by `EnvSession.step()`'s action-range validation, for instance) surfacing at the same call the real backend would eventually make it surface at in the fully-concurrent `SubprocessBackend` case: `recv()` is where a failure becomes visible to `VecPokemonEnv`, but here the work itself already happened at `send_*` time since there's nothing to overlap it with.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_pokemon_env_vec_env.py -k inprocess_backend -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Verify the misuse-guard tests can actually fail**

Temporarily comment out the `if self._pending is not None: raise RuntimeError(...)` check in `_dispatch` and confirm `test_inprocess_backend_send_step_while_a_previous_dispatch_is_unread_raises` goes red. Temporarily comment out the `if self._pending is None: raise RuntimeError(...)` check in `recv()` and confirm `test_inprocess_backend_recv_without_a_prior_send_raises` goes red. Restore both checks and confirm all three pass again.

- [ ] **Step 6: Commit**

```bash
git add src/pokemon_env/vec_env.py tests/unit/test_pokemon_env_vec_env.py
git commit -m "feat(pokemon_env): split EnvBackend into two-phase send/recv"
```

---

## Task 2: Rewire `VecPokemonEnv.step()`/`.reset()` to dispatch two-phase, prove send-before-recv ordering

**Files:**
- Modify: `src/pokemon_env/vec_env.py:123-139` (`VecPokemonEnv.step()`, `.reset()`)
- Modify: `tests/unit/fakes.py:66-119` (`FakeBackend`)
- Test: `tests/unit/test_pokemon_env_vec_env.py`

**Interfaces:**
- Consumes: `EnvBackend.send_reset() -> None`, `.send_step(action: int) -> None`, `.recv() -> StepResult` from Task 1.
- Produces: `VecPokemonEnv.step(actions: np.ndarray) -> VecStep`, `.reset() -> VecStep` — unchanged external signatures, so every existing `VecPokemonEnv`-level test in this file (fixtures `vec_env`, `two_badge_vec_env`, all the reward/episode/state_dict tests already in the file) continues to pass unmodified once `InProcessBackend` implements the two-phase Protocol from Task 1.

This is the task that actually establishes "every send before any recv" — the property the whole spec exists to deliver. It's provable today, without `SubprocessBackend`, because the property is about `VecPokemonEnv`'s call order, not about wall-clock overlap (that needs Task 3 plus real subprocesses, which is Task 4's manual Gate 2 measurement).

- [ ] **Step 1: Write the failing call-order test, using a new recording fake**

Add to `tests/unit/fakes.py`, after `FakeBackend` (which Step 3 below rewrites in place — add `RecordingBackend` as a new class right after it):

```python
class RecordingBackend:
    """Hand-written fake typed against `EnvBackend`, logging every
    send_step/send_reset/recv call into a list shared across every backend
    in one VecPokemonEnv. Exists to prove VecPokemonEnv issues every send_*
    before any recv() -- the property the concurrent-dispatch fix exists to
    establish. Nothing about that ordering is visible from the returned
    StepResults themselves, so this is the only way to test it."""

    def __init__(self, index: int, call_log: list[str]) -> None:
        self._index = index
        self._call_log = call_log
        self._pending: StepResult | None = None

    def _result(self) -> StepResult:
        return StepResult(
            frame=np.zeros((144, 160), dtype=np.uint8),
            aux=np.zeros(AUX_STATE_DIM, dtype=np.float32),
            reward=0.0,
            done=False,
            episode_id=0,
            components={},
            clipped=False,
        )

    def send_reset(self) -> None:
        self._call_log.append(f"send:{self._index}")
        self._pending = self._result()

    def send_step(self, action: int) -> None:
        self._call_log.append(f"send:{self._index}")
        self._pending = self._result()

    def recv(self) -> StepResult:
        self._call_log.append(f"recv:{self._index}")
        result, self._pending = self._pending, None
        return result

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def stats(self) -> dict:
        return {}

    def close(self) -> None:
        pass
```

Add to `tests/unit/test_pokemon_env_vec_env.py` (add `RecordingBackend` to the `from .fakes import ...` line):

```python
def test_step_dispatches_every_send_before_any_recv() -> None:
    """The property this whole fix exists to establish: VecPokemonEnv.step()
    must not block on backend i's reply before backend i+1 has even been
    told what to do -- that's the sequential-dispatch bug the spec measures
    at ~64x on 64 real subprocess workers. This test can't see wall-clock
    time (RecordingBackend does no real work), but call order is exactly
    what a return to the old `backend.step()`-per-iteration shape would
    break, and this catches that regardless of timing."""
    call_log: list[str] = []
    backends = [RecordingBackend(i, call_log) for i in range(3)]
    vec_env = VecPokemonEnv(backends, EnvConfig(n_envs=3))
    vec_env.reset()
    call_log.clear()

    vec_env.step(np.zeros(3, dtype=np.int64))

    assert call_log == ["send:0", "send:1", "send:2", "recv:0", "recv:1", "recv:2"]


def test_reset_dispatches_every_send_before_any_recv() -> None:
    call_log: list[str] = []
    backends = [RecordingBackend(i, call_log) for i in range(3)]
    vec_env = VecPokemonEnv(backends, EnvConfig(n_envs=3))

    vec_env.reset()

    assert call_log == ["send:0", "send:1", "send:2", "recv:0", "recv:1", "recv:2"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_pokemon_env_vec_env.py -k dispatches_every_send -v`
Expected: FAIL — `vec_env` fixture's underlying `InProcessBackend`-driven `VecPokemonEnv.step()`/`.reset()` still call the (now-removed) `backend.reset()`/`backend.step()`, so this fails with `AttributeError: 'RecordingBackend' object has no attribute 'reset'` (from `VecPokemonEnv` itself, since `VecPokemonEnv.step`/`.reset` haven't been rewritten yet).

- [ ] **Step 3: Rewrite `VecPokemonEnv.step()`/`.reset()` and `FakeBackend`**

In `src/pokemon_env/vec_env.py`, replace `VecPokemonEnv.reset()` and `.step()`:

```python
    def reset(self) -> VecStep:
        self._needs_reset[:] = False
        for backend in self._backends:
            backend.send_reset()
        return self._collect([backend.recv() for backend in self._backends])

    def step(self, actions: np.ndarray) -> VecStep:
        if len(actions) != self._config.n_envs:
            raise ValueError(
                f"actions has length {len(actions)}, expected {self._config.n_envs}"
            )
        for backend, action, needs_reset in zip(
            self._backends, actions, self._needs_reset, strict=True
        ):
            if needs_reset:
                backend.send_reset()
            else:
                backend.send_step(int(action))
        results = [backend.recv() for backend in self._backends]
        self._needs_reset = np.array([result.done for result in results], dtype=bool)
        return self._collect(results)
```

In `tests/unit/fakes.py`, replace `FakeBackend`'s `reset`/`step` methods:

```python
    def send_reset(self) -> None:
        self._pending = self._result()

    def send_step(self, action: int) -> None:
        self._pending = self._result()

    def recv(self) -> StepResult:
        result, self._pending = self._pending, None
        return result
```

and add `self._pending: StepResult | None = None` to `FakeBackend.__init__` (alongside the existing `self._step_count = step_count` line). Update the class docstring's `_result` comment, which currently reads "reset/step/state_dict/load_state_dict/close and this helper exist only to satisfy EnvBackend's structural Protocol surface -- no test calls them today; only stats() is exercised" — this stays accurate for `send_reset`/`send_step`/`recv`/`state_dict`/`load_state_dict`/`close`, so only the method names in that comment need updating: `reset/step` -> `send_reset/send_step/recv`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_pokemon_env_vec_env.py -v`
Expected: PASS — every test in the file, including the two new ordering tests and every pre-existing test in this file (they exercise `InProcessBackend` via the `vec_env`/`two_badge_vec_env` fixtures, which Task 1 already made two-phase-compatible).

- [ ] **Step 5: Verify the ordering test can actually fail**

Temporarily change `VecPokemonEnv.step()` back to the old interleaved shape (call `backend.recv()` immediately inside the same loop as `backend.send_step()`, i.e. move `results = [backend.recv() ...]` inside the `for` loop right after each dispatch) and confirm `test_step_dispatches_every_send_before_any_recv` goes red with something like `assert ['send:0', 'recv:0', 'send:1', 'recv:1', 'send:2', 'recv:2'] == ['send:0', 'send:1', 'send:2', 'recv:0', 'recv:1', 'recv:2']`. Revert the change and confirm it passes again.

- [ ] **Step 6: Commit**

```bash
git add src/pokemon_env/vec_env.py tests/unit/fakes.py tests/unit/test_pokemon_env_vec_env.py
git commit -m "feat(pokemon_env): dispatch every backend send before any recv"
```

---

## Task 3: `SubprocessBackend` two-phase dispatch + the explicit-error-vs-process-failure fix

**Files:**
- Modify: `src/pokemon_env/subprocess_backend.py:199-358` (`SubprocessBackend`)
- Modify: `tests/unit/test_pokemon_env_subprocess_backend.py`

**Interfaces:**
- Consumes: `EnvBackend` Protocol from Task 1 (`send_reset`, `send_step`, `recv`, plus the unchanged `state_dict`/`load_state_dict`/`stats`/`close`).
- Produces: `SubprocessBackend.send_step(action: int) -> None`, `.send_reset() -> None`, `.recv() -> StepResult` — this is the production dispatch path `build_subprocess_vec_env` wires into `VecPokemonEnv`, so no other file needs to change once this task lands.

This is the task that delivers the actual ~64x dispatch-loop fix described in the spec, and folds in the adjacent bug fix (an explicit worker error must crash loud, not respawn silently) because both changes touch the same method.

- [ ] **Step 1: Extend `FakeConnection` to script a send-time failure**

In `tests/unit/test_pokemon_env_subprocess_backend.py`, replace the `FakeConnection` class:

```python
class FakeConnection:
    """Stands in for a multiprocessing Pipe end. `poll_results`, `responses`,
    and `send_side_effects` are each consumed in order, so a test scripts the
    exact sequence the backend will observe."""

    def __init__(
        self,
        responses: list,
        poll_results: list[bool] | None = None,
        send_side_effects: list[BaseException | None] | None = None,
    ) -> None:
        self.sent: list = []
        self.responses = list(responses)
        self.poll_results = list(poll_results) if poll_results is not None else []
        self.send_side_effects = (
            list(send_side_effects) if send_side_effects is not None else []
        )
        self.closed = False
        # Recorded so a test can assert the configured timeout actually reaches
        # poll(). The failure this catches is poll(None), which blocks forever:
        # an unattended run stalls silently while the GPU keeps billing, which
        # is strictly worse than crashing.
        self.poll_timeouts: list[float | None] = []

    def send(self, obj: object) -> None:
        if self.send_side_effects:
            effect = self.send_side_effects.pop(0)
            if effect is not None:
                raise effect
        self.sent.append(obj)

    def poll(self, timeout: float | None = None) -> bool:
        self.poll_timeouts.append(timeout)
        return self.poll_results.pop(0) if self.poll_results else True

    def recv(self):
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True
```

And the `_backend` helper, to pass the new parameter through:

```python
def _backend(frame_buffer, responses, poll_results=None, index=0, send_side_effects=None):
    """Helper, not a test: a SubprocessBackend wired to scripted fakes.
    Returns (backend, connection, process) so tests can assert on all three."""
    connection = FakeConnection(responses, poll_results, send_side_effects)
    process = FakeProcess()

    def fake_spawn(shm_name, idx, config, rom_path, init_state):
        return connection, process

    backend = SubprocessBackend(
        index=index,
        shm_name=frame_buffer.name,
        config=EnvConfig(n_envs=3, max_steps=8),
        rom_path="unused.gb",
        init_state=b"init",
        frame_slot=frame_buffer.array[index],
        spawn_worker=fake_spawn,
    )
    return backend, connection, process
```

- [ ] **Step 2: Update every existing call site from `backend.reset()`/`backend.step(x)` to the two-phase form**

Replace these test bodies in `tests/unit/test_pokemon_env_subprocess_backend.py` (method names and assertions unchanged, only the call shape):

```python
def test_step_sends_the_step_command_with_its_action(frame_buffer) -> None:
    backend, connection, _ = _backend(frame_buffer, [_ok(), _ok()])
    backend.send_reset()
    backend.recv()

    backend.send_step(3)
    backend.recv()

    assert connection.sent[-1] == (Command.STEP, 3)
```

```python
def test_step_respawns_and_forces_done_when_the_worker_times_out(
    frame_buffer,
) -> None:
    """A dead worker must not take the run down. It respawns from init.state
    and forces done=True so the trainer resets its memory for that env --
    a respawned worker shares no history with the old one."""
    backend, _, _ = _backend(
        frame_buffer, [_ok(), _ok()], poll_results=[True, False, True]
    )
    backend.send_reset()
    backend.recv()

    backend.send_step(0)
    result = backend.recv()

    assert (result.done, result.reward) == (True, 0.0)
```

```python
def test_a_respawn_increments_the_respawn_counter(frame_buffer) -> None:
    """Respawn rate is a logged leading indicator of memory pressure, so the
    counter must actually move."""
    backend, _, _ = _backend(
        frame_buffer, [_ok(), _ok()], poll_results=[True, False, True]
    )
    backend.send_reset()
    backend.recv()
    backend.send_step(0)
    backend.recv()

    assert backend.respawns == 1
```

```python
def test_a_respawn_terminates_a_process_that_is_still_alive(frame_buffer) -> None:
    """A hung worker is alive but unresponsive; leaving it running leaks a
    process and its emulator for the rest of the run."""
    backend, _, process = _backend(
        frame_buffer, [_ok(), _ok()], poll_results=[True, False, True]
    )
    backend.send_reset()
    backend.recv()
    backend.send_step(0)
    backend.recv()

    assert process.terminated is True
```

```python
def test_to_result_takes_its_frame_from_the_shared_slot(frame_buffer) -> None:
    """The frame never rides the pipe -- the backend reads it from its own
    shared-memory slice. If it read the payload instead, the 1.5 GB per
    rollout the shared block exists to remove would come straight back."""
    frame_buffer.array[0] = 77
    backend, _, _ = _backend(frame_buffer, [_ok()])

    backend.send_reset()
    result = backend.recv()

    assert int(result.frame[0, 0]) == 77
```

```python
def test_reset_respawns_instead_of_propagating_a_dead_worker(frame_buffer) -> None:
    """VecPokemonEnv routes every autoreset through backend.send_reset()+
    recv(), so a worker that dies on an episode boundary must be recovered
    here. Before this guard existed the exception escaped VecPokemonEnv.step()
    and killed the whole run -- exactly what the respawn logic exists to
    prevent."""
    backend, _, _ = _backend(frame_buffer, [_ok(), _ok()], poll_results=[False, True])

    backend.send_reset()
    result = backend.recv()

    assert (backend.respawns, result.done) == (1, False)
```

```python
def test_reset_recovery_does_not_recurse_when_every_spawn_dies(frame_buffer) -> None:
    """_restart ends in the bare _reset_once, not another two-phase dispatch,
    so a worker that dies on every spawn raises loudly instead of looping
    forever. A silent infinite retry is the worse failure: the run neither
    progresses nor reports."""
    backend, _, _ = _backend(
        frame_buffer, [_ok(), _ok()], poll_results=[False, False, False]
    )

    backend.send_reset()
    with pytest.raises(TimeoutError, match="did not answer"):
        backend.recv()
```

```python
def test_episode_id_keeps_climbing_across_a_respawn(frame_buffer) -> None:
    """VecStep's contract is monotonic episode_id per env, and the transformer
    uses it to detect episode boundaries. A respawned worker's EnvSession
    restarts its own counter at 0, so without the offset an env at episode 3
    would emit 3 -> 0 and silently merge two distinct episodes."""
    backend, _, _ = _backend(
        frame_buffer, [_ok(3), _ok(0), _ok(0)], poll_results=[True, False, True]
    )
    backend.send_reset()
    before = backend.recv().episode_id

    backend.send_step(0)
    after = backend.recv().episode_id

    assert (before, after) == (3, 4)
```

```python
def test_state_dict_carries_the_episode_offset_across_a_resume(frame_buffer) -> None:
    """The offset lives in the parent, so the worker's checkpointed session
    cannot supply it. Dropped, a resume would restart the visible episode
    sequence from the restored session's own counter."""
    backend, _, _ = _backend(
        frame_buffer,
        [_ok(3), _ok(0), ("ok", {"state": {"step_count": 7}})],
        poll_results=[True, False, True],
    )
    backend.send_reset()
    backend.recv()
    backend.send_step(0)
    backend.recv()

    state = backend.state_dict()

    assert (state["episode_offset"], state["respawns"]) == (4, 1)
```

```python
def test_call_passes_the_configured_timeout_to_poll(frame_buffer) -> None:
    """The failure this catches is `poll(None)`, which blocks forever. An
    unattended run would stall silently while the GPU keeps billing --
    strictly worse than crashing, because nothing alerts."""
    backend, connection, _ = _backend(frame_buffer, [_ok()])

    backend.send_reset()
    backend.recv()

    assert connection.poll_timeouts == [pytest.approx(60.0)]
```

`test_call_raises_timeout_naming_the_env_index` and `test_a_worker_error_response_raises_naming_the_env` (both drive `backend.state_dict()`, not `step`/`reset`) are unaffected — leave them as-is.

- [ ] **Step 3: Write the failing tests for the new behavior — swallowed send failure, and error-vs-process-failure**

Add to `tests/unit/test_pokemon_env_subprocess_backend.py`:

```python
def test_send_step_swallows_a_broken_pipe_and_recv_respawns(frame_buffer) -> None:
    """Two-phase dispatch means send_step cannot propagate a broken pipe out
    of VecPokemonEnv's dispatch loop -- doing so would abort every backend
    after it before recv() is even called on the ones that already sent
    fine. The failure must be swallowed here and surfaced from recv()
    instead, on the same _restart() path a recv()-side timeout already
    takes."""
    backend, _, _ = _backend(
        frame_buffer,
        [_ok(), _ok()],
        poll_results=[True, True],
        send_side_effects=[None, BrokenPipeError("pipe gone"), None],
    )
    backend.send_reset()
    backend.recv()

    backend.send_step(0)  # underlying conn.send raises -- must not propagate
    result = backend.recv()

    assert (result.done, result.reward) == (True, 0.0)


def test_recv_reraises_an_explicit_worker_error_without_respawning(frame_buffer) -> None:
    """CLAUDE.md's own convention: programmer errors crash loud, operating
    errors get retried. Before this fix, a genuine bug in rewards.py/
    aux_state.py surfaced identically to a hung process -- silently
    discarded as an elevated respawn count instead of a traceback."""
    backend, _, _ = _backend(
        frame_buffer, [("error", "ValueError: boom")], poll_results=[True]
    )
    backend.send_step(0)

    with pytest.raises(RuntimeError, match="env 0 worker failed"):
        backend.recv()

    assert backend.respawns == 0


def test_recv_still_respawns_on_a_genuine_timeout(frame_buffer) -> None:
    """Pinned alongside the error-reply test above so the two paths cannot be
    silently merged back together by a future edit."""
    backend, _, _ = _backend(frame_buffer, [_ok()], poll_results=[False, True])
    backend.send_step(0)

    result = backend.recv()

    assert (backend.respawns, result.done) == (1, True)


def test_recv_without_a_prior_send_raises(frame_buffer) -> None:
    """The two-phase split makes call order significant for the first time --
    a stray recv() with nothing dispatched must be caught here, not silently
    block on a pipe read that was never primed."""
    backend, _, _ = _backend(frame_buffer, [_ok()])

    with pytest.raises(RuntimeError, match="no matching send_step/send_reset"):
        backend.recv()


def test_send_step_while_a_previous_dispatch_is_unread_raises(frame_buffer) -> None:
    """Guards the exact bug class this refactor introduces: a future edit to
    VecPokemonEnv that calls send_step twice before recv() would otherwise
    silently overwrite which command recv() answers for."""
    backend, _, _ = _backend(frame_buffer, [_ok()])
    backend.send_step(0)

    with pytest.raises(RuntimeError, match="previous dispatch has not been recv"):
        backend.send_step(1)
```

- [ ] **Step 4: Run all subprocess_backend tests to verify the new ones fail and confirm the extent of the rewrite needed**

Run: `pytest tests/unit/test_pokemon_env_subprocess_backend.py -v`
Expected: every test still calling `.reset()`/`.step()` fails with `AttributeError` (they were updated to the new call shape in Step 2, so this should already be clean if Step 2 landed first — if run before Step 2's edits are saved, expect widespread `AttributeError: 'SubprocessBackend' object has no attribute 'send_reset'`). The five new tests from Step 3 fail with the same `AttributeError` until Step 5 lands.

- [ ] **Step 5: Rewrite `SubprocessBackend`**

In `src/pokemon_env/subprocess_backend.py`, replace the `SubprocessBackend` class (lines 199-358) in full:

```python
class SubprocessBackend:
    """Parent-side handle on one worker. Respawns it from init.state on death
    or timeout rather than taking the whole run down.

    step/reset are two-phase: send_step()/send_reset() dispatch the command
    and return immediately, catching only a failure of the send itself.
    recv() blocks for the reply. VecPokemonEnv relies on this split to fire
    every backend's send before waiting on any reply -- see
    docs/superpowers/specs/2026-08-29-vec-env-step-concurrency-design.md."""

    def __init__(
        self,
        index: int,
        shm_name: str,
        config: EnvConfig,
        rom_path: str,
        init_state: bytes,
        frame_slot: np.ndarray,
        spawn_worker: SpawnWorker = spawn_real_worker,
    ) -> None:
        self._index = index
        self._shm_name = shm_name
        self._config = config
        self._rom_path = rom_path
        self._init_state = init_state
        self._frame_slot = frame_slot
        self._spawn_worker = spawn_worker
        self._respawns = 0
        # A respawned worker gets a fresh EnvSession whose episode_id restarts
        # at 0, but VecStep's contract is that episode_id is MONOTONIC per env
        # -- the transformer uses it to detect episode boundaries, and a
        # repeated id merges two distinct episodes in its attention mask. The
        # offset keeps the parent-visible sequence climbing across respawns.
        self._episode_offset = 0
        self._last_episode_id = -1
        # Two-phase dispatch state. None means "no send is outstanding";
        # recv() clears it back to None as soon as it reads it, so a stray
        # second recv() (or a recv() with no matching send) is caught here
        # rather than silently blocking on a pipe read that was never primed.
        self._pending_is_step: bool | None = None
        self._send_failed = False
        self._spawn()

    @property
    def respawns(self) -> int:
        return self._respawns

    def _spawn(self) -> None:
        self._conn, self._process = self._spawn_worker(
            self._shm_name, self._index, self._config, self._rom_path, self._init_state
        )

    def _call(self, command: Command, argument: object = None) -> dict:
        """Synchronous send+recv, for commands that are never dispatched
        two-phase (STATE_DICT, LOAD_STATE, STATS -- once per PPO update, not
        once per step) and for the internal reset used to recover after a
        respawn."""
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
        silently re-earn rewards for progress already banked.

        Ends in `_reset_once`, deliberately NOT a two-phase send_reset()+
        recv(): `_reset_once` recovers by calling back into here, so
        recovering twice would recurse forever on a worker that dies on
        every spawn. An unattended run that spins silently is worse than one
        that dies with a clear error."""
        self._respawns += 1
        self._episode_offset = self._last_episode_id + 1
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=5)
        self._close_connection()
        self._spawn()
        return self._reset_once()

    def _close_connection(self) -> None:
        """Release the old pipe's file descriptors. Respawns are expected over
        a multi-day run, so leaking one FD pair per respawn eventually
        exhausts the parent's descriptor table."""
        try:
            self._conn.close()
        except OSError:  # obs: allow LOG007 -- the pipe is already closed; nothing to report
            pass

    def _to_result(self, payload: dict) -> StepResult:
        episode_id = int(payload["episode_id"]) + self._episode_offset
        self._last_episode_id = max(self._last_episode_id, episode_id)
        return StepResult(
            frame=self._frame_slot,
            aux=payload["aux"],
            reward=payload["reward"],
            done=payload["done"],
            episode_id=episode_id,
            components=payload["components"],
            clipped=payload["clipped"],
        )

    def _reset_once(self) -> StepResult:
        """Bare reset with no recovery. Used by `_restart` so recovery cannot
        recurse."""
        return self._to_result(self._call(Command.RESET))

    def _dispatch(self, command: Command, argument: object, is_step: bool, label: str) -> None:
        if self._pending_is_step is not None:
            raise RuntimeError(
                f"env {self._index}: {label} called while a previous dispatch has "
                "not been recv()'d yet -- sends and recvs must alternate one-for-one"
            )
        self._pending_is_step = is_step
        try:
            self._conn.send((command, argument))
            self._send_failed = False
        except (BrokenPipeError, OSError):
            # The worker is already dead. Swallowed here, not raised: this
            # runs inside VecPokemonEnv's dispatch loop, which has no
            # try/except, so raising would abort dispatch to every backend
            # after this one before recv() is even called on the ones that
            # already sent fine. recv() surfaces it instead, on the same
            # _restart() path a recv()-side timeout already takes.
            self._send_failed = True

    def send_reset(self) -> None:
        """VecPokemonEnv routes every autoreset through here, so a worker that
        dies on an episode boundary must be recovered via recv() below --
        otherwise it propagates out of VecPokemonEnv.step() and kills the
        run."""
        self._dispatch(Command.RESET, None, is_step=False, label="send_reset")

    def send_step(self, action: int) -> None:
        self._dispatch(Command.STEP, action, is_step=True, label="send_step")

    def recv(self) -> StepResult:
        if self._pending_is_step is None:
            raise RuntimeError(
                f"env {self._index}: recv() called with no matching "
                "send_step/send_reset"
            )
        is_step = self._pending_is_step
        self._pending_is_step = None
        command_name = "STEP" if is_step else "RESET"
        try:
            if self._send_failed:
                raise BrokenPipeError(
                    f"env {self._index}: send failed before the worker could reply"
                )
            if not self._conn.poll(self._config.worker_timeout_s):
                raise TimeoutError(
                    f"env {self._index} did not answer {command_name} within "
                    f"{self._config.worker_timeout_s}s; a 24-frame tick takes about "
                    "1ms, so this is a hang, not slowness"
                )
            status, payload = self._conn.recv()
        except (TimeoutError, EOFError, BrokenPipeError):
            restarted = self._restart()
            if not is_step:
                return restarted
            # Force the episode boundary so the trainer resets its KV cache
            # for this env; a respawned worker shares no history with the
            # old one.
            return StepResult(
                frame=restarted.frame,
                aux=restarted.aux,
                reward=0.0,
                done=True,
                episode_id=restarted.episode_id,
                components={},
                clipped=False,
            )
        if status == "error":
            # A software bug in this project's own reward/observation code,
            # not a process failure -- the worker is still alive and
            # replied. Raised here, outside the except block above, so it
            # can never be caught by the respawn path and silently
            # discarded as an elevated respawn count. See "Adjacent bug
            # found during review" in the spec.
            raise RuntimeError(f"env {self._index} worker failed: {payload}")
        return self._to_result(payload)

    def state_dict(self) -> dict:
        """The worker's session state plus the parent-side bookkeeping the
        worker cannot know about. `episode_offset` in particular must survive:
        without it, a resume restarts the visible episode sequence from the
        restored session's own counter and breaks monotonicity across the
        crash."""
        return {
            "session": self._call(Command.STATE_DICT)["state"],
            "respawns": self._respawns,
            "episode_offset": self._episode_offset,
            "last_episode_id": self._last_episode_id,
        }

    def load_state_dict(self, state: dict) -> None:
        self._respawns = state["respawns"]
        self._episode_offset = state["episode_offset"]
        self._last_episode_id = state["last_episode_id"]
        self._call(Command.LOAD_STATE, state["session"])

    def stats(self) -> dict:
        """One round trip per update, not per step. The payload is ~1.3 KB per
        env, against the 168 KB a STATE_DICT round trip ships to extract the
        same coordinates."""
        return self._call(Command.STATS)["stats"]

    def close(self) -> None:
        try:
            self._conn.send((Command.CLOSE, None))
        except (BrokenPipeError, OSError):  # obs: allow LOG007 -- worker already gone at shutdown
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
        self._close_connection()
```

- [ ] **Step 6: Run all subprocess_backend tests to verify they pass**

Run: `pytest tests/unit/test_pokemon_env_subprocess_backend.py -v`
Expected: PASS (all tests, including the 5 new ones from Step 3)

- [ ] **Step 7: Verify each new test can actually fail**

Do these one at a time, confirming red then reverting before moving to the next:
- Comment out the `except (BrokenPipeError, OSError): self._send_failed = True` handling in `_dispatch` (let it propagate instead) — `test_send_step_swallows_a_broken_pipe_and_recv_respawns` should now raise `BrokenPipeError` out of `backend.send_step(0)` instead of passing.
- Move the `if status == "error": raise RuntimeError(...)` check to inside the `try` block (so it's caught by the surrounding `except (TimeoutError, EOFError, BrokenPipeError)`, which it won't match, but simulate the real regression by additionally adding `RuntimeError` to that except tuple) — `test_recv_reraises_an_explicit_worker_error_without_respawning`'s `assert backend.respawns == 0` should now fail (respawns becomes 1).
- Remove the `if self._pending_is_step is None: raise RuntimeError(...)` guard in `recv()` — `test_recv_without_a_prior_send_raises` should now fail (raises a different exception, e.g. `TypeError` from `is_step = self._pending_is_step` being `None` used downstream, or blocks — confirm it no longer raises the expected `RuntimeError`).
- Remove the `if self._pending_is_step is not None: raise RuntimeError(...)` guard in `_dispatch` — `test_send_step_while_a_previous_dispatch_is_unread_raises` should now fail (no exception raised).

Restore all four checks and confirm the full file passes again.

- [ ] **Step 8: Run the full unit suite and the pytest-expert audit**

Run: `pytest tests/unit/ -v` — expect all passing, no regressions in unrelated modules.
Run: `python ~/.claude/skills/pytest-expert/scripts/audit_tests.py tests/unit/test_pokemon_env_subprocess_backend.py tests/unit/test_pokemon_env_vec_env.py tests/unit/fakes.py` — resolve any findings before committing (expect none, given the tests above already follow the one-behavior/exact-value/named-exception conventions).

- [ ] **Step 9: Commit**

```bash
git add src/pokemon_env/subprocess_backend.py tests/unit/test_pokemon_env_subprocess_backend.py
git commit -m "fix(pokemon_env): overlap subprocess worker dispatch, stop swallowing worker bugs as respawns"
```

---

## Task 4: Gate 2 before/after measurement (manual, outside automated tests)

**Files:** none changed — this task records a measurement, per the spec's Testing section: "No unit test can prove the wall-clock win... Rerun Gate 2's `pokemon-ppo preflight --n-envs 16 32 64` before and after this change and record both numbers in the run log."

This cannot run in this dev sandbox (Apple Silicon, no CUDA, and the ROM is gitignored per CLAUDE.md's "Never commit" list) — it requires the real RunPod GPU pod with the ROM present. It is listed here so the plan carries the instruction through to whoever runs it, rather than leaving it as an unrecorded follow-up.

- [ ] **Step 1: Record the "before" number, on `main` at the commit immediately preceding this plan's Task 1 commit**

On the pod, with the ROM and `artifacts/init.state` present:

```bash
git checkout <commit-before-task-1>
uv run pokemon-ppo preflight --n-envs 16 32 64 | tee /tmp/gate2_before.log
```

Record the reported per-`n_envs` rollout throughput (steps/sec or seconds/1024-step-rollout, whichever the CLI reports) for each of 16/32/64.

- [ ] **Step 2: Record the "after" number, on the branch with Task 1-3's commits applied**

```bash
git checkout <this-plan's-branch>
uv run pokemon-ppo preflight --n-envs 16 32 64 | tee /tmp/gate2_after.log
```

- [ ] **Step 3: Compare and record in the run log**

Diff the two logs. The spec's projection is ~64x on the dispatch-loop component alone (not necessarily on total rollout wall-clock, since pickling/IPC overhead the env design spec already costed is additive). Record the actual before/after numbers and the resulting speedup factor in the project's run log (wherever `docs/superpowers/plans/` or a dated `docs/` note tracks pod run results in this repo — follow the existing convention rather than inventing a new location). If the speedup is far short of the projection, the spec's "Escalation path" section (a `multiprocessing.connection.wait()`-based gather loop) is the next step — do not implement it speculatively; only reach for it if this measurement shows the simple split still falls short of the rollout budget.

---

## Self-review notes (for whoever runs this plan)

- **Spec coverage:** `EnvBackend` Protocol change (Task 1), `InProcessBackend` (Task 1), `VecPokemonEnv.step()`/`.reset()` two-phase dispatch + ordering proof (Task 2), `SubprocessBackend` two-phase dispatch (Task 3), the adjacent error-vs-process-failure bug fix (Task 3, folded in per the spec's own instruction to fix it here rather than as a separate change), respawn-stays-per-backend semantics preserved (Task 3, every existing respawn test ported 1:1), Gate 2 remeasurement (Task 4). `state_dict`/`load_state_dict`/`stats`/`close` are explicitly untouched per the spec's "No change" bullet — verified no task modifies them beyond the doc comment on `_call`. The `wait()`-based escalation path is explicitly a non-goal for this plan (spec: "not proposed as the first implementation").
- **Type/name consistency:** `send_reset() -> None`, `send_step(action: int) -> None`, `recv() -> StepResult` are the exact names used in Task 1 (`InProcessBackend`), Task 2 (`VecPokemonEnv`, `FakeBackend`, `RecordingBackend`), and Task 3 (`SubprocessBackend`) — checked against each other during drafting, not just against the spec.
- **No placeholders:** every step above has runnable code or an exact shell command; no "add appropriate error handling"-style step exists in this plan.
