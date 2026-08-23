"""Thin wrapper around an injected `trackio` module, for testability.

Trackio is a best-effort live dashboard, not a correctness dependency --
the pipeline's JSON-lines logs and manifest checkpoints are the source of
truth. `log`/`finish` swallow any exception from the underlying module
rather than propagate: an internal trackio failure (e.g. its live-server
sync going down) must never fail real extraction work.

Root cause of a previously-unfixed bug: every `trackio.log()` call made
from a `ThreadPoolExecutor` worker thread failed with "Call trackio.init()
before trackio.log().", even right after a successful `trackio.init()`.
trackio's module-level `init`/`log`/`finish` functions track the active
run via a `contextvars.ContextVar` (`trackio.context_vars.current_run`),
and contextvars state set on one thread is NOT visible to a new OS thread
-- a `ThreadPoolExecutor` worker starts with a fresh, empty context, so it
always saw no current run. `trackio.init()` also returns the `Run`
instance it created, and that instance's own `log`/`finish` methods
operate on `self` with no thread-local state, so routing through the
returned instance (as `TrackioRun` does below) sidesteps the problem
entirely instead of depending on trackio's global/thread-local state.

`TrackioRunLike` implementations must never raise -- callers (e.g.
data_collection.pipeline) are not required to guard calls against
failures from this interface; `TrackioRun` and `NullTrackioRun` below
are expected to enforce that themselves.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class TrackioRunLike(Protocol):
    def log(self, metrics: dict) -> None: ...
    def finish(self) -> None: ...


class TrackioRun:
    def __init__(self, trackio_module, project: str, name: str) -> None:
        self._run = trackio_module.init(project=project, name=name)

    def log(self, metrics: dict) -> None:
        try:
            self._run.log(metrics)
        except Exception as exc:  # noqa: BLE001 -- must swallow any trackio failure, whatever its type
            logger.warning("trackio_log_failed", extra={"reason": str(exc)})

    def finish(self) -> None:
        try:
            self._run.finish()
        except Exception as exc:  # noqa: BLE001 -- must swallow any trackio failure, whatever its type
            logger.warning("trackio_finish_failed", extra={"reason": str(exc)})


class NullTrackioRun:
    """No-op TrackioRunLike, used as PipelineDeps.trackio_run's default so
    callers never need to special-case "no trackio configured"."""

    def log(self, metrics: dict) -> None:
        pass

    def finish(self) -> None:
        pass
