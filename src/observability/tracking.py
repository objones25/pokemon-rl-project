"""Thin wrapper around an injected `wandb` module, for testability.

W&B is a best-effort live dashboard, not a correctness dependency -- the
pipeline's JSON-lines logs and manifest checkpoints are the source of
truth. `log`/`finish` swallow any exception from the underlying module
rather than propagate: an internal W&B failure (e.g. its sync going down)
must never fail real extraction/training work.

Migrated from trackio (see git history): `wandb.init()` returns a `Run`
object whose `log`/`finish` are plain instance methods operating on
`self`, not the `wandb` module's own global `wandb.log()`/`wandb.finish()`
functions -- this project deliberately routes through the returned
instance rather than those module-level functions. That is what trackio's
own SDK required to avoid a real bug this project hit: trackio's
module-level `log`/`finish` read a `contextvars.ContextVar` that is not
inherited by a new OS thread, so calling them from a
`ThreadPoolExecutor` worker (data_collection.pipeline processes videos
concurrently) always failed with "Call trackio.init() before
trackio.log().", even right after a successful `init()`. Routing through
the `Run` instance sidesteps that class of bug regardless of whether the
tracking SDK in use has the same internal mechanism.

`ExperimentRunLike` implementations must never raise -- callers (e.g.
data_collection.pipeline) are not required to guard calls against
failures from this interface; `WandbRun` and `NullExperimentRun` below
are expected to enforce that themselves.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class ExperimentRunLike(Protocol):
    def log(self, metrics: dict) -> None: ...
    def finish(self) -> None: ...


class WandbRun:
    def __init__(self, wandb_module, project: str, name: str) -> None:
        self._run = wandb_module.init(project=project, name=name)

    def log(self, metrics: dict) -> None:
        try:
            self._run.log(metrics)
        except Exception as exc:  # noqa: BLE001 -- must swallow any wandb failure, whatever its type
            logger.warning("wandb_log_failed", extra={"reason": str(exc)})

    def finish(self) -> None:
        try:
            self._run.finish()
        except Exception as exc:  # noqa: BLE001 -- must swallow any wandb failure, whatever its type
            logger.warning("wandb_finish_failed", extra={"reason": str(exc)})


class NullExperimentRun:
    """No-op ExperimentRunLike, used as *Deps.wandb_run's default so
    callers never need to special-case "no experiment tracker configured"."""

    def log(self, metrics: dict) -> None:
        pass

    def finish(self) -> None:
        pass
