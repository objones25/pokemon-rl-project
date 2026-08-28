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
from typing import Protocol, Self

logger = logging.getLogger(__name__)


class ExperimentRunLike(Protocol):
    def log(self, metrics: dict) -> None: ...
    def summary(self, metrics: dict) -> None: ...
    def finish(self) -> None: ...
    def __enter__(self) -> Self: ...
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
        except Exception:  # must swallow any wandb failure, whatever its type
            logger.warning("wandb_log_failed", exc_info=True)

    def summary(self, metrics: dict) -> None:
        """Run-level bests, written explicitly. Without this the dashboard's
        summary column holds whatever the LAST update happened to log, which
        for a 48-hour run is the least interesting value in the history."""
        try:
            self._run.summary.update(metrics)
        except Exception:  # must swallow any wandb failure, whatever its type
            logger.warning("wandb_summary_failed", exc_info=True)

    def finish(self, exit_code: int = 0) -> None:
        try:
            self._run.finish(exit_code=exit_code)
        except Exception:  # must swallow any wandb failure, whatever its type
            logger.warning("wandb_finish_failed", exc_info=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(exit_code=1 if exc_type is not None else 0)


class NullExperimentRun:
    """No-op ExperimentRunLike, used as *Deps.wandb_run's default so
    callers never need to special-case "no experiment tracker configured"."""

    def log(self, metrics: dict) -> None:
        pass

    def summary(self, metrics: dict) -> None:
        pass

    def finish(self, exit_code: int = 0) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass
