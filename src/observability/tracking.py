"""Thin wrapper around an injected `trackio` module, for testability.

Trackio is a best-effort live dashboard, not a correctness dependency --
the pipeline's JSON-lines logs and manifest checkpoints are the source of
truth. `log`/`finish` swallow any exception from the underlying module
rather than propagate: an internal trackio failure (e.g. its global
"current run" state getting invalidated under concurrent access from
multiple pipeline worker threads) must never fail real extraction work.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TrackioRun:
    def __init__(self, trackio_module, project: str, name: str) -> None:
        self._trackio = trackio_module
        self._trackio.init(project=project, name=name)

    def log(self, metrics: dict) -> None:
        try:
            self._trackio.log(metrics)
        except Exception as exc:
            logger.warning("trackio_log_failed", extra={"reason": str(exc)})

    def finish(self) -> None:
        try:
            self._trackio.finish()
        except Exception as exc:
            logger.warning("trackio_finish_failed", extra={"reason": str(exc)})
