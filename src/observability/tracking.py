"""Thin wrapper around an injected `trackio` module, for testability."""

from __future__ import annotations


class TrackioRun:
    def __init__(self, trackio_module, project: str, name: str) -> None:
        self._trackio = trackio_module
        self._trackio.init(project=project, name=name)

    def log(self, metrics: dict) -> None:
        self._trackio.log(metrics)

    def finish(self) -> None:
        self._trackio.finish()
