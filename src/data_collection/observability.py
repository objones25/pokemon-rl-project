"""Structured logging, a thin Trackio wrapper, and contact-sheet previews."""

from __future__ import annotations

import json
import logging
from typing import TextIO

import numpy as np


class _JsonFormatter(logging.Formatter):
    _RESERVED = set(logging.LogRecord(
        "", 0, "", 0, "", (), None
    ).__dict__.keys()) | {"message"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value
        return json.dumps(payload)


def configure_logging(stream: TextIO | None = None) -> logging.Logger:
    logger = logging.getLogger("data_collection")
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class TrackioRun:
    def __init__(self, trackio_module, project: str, name: str) -> None:
        self._trackio = trackio_module
        self._trackio.init(project=project, name=name)

    def log(self, metrics: dict) -> None:
        self._trackio.log(metrics)

    def finish(self) -> None:
        self._trackio.finish()


def build_contact_sheet(frames: list[np.ndarray], cols: int = 8) -> np.ndarray:
    if not frames:
        return np.empty((0, 0), dtype=np.uint8)

    height, width = frames[0].shape
    rows = -(-len(frames) // cols)  # ceil division
    sheet = np.zeros((height * rows, width * cols), dtype=np.uint8)

    for i, frame in enumerate(frames):
        row, col = divmod(i, cols)
        sheet[row * height : (row + 1) * height, col * width : (col + 1) * width] = frame

    return sheet
