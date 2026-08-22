"""Root-logger JSON logging, configured once at application startup.

Every module gets its own logger via `logging.getLogger(__name__)` and lets
it propagate; nothing attaches handlers anywhere except here, at root, so
third-party library logs (yt-dlp, huggingface_hub, ...) get the same JSON
formatting as application logs instead of vanishing silently.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import logging.config
from typing import Any, TextIO

_LOG_RECORD_BUILTIN_ATTRS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "taskName", "thread", "threadName",
}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
        }
        if record.exc_info is not None:
            payload["exc_info"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _LOG_RECORD_BUILTIN_ATTRS:
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging(stream: TextIO | None = None) -> None:
    handler: dict[str, Any] = {
        "class": "logging.StreamHandler",
        "formatter": "json",
    }
    handler["stream"] = stream if stream is not None else "ext://sys.stdout"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {"()": "observability.logging_config.JSONFormatter"},
            },
            "handlers": {"stdout": handler},
            "root": {"level": "INFO", "handlers": ["stdout"]},
        }
    )
