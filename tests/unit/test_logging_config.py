import io
import json
import logging
import sys

from observability.logging_config import JSONFormatter, configure_logging


def test_json_formatter_emits_one_json_object_with_expected_fields(caplog) -> None:
    logger = logging.getLogger("observability.test.formatter")
    with caplog.at_level(logging.INFO, logger="observability.test.formatter"):
        logger.info("frames_kept", extra={"video_id": "abc123", "count": 42})

    payload = json.loads(JSONFormatter().format(caplog.records[0]))

    assert payload["message"] == "frames_kept"
    assert payload["level"] == "INFO"
    assert payload["video_id"] == "abc123"
    assert payload["count"] == 42
    assert "timestamp" in payload


def test_json_formatter_stringifies_non_serializable_extra_values(caplog) -> None:
    logger = logging.getLogger("observability.test.formatter")
    with caplog.at_level(logging.INFO, logger="observability.test.formatter"):
        logger.info("weird", extra={"thing": object()})

    payload = json.loads(JSONFormatter().format(caplog.records[0]))

    assert isinstance(payload["thing"], str)


def test_json_formatter_includes_exc_info_when_present(caplog) -> None:
    logger = logging.getLogger("test_logging_config_exc_info")
    formatter = JSONFormatter()

    try:
        raise ValueError("boom")
    except ValueError:
        record = logger.makeRecord(
            logger.name, logging.ERROR, __file__, 0, "failed", (), sys.exc_info()
        )

    payload = json.loads(formatter.format(record))

    assert "exc_info" in payload
    assert "ValueError: boom" in payload["exc_info"]


def test_configure_logging_writes_json_to_the_given_stream() -> None:
    original_handlers = list(logging.root.handlers)
    original_level = logging.root.level
    try:
        stream = io.StringIO()
        configure_logging(stream=stream)

        logging.getLogger("observability.test.configure").info("hello", extra={"x": 1})

        stream.seek(0)
        lines = [line for line in stream.read().splitlines() if line.strip()]
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["message"] == "hello"
        assert payload["x"] == 1
    finally:
        logging.root.handlers = original_handlers
        logging.root.setLevel(original_level)
