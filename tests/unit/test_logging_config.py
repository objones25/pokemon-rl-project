import io
import json
import logging

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


def test_configure_logging_writes_json_to_the_given_stream() -> None:
    stream = io.StringIO()
    configure_logging(stream=stream)

    logging.getLogger("observability.test.configure").info("hello", extra={"x": 1})

    stream.seek(0)
    lines = [line for line in stream.read().splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["message"] == "hello"
    assert payload["x"] == 1
