import io
import json
import logging

import numpy as np

from data_collection.observability import TrackioRun, build_contact_sheet, configure_logging


def test_configure_logging_emits_one_json_object_per_line() -> None:
    stream = io.StringIO()
    logger = configure_logging(stream=stream)

    logger.info("frames_kept", extra={"video_id": "abc123", "count": 42})

    stream.seek(0)
    lines = [line for line in stream.read().splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["message"] == "frames_kept"
    assert payload["video_id"] == "abc123"
    assert payload["count"] == 42
    assert payload["level"] == "INFO"


def test_configure_logging_returns_same_named_logger_on_repeat_calls() -> None:
    logger_a = configure_logging(stream=io.StringIO())
    logger_b = configure_logging(stream=io.StringIO())
    assert logger_a.name == logger_b.name == "data_collection"


class FakeTrackioModule:
    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.log_calls: list[dict] = []
        self.finished = False

    def init(self, project: str, name: str) -> None:
        self.init_calls.append({"project": project, "name": name})

    def log(self, metrics: dict) -> None:
        self.log_calls.append(metrics)

    def finish(self) -> None:
        self.finished = True


def test_trackio_run_forwards_calls() -> None:
    fake = FakeTrackioModule()
    run = TrackioRun(fake, project="pokemon-data-collection", name="run-1")

    run.log({"frames_per_sec": 12.5})
    run.finish()

    assert fake.init_calls == [{"project": "pokemon-data-collection", "name": "run-1"}]
    assert fake.log_calls == [{"frames_per_sec": 12.5}]
    assert fake.finished is True


def test_build_contact_sheet_grid_dimensions() -> None:
    frames = [np.full((144, 160), i, dtype=np.uint8) for i in range(10)]

    sheet = build_contact_sheet(frames, cols=4)

    # 10 frames at 4 cols -> 3 rows (ceil(10/4)), each cell 144x160.
    assert sheet.shape == (144 * 3, 160 * 4)


def test_build_contact_sheet_empty_input_returns_empty_array() -> None:
    sheet = build_contact_sheet([], cols=4)
    assert sheet.size == 0
