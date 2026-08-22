import logging

from observability.tracking import TrackioRun


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


class _RaisingOnceTrackioModule(FakeTrackioModule):
    """Simulates trackio's real, observed failure mode: log() starts
    raising after some internal state gets invalidated (concurrent
    threads sharing one global 'current run', a background sync thread,
    etc.) -- root cause we don't control from this side."""

    def log(self, metrics: dict) -> None:
        raise RuntimeError("Call trackio.init() before trackio.log().")

    def finish(self) -> None:
        raise RuntimeError("Call trackio.init() before trackio.finish().")


def test_trackio_run_forwards_calls() -> None:
    fake = FakeTrackioModule()
    run = TrackioRun(fake, project="pokemon-data-collection", name="run-1")

    run.log({"frames_per_sec": 12.5})
    run.finish()

    assert fake.init_calls == [{"project": "pokemon-data-collection", "name": "run-1"}]
    assert fake.log_calls == [{"frames_per_sec": 12.5}]
    assert fake.finished is True


def test_trackio_run_log_failure_is_swallowed_not_raised(caplog) -> None:
    """Trackio is a best-effort dashboard, not a correctness dependency --
    a failure inside it must never propagate up and fail real pipeline
    work (see pipeline.py's checkpointing, which must survive this)."""
    run = TrackioRun(_RaisingOnceTrackioModule(), project="p", name="r")

    with caplog.at_level(logging.WARNING, logger="observability.tracking"):
        run.log({"sampled": 200})  # must not raise

    assert any(r.message == "trackio_log_failed" for r in caplog.records)


def test_trackio_run_finish_failure_is_swallowed_not_raised(caplog) -> None:
    run = TrackioRun(_RaisingOnceTrackioModule(), project="p", name="r")

    with caplog.at_level(logging.WARNING, logger="observability.tracking"):
        run.finish()  # must not raise

    assert any(r.message == "trackio_finish_failed" for r in caplog.records)
