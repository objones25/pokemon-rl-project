import logging
from concurrent.futures import ThreadPoolExecutor

from observability.tracking import NullTrackioRun, TrackioRun


class FakeRun:
    """Stands in for the `Run` object trackio.init() returns. Its log/finish
    are plain instance methods -- no module-level or thread-local state --
    matching how the real trackio.Run works and why routing through the
    returned instance (not the trackio module's free functions) fixes
    concurrent multi-thread logging (see module docstring)."""

    def __init__(self) -> None:
        self.log_calls: list[dict] = []
        self.finished = False

    def log(self, metrics: dict) -> None:
        self.log_calls.append(metrics)

    def finish(self) -> None:
        self.finished = True


class FakeTrackioModule:
    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.run = FakeRun()

    def init(self, project: str, name: str):
        self.init_calls.append({"project": project, "name": name})
        return self.run

    def log(self, metrics: dict) -> None:
        raise AssertionError(
            "TrackioRun must call log()/finish() on the Run object returned "
            "by init(), not on the trackio module -- routing through the "
            "module is exactly the bug that broke concurrent logging."
        )

    def finish(self) -> None:
        raise AssertionError(
            "TrackioRun must call log()/finish() on the Run object returned "
            "by init(), not on the trackio module."
        )


class _RaisingRun:
    """Simulates trackio's observed failure mode at the Run level: log()/
    finish() start raising after some internal state gets invalidated."""

    def log(self, metrics: dict) -> None:
        raise RuntimeError("Call trackio.init() before trackio.log().")

    def finish(self) -> None:
        raise RuntimeError("Call trackio.init() before trackio.finish().")


class _RaisingTrackioModule:
    def init(self, project: str, name: str):
        return _RaisingRun()


def test_trackio_run_forwards_calls_to_returned_run_object() -> None:
    fake = FakeTrackioModule()
    run = TrackioRun(fake, project="pokemon-data-collection", name="run-1")

    run.log({"frames_per_sec": 12.5})
    run.finish()

    assert fake.init_calls == [{"project": "pokemon-data-collection", "name": "run-1"}]
    assert fake.run.log_calls == [{"frames_per_sec": 12.5}]
    assert fake.run.finished is True


def test_trackio_run_survives_concurrent_worker_threads() -> None:
    """Root-cause regression test: trackio's module-level log()/finish()
    read a contextvars.ContextVar that is NOT inherited by new OS threads,
    so calling through the module from a ThreadPoolExecutor worker always
    raised "Call trackio.init() before trackio.log().". Routing through the
    Run object init() returns sidesteps that entirely, since Run.log/finish
    are plain instance methods with no thread-local state."""
    fake = FakeTrackioModule()
    run = TrackioRun(fake, project="p", name="r")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda i: run.log({"i": i}), range(32)))

    assert len(fake.run.log_calls) == 32


def test_trackio_run_log_failure_is_swallowed_not_raised(caplog) -> None:
    """Trackio is a best-effort dashboard, not a correctness dependency --
    a failure inside it must never propagate up and fail real pipeline
    work (see pipeline.py's checkpointing, which must survive this)."""
    run = TrackioRun(_RaisingTrackioModule(), project="p", name="r")

    with caplog.at_level(logging.WARNING, logger="observability.tracking"):
        run.log({"sampled": 200})  # must not raise

    assert any(r.message == "trackio_log_failed" for r in caplog.records)


def test_trackio_run_finish_failure_is_swallowed_not_raised(caplog) -> None:
    run = TrackioRun(_RaisingTrackioModule(), project="p", name="r")

    with caplog.at_level(logging.WARNING, logger="observability.tracking"):
        run.finish()  # must not raise

    assert any(r.message == "trackio_finish_failed" for r in caplog.records)


def test_null_trackio_run_is_a_no_op() -> None:
    run = NullTrackioRun()

    run.log({"anything": 1})  # must not raise
    run.finish()  # must not raise
