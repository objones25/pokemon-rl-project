import logging
from concurrent.futures import ThreadPoolExecutor

from observability.tracking import NullExperimentRun, WandbRun


class FakeRun:
    """Stands in for the `Run` object wandb.init() returns. Its log/finish
    are plain instance methods -- no module-level or thread-local state --
    matching how the real wandb.Run works and why routing through the
    returned instance (not the wandb module's free functions) avoids the
    class of bug documented in the module docstring."""

    def __init__(self) -> None:
        self.log_calls: list[dict] = []
        self.finished = False

    def log(self, metrics: dict) -> None:
        self.log_calls.append(metrics)

    def finish(self) -> None:
        self.finished = True


class FakeWandbModule:
    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.run = FakeRun()

    def init(self, project: str, name: str):
        self.init_calls.append({"project": project, "name": name})
        return self.run

    def log(self, metrics: dict) -> None:
        raise AssertionError(
            "WandbRun must call log()/finish() on the Run object returned "
            "by init(), not on the wandb module -- routing through the "
            "module is exactly the bug that broke concurrent logging under "
            "trackio (see module docstring)."
        )

    def finish(self) -> None:
        raise AssertionError(
            "WandbRun must call log()/finish() on the Run object returned "
            "by init(), not on the wandb module."
        )


class _RaisingRun:
    """Simulates a tracking SDK's Run object failing at the log/finish
    level after some internal state gets invalidated."""

    def log(self, metrics: dict) -> None:
        raise RuntimeError("simulated wandb Run.log() failure")

    def finish(self) -> None:
        raise RuntimeError("simulated wandb Run.finish() failure")


class _RaisingWandbModule:
    def init(self, project: str, name: str):
        return _RaisingRun()


def test_wandb_run_forwards_calls_to_returned_run_object() -> None:
    fake = FakeWandbModule()
    run = WandbRun(fake, project="pokemon-data-collection", name="run-1")

    run.log({"frames_per_sec": 12.5})
    run.finish()

    assert fake.init_calls == [{"project": "pokemon-data-collection", "name": "run-1"}]
    assert fake.run.log_calls == [{"frames_per_sec": 12.5}]
    assert fake.run.finished is True


def test_wandb_run_survives_concurrent_worker_threads() -> None:
    """Regression test carried over from the trackio implementation:
    routing through the Run object init() returns must work correctly
    from a ThreadPoolExecutor worker, since data_collection.pipeline logs
    from concurrent per-video worker threads."""
    fake = FakeWandbModule()
    run = WandbRun(fake, project="p", name="r")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda i: run.log({"i": i}), range(32)))

    assert len(fake.run.log_calls) == 32


def test_wandb_run_log_failure_is_swallowed_not_raised(caplog) -> None:
    """W&B is a best-effort dashboard, not a correctness dependency -- a
    failure inside it must never propagate up and fail real pipeline work
    (see pipeline.py's checkpointing, which must survive this)."""
    run = WandbRun(_RaisingWandbModule(), project="p", name="r")

    with caplog.at_level(logging.WARNING, logger="observability.tracking"):
        run.log({"sampled": 200})  # must not raise

    assert any(r.message == "wandb_log_failed" for r in caplog.records)


def test_wandb_run_finish_failure_is_swallowed_not_raised(caplog) -> None:
    run = WandbRun(_RaisingWandbModule(), project="p", name="r")

    with caplog.at_level(logging.WARNING, logger="observability.tracking"):
        run.finish()  # must not raise

    assert any(r.message == "wandb_finish_failed" for r in caplog.records)


def test_null_experiment_run_is_a_no_op() -> None:
    run = NullExperimentRun()

    run.log({"anything": 1})  # must not raise
    run.finish()  # must not raise
