import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from observability.tracking import NullExperimentRun, WandbRun


class FakeWandbRun:
    """Hand-written fake for the Run object wandb.init() returns."""

    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.defined: list[tuple[str, str]] = []
        self.finished_with: object = "not-finished"
        self.id = "fake-run-id"
        # wandb's Run.summary is a dict-like whose .update() writes the
        # run-level summary; a plain dict matches that surface exactly.
        self.summary: dict = {}

    def log(self, metrics: dict) -> None:
        self.logged.append(metrics)

    def define_metric(self, name: str, step_metric: str) -> None:
        self.defined.append((name, step_metric))

    def finish(self, exit_code: int = 0) -> None:
        self.finished_with = exit_code


class FakeWandbModule:
    def __init__(self) -> None:
        self.run = FakeWandbRun()
        self.init_kwargs: dict = {}

    def init(self, **kwargs) -> FakeWandbRun:
        self.init_kwargs = kwargs
        return self.run


class _WandbModuleExposingTopLevelLogAndFinish(FakeWandbModule):
    """Regression fixture: would incorrectly satisfy a WandbRun that (bug)
    called log()/finish() on the wandb module itself rather than on the Run
    object init() returns -- routing through the module is exactly the bug
    that broke concurrent logging under trackio (see module docstring)."""

    def log(self, metrics: dict) -> None:
        raise AssertionError(
            "WandbRun must call log()/finish() on the Run object returned "
            "by init(), not on the wandb module."
        )

    def finish(self, exit_code: int = 0) -> None:
        raise AssertionError(
            "WandbRun must call log()/finish() on the Run object returned "
            "by init(), not on the wandb module."
        )


class _RaisingRun:
    """Simulates a tracking SDK's Run object failing at the log/finish
    level after some internal state gets invalidated."""

    @property
    def summary(self) -> dict:
        raise RuntimeError("simulated wandb Run.summary failure")

    def log(self, metrics: dict) -> None:
        raise RuntimeError("simulated wandb Run.log() failure")

    def finish(self, exit_code: int = 0) -> None:
        raise RuntimeError("simulated wandb Run.finish() failure")


class _RaisingWandbModule:
    def init(self, **kwargs) -> _RaisingRun:
        return _RaisingRun()


def test_wandb_run_forwards_calls_to_returned_run_object() -> None:
    fake = FakeWandbModule()
    run = WandbRun(fake, project="pokemon-data-collection", name="run-1")

    run.log({"frames_per_sec": 12.5})
    run.finish()

    assert fake.init_kwargs["project"] == "pokemon-data-collection"
    assert fake.init_kwargs["name"] == "run-1"
    assert fake.run.logged == [{"frames_per_sec": 12.5}]
    assert fake.run.finished_with == 0


def test_wandb_run_routes_log_and_finish_through_the_run_object_not_the_module() -> None:
    fake = _WandbModuleExposingTopLevelLogAndFinish()
    run = WandbRun(fake, project="p", name="r")

    run.log({"frames_per_sec": 12.5})
    run.finish()

    assert fake.run.logged == [{"frames_per_sec": 12.5}]
    assert fake.run.finished_with == 0


def test_wandb_run_survives_concurrent_worker_threads() -> None:
    """Regression test carried over from the trackio implementation:
    routing through the Run object init() returns must work correctly
    from a ThreadPoolExecutor worker, since data_collection.pipeline logs
    from concurrent per-video worker threads."""
    fake = FakeWandbModule()
    run = WandbRun(fake, project="p", name="r")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda i: run.log({"i": i}), range(32)))

    assert len(fake.run.logged) == 32


def test_wandb_run_log_failure_is_swallowed_not_raised(caplog) -> None:
    """W&B is a best-effort dashboard, not a correctness dependency -- a
    failure inside it must never propagate up and fail real pipeline work
    (see pipeline.py's checkpointing, which must survive this)."""
    run = WandbRun(_RaisingWandbModule(), project="p", name="r")

    with caplog.at_level(logging.WARNING, logger="observability.tracking"):
        run.log({"sampled": 200})  # must not raise

    assert any(r.message == "wandb_log_failed" for r in caplog.records)


def test_wandb_run_summary_writes_through_the_run_objects_summary_dict() -> None:
    """Run-level bests must reach `Run.summary`, not the history: logged as an
    ordinary metric they would be overwritten by the next update, and the
    dashboard's summary column would show the LAST value of a 48-hour run
    rather than its best."""
    fake = FakeWandbModule()
    run = WandbRun(fake, project="p", name="r")

    run.summary({"best/badges": 3.0})

    assert fake.run.summary == {"best/badges": 3.0}


def test_wandb_run_summary_failure_is_swallowed_not_raised(caplog) -> None:
    run = WandbRun(_RaisingWandbModule(), project="p", name="r")

    with caplog.at_level(logging.WARNING, logger="observability.tracking"):
        run.summary({"best/badges": 3.0})  # must not raise

    assert any(r.message == "wandb_summary_failed" for r in caplog.records)


def test_wandb_run_finish_failure_is_swallowed_not_raised(caplog) -> None:
    run = WandbRun(_RaisingWandbModule(), project="p", name="r")

    with caplog.at_level(logging.WARNING, logger="observability.tracking"):
        run.finish()  # must not raise

    assert any(r.message == "wandb_finish_failed" for r in caplog.records)


def test_null_experiment_run_is_a_no_op() -> None:
    run = NullExperimentRun()

    assert run.log({"anything": 1}) is None
    assert run.summary({"anything": 1}) is None
    assert run.finish() is None


def test_the_config_reaches_wandb_init() -> None:
    module = FakeWandbModule()

    WandbRun(module, project="p", name="n", config={"lr": 0.1})

    assert module.init_kwargs["config"] == {"lr": 0.1}


def test_a_run_id_is_passed_with_resume_allow_so_a_preempted_run_continues() -> None:
    module = FakeWandbModule()

    WandbRun(module, project="p", name="n", run_id="abc")

    assert (module.init_kwargs["id"], module.init_kwargs["resume"]) == ("abc", "allow")


def test_run_id_returns_the_underlying_run_objects_id() -> None:
    module = FakeWandbModule()

    run = WandbRun(module, project="p", name="n")

    assert run.run_id == "fake-run-id"


def test_no_resume_arguments_are_sent_when_no_run_id_is_given() -> None:
    module = FakeWandbModule()

    WandbRun(module, project="p", name="n")

    assert "resume" not in module.init_kwargs


def test_step_metrics_are_declared_as_x_axes() -> None:
    module = FakeWandbModule()

    WandbRun(module, project="p", name="n", step_metrics={"loss/*": "train/update"})

    assert module.run.defined == [("loss/*", "train/update")]


def test_log_never_passes_a_step_argument() -> None:
    """wandb drops a log whose step is below the current one, with no
    exception. The x-axis is declared instead."""
    module = FakeWandbModule()
    run = WandbRun(module, project="p", name="n")

    run.log({"loss": 1.0, "train/update": 3})

    assert module.run.logged == [{"loss": 1.0, "train/update": 3}]


def test_leaving_the_context_normally_finishes_with_exit_code_zero() -> None:
    module = FakeWandbModule()

    with WandbRun(module, project="p", name="n"):
        pass

    assert module.run.finished_with == 0


def test_an_exception_inside_the_context_marks_the_run_failed() -> None:
    module = FakeWandbModule()

    with pytest.raises(RuntimeError, match="boom"), WandbRun(module, project="p", name="n"):
        raise RuntimeError("boom")

    assert module.run.finished_with == 1


def test_null_experiment_run_is_usable_as_a_context_manager() -> None:
    """*Deps default to NullExperimentRun, so every call site that wraps the
    run in a `with` must work without a tracker configured."""
    with NullExperimentRun() as run:
        run.log({"a": 1.0})

    assert isinstance(run, NullExperimentRun)
