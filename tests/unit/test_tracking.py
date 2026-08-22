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


def test_trackio_run_forwards_calls() -> None:
    fake = FakeTrackioModule()
    run = TrackioRun(fake, project="pokemon-data-collection", name="run-1")

    run.log({"frames_per_sec": 12.5})
    run.finish()

    assert fake.init_calls == [{"project": "pokemon-data-collection", "name": "run-1"}]
    assert fake.log_calls == [{"frames_per_sec": 12.5}]
    assert fake.finished is True
