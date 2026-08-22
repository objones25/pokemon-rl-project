from click.testing import CliRunner

from data_collection import pipeline
from data_collection.cli import main


def test_cli_help_lists_curate_and_run_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "curate" in result.output
    assert "run" in result.output


def test_curate_command_requires_url_argument() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["curate"])

    assert result.exit_code != 0


def test_run_command_requires_repo_id_option() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["run"])

    assert result.exit_code != 0


def test_run_command_fails_fast_with_no_hf_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("data_collection.cli.get_token", lambda: None)
    registry_path = tmp_path / "video_sources.yaml"
    registry_path.write_text("videos: []\n")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--repo-id", "me/pokemon-frames", "--registry", str(registry_path)]
    )

    assert result.exit_code != 0
    assert "HF_TOKEN" in result.output


class _FakeTrackioModule:
    """`cli.run` constructs a real TrackioRun before calling `pipeline.run_pipeline`,
    so this test must fake the `trackio` module itself, not just `run_pipeline`,
    or it makes a real call into the trackio library on every test run."""

    def init(self, project: str, name: str) -> None:
        pass

    def log(self, metrics: dict) -> None:
        pass

    def finish(self) -> None:
        pass


def test_run_command_exits_nonzero_when_pipeline_reports_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("data_collection.cli.get_token", lambda: "fake-token")
    monkeypatch.setattr("data_collection.cli.trackio", _FakeTrackioModule())
    monkeypatch.setattr(
        "data_collection.cli.pipeline.run_pipeline",
        lambda sources, deps: pipeline.PipelineResult(completed=0, failed=1),
    )
    registry_path = tmp_path / "video_sources.yaml"
    registry_path.write_text("videos: []\n")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--repo-id", "me/pokemon-frames", "--registry", str(registry_path)]
    )

    assert result.exit_code != 0
