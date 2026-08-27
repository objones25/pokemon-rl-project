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


def test_run_command_fails_fast_with_no_wandb_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("data_collection.cli.get_token", lambda: "fake-token")
    # main()'s own load_dotenv() call would otherwise repopulate WANDB_API_KEY
    # from a real .env file on the machine running this test, silently
    # defeating the "missing credentials" scenario and -- worse -- letting a
    # real wandb.init() fire against a real account. Neutralize it so this
    # test's outcome depends only on the deleted env var, not on what's in
    # some developer's local .env.
    monkeypatch.setattr("data_collection.cli.load_dotenv", lambda: None)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    registry_path = tmp_path / "video_sources.yaml"
    registry_path.write_text("videos: []\n")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--repo-id", "me/pokemon-frames", "--registry", str(registry_path)]
    )

    assert result.exit_code != 0
    assert "WANDB_API_KEY" in result.output


class _FakeWandbModule:
    """`cli.run` constructs a real WandbRun before calling `pipeline.run_pipeline`,
    so this test must fake the `wandb` module itself, not just `run_pipeline`,
    or it makes a real call into the wandb library on every test run."""

    def init(self, **kwargs) -> None:
        pass

    def log(self, metrics: dict) -> None:
        pass

    def finish(self) -> None:
        pass


def test_run_command_exits_nonzero_when_pipeline_reports_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("data_collection.cli.get_token", lambda: "fake-token")
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setattr("data_collection.cli.wandb", _FakeWandbModule())
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


def test_run_command_exits_zero_when_pipeline_reports_no_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("data_collection.cli.get_token", lambda: "fake-token")
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setattr("data_collection.cli.wandb", _FakeWandbModule())
    monkeypatch.setattr(
        "data_collection.cli.pipeline.run_pipeline",
        lambda sources, deps: pipeline.PipelineResult(completed=0, failed=0),
    )
    registry_path = tmp_path / "video_sources.yaml"
    registry_path.write_text("videos: []\n")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--repo-id", "me/pokemon-frames", "--registry", str(registry_path)]
    )

    assert result.exit_code == 0, result.output


def test_run_command_wires_batch_and_checkpoint_options_onto_deps(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("data_collection.cli.get_token", lambda: "fake-token")
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setattr("data_collection.cli.wandb", _FakeWandbModule())
    captured = {}
    monkeypatch.setattr(
        "data_collection.cli.pipeline.run_pipeline",
        lambda sources, deps: captured.update(deps=deps) or pipeline.PipelineResult(completed=0, failed=0),
    )
    registry_path = tmp_path / "video_sources.yaml"
    registry_path.write_text("videos: []\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--repo-id", "me/pokemon-frames",
            "--registry", str(registry_path),
            "--batch-size", "50",
            "--checkpoint-interval", "999",
            "--max-concurrent-videos", "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["deps"].batch_size == 50
    assert captured["deps"].checkpoint_interval_samples == 999
    assert captured["deps"].max_concurrent_videos == 3


def test_curate_command_delegates_to_run_curation(tmp_path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "data_collection.cli.curation.run_curation",
        lambda **kwargs: captured.update(kwargs),
    )
    registry_path = tmp_path / "video_sources.yaml"
    approved_dir = tmp_path / "approved"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "curate", "https://youtube.com/watch?v=abc123",
            "--game", "red",
            "--registry", str(registry_path),
            "--approved-dir", str(approved_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["video_url"] == "https://youtube.com/watch?v=abc123"
    assert captured["game"] == "red"
    assert captured["registry_path"] == registry_path
    assert captured["approved_dir"] == approved_dir
