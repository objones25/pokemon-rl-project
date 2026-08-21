from click.testing import CliRunner

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
