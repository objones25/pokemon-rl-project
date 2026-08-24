from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image

from contrastive_pretrain.cli import main
from contrastive_pretrain.config import TrainingConfig


def test_preview_command_writes_contact_sheet(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(2):
        frame = np.full((144, 160), 50 + i, dtype=np.uint8)
        Image.fromarray(frame).save(frames_dir / f"frame_{i}.png")

    out_path = tmp_path / "preview.png"
    runner = CliRunner()
    result = runner.invoke(
        main, ["preview", "--frames-dir", str(frames_dir), "--out", str(out_path)]
    )

    assert result.exit_code == 0
    assert out_path.exists()
    saved = np.array(Image.open(out_path))
    assert saved.shape == (144 * 2, 480)


def test_preview_command_errors_on_empty_directory(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(main, ["preview", "--frames-dir", str(frames_dir)])

    assert result.exit_code != 0
    assert "No .png frames found" in result.output


def test_preview_command_caps_frames_at_limit(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(5):
        frame = np.full((144, 160), 50 + i, dtype=np.uint8)
        Image.fromarray(frame).save(frames_dir / f"frame_{i}.png")

    out_path = tmp_path / "preview.png"
    runner = CliRunner()
    result = runner.invoke(
        main, ["preview", "--frames-dir", str(frames_dir), "--out", str(out_path), "--limit", "2"]
    )

    assert result.exit_code == 0
    saved = np.array(Image.open(out_path))
    assert saved.shape == (144 * 2, 480)


def test_preview_command_errors_on_mismatched_frame_sizes(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    Image.fromarray(np.full((144, 160), 50, dtype=np.uint8)).save(frames_dir / "a.png")
    Image.fromarray(np.full((100, 100), 50, dtype=np.uint8)).save(frames_dir / "b.png")

    runner = CliRunner()
    result = runner.invoke(main, ["preview", "--frames-dir", str(frames_dir)])

    assert result.exit_code != 0
    assert "mismatched shapes" in result.output


class _FakeHfApi:
    def create_repo(self, repo_id: str, repo_type: str, exist_ok: bool, private: bool) -> None:
        pass


class _FakeTrackioModule:
    """`cli.train` constructs a real TrackioRun before calling `run_training`,
    so this test must fake the `trackio` module itself, not just `run_training`,
    or it makes a real call into the trackio library on every test run."""

    def init(self, project: str, name: str) -> None:
        pass

    def log(self, metrics: dict) -> None:
        pass

    def finish(self) -> None:
        pass


def test_cli_help_lists_train_and_export_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "export-frozen-encoder" in result.output


def test_train_command_builds_deps_from_config_and_calls_run_training(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("batch_size: 8\n")
    captured = {}

    monkeypatch.setattr("contrastive_pretrain.cli.run_training", lambda deps: captured.update(deps=deps))
    monkeypatch.setattr("contrastive_pretrain.cli.RealHfClient", lambda *a, **k: object())
    monkeypatch.setattr("contrastive_pretrain.cli.HfApi", lambda: _FakeHfApi())
    monkeypatch.setattr("contrastive_pretrain.cli.get_token", lambda: "fake-token")
    monkeypatch.setattr("contrastive_pretrain.cli.trackio", _FakeTrackioModule())

    runner = CliRunner()
    result = runner.invoke(main, ["train", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert captured["deps"].config.batch_size == 8


def test_train_command_fails_fast_with_no_hf_credentials(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("batch_size: 8\n")
    monkeypatch.setattr("contrastive_pretrain.cli.get_token", lambda: None)

    runner = CliRunner()
    result = runner.invoke(main, ["train", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "HF_TOKEN" in result.output


def test_export_frozen_encoder_command_requires_checkpoint_option() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["export-frozen-encoder"])

    assert result.exit_code != 0
