from pathlib import Path

import numpy as np
import torch
from click.testing import CliRunner
from PIL import Image

from contrastive_pretrain.checkpoint import save_checkpoint
from contrastive_pretrain.cli import main
from contrastive_pretrain.model import build_encoder


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


def test_preview_command_with_limit_one_selects_a_single_frame(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(5):
        frame = np.full((144, 160), 50 + i, dtype=np.uint8)
        Image.fromarray(frame).save(frames_dir / f"frame_{i}.png")

    out_path = tmp_path / "preview.png"
    runner = CliRunner()
    result = runner.invoke(
        main, ["preview", "--frames-dir", str(frames_dir), "--out", str(out_path), "--limit", "1"]
    )

    assert result.exit_code == 0, result.output
    saved = np.array(Image.open(out_path))
    assert saved.shape == (144, 480)  # one row, one (original|view_a|view_b) triple wide


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


class _FakeWandbModule:
    """`cli.train` constructs a real WandbRun before calling `run_training`,
    so this test must fake the `wandb` module itself, not just `run_training`,
    or it makes a real call into the wandb library on every test run."""

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
    monkeypatch.setattr("contrastive_pretrain.cli.RealHfClient", lambda *_a, **_k: "the-frozen-encoder-client")
    monkeypatch.setattr("contrastive_pretrain.cli.HfApi", lambda: _FakeHfApi())
    monkeypatch.setattr("contrastive_pretrain.cli.get_token", lambda: "fake-token")
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setattr("contrastive_pretrain.cli.wandb", _FakeWandbModule())

    runner = CliRunner()
    result = runner.invoke(main, ["train", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert captured["deps"].config.batch_size == 8
    assert captured["deps"].frozen_encoder_client == "the-frozen-encoder-client"


def test_train_command_fails_fast_with_no_hf_credentials(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("batch_size: 8\n")
    monkeypatch.setattr("contrastive_pretrain.cli.get_token", lambda: None)

    runner = CliRunner()
    result = runner.invoke(main, ["train", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "HF_TOKEN" in result.output


def test_train_command_fails_fast_with_no_wandb_credentials(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("batch_size: 8\n")
    monkeypatch.setattr("contrastive_pretrain.cli.get_token", lambda: "fake-token")
    # main()'s own load_dotenv() call would otherwise repopulate WANDB_API_KEY
    # from a real .env file on the machine running this test, silently
    # defeating the "missing credentials" scenario and -- worse -- letting a
    # real wandb.init() fire against a real account. Neutralize it so this
    # test's outcome depends only on the deleted env var, not on what's in
    # some developer's local .env.
    monkeypatch.setattr("contrastive_pretrain.cli.load_dotenv", lambda: None)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(main, ["train", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "WANDB_API_KEY" in result.output


def test_export_frozen_encoder_command_requires_checkpoint_option() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["export-frozen-encoder"])

    assert result.exit_code != 0


def test_export_frozen_encoder_command_success_path(tmp_path, monkeypatch) -> None:
    """Prior coverage only checked the missing---checkpoint failure path;
    the command body itself (load checkpoint -> build encoder -> compute
    latent stats -> push) was never exercised, so a wiring bug (wrong
    load_state_dict key, wrong arg order into push_frozen_encoder) could
    ship silently. Monkeypatches build_val_dataset/HfApi/RealHfClient the
    same way the `train` command test already does, so this needs no
    network or HF credentials."""
    encoder, _ = build_encoder(pretrained=False)
    checkpoint_path = tmp_path / "checkpoint_step00000001.pt"
    save_checkpoint(checkpoint_path, {"model": encoder.state_dict()})

    config_path = tmp_path / "config.yaml"
    config_path.write_text("batch_size: 2\n")

    captured = {}

    monkeypatch.setattr(
        "contrastive_pretrain.cli.build_val_dataset",
        lambda _config: [
            {"original": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8)}
            for _ in range(2)
        ],
    )
    monkeypatch.setattr("contrastive_pretrain.cli.HfApi", lambda: _FakeHfApi())
    monkeypatch.setattr("contrastive_pretrain.cli.RealHfClient", lambda *_a, **_k: "the-client")

    def _fake_push(client, pushed_encoder, latent_mean, latent_std):
        captured["client"] = client
        captured["encoder"] = pushed_encoder
        captured["latent_mean"] = latent_mean
        captured["latent_std"] = latent_std

    monkeypatch.setattr("contrastive_pretrain.cli.push_frozen_encoder", _fake_push)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "export-frozen-encoder",
            "--checkpoint",
            str(checkpoint_path),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["client"] == "the-client"
    assert captured["latent_mean"].shape == (2048,)
    assert captured["latent_std"].shape == (2048,)
    # The loaded checkpoint's weights actually landed on the encoder that
    # gets pushed, not a fresh/mismatched one.
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, captured["encoder"].state_dict()[key])
