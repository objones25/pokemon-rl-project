from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image

from contrastive_pretrain.cli import main


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
