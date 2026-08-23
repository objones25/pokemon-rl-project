"""Console entry point: `contrastive-pretrain preview --frames-dir <dir> --out <path>`
generates a visual (original, view A, view B) contact sheet for spot-checking
the augmentation policy before spending GPU time on a training run."""

from __future__ import annotations

from pathlib import Path

import click
import torch
from PIL import Image
from torchvision.io import ImageReadMode, read_image

from contrastive_pretrain.augmentation import AugmentationConfig, make_pair
from observability.visualization import build_augmentation_contact_sheet


def _load_grayscale_frames(frames_dir: Path) -> list[torch.Tensor]:
    paths = sorted(frames_dir.glob("*.png"))
    return [read_image(str(p), mode=ImageReadMode.GRAY) for p in paths]


@click.group()
def main() -> None:
    """Pokemon Red/Blue contrastive-pretraining tools."""


@main.command()
@click.option("--frames-dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("augmentation_preview.png"))
@click.option("--seed", type=int, default=0)
def preview(frames_dir: Path, out: Path, seed: int) -> None:
    """Build an (original, view A, view B) contact sheet from sample frames."""
    frames = _load_grayscale_frames(frames_dir)
    if not frames:
        raise click.ClickException(f"No .png frames found in {frames_dir}")

    config = AugmentationConfig()
    rng = torch.Generator().manual_seed(seed)
    triples = []
    for frame in frames:
        view_a, view_b = make_pair(frame, config, rng)
        triples.append(
            (frame.squeeze(0).numpy(), view_a.squeeze(0).numpy(), view_b.squeeze(0).numpy())
        )

    sheet = build_augmentation_contact_sheet(triples)
    Image.fromarray(sheet).save(out)
    click.echo(f"Wrote {len(frames)}-frame augmentation preview to {out}")
