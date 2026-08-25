"""Console entry point: `contrastive-pretrain preview --frames-dir <dir> --out <path>`
generates a visual (original, view A, view B) contact sheet for spot-checking
the augmentation policy before spending GPU time on a training run."""

from __future__ import annotations

from pathlib import Path

import click
import torch
import trackio
from huggingface_hub import HfApi, get_token
from PIL import Image
from torchvision.io import ImageReadMode, read_image

from contrastive_pretrain.augmentation import AugmentationConfig, make_pair
from contrastive_pretrain.checkpoint import load_checkpoint
from contrastive_pretrain.config import load_config
from contrastive_pretrain.dataset import build_val_dataset
from contrastive_pretrain.encoder_io import compute_latent_stats, push_frozen_encoder
from contrastive_pretrain.model import build_encoder
from contrastive_pretrain.train import TrainingDeps, run_training
from hf_storage.client import RealHfClient
from observability.tracking import TrackioRun
from observability.visualization import build_augmentation_contact_sheet

_DEFAULT_CONFIG = Path("configs/contrastive_pretrain.yaml")


def _load_grayscale_frames(frames_dir: Path) -> list[torch.Tensor]:
    paths = sorted(frames_dir.glob("*.png"))
    frames = [read_image(str(p), mode=ImageReadMode.GRAY) for p in paths]
    if frames:
        shapes = {tuple(f.shape) for f in frames}
        if len(shapes) > 1:
            raise click.ClickException(
                f"Frames in {frames_dir} have mismatched shapes: {sorted(shapes)}. "
                "All frames must be the same size to build a contact sheet."
            )
    return frames


@click.group()
def main() -> None:
    """Pokemon Red/Blue contrastive-pretraining tools."""


@main.command()
@click.option("--frames-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("augmentation_preview.png"))
@click.option("--seed", type=int, default=0)
@click.option(
    "--limit",
    type=int,
    default=12,
    help="Maximum frames in the preview sheet (sampled evenly across the sorted file list).",
)
def preview(frames_dir: Path, out: Path, seed: int, limit: int) -> None:
    """Build an (original, view A, view B) contact sheet from sample frames."""
    frames = _load_grayscale_frames(frames_dir)
    if not frames:
        raise click.ClickException(f"No .png frames found in {frames_dir}")
    if len(frames) > limit:
        indices = [round(i * (len(frames) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [0]
        frames = [frames[i] for i in indices]

    config = AugmentationConfig()
    rng = torch.Generator().manual_seed(seed)
    triples = []
    for frame in frames:
        view_a, view_b = make_pair(frame, config, rng)
        triples.append(
            (frame.squeeze(0).numpy(), view_a.squeeze(0).numpy(), view_b.squeeze(0).numpy())
        )

    sheet = build_augmentation_contact_sheet(triples)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    click.echo(f"Wrote {len(frames)}-frame augmentation preview to {out}")


@main.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), default=_DEFAULT_CONFIG)
def train(config_path: Path) -> None:
    """Run (or resume) contrastive pretraining on an A100 pod."""
    if get_token() is None:
        raise click.ClickException(
            "No Hugging Face credentials found. Set HF_TOKEN (e.g. in a "
            ".env file) or run `hf auth login` before using this command."
        )

    config = load_config(config_path)
    HfApi().create_repo(config.frozen_encoder_repo_id, repo_type="model", exist_ok=True, private=True)
    frozen_client = RealHfClient(HfApi(), config.frozen_encoder_repo_id, repo_type="model")
    trackio_run = TrackioRun(trackio, project="pokemon-contrastive-pretrain", name=config.frozen_encoder_repo_id)

    deps = TrainingDeps(config=config, frozen_encoder_client=frozen_client, trackio_run=trackio_run)
    run_training(deps)


@main.command(name="export-frozen-encoder")
@click.option("--checkpoint", "checkpoint_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), default=_DEFAULT_CONFIG)
def export_frozen_encoder_command(checkpoint_path: Path, config_path: Path) -> None:
    """Manually (re-)export a frozen encoder artifact from a saved training checkpoint."""
    config = load_config(config_path)
    encoder, _ = build_encoder(pretrained=False)
    state = load_checkpoint(checkpoint_path)
    encoder.load_state_dict(state["model"])

    val_dataset = build_val_dataset(config)
    latent_mean, latent_std = compute_latent_stats(encoder, val_dataset, device=torch.device("cpu"))

    HfApi().create_repo(config.frozen_encoder_repo_id, repo_type="model", exist_ok=True, private=True)
    client = RealHfClient(HfApi(), config.frozen_encoder_repo_id, repo_type="model")
    push_frozen_encoder(client, encoder, latent_mean, latent_std)
    click.echo(f"Exported frozen encoder from {checkpoint_path} to {config.frozen_encoder_repo_id}")
