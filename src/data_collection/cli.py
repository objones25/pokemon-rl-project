"""Console entry points: `data-collection curate <url>` (Phase A) and
`data-collection run --repo-id <repo>` (Phase B)."""

from __future__ import annotations

from pathlib import Path

import click
import trackio
from dotenv import load_dotenv
from huggingface_hub import HfApi, get_token

from data_collection import curation, extract, pipeline
from data_collection.hf_uploader import HfUploader
from data_collection.registry import load_registry
from hf_storage.client import HfClient, RealHfClient
from observability.logging_config import configure_logging
from observability.tracking import TrackioRun

_DEFAULT_REGISTRY = Path("configs/video_sources.yaml")
_DEFAULT_APPROVED_DIR = Path("configs/templates/approved")


@click.group()
def main() -> None:
    """Pokemon Red/Blue data collection pipeline."""
    load_dotenv()
    configure_logging()


@main.command()
@click.argument("url")
@click.option("--game", type=click.Choice(["red", "blue"]), required=True)
@click.option("--registry", type=click.Path(path_type=Path), default=_DEFAULT_REGISTRY)
@click.option("--approved-dir", type=click.Path(path_type=Path), default=_DEFAULT_APPROVED_DIR)
def curate(url: str, game: str, registry: Path, approved_dir: Path) -> None:
    """Phase A: interactively review and approve a candidate video."""
    curation.run_curation(
        video_url=url,
        approved_dir=approved_dir,
        registry_path=registry,
        game=game,
    )


@main.command()
@click.option("--repo-id", required=True, help="HF dataset repo, e.g. me/pokemon-frames")
@click.option("--registry", type=click.Path(path_type=Path), default=_DEFAULT_REGISTRY)
@click.option("--batch-size", type=int, default=2000)
@click.option(
    "--checkpoint-interval",
    type=int,
    default=5000,
    help="Sampled frames between progress checkpoints (each one commits manifest.json to the "
    "Hub -- too small a value burns through HF's hourly commit quota on a long video).",
)
@click.option(
    "--max-concurrent-videos",
    type=int,
    default=1,
    help="Process this many videos in parallel (each has its own ffmpeg subprocess).",
)
def run(
    repo_id: str,
    registry: Path,
    batch_size: int,
    checkpoint_interval: int,
    max_concurrent_videos: int,
) -> None:
    """Phase B: unattended extraction across all approved, incomplete videos."""
    if get_token() is None:
        raise click.ClickException(
            "No Hugging Face credentials found. Set HF_TOKEN (e.g. in a "
            ".env file) or run `hf auth login` before using this command."
        )

    sources = load_registry(registry)

    client: HfClient = RealHfClient(HfApi(), repo_id)
    uploader = HfUploader(client, repo_id)
    trackio_run = TrackioRun(trackio, project="pokemon-data-collection", name=repo_id)

    def frame_source(video_source, resume_seconds: float = 0.0):
        stream_url, _, _, headers = extract.get_stream_info(video_source.url)
        return extract.stream_frames(
            stream_url,
            crop_x=video_source.crop_x,
            crop_y=video_source.crop_y,
            crop_w=video_source.crop_w,
            crop_h=video_source.crop_h,
            start_seconds=resume_seconds,
            headers=headers,
        )

    deps = pipeline.PipelineDeps(
        frame_source=frame_source,
        uploader=uploader,
        trackio_run=trackio_run,
        batch_size=batch_size,
        checkpoint_interval_samples=checkpoint_interval,
        max_concurrent_videos=max_concurrent_videos,
    )
    result = pipeline.run_pipeline(sources, deps)
    if result.failed > 0:
        raise click.ClickException(
            f"{result.failed} of {len(sources)} video(s) failed -- see logs for details."
        )
