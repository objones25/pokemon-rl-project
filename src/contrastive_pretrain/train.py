"""Training loop orchestration: startup memory probe, resume-from-checkpoint,
per-step training under bf16 autocast, periodic checkpointing, periodic
frozen-artifact export, and structured logging / Trackio reporting.

run_memory_probe and check_finite_loss are the two places this module
deliberately fails fast rather than silently degrading -- see the design
spec's rationale for why a batch-size OOM is not auto-retried smaller,
and why a non-finite loss is not silently skipped.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from contrastive_pretrain.losses import nt_xent_loss

logger = logging.getLogger(__name__)


def run_memory_probe(probe_step: Callable[[], None], batch_size: int) -> None:
    try:
        probe_step()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"Out of memory at batch_size={batch_size}. Lower batch_size in "
            "the config, or add gradient accumulation, and retry."
        ) from exc


def check_finite_loss(loss: torch.Tensor, global_step: int) -> None:
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"non-finite loss ({loss.item()}) at step {global_step} -- stopping (fail-fast).")


def compute_val_loss(
    encoder: nn.Module,
    projector: nn.Module,
    val_batches: Iterable[dict],
    temperature: float,
    device: torch.device,
    max_batches: int,
) -> float:
    was_encoder_training = encoder.training
    was_projector_training = projector.training
    encoder.eval()
    projector.eval()

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(val_batches):
            if i >= max_batches:
                break
            view_a = batch["view_a"].to(device).float()
            view_b = batch["view_b"].to(device).float()
            z_a = projector(encoder(view_a))
            z_b = projector(encoder(view_b))
            loss = nt_xent_loss(z_a, z_b, temperature)
            total_loss += loss.item()
            n_batches += 1

    encoder.train(was_encoder_training)
    projector.train(was_projector_training)

    if n_batches == 0:
        raise RuntimeError("val_batches produced no batches")
    return total_loss / n_batches


import time
from dataclasses import dataclass, field
from pathlib import Path

from contrastive_pretrain.checkpoint import (
    build_checkpoint_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_optimizer_and_scheduler,
    save_checkpoint,
)
from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.dataset import build_dataloader, build_train_dataset, build_val_dataset
from contrastive_pretrain.encoder_io import compute_latent_stats, push_frozen_encoder
from contrastive_pretrain.model import build_encoder, build_projector
from hf_storage.client import HfClient
from observability.tracking import NullTrackioRun, TrackioRunLike
from observability.visualization import build_augmentation_contact_sheet

# objones25/pokemon-frames has 296,000 rows (confirmed via the HF Hub API
# at spec time) -- used only to size the cosine LR schedule's T_max for a
# streaming dataset where "total steps" isn't otherwise knowable upfront.
_DATASET_ROW_COUNT = 296_000


@dataclass
class TrainingDeps:
    config: TrainingConfig
    frozen_encoder_client: HfClient
    trackio_run: TrackioRunLike = field(default_factory=NullTrackioRun)
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


def run_training(deps: TrainingDeps) -> None:
    config = deps.config
    device = deps.device

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.fp32_precision = "tf32"

    encoder, embedding_dim = build_encoder(pretrained=config.pretrained)
    projector = build_projector(in_dim=embedding_dim)
    encoder.to(device, memory_format=torch.channels_last)
    projector.to(device)

    checkpoint_dir = Path(config.network_volume_checkpoint_dir)
    latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    state = load_checkpoint(latest_checkpoint_path) if latest_checkpoint_path is not None else None
    if state is not None:
        encoder.load_state_dict(state["model"])  # BEFORE torch.compile -- see checkpoint.py's docstring
        logger.info(
            "resumed_from_checkpoint",
            extra={"path": str(latest_checkpoint_path), "global_step": state["global_step"]},
        )

    compiled_encoder = torch.compile(encoder, mode="default")

    def _probe_step() -> None:
        dummy = torch.zeros(config.batch_size, 1, 144, 160, device=device).to(
            memory_format=torch.channels_last
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z = projector(compiled_encoder(dummy))
            loss = nt_xent_loss(z, z, config.temperature)
        loss.backward()
        compiled_encoder.zero_grad(set_to_none=True)
        projector.zero_grad(set_to_none=True)

    run_memory_probe(_probe_step, config.batch_size)

    optimizer = torch.optim.AdamW(
        list(compiled_encoder.parameters()) + list(projector.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch_estimate = max(1, _DATASET_ROW_COUNT // config.batch_size)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=config.warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.max_epochs * steps_per_epoch_estimate - config.warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
    )

    train_dataset = build_train_dataset(config)
    dataloader = build_dataloader(
        train_dataset, config.batch_size, config.num_workers, config.checkpoint_interval_steps
    )
    val_dataset = build_val_dataset(config)

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    if state is not None:
        restore_optimizer_and_scheduler(optimizer, scheduler, state)
        dataloader.load_state_dict(state["dataloader"])
        global_step = state["global_step"]
        start_epoch = state["epoch"]
        best_val_loss = state["best_val_loss"]

    for epoch in range(start_epoch, config.max_epochs):
        train_dataset.set_epoch(epoch)
        contact_sheet_logged_this_epoch = False
        for batch in dataloader:
            step_start = time.monotonic()
            view_a = batch["view_a"].to(device, non_blocking=True).float().to(memory_format=torch.channels_last)
            view_b = batch["view_b"].to(device, non_blocking=True).float().to(memory_format=torch.channels_last)
            data_wait_s = time.monotonic() - step_start

            # Logged every step (not gated behind the 50-step interval below):
            # data_wait_s is a plain wall-clock float, not a GPU read, so
            # this costs no extra device sync -- and the whole point of
            # this metric is catching a streaming-throughput bottleneck
            # immediately, not up to 50 steps late.
            logger.info("data_wait", extra={"global_step": global_step, "data_wait_s": data_wait_s})

            if not contact_sheet_logged_this_epoch:
                # Per the design spec's observability section: the same
                # human-in-the-loop augmentation sanity check the standalone
                # `preview` CLI command does, now running against real
                # training-time batches once per epoch. Uses the batch's
                # original CPU uint8 tensors (batch["original"]/["view_a"]/
                # ["view_b"]), not the device/channels_last-converted
                # view_a/view_b locals above.
                triples = [
                    (
                        batch["original"][i].squeeze(0).numpy(),
                        batch["view_a"][i].squeeze(0).numpy(),
                        batch["view_b"][i].squeeze(0).numpy(),
                    )
                    for i in range(min(4, batch["original"].shape[0]))
                ]
                contact_sheet = build_augmentation_contact_sheet(triples)
                deps.trackio_run.log({"augmentation_contact_sheet": contact_sheet})
                contact_sheet_logged_this_epoch = True

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                z_a = projector(compiled_encoder(view_a))
                z_b = projector(compiled_encoder(view_b))
                loss = nt_xent_loss(z_a, z_b, config.temperature)
            check_finite_loss(loss, global_step)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 50 == 0:
                metrics = {
                    "global_step": global_step,
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                }
                logger.info("train_step", extra=metrics)
                deps.trackio_run.log(metrics)

            if global_step % config.checkpoint_interval_steps == 0:
                ckpt_state = build_checkpoint_state(
                    epoch, global_step, encoder, optimizer, scheduler, dataloader.state_dict(), best_val_loss
                )
                save_checkpoint(checkpoint_dir / f"checkpoint_step{global_step:08d}.pt", ckpt_state)
                logger.info("checkpoint_saved", extra={"global_step": global_step})

        val_dataloader = build_dataloader(val_dataset, config.batch_size, 0, 1, pin_memory=False)
        val_loss = compute_val_loss(
            compiled_encoder, projector, val_dataloader, config.temperature, device, max_batches=20
        )
        logger.info("epoch_complete", extra={"epoch": epoch, "val_loss": val_loss, "best_val_loss": best_val_loss})
        deps.trackio_run.log({"val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            latent_mean, latent_std = compute_latent_stats(compiled_encoder, val_dataset, device)
            push_frozen_encoder(deps.frozen_encoder_client, encoder, latent_mean, latent_std)
            logger.info("frozen_artifact_pushed", extra={"epoch": epoch, "val_loss": val_loss})

    deps.trackio_run.finish()
