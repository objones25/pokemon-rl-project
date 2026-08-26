"""Training loop orchestration: startup memory probe, resume-from-checkpoint,
per-step training under autocast (bf16 on CUDA/CPU, fp16 on MPS -- see
autocast_dtype), periodic checkpointing, periodic frozen-artifact export,
and structured logging / Weights & Biases reporting.

run_memory_probe and check_finite_loss are the two places this module
deliberately fails fast rather than silently degrading -- see the design
spec's rationale for why a batch-size OOM is not auto-retried smaller,
and why a non-finite loss is not silently skipped.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import torch
from torch import nn

from contrastive_pretrain.losses import nt_xent_loss

logger = logging.getLogger(__name__)


def autocast_dtype(device: torch.device) -> torch.dtype:
    """bf16 everywhere except MPS. The target production hardware is a CUDA
    A100 (bf16-native, Ampere+), and CPU's own recommended autocast dtype is
    also bf16 -- but MPS has weak bf16 kernel coverage and torch's own
    per-device guidance there is fp16 (see scripts/check_env.py's autocast
    table in the pytorch skill). No GradScaler is used anywhere in this loop
    (torch reports none available on MPS either), so this only changes local
    MPS dev-loop numerics -- never the CUDA A100 production path.
    """
    return torch.float16 if device.type == "mps" else torch.bfloat16


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
        raise RuntimeError(
            f"non-finite loss ({loss.item()}) at step {global_step} -- stopping (fail-fast)."
        )


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
    with torch.inference_mode():
        for i, batch in enumerate(val_batches):
            if i >= max_batches:
                break
            view_a = batch["view_a"].to(device).float()
            view_b = batch["view_b"].to(device).float()
            # Matches the training step's autocast context -- otherwise
            # validation runs in full fp32 at the training batch size, and
            # the startup memory probe (which only exercises the training
            # step's autocast dtype) doesn't actually cover validation's
            # memory profile.
            with torch.autocast(device_type=device.type, dtype=autocast_dtype(device)):
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

import wandb

from contrastive_pretrain.checkpoint import (
    build_checkpoint_state,
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    restore_optimizer_and_scheduler,
    save_checkpoint,
)
from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.dataset import (
    build_dataloader,
    build_train_dataset,
    build_val_dataset,
)
from contrastive_pretrain.encoder_io import compute_latent_stats, push_frozen_encoder
from contrastive_pretrain.model import build_encoder, build_projector
from hf_storage.client import AtomicHfClient
from observability.tracking import ExperimentRunLike, NullExperimentRun
from observability.visualization import build_augmentation_contact_sheet

# objones25/pokemon-frames has 296,040 rows across 7 videos; excluding the
# two default val_video_ids (D1SrSFZrV7A: 51,967 rows, YW29l3jJXr4: 45,880
# rows) leaves 198,193 training rows. Confirmed by reading just the
# video_id column (column-pruned, no image bytes) out of every parquet
# shard via the HF Hub API -- not a guess, and not the whole-dataset count:
# the val split is ~33% of the dataset here, not a rounding error, and
# build_train_dataset always excludes it before training. Used only to
# size the cosine LR schedule's T_max for a streaming dataset where "total
# steps" isn't otherwise knowable upfront -- if TrainingConfig.val_video_ids
# is ever changed from its default, this constant needs re-deriving the
# same way.
_TRAIN_ROW_COUNT = 198_193


@dataclass
class TrainingDeps:
    config: TrainingConfig
    frozen_encoder_client: AtomicHfClient
    wandb_run: ExperimentRunLike = field(default_factory=NullExperimentRun)
    device: torch.device = field(
        default_factory=lambda: (
            torch.accelerator.current_accelerator(check_available=True)
            or torch.device("cpu")
        )
    )


def run_training(deps: TrainingDeps) -> None:
    config = deps.config
    device = deps.device

    # cudnn and TF32 are CUDA/NVIDIA-specific -- setting them unconditionally
    # would be a no-op at best on other accelerators (MPS, CPU) and is simply
    # wrong to ship as if it applied everywhere, given the target production
    # hardware is a CUDA A100 but this code also runs in non-CUDA dev/test
    # environments.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = "tf32"

    encoder, embedding_dim = build_encoder(pretrained=config.pretrained)
    projector = build_projector(in_dim=embedding_dim)
    # nn.Module.to()'s @overload stubs omit memory_format entirely (verified
    # via inspect.getsource on the installed torch), even though the method's
    # own docstring documents `to(memory_format=torch.channels_last)` as a
    # valid call form -- a stub gap, not a real type error.
    encoder.to(device, memory_format=torch.channels_last)  # type: ignore[call-overload]
    projector.to(device)

    checkpoint_dir = Path(config.network_volume_checkpoint_dir)
    latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    state = (
        load_checkpoint(latest_checkpoint_path)
        if latest_checkpoint_path is not None
        else None
    )
    if state is not None:
        encoder.load_state_dict(
            state["model"]
        )  # BEFORE torch.compile -- see checkpoint.py's docstring
        # The optimizer below is built over BOTH modules' parameters, so the
        # projector must be restored too -- otherwise the old projector's Adam
        # moments would be loaded onto a freshly-random projection head.
        projector.load_state_dict(state["projector"])
        logger.info(
            "resumed_from_checkpoint",
            extra={
                "path": str(latest_checkpoint_path),
                "global_step": state["global_step"],
            },
        )

    compiled_encoder = torch.compile(encoder, mode="default")
    # torch.compile's stub types its return as a generic Callable, not
    # nn.Module, since it also accepts/returns plain functions -- given an
    # nn.Module input it always returns an OptimizedModule (an nn.Module
    # subclass) at runtime. Asserted, not just annotated, so the claim is
    # backed by a real runtime check rather than silently trusted.
    assert isinstance(compiled_encoder, nn.Module)

    def _probe_step() -> None:
        # TWO forward passes, mirroring the real training step: view_a and
        # view_b each go through the encoder independently and BOTH autograd
        # graphs stay alive until the single shared backward below. Encoder
        # activations dominate memory here (the maxpool-dropped stem keeps
        # 4x the spatial resolution of a stock ResNet-50), so a single-forward
        # probe would under-measure real training memory by roughly 2x and
        # could pass at a batch_size that OOMs on the first real step.
        dummy_a = torch.zeros(config.batch_size, 1, 144, 160, device=device).to(
            memory_format=torch.channels_last
        )
        dummy_b = torch.zeros(config.batch_size, 1, 144, 160, device=device).to(
            memory_format=torch.channels_last
        )
        with torch.autocast(device_type=device.type, dtype=autocast_dtype(device)):
            z_a = projector(compiled_encoder(dummy_a))
            z_b = projector(compiled_encoder(dummy_b))
            loss = nt_xent_loss(z_a, z_b, config.temperature)
        loss.backward()
        compiled_encoder.zero_grad(set_to_none=True)
        projector.zero_grad(set_to_none=True)

    run_memory_probe(_probe_step, config.batch_size)

    optimizer = torch.optim.AdamW(
        list(compiled_encoder.parameters()) + list(projector.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch_estimate = max(1, _TRAIN_ROW_COUNT // config.batch_size)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=config.warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            1, config.max_epochs * steps_per_epoch_estimate - config.warmup_steps
        ),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
    )

    train_dataset = build_train_dataset(config)
    dataloader = build_dataloader(
        train_dataset,
        config.batch_size,
        config.num_workers,
        config.checkpoint_interval_steps,
    )
    val_dataset = build_val_dataset(config)

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    if state is not None:
        restore_optimizer_and_scheduler(optimizer, scheduler, state)
        # An epoch-boundary checkpoint deliberately stores no dataloader state
        # (see the epoch-boundary _save_checkpoint call below): a freshly-built
        # dataloader is exactly right for starting the next epoch. Mid-epoch
        # checkpoints DO carry state, and restoring it correctly continues the
        # partial epoch -- but ONLY if it was built over the same data source.
        # Setting or clearing local_cache_dir adds/drops build_train_dataset's
        # pre-shuffle resize-map stage, changing the nesting of the underlying
        # datasets.IterableDataset's state dict. load_state_dict() accepts the
        # mismatched shape without complaint (silently resetting position to
        # zero) and then dies with KeyError: 'examples_iterable' on the first
        # batch -- i.e. hours in, after a full cache build. `.get`, not
        # `[...]`: checkpoints written before this field existed carry no key,
        # and treating those as local_cache_dir=None matches the behavior they
        # were actually saved under.
        #
        # Mid-epoch train resume is APPROXIMATE, not exact, and that is a
        # property of the pipeline rather than a bug here. build_train_dataset
        # ends in .shuffle(buffer_size=...), and a shuffle buffer's *contents*
        # are not part of the checkpointed state -- datasets says so itself on
        # load ("Loading a state dict of a shuffle buffer of a dataset without
        # the buffer content. The shuffle buffer will be refilled..."). So a
        # resumed run refills the buffer from the checkpointed source position
        # and serves a different order than the un-interrupted run would have:
        # up to shuffle_buffer_size x num_workers rows get re-seen or skipped
        # within that epoch. Measured directly, not inferred.
        #
        # Benign for SimCLR -- the objective is over augmented pairs and
        # nothing depends on epoch-exact sample coverage -- but do NOT build
        # anything on top of this that assumes exact resume (per-sample
        # curricula, epoch-level dedup, "each row seen once per epoch"
        # accounting). build_val_dataset has no .shuffle() and DOES resume
        # exactly; that difference is deliberate, and is what
        # test_build_val_dataloader_resumes_from_exact_position_over_streamed_parquet_shards
        # pins.
        if state["dataloader"]:
            if state.get("local_cache_dir") == config.local_cache_dir:
                dataloader.load_state_dict(state["dataloader"])
            else:
                logger.info(
                    "dataloader_state_skipped_data_source_changed",
                    extra={
                        "checkpoint_local_cache_dir": state.get("local_cache_dir"),
                        "config_local_cache_dir": config.local_cache_dir,
                    },
                )
        global_step = state["global_step"]
        start_epoch = state["epoch"]
        best_val_loss = state["best_val_loss"]

    def _save_checkpoint(epoch: int, dataloader_state: dict | None) -> None:
        ckpt_state = build_checkpoint_state(
            epoch,
            global_step,
            encoder,
            projector,
            optimizer,
            scheduler,
            dataloader_state,
            best_val_loss,
            local_cache_dir=config.local_cache_dir,
        )
        save_checkpoint(
            checkpoint_dir / f"checkpoint_step{global_step:08d}.pt", ckpt_state
        )
        logger.info("checkpoint_saved", extra={"global_step": global_step})
        # AFTER the save, never before: pruning first would briefly leave the
        # run with one fewer resume point than intended, and a crash in the
        # window between would resume further back than necessary. Pruning
        # after also means the checkpoint just written is always among the
        # survivors, since it sorts highest.
        pruned = prune_checkpoints(checkpoint_dir, config.checkpoint_keep_last_n)
        if pruned:
            logger.info(
                "checkpoints_pruned",
                extra={
                    "global_step": global_step,
                    "deleted_count": len(pruned),
                    "keep_last_n": config.checkpoint_keep_last_n,
                },
            )

    prev_step_end = time.monotonic()
    for epoch in range(start_epoch, config.max_epochs):
        train_dataset.set_epoch(epoch)
        contact_sheet_logged_this_epoch = False
        for batch in dataloader:
            # The blocking wait for the next batch happens inside the `for`
            # loop's implicit __next__() call, i.e. BETWEEN iterations, not
            # inside the loop body -- so data_wait_s is measured as the gap
            # since the end of the previous iteration, not as time spent
            # inside this one (which would just measure the trivial H2D
            # transfer below and read ~0 regardless of streaming stalls).
            data_wait_s = time.monotonic() - prev_step_end
            view_a = (
                batch["view_a"]
                .to(device, non_blocking=True)
                .float()
                .to(memory_format=torch.channels_last)
            )
            view_b = (
                batch["view_b"]
                .to(device, non_blocking=True)
                .float()
                .to(memory_format=torch.channels_last)
            )

            # Logged every step (not gated behind the 50-step interval below):
            # data_wait_s is a plain wall-clock float, not a GPU read, so
            # this costs no extra device sync -- and the whole point of
            # this metric is catching a streaming-throughput bottleneck
            # immediately, not up to 50 steps late.
            logger.info(
                "data_wait",
                extra={"global_step": global_step, "data_wait_s": data_wait_s},
            )

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
                # A raw ndarray isn't rendered as an image by wandb.Run.log --
                # it must be wrapped in wandb.Image first (WandbRun.log
                # swallows all exceptions by design, so a wrapper mismatch
                # would fail silently rather than error), or the sanity-check
                # image would never actually appear anywhere.
                deps.wandb_run.log(
                    {
                        "augmentation_contact_sheet": wandb.Image(
                            contact_sheet, caption=f"epoch {epoch}"
                        )
                    }
                )
                contact_sheet_logged_this_epoch = True

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype(device)):
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
                deps.wandb_run.log(metrics)

            if global_step % config.checkpoint_interval_steps == 0:
                # Mid-epoch: the dataloader's state is a real mid-stream
                # position, and restoring it correctly continues this partial
                # epoch on resume.
                _save_checkpoint(epoch, dataloader.state_dict())

            # Last thing in the loop body, right before the loop returns to
            # `for batch in dataloader` and blocks on the next batch.
            prev_step_end = time.monotonic()

        val_dataloader = build_dataloader(
            val_dataset, config.batch_size, 0, 1, pin_memory=False, drop_last=False
        )
        val_loss = compute_val_loss(
            compiled_encoder,
            projector,
            val_dataloader,
            config.temperature,
            device,
            max_batches=20,
        )
        logger.info(
            "epoch_complete",
            extra={
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            },
        )
        deps.wandb_run.log({"val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            latent_mean, latent_std = compute_latent_stats(
                compiled_encoder, val_dataset, device
            )
            push_frozen_encoder(
                deps.frozen_encoder_client, encoder, latent_mean, latent_std
            )
            logger.info(
                "frozen_artifact_pushed", extra={"epoch": epoch, "val_loss": val_loss}
            )

        # Checkpoint at every epoch boundary, not just every
        # checkpoint_interval_steps: best_val_loss just changed in memory
        # above but isn't durable until this call. Without it, a crash
        # between a val-loss improvement and the next periodic mid-epoch
        # save would resume with a stale (too-high) best_val_loss, letting
        # a later, actually-worse model pass the improvement check and
        # overwrite the genuinely better encoder already published to the
        # Hub.
        #
        # Stores `epoch + 1` (this epoch is DONE -- resume must start at the
        # next one, since start_epoch = state["epoch"]) and no dataloader
        # state. The iterator that just finished this epoch is exhausted;
        # measured against a datasets.IterableDataset (which, unlike a plain
        # torch IterableDataset, implements state_dict/load_state_dict, so
        # StatefulDataLoader delegates to it), restoring an exhausted state
        # starves the resumed epoch of batches. Because this loop always
        # calls train_dataset.set_epoch(epoch) with a new value each epoch,
        # measured behavior in this exact shape is that only the resumed
        # epoch is starved (zero batches) -- the epoch after it recovers
        # fully. Not saving stale dataloader state here still eliminates
        # that starved epoch entirely, at no cost to the epochs after it.
        _save_checkpoint(epoch + 1, None)

        # Reset here, not just at the end of each batch iteration above:
        # validation, the conditional Hub push, and this checkpoint save all
        # ran since the last batch's prev_step_end update. Without this,
        # data_wait_s for the next epoch's first batch would include all of
        # that epoch-boundary work, misreporting it as a streaming stall.
        prev_step_end = time.monotonic()

    deps.wandb_run.finish()
