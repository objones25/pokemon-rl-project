"""Paired policy + env checkpoints, committed by a manifest.

checkpointing.io.save_checkpoint is already atomic per file (.tmp + replace).
The failure it cannot see is ONE OF THE TWO landing: a crash between the policy
write and the env write leaves a policy that believes in a game position the
emulator no longer occupies. The manifest is written last and is the commit
point; a resume that finds no manifest, or files whose sizes disagree with it,
falls back to the previous update.

Filename globs are distinct from contrastive_pretrain's and from the env's so
one network volume can hold every run's checkpoints without any of them
pruning another's resume point."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from checkpointing.io import load_checkpoint, prune_checkpoints, save_checkpoint
from pokemon_env.checkpoint import (
    ENV_CHECKPOINT_PATTERN,
    build_env_checkpoint_state,
    restore_env_checkpoint,
)
from ppo.config import PPOConfig
from ppo.normalizer import ReturnScaler
from sequence_model.checkpoint import (
    build_policy_checkpoint_state,
    capture_rng_state,
    rebuild_cache,
    restore_policy_checkpoint,
    restore_rng_state,
)
from sequence_model.config import PolicyConfig

logger = logging.getLogger(__name__)

POLICY_PATTERN = "policy_update*.pt"
# Reuses pokemon_env.checkpoint's own constant rather than a second literal of
# the same string, so a future rename there can't silently desync pruning and
# resume from what that module actually writes.
ENV_PATTERN = ENV_CHECKPOINT_PATTERN
MANIFEST_PATTERN = "manifest_update*.json"


@dataclass(frozen=True)
class ResumeResult:
    update: int
    global_step: int
    cache: object | None
    wandb_run_id: str | None


def write_checkpoint(
    directory: Path,
    update: int,
    global_step: int,
    policy,
    optimizer,
    scheduler,
    cache,
    vec_env,
    scaler: ReturnScaler,
    config: PPOConfig,
    init_state_hash: str,
    wandb_run_id: str | None,
    git_commit: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    # Zero-padded: checkpointing.io.find_latest_checkpoint/prune_checkpoints
    # both sort by filename, and that module's own docstring states the
    # precondition explicitly ("callers must zero-pad: step900 sorts after
    # step1300 as a string"). Six digits covers 999,999 updates, far beyond
    # any run this design contemplates. Unpadded, prune_checkpoints would
    # delete the newest checkpoints and keep old ones past update 9 -- silent
    # data loss on exactly the preemptible run this task exists to protect.
    policy_file = directory / f"policy_update{update:06d}.pt"
    env_file = directory / f"env_update{update:06d}.pt"

    save_checkpoint(
        policy_file,
        build_policy_checkpoint_state(
            update, global_step, policy, optimizer, scheduler, cache, capture_rng_state()
        ),
    )
    save_checkpoint(env_file, build_env_checkpoint_state(update, vec_env, init_state_hash))

    manifest = {
        "update": update,
        "global_step": global_step,
        "policy_file": policy_file.name,
        "env_file": env_file.name,
        "sizes": {
            policy_file.name: policy_file.stat().st_size,
            env_file.name: env_file.stat().st_size,
        },
        "return_scaler": scaler.state_dict(),
        "torch_version": torch.__version__,
        "git_commit": git_commit,
        "frozen_encoder_revision": config.frozen_encoder_revision,
        "wandb_run_id": wandb_run_id,
    }
    manifest_file = directory / f"manifest_update{update:06d}.json"
    # Written LAST. Everything above may exist without this; nothing resumes
    # from a checkpoint this file does not name.
    manifest_file.write_text(json.dumps(manifest, indent=2))

    for pattern in (POLICY_PATTERN, ENV_PATTERN, MANIFEST_PATTERN):
        prune_checkpoints(directory, config.keep_last_n, pattern)
    logger.info("checkpoint_written", extra={"update": update, "path": str(manifest_file)})
    return manifest_file


def resume(
    directory: Path,
    policy,
    optimizer,
    scheduler,
    vec_env,
    scaler: ReturnScaler,
    policy_config: PolicyConfig,
    config: PPOConfig,
    init_state_hash: str,
) -> ResumeResult | None:
    for manifest_file in sorted(directory.glob(MANIFEST_PATTERN), reverse=True):
        manifest = json.loads(manifest_file.read_text())
        if not _files_intact(directory, manifest):
            logger.warning(
                "checkpoint_incomplete_skipped", extra={"manifest": str(manifest_file)}
            )
            continue

        # Raised, not skipped: the manifest records the encoder revision
        # precisely so a mid-run encoder change is detectable, and every older
        # manifest in the directory carries the same stale revision -- falling
        # back to one of those would resume against the same wrong features
        # while looking like it recovered. Mirrors restore_env_checkpoint's
        # init_state_hash check, which raises for the same reason.
        if manifest["frozen_encoder_revision"] != config.frozen_encoder_revision:
            raise ValueError(
                f"checkpoint was written against frozen encoder revision "
                f"{manifest['frozen_encoder_revision']}, but this run is configured for "
                f"{config.frozen_encoder_revision}. Every latent the policy was trained "
                "on came from the old encoder; resuming would feed it features from a "
                "different one."
            )

        policy_state = load_checkpoint(directory / manifest["policy_file"])
        restore_policy_checkpoint(policy, optimizer, scheduler, policy_state)
        restore_env_checkpoint(
            vec_env, load_checkpoint(directory / manifest["env_file"]), init_state_hash
        )
        scaler.load_state_dict(manifest["return_scaler"])
        restore_rng_state(policy_state.get("rng"))

        cache = _restore_cache(policy_state, policy_config)
        logger.info(
            "resumed_from_checkpoint",
            extra={"update": manifest["update"], "global_step": manifest["global_step"]},
        )
        return ResumeResult(
            update=int(manifest["update"]),
            global_step=int(manifest["global_step"]),
            cache=cache,
            wandb_run_id=manifest.get("wandb_run_id"),
        )
    return None


def _files_intact(directory: Path, manifest: dict) -> bool:
    for name, size in manifest["sizes"].items():
        path = directory / name
        if not path.exists() or path.stat().st_size != size:
            return False
    return True


def _restore_cache(policy_state: dict, policy_config: PolicyConfig):
    """A context_len change at a curriculum boundary invalidates the ring
    buffer. Compared BEFORE calling rebuild_cache rather than catching its
    ValueError, which is raised for several distinct reasons."""
    cache_state = policy_state.get("cache")
    if cache_state is None:
        return None
    saved_context = int(cache_state["k"].shape[3])
    if saved_context != policy_config.context_len:
        logger.warning(
            "cache_dropped_context_changed",
            extra={"saved": saved_context, "live": policy_config.context_len},
        )
        return None
    return rebuild_cache(cache_state, policy_config)
