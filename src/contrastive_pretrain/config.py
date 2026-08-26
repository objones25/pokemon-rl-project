"""Training hyperparameters and paths, loaded from configs/contrastive_pretrain.yaml.
Mirrors data_collection.registry's dataclass + yaml.safe_load pattern."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    dataset_repo_id: str = "objones25/pokemon-frames"
    frozen_encoder_repo_id: str = "objones25/pokemon-contrastive-encoder"
    val_video_ids: tuple[str, ...] = ("D1SrSFZrV7A", "YW29l3jJXr4")
    pretrained: bool = True
    # 512, not 1024: 1024 OOM'd in practice on an 80GB A100 pod (the design
    # spec's memory arithmetic was an estimate, not a measurement). Dropping
    # back to 512 also reverts learning_rate/warmup_steps to their coupled
    # batch=512 baseline (spec's sqrt(2) LR-scaling heuristic and
    # warmup-covers-the-same-number-of-examples rule) rather than leaving
    # them tuned for a batch size this pod can't actually run.
    batch_size: int = 512
    num_workers: int = 8
    shuffle_buffer_size: int = 10_000
    seed: int = 0
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 1e-6
    temperature: float = 0.1
    max_epochs: int = 100
    checkpoint_interval_steps: int = 1000
    # /workspace, not /runpod-volume -- /runpod-volume is where a network
    # volume mounts for a Serverless *worker*; a persistent training Pod
    # (what this config targets) mounts its network volume at /workspace by
    # default. Confirmed against current RunPod docs, not assumed.
    network_volume_checkpoint_dir: str = "/workspace/contrastive_pretrain/checkpoints"
    # None (default): stream objones25/pokemon-frames directly from the
    # Hub, resizing per-row on the fly, exactly as before this field
    # existed. Set to a /workspace-backed directory to cache already-
    # resized frames there instead -- build_train_dataset/build_val_dataset
    # populate it automatically on first use (see
    # contrastive_pretrain.resize_cache.ensure_local_cache), eliminating
    # both the per-epoch network round-trips and per-epoch resize work on
    # every call after that. Note this is not a train-only cost: cli.py's
    # export-frozen-encoder calls build_val_dataset alone, which also triggers
    # ensure_local_cache -- so running it on a fresh pod before `train` builds
    # the entire cache, not just a lightweight val-only subset.
    local_cache_dir: str | None = None
    # Checkpoints are ~336MB each (ResNet-50 encoder + projector + AdamW
    # moments, measured). Unpruned, a 100-epoch run at
    # checkpoint_interval_steps=1000 writes ~138 of them -- ~46GB, which on
    # its own nearly fills the 50GB network volume that must also hold the
    # resize cache. 3 is a resume tail, not an archive: the *best* encoder is
    # never only here, it's pushed to the Hub on every val-loss improvement.
    checkpoint_keep_last_n: int = 3
    # None (default): resize shards single-threaded, as before this field
    # existed. The resize is ~8.9s per 500-row shard measured on a real
    # shard, so a full 367-shard build is roughly an hour of pure CPU work.
    # Set to the pod's core count to parallelize it (build-cache also takes
    # --num-proc, since a CPU build pod is sized differently than the GPU
    # training pod this config otherwise targets).
    resize_cache_num_proc: int | None = None


def load_config(path: str | Path) -> TrainingConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if "val_video_ids" in data:
        data["val_video_ids"] = tuple(data["val_video_ids"])
    valid_fields = {f.name for f in fields(TrainingConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")
    return TrainingConfig(**data)
