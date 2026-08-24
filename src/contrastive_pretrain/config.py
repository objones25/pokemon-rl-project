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
    batch_size: int = 1024
    num_workers: int = 8
    shuffle_buffer_size: int = 10_000
    seed: int = 0
    learning_rate: float = 4e-4
    warmup_steps: int = 500
    weight_decay: float = 1e-6
    temperature: float = 0.1
    max_epochs: int = 100
    checkpoint_interval_steps: int = 1000
    network_volume_checkpoint_dir: str = "/runpod-volume/contrastive_pretrain/checkpoints"


def load_config(path: str | Path) -> TrainingConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if "val_video_ids" in data:
        data["val_video_ids"] = tuple(data["val_video_ids"])
    valid_fields = {f.name for f in fields(TrainingConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")
    return TrainingConfig(**data)
