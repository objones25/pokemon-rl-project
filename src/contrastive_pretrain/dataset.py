"""Streaming dataset construction. Augmentation randomness is a per-row
deterministic seed derived from (base_seed, video_id, timestamp_s), NOT
a shared torch.Generator -- a shared Generator would be copied identically
into every StatefulDataLoader worker process, producing correlated (not
independent) augmentation sequences across workers. This also means
resuming needs zero augmentation-RNG checkpoint state: the seed for a
given row is always re-derivable from data the row already carries.
"""

from __future__ import annotations

import hashlib

import torch
from torchvision.transforms.functional import pil_to_tensor

from contrastive_pretrain.augmentation import AugmentationConfig, make_pair


def row_seed(base_seed: int, video_id: str, timestamp_s: float) -> int:
    digest = hashlib.sha256(f"{base_seed}:{video_id}:{timestamp_s}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def to_pair_transform(example: dict, augmentation_config: AugmentationConfig, base_seed: int) -> dict:
    frame = pil_to_tensor(example["image"])  # (1, H, W) uint8
    seed = row_seed(base_seed, example["video_id"], example["timestamp_s"])
    rng = torch.Generator().manual_seed(seed)
    view_a, view_b = make_pair(frame, augmentation_config, rng)
    return {"original": frame, "view_a": view_a, "view_b": view_b}
