"""Frozen-encoder inference and the latent statistics that normalize it.

Owns two things the sequence-model spec's handoff assigned to PPO, because
they belong wherever the frozen encoder lives: batched inference, and fetching
plus validating latent_stats.json. InputAdapter already raises on bad stats,
but it can only do so once someone hands it the values -- fetching them is
this module's job."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from contrastive_pretrain.model import EMBEDDING_DIM
from hf_storage.client import HfClient

LATENT_STATS_FILENAME = "latent_stats.json"
FRAME_HEIGHT = 144
FRAME_WIDTH = 160


def load_latent_stats(client: HfClient) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, std), each (2048,) float32, validated against InputAdapter's
    contract before anyone builds a policy with them."""
    payload = client.download_bytes(LATENT_STATS_FILENAME)
    if payload is None:
        raise FileNotFoundError(
            f"{LATENT_STATS_FILENAME} missing from the frozen-encoder repo; the policy "
            "cannot normalize latents without it, and unnormalized contrastive latents "
            "cause immediate PPO policy collapse"
        )
    stats = json.loads(payload)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)

    for name, tensor in (("latent_mean", mean), ("latent_std", std)):
        if tensor.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"{name} has {tensor.numel()} elements, expected {EMBEDDING_DIM}"
            )

    # Finiteness is checked BEFORE the sign check, because `NaN <= 0` is False:
    # a NaN std sails past the non-positive test, normalizes every latent to
    # NaN, and reaches the value head at the first update with nothing raised
    # anywhere along the way.
    for name, tensor in (("latent_mean", mean), ("latent_std", std)):
        non_finite = int((~torch.isfinite(tensor)).sum())
        if non_finite:
            raise ValueError(
                f"{name} has {non_finite} non-finite entries (NaN or inf). These pass "
                "the positivity check silently -- NaN <= 0 is False -- and turn every "
                "normalized latent into NaN at the first update."
            )

    non_positive = int((std <= 0).sum())
    if non_positive:
        raise ValueError(
            f"latent_std has {non_positive} non-positive entries. A dead encoder channel "
            "divides by InputAdapter's 1e-6 floor and feeds ~1e6-scale inputs to the "
            "value head."
        )
    return mean, std


class LatentEncoder:
    """Batched frozen-CNN inference: (N, 1, 144, 160) uint8 -> (N, 2048)."""

    def __init__(self, encoder: nn.Module, device: torch.device) -> None:
        self._encoder = encoder.to(device).to(memory_format=torch.channels_last).eval()
        self._device = device

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """@torch.no_grad(), deliberately NOT @torch.inference_mode().

        Latents recorded during rollout become inputs to forward_chunk at the
        PPO update, and a tensor created under inference_mode raises "Inference
        tensors cannot be saved for backward" the moment the adapter tries to
        save it -- at the first update, on a paid GPU, not at rollout.
        Cloning is not a fix: inside an inference_mode context .clone()
        returns another inference tensor.

        Pixels are cast to float but NOT rescaled to [0, 1]: the published
        artifact has Conv+BN fused, so no BatchNorm remains to absorb a
        different input scale and the features would be wrong with no error."""
        if frames.ndim != 4 or frames.shape[1:] != (1, FRAME_HEIGHT, FRAME_WIDTH):
            raise ValueError(
                f"frames has shape {tuple(frames.shape)}, expected "
                f"(N, 1, {FRAME_HEIGHT}, {FRAME_WIDTH})"
            )
        batch = (
            torch.from_numpy(frames)
            .to(self._device, non_blocking=True)
            .float()
            .to(memory_format=torch.channels_last)
        )
        return self._encoder(batch)
