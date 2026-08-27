import json

import numpy as np
import pytest
import torch
from torch import nn

from pokemon_env.encoder import LatentEncoder, load_latent_stats


class FakeStatsClient:
    """Hand-written fake typed against hf_storage.client.HfClient."""

    def __init__(self, payload: bytes | None) -> None:
        self._payload = payload
        self.requested: list[str] = []

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        raise AssertionError("the encoder never uploads")

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        self.requested.append(path_in_repo)
        return self._payload


def _stats_payload(mean: list[float], std: list[float]) -> bytes:
    """Helper, not a test."""
    return json.dumps({"mean": mean, "std": std}).encode()


class TinyEncoder(nn.Module):
    """Stands in for the frozen ResNet: same contract, 400x smaller.

    Seeds itself so the suite has no unseeded randomness -- a flaky failure
    you cannot reproduce is a failure you cannot fix."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.head = nn.Linear(144 * 160, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.flatten(1))


def test_load_latent_stats_returns_mean_and_std_tensors() -> None:
    client = FakeStatsClient(_stats_payload([0.0] * 2048, [1.0] * 2048))

    mean, std = load_latent_stats(client)

    assert (tuple(mean.shape), tuple(std.shape)) == ((2048,), (2048,))


def test_load_latent_stats_requests_the_documented_filename() -> None:
    client = FakeStatsClient(_stats_payload([0.0] * 2048, [1.0] * 2048))

    load_latent_stats(client)

    assert client.requested == ["latent_stats.json"]


def test_load_latent_stats_rejects_a_zero_standard_deviation() -> None:
    """A dead encoder channel with std 0 divides by InputAdapter's 1e-6 floor
    and feeds ~1e6-scale inputs to a value head the architecture plan calls
    hypersensitive to input scale."""
    std = [1.0] * 2048
    std[7] = 0.0
    client = FakeStatsClient(_stats_payload([0.0] * 2048, std))

    with pytest.raises(ValueError, match="latent_std has 1 non-positive"):
        load_latent_stats(client)


def test_load_latent_stats_rejects_a_wrong_length_vector() -> None:
    client = FakeStatsClient(_stats_payload([0.0] * 512, [1.0] * 512))

    with pytest.raises(ValueError, match="expected 2048"):
        load_latent_stats(client)


def test_load_latent_stats_raises_when_the_file_is_missing() -> None:
    client = FakeStatsClient(None)

    with pytest.raises(FileNotFoundError, match="latent_stats.json"):
        load_latent_stats(client)


def test_encode_returns_one_latent_row_per_frame() -> None:
    encoder = LatentEncoder(TinyEncoder(), torch.device("cpu"))
    frames = np.zeros((3, 1, 144, 160), dtype=np.uint8)

    latents = encoder.encode(frames)

    assert tuple(latents.shape) == (3, 2048)


def test_encode_output_is_not_an_inference_tensor() -> None:
    """THE test. Latents recorded at rollout become inputs to forward_chunk at
    the PPO update, and a tensor made under inference_mode raises 'Inference
    tensors cannot be saved for backward' the moment the adapter tries to save
    it -- at the first update, on a paid GPU. requires_grad is False either
    way, so is_inference() is the only check that distinguishes them."""
    encoder = LatentEncoder(TinyEncoder(), torch.device("cpu"))
    frames = np.zeros((2, 1, 144, 160), dtype=np.uint8)

    latents = encoder.encode(frames)

    assert latents.is_inference() is False


def test_encode_rejects_a_transposed_frame_batch() -> None:
    """GrayscaleResNetEncoder rejects (N, 1, 160, 144), but only after the
    frame has already crossed IPC. Catching it here names the real problem."""
    encoder = LatentEncoder(TinyEncoder(), torch.device("cpu"))

    with pytest.raises(ValueError, match=r"expected \(N, 1, 144, 160\)"):
        encoder.encode(np.zeros((2, 1, 160, 144), dtype=np.uint8))


def test_encode_does_not_rescale_pixels_to_unit_range() -> None:
    """The published artifact has Conv+BN fused, so no BatchNorm remains to
    absorb a different input scale; rescaling to [0,1] produces wrong features
    with no error raised. A constant-255 frame must reach the encoder as 255.

    Threshold is 20.0, not 1.0: with TinyEncoder's seeded init, feeding pixels
    as 255 produces max|latent| ~= 532, but even a buggy `/ 255.0` (feeding
    all-ones) still lands at ~2.1 -- comfortably above a threshold of 1.0 --
    because summing 23040 zero-mean weights against an all-ones input has
    order-1 variance on its own. 20.0 sits with an order-of-magnitude margin
    on both sides of the two deterministic (seed=0) outcomes."""
    encoder = LatentEncoder(TinyEncoder(), torch.device("cpu"))
    frames = np.full((1, 1, 144, 160), 255, dtype=np.uint8)

    latents = encoder.encode(frames)

    assert latents.abs().max().item() > 20.0


def test_load_latent_stats_rejects_a_nan_standard_deviation() -> None:
    """NaN <= 0 is False, so a NaN std sails past the positivity check, turns
    every normalized latent into NaN, and reaches the value head at the first
    update with nothing raised anywhere. This function's whole job is to stop
    exactly that."""
    std = [1.0] * 2048
    std[7] = float("nan")
    client = FakeStatsClient(_stats_payload([0.0] * 2048, std))

    with pytest.raises(ValueError, match="latent_std has 1 non-finite"):
        load_latent_stats(client)


def test_load_latent_stats_rejects_an_infinite_mean() -> None:
    mean = [0.0] * 2048
    mean[3] = float("inf")
    client = FakeStatsClient(_stats_payload(mean, [1.0] * 2048))

    with pytest.raises(ValueError, match="latent_mean has 1 non-finite"):
        load_latent_stats(client)
