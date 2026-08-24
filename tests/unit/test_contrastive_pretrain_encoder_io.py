import json

import pytest
import torch
import torch.nn as nn

from contrastive_pretrain.encoder_io import (
    compute_latent_stats,
    export_frozen_encoder,
    fuse_conv_bn_modules,
    push_frozen_encoder,
)
from contrastive_pretrain.model import build_encoder
from hf_storage.client import HfClient


def test_fuse_conv_bn_modules_preserves_output() -> None:
    torch.manual_seed(0)
    module = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4))
    module.eval()
    with torch.no_grad():
        # Non-trivial BN stats so fusion isn't testing a no-op identity case.
        module[1].running_mean.copy_(torch.randn(4))
        module[1].running_var.copy_(torch.rand(4) + 0.5)
        module[1].weight.copy_(torch.randn(4))
        module[1].bias.copy_(torch.randn(4))

    x = torch.randn(2, 1, 8, 8)
    before = module(x)
    fused = fuse_conv_bn_modules(module)
    after = fused(x)

    assert isinstance(fused[1], nn.Identity)
    assert torch.allclose(before, after, atol=1e-5)


def test_fuse_conv_bn_modules_on_real_encoder() -> None:
    encoder, _ = build_encoder(pretrained=False)
    encoder.eval()
    x = torch.rand(1, 1, 144, 160)

    with torch.no_grad():
        before = encoder(x)
        fused = fuse_conv_bn_modules(encoder)
        after = fused(x)

    assert torch.allclose(before, after, atol=1e-3)


class _FakeHfClient:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.upload_calls: list[str] = []

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.upload_calls.append(path_in_repo)
        self.files[path_in_repo] = data

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.files.get(path_in_repo)


class _FlakyThenWorksHfClient:
    """Fails upload_bytes twice, then succeeds -- verifies push_frozen_encoder
    actually retries rather than propagating the first failure."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.attempts: dict[str, int] = {}

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.attempts[path_in_repo] = self.attempts.get(path_in_repo, 0) + 1
        if self.attempts[path_in_repo] < 3:
            raise RuntimeError("transient upload failure")
        self.files[path_in_repo] = data

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.files.get(path_in_repo)


def test_export_frozen_encoder_round_trips_weights() -> None:
    encoder, _ = build_encoder(pretrained=False)
    encoder.eval()
    x = torch.rand(1, 1, 144, 160)
    with torch.no_grad():
        expected = encoder(x)

    weights_bytes, config_bytes = export_frozen_encoder(encoder)

    from safetensors.torch import load as safetensors_load

    reloaded_encoder, _ = build_encoder(pretrained=False)
    reloaded_encoder.eval()
    fuse_conv_bn_modules(reloaded_encoder)
    reloaded_encoder.load_state_dict(safetensors_load(weights_bytes))
    with torch.no_grad():
        actual = reloaded_encoder(x)

    assert torch.allclose(expected, actual, atol=1e-3)
    config = json.loads(config_bytes)
    assert config == {
        "embedding_dim": 2048,
        "stem": "no_maxpool",
        "input_channels": 1,
        "input_size": [160, 144],
        "pretrained_init": True,
    }


def test_export_frozen_encoder_does_not_mutate_input_module() -> None:
    encoder, _ = build_encoder(pretrained=False)

    export_frozen_encoder(encoder)

    # Fusion replaces BN layers with nn.Identity(), so if the original module were
    # fused in place, bn1 would be Identity. Verify it's still BatchNorm2d.
    assert isinstance(encoder.backbone.bn1, torch.nn.BatchNorm2d)


def test_push_frozen_encoder_uploads_three_files() -> None:
    encoder, _ = build_encoder(pretrained=False)
    client = _FakeHfClient()

    push_frozen_encoder(
        client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048),
        sleep_func=lambda _: None,
    )

    assert set(client.upload_calls) == {"model.safetensors", "config.json", "latent_stats.json"}
    stats = json.loads(client.files["latent_stats.json"])
    assert len(stats["mean"]) == 2048
    assert len(stats["std"]) == 2048


def test_push_frozen_encoder_retries_transient_upload_failures() -> None:
    encoder, _ = build_encoder(pretrained=False)
    client = _FlakyThenWorksHfClient()

    push_frozen_encoder(
        client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048),
        max_retries=5, base_delay=0.0, sleep_func=lambda _: None,
    )

    assert set(client.files.keys()) == {"model.safetensors", "config.json", "latent_stats.json"}


def test_load_frozen_encoder_from_client_matches_exported_weights() -> None:
    from contrastive_pretrain.encoder_io import _load_frozen_encoder_from_client

    encoder, _ = build_encoder(pretrained=False)
    encoder.eval()
    x = torch.rand(1, 1, 144, 160)
    with torch.no_grad():
        expected = encoder(x)
    client = _FakeHfClient()
    push_frozen_encoder(client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048))

    loaded = _load_frozen_encoder_from_client(client)

    with torch.no_grad():
        actual = loaded(x)
    assert torch.allclose(expected, actual, atol=1e-3)
    assert all(not p.requires_grad for p in loaded.parameters())
    assert loaded.training is False


def test_compute_latent_stats_shapes() -> None:
    encoder, dim = build_encoder(pretrained=False)
    rows = [
        {"original": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8)}
        for _ in range(5)
    ]

    mean, std = compute_latent_stats(encoder, rows, device=torch.device("cpu"), max_examples=5)

    assert mean.shape == (dim,)
    assert std.shape == (dim,)


def test_load_frozen_encoder_raises_on_revision_parameter() -> None:
    from contrastive_pretrain.encoder_io import load_frozen_encoder

    with pytest.raises(NotImplementedError):
        load_frozen_encoder("objones25/test-repo", revision="v1")
