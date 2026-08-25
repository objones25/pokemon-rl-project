import pytest
import torch
import torch.nn as nn

from contrastive_pretrain.model import EMBEDDING_DIM, build_encoder, build_projector


def test_build_encoder_returns_2048_dim() -> None:
    _, dim = build_encoder(pretrained=False)
    assert dim == EMBEDDING_DIM == 2048


def test_build_encoder_output_shape() -> None:
    encoder, dim = build_encoder(pretrained=False)
    x = torch.randint(0, 256, (2, 1, 144, 160), dtype=torch.uint8).float()

    out = encoder(x)

    assert out.shape == (2, dim)


def test_build_encoder_has_no_maxpool() -> None:
    encoder, _ = build_encoder(pretrained=False)
    assert isinstance(encoder.backbone.maxpool, nn.Identity)


def test_build_encoder_rejects_wrong_channel_count() -> None:
    encoder, _ = build_encoder(pretrained=False)
    x = torch.zeros(2, 3, 144, 160)

    with pytest.raises(ValueError, match="1-channel"):
        encoder(x)


def test_build_encoder_rejects_transposed_spatial_dims() -> None:
    """The backbone is fully convolutional and ends in an adaptive average
    pool, so a transposed (N, 1, W, H) input would run without error and
    silently produce wrong features -- the exact failure mode a downstream
    consumer trusting a (N, 1, 160, 144) docstring would have hit."""
    encoder, _ = build_encoder(pretrained=False)
    x = torch.zeros(2, 1, 160, 144)

    with pytest.raises(ValueError, match="144x160"):
        encoder(x)


def test_build_encoder_rejects_non_4d_input() -> None:
    encoder, _ = build_encoder(pretrained=False)

    with pytest.raises(ValueError, match="4-D"):
        encoder(torch.zeros(1, 144, 160))


def test_build_projector_output_shape() -> None:
    projector = build_projector()
    x = torch.randn(4, 2048)

    out = projector(x)

    assert out.shape == (4, 128)
