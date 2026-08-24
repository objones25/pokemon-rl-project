import torch
import torch.nn as nn

from contrastive_pretrain.encoder_io import fuse_conv_bn_modules
from contrastive_pretrain.model import build_encoder


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
