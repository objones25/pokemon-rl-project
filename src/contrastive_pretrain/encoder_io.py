"""Frozen-encoder export/load -- the interface contract the PPO stage
depends on. This module owns: Conv+BN fusion (applied only at export,
never during training, since BatchNorm needs live batch stats while
training), packaging the backbone-only weights as safetensors + a JSON
config, pushing that to an HF Hub model repo with rate-limit-aware
retry, and load_frozen_encoder() -- the single function downstream
consumers call.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.fusion as fusion


def fuse_conv_bn_modules(module: nn.Module) -> nn.Module:
    """Recursively fuses every adjacent (Conv2d, BatchNorm2d) pair in
    `module` in-place, for eval-mode inference only. Relies on
    torchvision's Bottleneck blocks (and this project's own
    GrayscaleResNetEncoder) declaring/iterating children in
    conv-then-bn order, which fuse_conv_bn_eval requires."""
    module.eval()
    for child in module.children():
        fuse_conv_bn_modules(child)

    children = list(module.named_children())
    for i in range(len(children) - 1):
        name_a, mod_a = children[i]
        name_b, mod_b = children[i + 1]
        if isinstance(mod_a, nn.Conv2d) and isinstance(mod_b, nn.BatchNorm2d):
            fused_conv = fusion.fuse_conv_bn_eval(mod_a, mod_b)
            setattr(module, name_a, fused_conv)
            setattr(module, name_b, nn.Identity())

    return module
