"""Frozen-encoder export/load -- the interface contract the PPO stage
depends on. This module owns: Conv+BN fusion (applied only at export,
never during training, since BatchNorm needs live batch stats while
training), packaging the backbone-only weights as safetensors + a JSON
config, pushing that to an HF Hub model repo with rate-limit-aware
retry, and load_frozen_encoder() -- the single function downstream
consumers call.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.utils.fusion as fusion
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from contrastive_pretrain.model import EMBEDDING_DIM, build_encoder
from hf_storage.client import HfClient, RealHfClient
from hf_storage.retry import rate_limit_aware_backoff, retry_with_backoff

_INPUT_SIZE = [160, 144]  # [W, H], matches the augmentation spec's convention
# The unambiguous tensor-shape statement of the same thing, per-sample NCHW
# (channels, height, width) -- _INPUT_SIZE's [W, H] ordering is the opposite
# of the tensor's, and a caller who mixes the two gets a transposed input.
_INPUT_SHAPE_NCHW = [1, 144, 160]
# Nothing in this pipeline normalizes pixels: frames are uint8 [0, 255] cast
# to float and fed straight to the backbone. During training BatchNorm makes
# the network scale-invariant, but the exported artifact has Conv+BN FUSED --
# there is no BatchNorm left to absorb a caller who feeds [0, 1] input, and
# the wrong scale produces wrong features with no error. Hence: stated in the
# published config, not just in a docstring.
_INPUT_SCALE = "uint8_0_255"


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


def export_frozen_encoder(encoder: nn.Module) -> tuple[bytes, bytes]:
    """`encoder` is the live, unfused training module -- this function
    deep-copies it before fusing, so the caller's training model is
    never mutated by an export call.

    `.contiguous()` is required here because the training-side encoder
    (contrastive_pretrain.train.run_training) is moved to
    memory_format=torch.channels_last for conv performance, which leaves
    its parameter tensors non-contiguous in the standard (NCHW) sense --
    safetensors refuses to save those directly."""
    fused = fuse_conv_bn_modules(copy.deepcopy(encoder).eval())
    state_dict = {name: tensor.contiguous() for name, tensor in fused.state_dict().items()}
    weights_bytes = safetensors_save(state_dict)
    config = {
        "embedding_dim": EMBEDDING_DIM,
        "stem": "no_maxpool",
        "input_channels": 1,
        "input_size": _INPUT_SIZE,
        "input_shape_nchw": _INPUT_SHAPE_NCHW,
        "input_scale": _INPUT_SCALE,
        "pretrained_init": True,
    }
    config_bytes = json.dumps(config).encode("utf-8")
    return weights_bytes, config_bytes


def push_frozen_encoder(
    client: HfClient,
    encoder: nn.Module,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    max_retries: int = 5,
    base_delay: float = 2.0,
    rate_limit_delay: float = 120.0,
    sleep_func: Callable[[float], None] = time.sleep,
) -> None:
    weights_bytes, config_bytes = export_frozen_encoder(encoder)
    stats_bytes = json.dumps(
        {"mean": latent_mean.tolist(), "std": latent_std.tolist()}
    ).encode("utf-8")

    backoff = rate_limit_aware_backoff(base_delay, rate_limit_delay)
    for data, path_in_repo in (
        (weights_bytes, "model.safetensors"),
        (config_bytes, "config.json"),
        (stats_bytes, "latent_stats.json"),
    ):
        retry_with_backoff(
            lambda data=data, path_in_repo=path_in_repo: client.upload_bytes(data, path_in_repo),
            max_retries=max_retries,
            base_delay=base_delay,
            sleep_func=sleep_func,
            backoff_seconds=backoff,
        )


def _load_frozen_encoder_from_client(client: HfClient) -> nn.Module:
    config_bytes = client.download_bytes("config.json")
    weights_bytes = client.download_bytes("model.safetensors")
    if config_bytes is None or weights_bytes is None:
        raise FileNotFoundError("model.safetensors or config.json missing from the target repo")
    config = json.loads(config_bytes)

    encoder, _ = build_encoder(pretrained=False)
    fused = fuse_conv_bn_modules(encoder.eval())
    fused.load_state_dict(safetensors_load(weights_bytes))
    for param in fused.parameters():
        param.requires_grad_(False)
    fused.eval()
    return fused


def load_frozen_encoder(repo_id: str, revision: str | None = None) -> nn.Module:
    """The PPO-facing entrypoint: downloads config.json + model.safetensors
    from `repo_id` (an HF Hub model repo) and returns a frozen, eval-mode
    module mapping (N, 1, 144, 160) grayscale input to (N, 2048) float
    features.

    Input contract (also published in the repo's config.json as
    `input_shape_nchw` / `input_scale`):
      - shape (N, 1, 144, 160) NCHW -- height 144, width 160. Passing a
        transposed (N, 1, 160, 144) tensor is a ValueError, not silently
        wrong features.
      - pixel scale uint8 [0, 255], cast to float. Do NOT rescale to
        [0, 1]: this artifact has Conv+BN fused, so no BatchNorm remains
        to absorb a different input scale, and the features would be
        wrong with no error raised.

    Raw, unnormalized output -- the affine normalization layer is the
    caller's job, using that repo's latent_stats.json."""
    if revision is not None:
        raise NotImplementedError(
            "revision pinning is not yet supported — hf_storage.client.HfClient has no revision parameter"
        )
    from huggingface_hub import HfApi

    client = RealHfClient(HfApi(), repo_id, repo_type="model")
    return _load_frozen_encoder_from_client(client)


def compute_latent_stats(
    encoder: nn.Module,
    rows: Iterable[dict],
    device: torch.device,
    max_examples: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`rows` yields dicts with an "original" (1, H, W) uint8 tensor --
    the same shape contrastive_pretrain.dataset.to_pair_transform
    produces. Restores the encoder's original train/eval mode on exit."""
    was_training = encoder.training
    encoder.eval()
    features = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            if i >= max_examples:
                break
            frame = row["original"].unsqueeze(0).to(device).float()
            features.append(encoder(frame).squeeze(0).cpu())
    encoder.train(was_training)

    stacked = torch.stack(features)
    return stacked.mean(dim=0), stacked.std(dim=0)
