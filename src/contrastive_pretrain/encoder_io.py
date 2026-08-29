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
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from torch import nn
from torch.nn.utils import fusion

from contrastive_pretrain.model import EMBEDDING_DIM, build_encoder
from hf_storage.client import AtomicHfClient, HfClient, RealHfClient
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


def fuse_conv_bn_modules[ModuleT: nn.Module](module: ModuleT) -> ModuleT:
    """Recursively fuses every adjacent (Conv2d, BatchNorm2d) pair in
    `module` in-place, for eval-mode inference only. Relies on
    torchvision's Bottleneck blocks (and this project's own
    GrayscaleResNetEncoder) declaring/iterating children in
    conv-then-bn order, which fuse_conv_bn_eval requires.

    Generic over its input type: this always mutates and returns the exact
    same object it was given, so preserving the caller's concrete type
    (e.g. GrayscaleResNetEncoder, or nn.Sequential for indexing) is both
    accurate and useful, not just a wider annotation."""
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


def _validate_latent_stats(mean: torch.Tensor, std: torch.Tensor) -> None:
    """Deliberately duplicates src/pokemon_env/encoder.py's load_latent_stats
    guard rather than importing it -- CLAUDE.md's codebase map keeps
    contrastive_pretrain and pokemon_env independent by design, and this
    invariant happening to be identical on both sides of that boundary is
    a coincidence, not a reason to couple the two sub-projects."""
    for name, tensor in (("latent_mean", mean), ("latent_std", std)):
        non_finite = torch.nonzero(~torch.isfinite(tensor)).flatten().tolist()
        if non_finite:
            raise ValueError(
                f"{name} has {len(non_finite)} non-finite entries at indices "
                f"{non_finite[:20]}{' ...' if len(non_finite) > 20 else ''}; refusing to publish"
            )
    non_positive = torch.nonzero(std <= 0).flatten().tolist()
    if non_positive:
        raise ValueError(
            f"latent_std has {len(non_positive)} non-positive entries at indices "
            f"{non_positive[:20]}{' ...' if len(non_positive) > 20 else ''}; refusing to publish. "
            "A dead or under-sampled encoder channel would divide by the 1e-6 floor "
            "downstream and feed ~1e6-scale inputs to the policy's value head."
        )


def push_frozen_encoder(
    client: AtomicHfClient,
    encoder: nn.Module,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    max_retries: int = 5,
    base_delay: float = 2.0,
    rate_limit_delay: float = 120.0,
    sleep_func: Callable[[float], None] = time.sleep,
) -> None:
    """Publishes weights, config, and latent stats as a single atomic Hub
    commit -- not three independent uploads. Three separate commits could
    leave the repo with weights but no config.json (or vice versa) on a
    mid-publish failure, which _load_frozen_encoder_from_client then hard-
    fails on; one commit either lands completely or not at all, and the
    retry below retries the whole publish as one unit."""
    _validate_latent_stats(latent_mean, latent_std)
    weights_bytes, config_bytes = export_frozen_encoder(encoder)
    stats_bytes = json.dumps(
        {"mean": latent_mean.tolist(), "std": latent_std.tolist()}
    ).encode("utf-8")
    files = {
        "model.safetensors": weights_bytes,
        "config.json": config_bytes,
        "latent_stats.json": stats_bytes,
    }

    backoff = rate_limit_aware_backoff(base_delay, rate_limit_delay)
    retry_with_backoff(
        lambda: client.upload_many_bytes(files, commit_message="Publish frozen encoder artifact"),
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
    # build_encoder(pretrained=False) below always constructs the same fixed
    # architecture regardless of what config.json says -- so a repo whose
    # config doesn't match what we're about to build must fail loudly here,
    # not be silently ignored. Checked, not just parsed.
    if config.get("input_shape_nchw") != _INPUT_SHAPE_NCHW or config.get("input_scale") != _INPUT_SCALE:
        raise ValueError(
            f"config.json contract mismatch: expected input_shape_nchw={_INPUT_SHAPE_NCHW}, "
            f"input_scale={_INPUT_SCALE!r}, got input_shape_nchw={config.get('input_shape_nchw')!r}, "
            f"input_scale={config.get('input_scale')!r}. This repo's artifact was not exported "
            "with a contract this loader understands."
        )

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
    caller's job, using that repo's latent_stats.json.

    `revision` pins the download to a resolved commit so a later push to
    `repo_id` cannot change the features underneath a running PPO agent --
    PPOConfig.frozen_encoder_revision requires exactly this. Passed straight
    through to RealHfClient; omit it only for exploratory/local use where
    `repo_id`'s branch head is acceptable."""
    from huggingface_hub import HfApi

    client = RealHfClient(HfApi(), repo_id, repo_type="model", revision=revision)
    return _load_frozen_encoder_from_client(client)


def compute_latent_stats(
    encoder: nn.Module,
    rows: Iterable[dict],
    device: torch.device,
    max_examples: int = 2000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`rows` yields dicts with an "original" (1, H, W) uint8 tensor --
    the same shape contrastive_pretrain.dataset.to_pair_transform
    produces. Restores the encoder's original train/eval mode on exit.

    Uses reservoir sampling (Algorithm R): visits the FULL `rows` stream
    once and produces a uniform random sample of size `max_examples`
    regardless of the stream's order. A take-first-N implementation over
    an unshuffled stream (build_val_dataset deliberately never shuffles,
    for compute_val_loss's resume-without-shuffle needs) can miss a real
    feature channel entirely if it only fires late in the stream -- this
    produced this project's 2026-08-28 nine-dead-latent-channel incident.
    `seed` makes the sample reproducible across a resumed run, matching
    how build_train_dataset already threads a seed through.

    Only the `max_examples` rows the reservoir ends up retaining are ever
    encoded, and in a single batched forward pass after the stream
    traversal finishes -- not one encoder forward per row of the (held-out
    videos can total 100k+ rows) stream. The reservoir holds raw uint8
    frame tensors during traversal (cheap), so visiting every row for
    representativeness no longer multiplies the encoder-forward cost,
    which is what actually dominates this function's runtime."""
    assert max_examples > 0, f"max_examples must be positive, got {max_examples}"

    was_training = encoder.training
    encoder.eval()
    generator = torch.Generator().manual_seed(seed)
    reservoir: list[torch.Tensor] = []
    for i, row in enumerate(rows):
        frame = row["original"]
        if len(reservoir) < max_examples:
            reservoir.append(frame)
        else:
            j = int(torch.randint(0, i + 1, (1,), generator=generator).item())
            if j < max_examples:
                reservoir[j] = frame

    if not reservoir:
        encoder.train(was_training)
        raise ValueError(
            "compute_latent_stats received an empty `rows` stream -- the held-out "
            "video set (config.val_video_ids) produced zero rows to sample from"
        )

    batch = torch.stack(reservoir).to(device).float()
    with torch.inference_mode():
        features = encoder(batch).cpu()
    encoder.train(was_training)

    assert features.shape[0] <= max_examples, "reservoir grew past its cap"
    mean, std = features.mean(dim=0), features.std(dim=0)
    assert mean.shape == std.shape == (features.shape[1],)
    return mean, std
