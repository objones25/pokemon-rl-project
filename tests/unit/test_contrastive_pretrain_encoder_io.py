import json
from unittest.mock import patch

import pytest
import torch
from torch import nn

from contrastive_pretrain.encoder_io import (
    compute_latent_stats,
    export_frozen_encoder,
    fuse_conv_bn_modules,
    push_frozen_encoder,
)
from contrastive_pretrain.model import GrayscaleResNetEncoder, build_encoder


def test_fuse_conv_bn_modules_preserves_output() -> None:
    torch.manual_seed(0)
    module = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4))
    module.eval()
    bn = module[1]
    assert isinstance(bn, nn.BatchNorm2d)  # narrows for the buffer/param access below
    # weight/bias are only Optional because affine=False would drop them, and
    # running_mean/running_var only because track_running_stats=False would
    # drop them; this BatchNorm2d uses both defaults (True), so all four
    # always exist.
    assert bn.weight is not None
    assert bn.bias is not None
    assert bn.running_mean is not None
    assert bn.running_var is not None
    with torch.no_grad():
        # Non-trivial BN stats so fusion isn't testing a no-op identity case.
        bn.running_mean.copy_(torch.randn(4))
        bn.running_var.copy_(torch.rand(4) + 0.5)
        bn.weight.copy_(torch.randn(4))
        bn.bias.copy_(torch.randn(4))

    x = torch.randn(2, 1, 8, 8)
    before = module(x)
    fused = fuse_conv_bn_modules(module)
    after = fused(x)

    assert isinstance(fused[1], nn.Identity)
    assert torch.allclose(before, after, atol=1e-5)


def test_fuse_conv_bn_modules_on_real_encoder(encoder: nn.Module) -> None:
    encoder.eval()
    x = torch.rand(1, 1, 144, 160)

    with torch.no_grad():
        before = encoder(x)
        fused = fuse_conv_bn_modules(encoder)
        after = fused(x)

    assert torch.allclose(before, after, atol=1e-3)


class _FlakyThenWorksHfClient:
    """Fails the whole publish commit twice, then succeeds -- verifies
    push_frozen_encoder actually retries rather than propagating the first
    failure. A single attempt counter (not per-path) matches
    upload_many_bytes's one-commit-for-everything semantics: either the
    whole batch lands or none of it does, so there's no such thing as a
    partial per-file attempt count anymore."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.attempts = 0

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        # Never called by push_frozen_encoder (it only calls
        # upload_many_bytes) -- present only so this fake structurally
        # satisfies AtomicHfClient's base HfClient requirement.
        self.files[path_in_repo] = data

    def upload_many_bytes(self, files: dict[str, bytes], commit_message: str) -> None:
        self.attempts += 1
        if self.attempts < 3:
            raise RuntimeError("transient upload failure")
        self.files.update(files)

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.files.get(path_in_repo)


def test_export_frozen_encoder_round_trips_weights(encoder: nn.Module) -> None:
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
    # input_shape_nchw and input_scale are the unambiguous half of the
    # downstream (sequence-model/PPO) interface contract: input_size is
    # [W, H], the opposite order from the tensor's, and nothing in this
    # pipeline normalizes pixels -- but the exported artifact has Conv+BN
    # fused, so there is no BatchNorm left to absorb a caller who feeds
    # [0, 1] input instead of [0, 255].
    assert config == {
        "embedding_dim": 2048,
        "stem": "no_maxpool",
        "input_channels": 1,
        "input_size": [160, 144],
        "input_shape_nchw": [1, 144, 160],
        "input_scale": "uint8_0_255",
        "pretrained_init": True,
    }


def test_export_frozen_encoder_handles_channels_last_encoder(encoder: nn.Module) -> None:
    """Regression test: contrastive_pretrain.train.run_training calls
    encoder.to(device, memory_format=torch.channels_last) on the training
    encoder for conv performance -- that leaves parameter tensors
    non-contiguous in the standard sense, which used to make
    safetensors_save raise ValueError('non contiguous tensor')."""
    # nn.Module.to()'s @overload stubs omit memory_format -- see train.py's
    # matching comment; same stub gap, not a real type error.
    encoder.to(torch.device("cpu"), memory_format=torch.channels_last)  # type: ignore[call-overload]

    weights_bytes, _ = export_frozen_encoder(encoder)

    from safetensors.torch import load as safetensors_load

    reloaded = safetensors_load(weights_bytes)
    assert all(tensor.is_contiguous() for tensor in reloaded.values())


def test_export_frozen_encoder_does_not_mutate_input_module(encoder: nn.Module) -> None:
    assert isinstance(encoder, GrayscaleResNetEncoder)  # narrows for .backbone below

    export_frozen_encoder(encoder)

    # Fusion replaces BN layers with nn.Identity(), so if the original module were
    # fused in place, bn1 would be Identity. Verify it's still BatchNorm2d.
    assert isinstance(encoder.backbone.bn1, torch.nn.BatchNorm2d)


def test_export_frozen_encoder_does_not_strip_compiled_wrapper_prefix(encoder: nn.Module) -> None:
    """Pins a hazard, not a guarantee: export_frozen_encoder's deepcopy +
    fuse_conv_bn_modules path only fuses Conv2d/BatchNorm2d children in
    place -- it never inspects or renames a wrapping module's own
    attribute name, so a torch.compile-wrapped encoder's `_orig_mod.`
    state_dict prefix survives export untouched. checkpoint.py's
    build_checkpoint_state docstring is explicit that avoiding this is the
    CALLER's responsibility ("callers must never pass a torch.compile-
    wrapped model here"), not export_frozen_encoder's job to detect or
    strip. This test documents that contract: if train.py's
    push_frozen_encoder call site is ever mutated to pass
    compiled_encoder instead of the raw encoder, the published artifact's
    state_dict keys would carry `_orig_mod.` and load_frozen_encoder's
    plain (uncompiled) encoder would fail to load it -- silently breaking
    the downstream PPO consumer."""

    class _FakeCompiled(nn.Module):
        def __init__(self, mod: nn.Module) -> None:
            super().__init__()
            self._orig_mod = mod

        def forward(self, *args, **kwargs):
            return self._orig_mod(*args, **kwargs)

    weights_bytes, _ = export_frozen_encoder(_FakeCompiled(encoder))

    from safetensors.torch import load as safetensors_load

    keys = safetensors_load(weights_bytes).keys()
    assert all(k.startswith("_orig_mod.") for k in keys)


def test_push_frozen_encoder_uploads_three_files(fake_hf_client, encoder: nn.Module) -> None:
    push_frozen_encoder(
        fake_hf_client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048),
        sleep_func=lambda _: None,
    )

    assert set(fake_hf_client.upload_calls) == {"model.safetensors", "config.json", "latent_stats.json"}
    stats = json.loads(fake_hf_client.files["latent_stats.json"])
    assert len(stats["mean"]) == 2048
    assert len(stats["std"]) == 2048


def test_push_frozen_encoder_publishes_all_three_files_as_one_atomic_commit(
    fake_hf_client, encoder: nn.Module
) -> None:
    """Regression test for the non-atomic-publish fix: weights, config,
    and latent stats must land as a single Hub commit, not three
    independent ones -- otherwise a mid-publish failure could leave the
    repo with weights but no config.json, which
    _load_frozen_encoder_from_client then hard-fails on."""
    push_frozen_encoder(
        fake_hf_client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048),
        sleep_func=lambda _: None,
    )

    assert len(fake_hf_client.commits) == 1
    assert fake_hf_client.commits[0]["paths"] == ["config.json", "latent_stats.json", "model.safetensors"]


def test_push_frozen_encoder_retries_transient_upload_failures(encoder: nn.Module) -> None:
    client = _FlakyThenWorksHfClient()

    push_frozen_encoder(
        client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048),
        max_retries=5, base_delay=0.0, sleep_func=lambda _: None,
    )

    assert set(client.files.keys()) == {"model.safetensors", "config.json", "latent_stats.json"}


class _AlwaysRateLimitedClient:
    """Mirrors test_hf_uploader.py's _AlwaysRateLimitedClient precedent --
    a real observed HF Hub 429 message, not a generic RuntimeError, so
    push_frozen_encoder's rate_limit_aware_backoff actually has to match
    is_rate_limited's string check rather than just any failure."""

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        # Never called by push_frozen_encoder -- present only so this fake
        # structurally satisfies AtomicHfClient's base HfClient requirement.
        raise NotImplementedError

    def upload_many_bytes(self, files: dict[str, bytes], commit_message: str) -> None:
        raise RuntimeError(
            "429 Too Many Requests for url: ...\n"
            "You have exceeded the rate limit for repository commits (256 per hour)."
        )

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return None


def test_push_frozen_encoder_waits_the_dedicated_rate_limit_delay_not_the_short_schedule(
    encoder: nn.Module,
) -> None:
    """Regression test for the exact mechanism behind the documented HF
    rate-limit incident (objones25/pokemon-frames hit the 256
    commits/hour quota): a rate-limit error must select rate_limit_delay
    (minutes-scale), not the ordinary seconds-scale exponential backoff --
    retrying sooner just re-hits the same quota wall. Prior coverage only
    exercised a generic transient RuntimeError, which can't tell these two
    backoff schedules apart."""
    client = _AlwaysRateLimitedClient()
    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match="429"):
        push_frozen_encoder(
            client,
            encoder,
            latent_mean=torch.zeros(2048),
            latent_std=torch.ones(2048),
            max_retries=3,
            base_delay=1.0,
            rate_limit_delay=120.0,
            sleep_func=sleeps.append,
        )

    assert sleeps
    assert all(s == 120.0 for s in sleeps)


def test_load_frozen_encoder_from_client_matches_exported_weights(
    fake_hf_client, encoder: nn.Module
) -> None:
    from contrastive_pretrain.encoder_io import _load_frozen_encoder_from_client

    encoder.eval()
    x = torch.rand(1, 1, 144, 160)
    with torch.no_grad():
        expected = encoder(x)
    push_frozen_encoder(fake_hf_client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048))

    loaded = _load_frozen_encoder_from_client(fake_hf_client)

    with torch.no_grad():
        actual = loaded(x)
    assert torch.allclose(expected, actual, atol=1e-3)
    assert all(not p.requires_grad for p in loaded.parameters())
    assert loaded.training is False


def test_load_frozen_encoder_from_client_rejects_mismatched_config(
    fake_hf_client, encoder: nn.Module
) -> None:
    """The architecture build_encoder(pretrained=False) always produces is
    fixed regardless of what config.json says, so a repo whose published
    contract doesn't match must fail loudly here rather than be silently
    ignored -- a caller would otherwise get an encoder trained/exported
    under different shape/scale assumptions with no error."""
    from contrastive_pretrain.encoder_io import _load_frozen_encoder_from_client

    push_frozen_encoder(fake_hf_client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048))
    stored_config = json.loads(fake_hf_client.files["config.json"])
    stored_config["input_shape_nchw"] = [1, 160, 144]  # transposed -- a real mismatch
    fake_hf_client.files["config.json"] = json.dumps(stored_config).encode("utf-8")

    with pytest.raises(ValueError, match="contract mismatch"):
        _load_frozen_encoder_from_client(fake_hf_client)


def test_load_frozen_encoder_from_client_raises_when_files_missing(fake_hf_client) -> None:
    from contrastive_pretrain.encoder_io import _load_frozen_encoder_from_client

    with pytest.raises(FileNotFoundError, match="model.safetensors or config.json"):
        _load_frozen_encoder_from_client(fake_hf_client)


def test_compute_latent_stats_shapes(encoder_and_dim: tuple[nn.Module, int]) -> None:
    encoder, dim = encoder_and_dim
    rows = [
        {"original": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8)}
        for _ in range(5)
    ]

    mean, std = compute_latent_stats(encoder, rows, device=torch.device("cpu"), max_examples=5)

    assert mean.shape == (dim,)
    assert std.shape == (dim,)


def test_compute_latent_stats_truncates_at_max_examples(encoder_and_dim: tuple[nn.Module, int]) -> None:
    """A shape-only assertion can't distinguish truncation from processing
    all 10 rows -- mean/std are always shape (dim,) regardless of how many
    rows got averaged. Spying on encoder.forward (which compute_latent_stats
    reaches via `encoder(frame)` -> nn.Module.__call__ -> self.forward)
    gives a genuinely black-box count of rows actually processed."""
    encoder, dim = encoder_and_dim
    rows = [
        {"original": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8)}
        for _ in range(10)
    ]

    with patch.object(encoder, "forward", wraps=encoder.forward) as spy:
        mean, std = compute_latent_stats(encoder, rows, device=torch.device("cpu"), max_examples=3)

    assert spy.call_count == 3  # truncated at max_examples, not all 10 rows
    assert mean.shape == (dim,)
    assert std.shape == (dim,)


@pytest.mark.parametrize("was_training", [True, False])
def test_compute_latent_stats_restores_original_training_mode(
    encoder: nn.Module, was_training: bool
) -> None:
    encoder.train(was_training)
    rows = [{"original": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8)} for _ in range(2)]

    compute_latent_stats(encoder, rows, device=torch.device("cpu"), max_examples=2)

    assert encoder.training is was_training


def test_load_frozen_encoder_raises_on_revision_parameter() -> None:
    from contrastive_pretrain.encoder_io import load_frozen_encoder

    with pytest.raises(NotImplementedError, match="revision"):
        load_frozen_encoder("objones25/test-repo", revision="v1")
