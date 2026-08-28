"""The gates themselves are CUDA measurements; what is unit-testable is that
they ask the right question with the right shapes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ppo.preflight import sdpa_backend_report, sdpa_params_for, throughput_report
from sequence_model.config import PolicyConfig


class _RecordingVecEnv:
    """Fake `build_env`-supplied vec env. Appends to a list shared with its
    paired `_RecordingFrameBuffer` so a test can assert call order across
    both objects, not just within one. `fail_after` scripts a `step()`
    exception `N` calls in, to exercise the `finally` cleanup path."""

    def __init__(self, calls: list[str], n_envs: int, fail_after: int | None = None) -> None:
        self._calls = calls
        self._n_envs = n_envs
        self._fail_after = fail_after
        self._step_calls = 0

    def reset(self) -> None:
        self._calls.append("env.reset")

    def step(self, actions: np.ndarray) -> None:
        if self._fail_after is not None and self._step_calls >= self._fail_after:
            raise RuntimeError("pyboy worker crashed")
        self._step_calls += 1
        self._calls.append("env.step")

    def close(self) -> None:
        self._calls.append("env.close")


class _RecordingFrameBuffer:
    """Fake `build_env`-supplied frame buffer, sharing `calls` with its
    paired `_RecordingVecEnv`."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def close(self) -> None:
        self._calls.append("buffer.close")

    def unlink(self) -> None:
        self._calls.append("buffer.unlink")


def _recording_build_env(calls: list[str], fail_after: int | None = None):
    def build(n_envs: int) -> tuple[_RecordingVecEnv, _RecordingFrameBuffer]:
        return _RecordingVecEnv(calls, n_envs, fail_after), _RecordingFrameBuffer(calls)

    return build


def _policy_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=128,
        n_layers=2,
        n_heads=8,
        head_dim=16,
        n_kv_heads=2,
        d_ff=64,
        context_len=8,
        latent_dim=8,
        aux_state_dim=4,
    )


def test_the_query_has_query_head_width_and_the_key_has_kv_head_width() -> None:
    """attention.py calls SDPA with enable_gqa=True, so k and v are NOT
    expanded. A gate run with symmetric head counts measures a call the model
    never makes."""
    params = sdpa_params_for(
        _policy_config(), minibatch_envs=4, seq_len=16, device=torch.device("cpu")
    )

    assert (params.query.shape[1], params.key.shape[1]) == (8, 2)


def test_enable_gqa_is_set_on_the_params() -> None:
    """enable_gqa is itself an SDPAParams field and can disqualify a backend
    on its own."""
    params = sdpa_params_for(
        _policy_config(), minibatch_envs=4, seq_len=16, device=torch.device("cpu")
    )

    assert params.enable_gqa is True


def test_the_mask_is_a_bool_tensor_broadcastable_over_heads() -> None:
    params = sdpa_params_for(
        _policy_config(), minibatch_envs=4, seq_len=16, device=torch.device("cpu")
    )

    # SDPAParams.attn_mask is Tensor | None -- None is a legal SDPA call, just
    # not the one this model makes, and that distinction is what the gate exists
    # to measure.
    assert params.attn_mask is not None
    assert (params.attn_mask.dtype, params.attn_mask.shape) == (torch.bool, (4, 1, 16, 16))


def test_the_report_names_every_candidate_backend() -> None:
    """A key-set-only assertion would also pass if the report hardcoded
    placeholder booleans or symmetric shapes instead of the real measurement --
    it wouldn't catch either bug. Pin the full content instead, including the
    asymmetric query/key widths that are this gate's entire point.

    The (False, False, False) pin is empirically verified against torch 2.13
    on CPU: can_use_flash_attention, can_use_efficient_attention, and
    can_use_cudnn_attention all currently require a CUDA tensor. A future
    torch adding CPU support to any of the three would need this test
    revisited.

    cudnn is the one candidate that can actually serve this model's real
    call on real hardware: verified by reading torch 2.13's own dispatch
    source (aten/src/ATen/native/transformers/cuda/sdp_utils.cpp) --
    can_use_flash_attention's general_constraints include
    check_for_attn_mask, which rejects any non-null attn_mask outright, and
    can_use_mem_efficient_attention's dense_constraints instantiate
    check_batch_size_and_num_heads_dense<false /*supports_gqa*/>, so GQA is
    compiled out of that backend entirely regardless of enable_gqa. Only
    cudnn's dense_constraints instantiate the GQA-supporting template AND
    tolerate an explicit mask (check_attn_mask_shape, not
    check_for_attn_mask) -- the one combination this model's causal +
    sliding-window + episode-boundary mask actually needs. A report that
    never asks the question can't ever tell a real "gate 1 fails, restructure
    now" from "the report itself is incomplete"."""
    report = sdpa_backend_report(
        _policy_config(), minibatch_envs=4, seq_len=16, device=torch.device("cpu")
    )

    assert report == {
        "flash": False,
        "efficient": False,
        "cudnn": False,
        "shapes": {"query": [4, 8, 16, 16], "key": [4, 2, 16, 16], "enable_gqa": True},
    }


def test_throughput_report_closes_the_env_then_closes_then_unlinks_the_buffer() -> None:
    """The cleanup order is a binding constraint (close-then-unlink, or a
    failed gate leaks shared memory) and needs no real env to verify --
    build_env is injected exactly so this is fake-able."""
    calls: list[str] = []

    throughput_report(_recording_build_env(calls), n_envs_candidates=[4], steps=2)

    assert calls == [
        "env.reset",
        "env.step",
        "env.step",
        "env.close",
        "buffer.close",
        "buffer.unlink",
    ]


def test_throughput_report_still_cleans_up_when_a_step_raises() -> None:
    """A crashed worker mid-rollout must not skip cleanup, or the gate leaks
    the shared-memory frame buffer on every failed run."""
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="pyboy worker crashed"):
        throughput_report(_recording_build_env(calls, fail_after=0), n_envs_candidates=[4], steps=2)

    assert calls == ["env.reset", "env.close", "buffer.close", "buffer.unlink"]
