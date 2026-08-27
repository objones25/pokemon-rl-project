"""The gates themselves are CUDA measurements; what is unit-testable is that
they ask the right question with the right shapes."""

from __future__ import annotations

import torch

from ppo.preflight import sdpa_backend_report, sdpa_params_for
from sequence_model.config import PolicyConfig


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

    assert (params.attn_mask.dtype, params.attn_mask.shape) == (torch.bool, (4, 1, 16, 16))


def test_the_report_names_every_candidate_backend() -> None:
    """A key-set-only assertion would also pass if the report hardcoded
    placeholder booleans or symmetric shapes instead of the real measurement --
    it wouldn't catch either bug. Pin the full content instead, including the
    asymmetric query/key widths that are this gate's entire point."""
    report = sdpa_backend_report(
        _policy_config(), minibatch_envs=4, seq_len=16, device=torch.device("cpu")
    )

    assert report == {
        "flash": False,
        "efficient": False,
        "shapes": {"query": [4, 8, 16, 16], "key": [4, 2, 16, 16], "enable_gqa": True},
    }
