"""Measurements that must pass before a paid run starts.

Gate 1 asks torch directly which SDPA backends can serve the model's real
call, rather than reading kernel names out of a profile. Verified against
torch 2.13 by introspection: torch.nn.attention exposes
can_use_flash_attention(params, debug=False),
can_use_efficient_attention(params, debug=False), and
can_use_cudnn_attention(params, debug=False), and
torch.backends.cuda.SDPAParams takes seven positional arguments --
(query, key, value, attn_mask, dropout, is_causal, enable_gqa).
attention.py's forward_chunk calls F.scaled_dot_product_attention with
enable_gqa=True, so k and v are never expanded to query-head width -- the
gate must build SDPAParams with the same asymmetric shapes or it measures a
call the model never makes.

All three backends are probed, not just flash and efficient: reading torch
2.13's own dispatch source (aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
shows flash's general_constraints include check_for_attn_mask, which rejects
ANY explicit attn_mask outright -- and this model's causal + sliding-window +
episode-boundary mask can never collapse to is_causal alone. Efficient
attention's dense_constraints instantiate
check_batch_size_and_num_heads_dense<false /*supports_gqa*/> -- GQA support is
compiled out of that backend entirely, so enable_gqa=True cannot make it
usable no matter the shapes. Only cudnn's dense_constraints instantiate the
GQA-supporting template AND tolerate an explicit mask
(check_attn_mask_shape, not check_for_attn_mask) -- the one backend that can
actually serve this exact call. A gate that never asks about cudnn cannot
tell "restructure the model, nothing fused will ever work" from "the report
just never checked the one candidate that would have worked."

Gate 2 answers the environment spec's own open question -- whether 64 envs
is right for the per-step cost -- with a measured number. 64 PyBoy emulators
are 64 processes, so vCPU is the binding constraint here, not VRAM."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import torch
from torch.backends.cuda import (
    SDPAParams,
    can_use_cudnn_attention,
    can_use_efficient_attention,
    can_use_flash_attention,
)

from sequence_model.config import PolicyConfig

logger = logging.getLogger(__name__)


def sdpa_params_for(
    policy_config: PolicyConfig, minibatch_envs: int, seq_len: int, device: torch.device
) -> SDPAParams:
    """The model's real call shape. attention.py passes enable_gqa=True, so k
    and v keep n_kv_heads width and are never expanded to query-head width."""
    query = torch.zeros(
        minibatch_envs,
        policy_config.n_heads,
        seq_len,
        policy_config.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.zeros(
        minibatch_envs,
        policy_config.n_kv_heads,
        seq_len,
        policy_config.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    mask = torch.ones(minibatch_envs, 1, seq_len, seq_len, dtype=torch.bool, device=device)
    return SDPAParams(query, key, key.clone(), mask, 0.0, False, True)


def sdpa_backend_report(
    policy_config: PolicyConfig, minibatch_envs: int, seq_len: int, device: torch.device
) -> dict:
    """Gate 1. A materialized bool mask rules out FlashAttention, and GQA is
    compiled out of efficient attention entirely regardless of enable_gqa (see
    this module's docstring) -- cudnn is the one backend that can actually
    serve this exact call, so it must be probed too, not just the two backends
    the original design handoff knew about. If none of the three is usable,
    MATH would materialize roughly 537 MB of scores at (8, 8, 2048) in bf16,
    and the restructure decision surfaces here rather than after the money is
    spent. debug=True makes torch print its rejection reason for whichever
    backend is disqualified."""
    params = sdpa_params_for(policy_config, minibatch_envs, seq_len, device)
    report = {
        "flash": bool(can_use_flash_attention(params, debug=True)),
        "efficient": bool(can_use_efficient_attention(params, debug=True)),
        "cudnn": bool(can_use_cudnn_attention(params, debug=True)),
        "shapes": {
            "query": list(params.query.shape),
            "key": list(params.key.shape),
            "enable_gqa": params.enable_gqa,
        },
    }
    logger.info("sdpa_backend_report", extra=report)
    return report


def throughput_report(
    build_env: Callable[[int], tuple], n_envs_candidates: list[int], steps: int
) -> dict:
    """Gate 2. Answers the env spec's own open question -- whether 64 envs is
    right for the per-step cost -- with a number, and its measured iteration
    time sets checkpoint_every_updates.

    64 PyBoy workers are 64 processes, so vCPU is the binding constraint
    here, not VRAM. The env and its shared-memory frame buffer are closed
    (and the buffer unlinked) in a finally, or a failed gate at one env
    count leaks shared memory for the rest."""
    results: dict[str, float] = {}
    for n_envs in n_envs_candidates:
        vec_env, buffer = build_env(n_envs)
        try:
            vec_env.reset()
            actions = torch.zeros(n_envs, dtype=torch.int64).numpy()
            started = time.monotonic()
            for _ in range(steps):
                vec_env.step(actions)
            # Guarded, not a `steps >= 1` precondition: elapsed can still read
            # 0.0 with steps >= 1 on a coarse clock or a fast enough env, and
            # a zero denominator must not raise mid-gate.
            elapsed = max(time.monotonic() - started, 1e-9)
        finally:
            vec_env.close()
            buffer.close()
            buffer.unlink()
        results[f"env_steps_per_sec_at_{n_envs}"] = steps * n_envs / elapsed
    logger.info("throughput_report", extra=results)
    return results
