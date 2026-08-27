import math

import pytest
import torch
import torch.nn.functional as F

from sequence_model.attention import GroupedQueryAttention
from sequence_model.config import PolicyConfig
from sequence_model.masks import build_chunk_mask
from sequence_model.telemetry import attention_logit_max


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )


def test_forward_chunk_preserves_input_shape(tiny_config: PolicyConfig) -> None:
    torch.manual_seed(0)
    attn = GroupedQueryAttention(tiny_config)
    x = torch.randn(2, 5, 32)
    mask = build_chunk_mask(torch.arange(5).expand(2, 5), torch.zeros(2, 5, dtype=torch.long), 8)

    out = attn.forward_chunk(x, *_tables(5, 2, tiny_config), mask)

    assert tuple(out.shape) == (2, 5, 32)


def test_forward_chunk_is_causal(tiny_config: PolicyConfig) -> None:
    """Changing the last token must leave every earlier output untouched."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(tiny_config)
    x = torch.randn(1, 5, 32)
    mask = build_chunk_mask(torch.arange(5).unsqueeze(0), torch.zeros(1, 5, dtype=torch.long), 8)
    cos, sin = _tables(5, 1, tiny_config)
    changed = x.clone()
    changed[0, 4] = torch.randn(32)

    before = attn.forward_chunk(x, cos, sin, mask)
    after = attn.forward_chunk(changed, cos, sin, mask)

    assert (before[0, :4] - after[0, :4]).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_gqa_repeats_kv_head_j_to_query_heads_j_times_n_rep(tiny_config: PolicyConfig) -> None:
    """GroupedQueryAttention.forward_chunk (enable_gqa=True internally)
    must map KV head j to query heads [j*n_rep, (j+1)*n_rep). x.repeat()
    gives the same shape with heads interleaved instead -- a query-head
    permutation that trains fine from scratch and breaks checkpoint
    interop and fused kernels, which is a different bug class than a
    quality regression.

    Builds the reference by expand-reshape from the module's OWN
    projections (via the private _project), not a hand-rolled q/k/v, so
    this exercises this repo's class rather than a bare PyTorch property."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(tiny_config)
    x = torch.randn(1, 5, 32)
    mask = build_chunk_mask(torch.arange(5).unsqueeze(0), torch.zeros(1, 5, dtype=torch.long), 8)
    cos, sin = _tables(5, 1, tiny_config)
    q, k, v = attn._project(x, cos, sin)
    n_rep = tiny_config.n_heads // tiny_config.n_kv_heads

    expanded = F.scaled_dot_product_attention(
        q,
        k.unsqueeze(2).expand(1, 2, n_rep, 5, 8).reshape(1, 4, 5, 8),
        v.unsqueeze(2).expand(1, 2, n_rep, 5, 8).reshape(1, 4, 5, 8),
        attn_mask=mask,
    )
    expected = attn._merge_heads(expanded)
    interleaved = F.scaled_dot_product_attention(
        q,
        k.repeat(1, n_rep, 1, 1),
        v.repeat(1, n_rep, 1, 1),
        attn_mask=mask,
    )
    wrong = attn._merge_heads(interleaved)

    actual = attn.forward_chunk(x, cos, sin, mask)

    assert (actual - expected).abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert (actual - wrong).abs().max().item() > 1e-4


def test_qk_norm_parameters_exist_when_enabled(tiny_config: PolicyConfig) -> None:
    attn = GroupedQueryAttention(tiny_config)

    names = sorted(n for n, _ in attn.named_parameters() if "norm" in n)

    assert names == ["k_norm.weight", "q_norm.weight"]


def test_qk_norm_parameters_absent_when_disabled(tiny_config: PolicyConfig) -> None:
    from dataclasses import replace

    attn = GroupedQueryAttention(replace(tiny_config, qk_norm=False))

    assert [n for n, _ in attn.named_parameters() if "norm" in n] == []


def test_projections_have_no_biases(tiny_config: PolicyConfig) -> None:
    attn = GroupedQueryAttention(tiny_config)

    assert [n for n, _ in attn.named_parameters() if n.endswith("bias")] == []


def test_attention_diagnostics_probabilities_sum_to_one_along_unmasked_rows(
    tiny_config: PolicyConfig,
) -> None:
    torch.manual_seed(0)
    attn = GroupedQueryAttention(tiny_config)
    x = torch.randn(2, 5, 32)
    mask = build_chunk_mask(torch.arange(5).expand(2, 5), torch.zeros(2, 5, dtype=torch.long), 8)

    _, _, probabilities = attn.attention_diagnostics(x, *_tables(5, 2, tiny_config), mask)

    row_sums = probabilities.sum(dim=-1)
    assert row_sums.flatten().tolist() == pytest.approx([1.0] * row_sums.numel(), abs=1e-5)


def test_attention_diagnostics_gives_masked_positions_exactly_zero_probability(
    tiny_config: PolicyConfig,
) -> None:
    """Query 0 in a causal mask may attend only to key 0, so it must
    assign exactly zero probability to every later key."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(tiny_config)
    x = torch.randn(1, 5, 32)
    mask = build_chunk_mask(torch.arange(5).unsqueeze(0), torch.zeros(1, 5, dtype=torch.long), 8)

    _, _, probabilities = attn.attention_diagnostics(x, *_tables(5, 1, tiny_config), mask)

    future_mass = probabilities[0, :, 0, 1:].abs().max().item()
    assert future_mass == pytest.approx(0.0, abs=1e-8)


def test_attention_diagnostics_returns_post_rope_q_and_k(tiny_config: PolicyConfig) -> None:
    """attention_logit_max requires post-RoPE, post-QK-norm q/k; feeding it
    the pair attention_diagnostics returns must produce a finite result."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(tiny_config)
    x = torch.randn(1, 5, 32)
    mask = build_chunk_mask(torch.arange(5).unsqueeze(0), torch.zeros(1, 5, dtype=torch.long), 8)

    q, k, _ = attn.attention_diagnostics(x, *_tables(5, 1, tiny_config), mask)

    assert math.isfinite(attention_logit_max(q, k, mask))


def _tables(seq_len: int, batch: int, config: PolicyConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper, not a test."""
    from sequence_model.rope import rope_tables

    return rope_tables(torch.arange(seq_len).expand(batch, seq_len), config.head_dim, config.rope_theta)
