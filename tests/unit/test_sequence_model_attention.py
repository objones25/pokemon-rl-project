import pytest
import torch
import torch.nn.functional as F

from sequence_model.attention import GroupedQueryAttention
from sequence_model.config import PolicyConfig
from sequence_model.masks import build_chunk_mask


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
    """enable_gqa=True must map KV head j to query heads [j*n_rep,
    (j+1)*n_rep). x.repeat() gives the same shape with heads
    interleaved -- a query-head permutation that trains fine and breaks
    checkpoint interop and fused kernels."""
    torch.manual_seed(0)
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    expected = F.scaled_dot_product_attention(
        q,
        k.unsqueeze(2).expand(1, 2, 2, 3, 8).reshape(1, 4, 3, 8),
        v.unsqueeze(2).expand(1, 2, 2, 3, 8).reshape(1, 4, 3, 8),
    )

    actual = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    assert (actual - expected).abs().max().item() == pytest.approx(0.0, abs=1e-6)


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


def _tables(seq_len: int, batch: int, config: PolicyConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper, not a test."""
    from sequence_model.rope import rope_tables

    return rope_tables(torch.arange(seq_len).expand(batch, seq_len), config.head_dim, config.rope_theta)
