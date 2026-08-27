import pytest
import torch

from sequence_model.block import TransformerBlock
from sequence_model.config import PolicyConfig
from sequence_model.masks import build_chunk_mask
from sequence_model.rope import rope_tables


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )


def test_block_preserves_shape(tiny_config: PolicyConfig) -> None:
    torch.manual_seed(0)
    block = TransformerBlock(tiny_config)
    x = torch.randn(2, 5, 32)
    cos, sin = rope_tables(torch.arange(5).expand(2, 5), 8, 10000.0)
    mask = build_chunk_mask(torch.arange(5).expand(2, 5), torch.zeros(2, 5, dtype=torch.long), 8)

    out = block.forward_chunk(x, cos, sin, mask)

    assert tuple(out.shape) == (2, 5, 32)


def test_block_is_residual_so_zeroed_sublayers_return_the_input(
    tiny_config: PolicyConfig,
) -> None:
    """Pre-norm with both output projections zeroed must be the identity.
    A block that is not residual, or that norms the residual stream
    itself, fails this."""
    torch.manual_seed(0)
    block = TransformerBlock(tiny_config)
    torch.nn.init.zeros_(block.attention.o_proj.weight)
    torch.nn.init.zeros_(block.mlp.down_proj.weight)
    x = torch.randn(1, 5, 32)
    cos, sin = rope_tables(torch.arange(5).unsqueeze(0), 8, 10000.0)
    mask = build_chunk_mask(torch.arange(5).unsqueeze(0), torch.zeros(1, 5, dtype=torch.long), 8)

    out = block.forward_chunk(x, cos, sin, mask)

    assert (out - x).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_block_declares_its_four_norm_weights(tiny_config: PolicyConfig) -> None:
    block = TransformerBlock(tiny_config)

    names = sorted(n for n, _ in block.named_parameters() if n.endswith("_norm.weight"))

    assert names == [
        "attention.k_norm.weight",
        "attention.q_norm.weight",
        "attn_norm.weight",
        "mlp_norm.weight",
    ]
