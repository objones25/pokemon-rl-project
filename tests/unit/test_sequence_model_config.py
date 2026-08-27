from pathlib import Path

import pytest

from sequence_model.config import PolicyConfig, load_config


def test_default_config_head_dim_times_n_heads_equals_d_model() -> None:
    config = PolicyConfig()

    assert config.n_heads * config.head_dim == config.d_model


def test_default_config_n_rep_is_query_heads_per_kv_head() -> None:
    config = PolicyConfig()

    assert config.n_rep == 4


def test_load_config_overrides_only_named_fields(tmp_path: Path) -> None:
    path = tmp_path / "seq.yaml"
    path.write_text("n_layers: 3\nd_ff: 1600\n")

    config = load_config(path)

    assert (config.d_model, config.n_layers, config.d_ff, config.context_len) == (512, 3, 1600, 1024)


def test_load_config_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "seq.yaml"
    path.write_text("d_modle: 64\n")

    with pytest.raises(ValueError, match=r"unknown config field\(s\): \['d_modle'\]"):
        load_config(path)


def test_config_rejects_n_heads_not_divisible_by_n_kv_heads() -> None:
    with pytest.raises(ValueError, match="n_heads=8 is not divisible by n_kv_heads=3"):
        PolicyConfig(n_heads=8, n_kv_heads=3)


def test_config_rejects_d_model_not_equal_to_n_heads_times_head_dim() -> None:
    with pytest.raises(ValueError, match="d_model=512 != n_heads=8 x head_dim=32"):
        PolicyConfig(d_model=512, n_heads=8, head_dim=32)


def test_config_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError, match="head_dim=63 must be even"):
        PolicyConfig(d_model=63, n_heads=1, n_kv_heads=1, head_dim=63)


def test_config_rejects_zero_context_len() -> None:
    """context_len=0 makes the window term (q - k) < context_len False even
    on the diagonal, producing a fully-masked row and all-NaN from
    softmax."""
    with pytest.raises(ValueError, match="context_len=0 must be >= 1"):
        PolicyConfig(context_len=0)


def test_default_config_matches_the_full_production_defaults_contract() -> None:
    """Nine modules consume these fields and configs/sequence_model.yaml
    pins only 7 of 15, so a typo in any of the other 8 -- latent_dim,
    aux_state_dim, action_dim, action_embed_dim, reward_feat_dim, qk_norm,
    rms_norm_eps, rope_theta -- passes the suite silently without this."""
    config = PolicyConfig()

    non_float_fields = (
        config.d_model, config.n_layers, config.n_heads, config.head_dim,
        config.n_kv_heads, config.d_ff, config.context_len, config.latent_dim,
        config.aux_state_dim, config.action_dim, config.action_embed_dim,
        config.reward_feat_dim, config.qk_norm,
    )

    assert non_float_fields == (512, 8, 8, 64, 2, 1408, 1024, 2048, 32, 7, 32, 8, True)
    assert config.rope_theta == pytest.approx(1e4)
    assert config.rms_norm_eps == pytest.approx(1e-6)
