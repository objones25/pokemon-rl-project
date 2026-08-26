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
