"""Architecture and sizing for the temporal sequence model, loaded from
configs/sequence_model.yaml via config_io.load_dataclass_config.

The defaults are the ones the design spec's arithmetic produced (22.6M
backbone parameters, 4.00 KiB/token KV cache, 256 MiB at 64 envs x 1024
context) -- changing d_model/n_layers/n_kv_heads without re-running
config_budget.py invalidates every memory number in that spec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config_io import load_dataclass_config


@dataclass(frozen=True)
class PolicyConfig:
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64
    n_kv_heads: int = 2
    d_ff: int = 1408
    context_len: int = 1024
    rope_theta: float = 1e4
    latent_dim: int = 2048
    aux_state_dim: int = 32
    action_dim: int = 7
    action_embed_dim: int = 32
    reward_feat_dim: int = 8
    qk_norm: bool = True
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} is not divisible by n_kv_heads={self.n_kv_heads}; "
                "GQA requires an integer number of query heads per KV head"
            )
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                f"d_model={self.d_model} != n_heads={self.n_heads} x head_dim={self.head_dim}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even; RoPE pairs it in halves")
        if self.context_len < 1:
            raise ValueError(
                f"context_len={self.context_len} must be >= 1; with context_len=0 the window "
                "term (q - k) < context_len is False even on the diagonal, producing a fully-"
                "masked row and all-NaN from softmax"
            )

    @property
    def n_rep(self) -> int:
        """Query heads per KV head."""
        return self.n_heads // self.n_kv_heads

    @property
    def episode_start_action(self) -> int:
        """Embedding row for 'no previous action'. Feeding a real action
        index at episode reset teaches the model a lie, so this gets its
        own row rather than reusing action 0."""
        return self.action_dim


def load_config(path: str | Path) -> PolicyConfig:
    return load_dataclass_config(PolicyConfig, path)
