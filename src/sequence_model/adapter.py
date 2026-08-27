"""Fuses one timestep into one token.

Layout: [normalized latent | aux_state | embed(prev_action) |
proj(prev_reward)] -> Linear -> d_model. One token per timestep, not three
interleaved: a 1024-step horizon then costs exactly 1024 KV positions
rather than 3072, and one forward yields that step's logits and value
directly.

Causality note: the token at position t carries s_t but a_{t-1} and
r_{t-1}, because that is exactly what is known when a_t must be chosen.

The latent normalizer's mean/std come from the frozen encoder repo's
latent_stats.json and are frozen buffers. PPO value networks are
hypersensitive to unscaled input variance, and raw contrastive latents
cause immediate policy collapse -- but adaptive (running) statistics
would put a second moving target under an objective that already moves."""

from __future__ import annotations

import torch
from torch import nn

from sequence_model.config import PolicyConfig


class InputAdapter(nn.Module):
    def __init__(
        self, config: PolicyConfig, latent_mean: torch.Tensor, latent_std: torch.Tensor
    ) -> None:
        super().__init__()
        if latent_mean.shape != (config.latent_dim,):
            raise ValueError(
                f"latent_mean has shape {tuple(latent_mean.shape)}, expected "
                f"({config.latent_dim},) to match config.latent_dim; a mismatched stats "
                "vector broadcasts silently instead of raising"
            )
        if latent_std.shape != (config.latent_dim,):
            raise ValueError(
                f"latent_std has shape {tuple(latent_std.shape)}, expected "
                f"({config.latent_dim},) to match config.latent_dim; a mismatched stats "
                "vector broadcasts silently instead of raising"
            )
        if not torch.all(latent_std > 0):
            raise ValueError(
                "latent_std must be strictly positive in every dimension; a dead encoder "
                "channel with std 0 divides by ~1e-6 and feeds ~1e6-scale inputs to the "
                "PPO value head"
            )
        self.config = config
        self.register_buffer("latent_mean", latent_mean)
        self.register_buffer("latent_std", latent_std)
        self.action_embed = nn.Embedding(config.episode_start_action + 1, config.action_embed_dim)
        self.reward_proj = nn.Linear(1, config.reward_feat_dim, bias=False)
        fused_dim = (
            config.latent_dim
            + config.aux_state_dim
            + config.action_embed_dim
            + config.reward_feat_dim
        )
        self.proj = nn.Linear(fused_dim, config.d_model, bias=False)

    # nn.Module.__getattr__ is typed as returning `Tensor | Module`, so a
    # registered buffer reads as possibly-a-Module at every use site. These
    # bare annotations are the documented way to type buffers: no assignment,
    # no runtime effect, and register_buffer still owns the actual values.
    latent_mean: torch.Tensor
    latent_std: torch.Tensor

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / (self.latent_std + 1e-6)

    def forward(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> torch.Tensor:
        """`latent` (..., latent_dim), `aux_state` (..., aux_state_dim),
        `prev_action` (...) int64 with config.episode_start_action for
        the first step of an episode, `prev_reward` (...) float."""
        fused = torch.cat(
            [
                self.normalize_latent(latent),
                aux_state,
                self.action_embed(prev_action),
                self.reward_proj(prev_reward.unsqueeze(-1)),
            ],
            dim=-1,
        )
        return self.proj(fused)
