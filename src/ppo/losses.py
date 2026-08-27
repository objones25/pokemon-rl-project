"""The clipped surrogate, the value loss, and the entropy bonus.

`logits` are raw -- log_softmax is applied here, and the caller must never
apply a softmax before handing them over.

Value clipping is off (clip_range_vf defaults to None), matching SB3.
SB3's own docstring says that clipping "depends on the reward scaling", and
the reward scale here is set by ReturnScaler at runtime rather than known in
advance."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ppo.config import PPOConfig


@dataclass(frozen=True)
class LossOutput:
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor
    clip_fraction: float
    approx_kl: float
    max_abs_ratio_dev: float


def ppo_losses(
    logits: torch.Tensor,
    value: torch.Tensor,
    action: torch.Tensor,
    logprob_old: torch.Tensor,
    advantage: torch.Tensor,
    value_target: torch.Tensor,
    config: PPOConfig,
) -> LossOutput:
    """All tensors are (B, T) except `logits`, which is (B, T, action_dim)."""
    log_probabilities = F.log_softmax(logits, dim=-1)
    logprob = log_probabilities.gather(-1, action.unsqueeze(-1)).squeeze(-1)

    log_ratio = logprob - logprob_old
    ratio = log_ratio.exp()

    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range) * advantage
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = F.mse_loss(value, value_target)
    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1).mean()

    total = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

    with torch.no_grad():
        # Schulman's low-variance estimator, the same one SB3 reports.
        approx_kl = float(((ratio - 1.0) - log_ratio).mean())
        clip_fraction = float(((ratio - 1.0).abs() > config.clip_range).float().mean())
        max_abs_ratio_dev = float((ratio - 1.0).abs().max())

    return LossOutput(
        policy=policy_loss,
        value=value_loss,
        entropy=entropy,
        total=total,
        clip_fraction=clip_fraction,
        approx_kl=approx_kl,
        max_abs_ratio_dev=max_abs_ratio_dev,
    )
