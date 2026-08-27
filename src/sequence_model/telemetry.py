"""Leading indicators for the sequence model.

Loss and grad norm are both late. These are not:

  attention_logit_max  Wortsman et al. found every run whose attention
                       logits passed 1e4 diverged. This is the mechanism
                       behind the value-loss spikes the architecture plan
                       worries about.
  residual_norm        Final-layer output norm predicted divergence 20-30%
                       of training ahead in Chameleon's runs.
  attention_distance_mass
                       The honest check on this whole component. A 1024-step
                       context is only worth its cost if attention actually
                       reaches past a few steps; if the mass never leaves
                       the "1" and "2-8" buckets, the design is wrong and
                       the histogram says so in hour one rather than week two.

These take already-computed tensors so they stay pure and CPU-testable.
SDPA does not return attention probabilities, so the caller computes them
explicitly on a sampled minibatch every N updates rather than on the hot
path."""

from __future__ import annotations

import math

import torch

# (label, inclusive lower bound, exclusive upper bound) over q_index - k_index.
DISTANCE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("past+self", -1024, 1),
    ("1", 1, 2),
    ("2-8", 2, 9),
    ("9-64", 9, 65),
    ("65-256", 65, 257),
    ("257-1024", 257, 1025),
)


def attention_logit_max(
    q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor
) -> float:
    """`q` (B, H, L, D), `k` (B, H_kv, L, D), `mask` broadcastable
    (B, 1, L, L) bool. Returns the largest scaled logit at an UNMASKED
    position -- masked positions are never attended to, so including them
    would raise false alarms on every step."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    n_rep = q.shape[1] // k.shape[1]
    k_expanded = k.repeat_interleave(n_rep, dim=1)
    scores = (q.float() @ k_expanded.float().transpose(-2, -1)) * scale
    return scores.masked_fill(~mask, float("-inf")).max().item()


def attention_distance_mass(weights: torch.Tensor) -> dict[str, float]:
    """`weights` (B, H, L, L), post-softmax. Returns the fraction of total
    attention mass falling in each distance bucket."""
    seq_len = weights.shape[-1]
    query_index = torch.arange(seq_len, device=weights.device).view(-1, 1)
    key_index = torch.arange(seq_len, device=weights.device).view(1, -1)
    distance = query_index - key_index
    total = weights.sum()

    mass: dict[str, float] = {}
    for label, low, high in DISTANCE_BUCKETS:
        in_bucket = (distance >= low) & (distance < high)
        mass[label] = (weights * in_bucket).sum().div(total).item()
    return mass


def residual_norm(hidden: torch.Tensor) -> float:
    """`hidden` (B, L, d_model). Mean per-token L2 norm of the residual
    stream leaving the final block."""
    return hidden.float().norm(dim=-1).mean().item()
