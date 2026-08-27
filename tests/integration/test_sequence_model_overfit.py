"""Overfit-one-batch gate for the sequence model policy.

CLAUDE.md's stated first move when a model trains but will not learn is to
overfit a single batch before touching hyperparameters, data, or
architecture. This test makes that check permanent: it collapses the search
space to model/loss/step, and it is marked `slow` so it stays deselected by
default.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sequence_model.config import PolicyConfig
from sequence_model.policy import ChunkOutput, RecurrentTransformerPolicy


@pytest.mark.slow
def test_tiny_policy_overfits_a_single_batch_to_near_zero_loss() -> None:
    """Behaviour-cloning proxy: 200 AdamW steps on ONE fixed batch of
    random targets must drive cross-entropy near zero. If this fails, the
    defect is in the model, the loss, or the step order -- not in
    hyperparameters, data, or architecture size."""
    torch.manual_seed(0)
    config = PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )
    policy = RecurrentTransformerPolicy(config, torch.zeros(16), torch.ones(16))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-5)
    batch = _fixed_batch()
    targets = torch.randint(0, 7, (2, 8))

    final_loss = _train_steps(policy, optimizer, batch, targets, steps=200)

    assert final_loss < 0.05


def _fixed_batch() -> tuple[torch.Tensor, ...]:
    """Helper, not a test."""
    return (
        torch.randn(2, 8, 16),
        torch.randn(2, 8, 4),
        torch.randint(0, 7, (2, 8)),
        torch.randn(2, 8),
        torch.arange(8).expand(2, 8),
        torch.zeros(2, 8, dtype=torch.long),
    )


def _train_steps(
    policy: RecurrentTransformerPolicy,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, ...],
    targets: torch.Tensor,
    steps: int,
) -> float:
    """Helper, not a test: zero_grad -> forward -> loss -> backward -> step."""
    latent, aux, action, reward, abs_pos, episode_id = batch
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        optimizer.zero_grad()
        out: ChunkOutput = policy.forward_chunk(
            latent, aux, action, reward, abs_pos, episode_id, burn_in=0
        )
        loss = F.cross_entropy(out.logits.reshape(-1, 7), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
    return loss.item()
