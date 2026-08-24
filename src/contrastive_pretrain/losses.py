"""Standard SimCLR NT-Xent (InfoNCE) loss: every other view in the batch
is an implicit negative for a given anchor -- no explicit negative
mining needed."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    n = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)  # (2N, D)
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature  # (2N, 2N)

    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    positive_idx = torch.arange(2 * n, device=z.device)
    positive_idx = (positive_idx + n) % (2 * n)

    return F.cross_entropy(sim, positive_idx)
