"""Running scale for value targets.

Rewards are clipped to [0, 1] per step and gamma is 0.997, so an undiscounted
horizon puts value targets in the tens -- against a critic the architecture
plan calls hypersensitive to input scale. Dividing by a running std of the
return keeps the target near unit scale for the whole run.

No mean shift, deliberately: subtracting a mean would change the sign of an
advantage computed against the unshifted value, and sign is the only part of
the advantage the policy gradient cannot recover from being wrong."""

from __future__ import annotations

import torch

EPSILON = 1e-8


class ReturnScaler:
    def __init__(self, gamma: float) -> None:
        self._gamma = gamma
        self._count = 0.0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def scale(self) -> float:
        """Starts at 1.0 so the first update is unscaled rather than divided by
        a variance estimated from nothing."""
        if self._count < 2:
            return 1.0
        return float(max((self._m2 / self._count) ** 0.5, EPSILON))

    def update(self, returns: torch.Tensor) -> None:
        """Chan et al.'s parallel variance update, so a whole update's returns
        fold in with one pass and no history is retained."""
        batch = returns.detach().flatten().float()
        batch_count = float(batch.numel())
        if batch_count == 0:
            return
        batch_mean = float(batch.mean())
        batch_m2 = float(((batch - batch_mean) ** 2).sum())

        delta = batch_mean - self._mean
        total = self._count + batch_count
        self._m2 += batch_m2 + delta * delta * self._count * batch_count / total
        self._mean += delta * batch_count / total
        self._count = total

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        return returns / self.scale

    def state_dict(self) -> dict:
        return {"count": self._count, "mean": self._mean, "m2": self._m2}

    def load_state_dict(self, state: dict) -> None:
        self._count = float(state["count"])
        self._mean = float(state["mean"])
        self._m2 = float(state["m2"])
