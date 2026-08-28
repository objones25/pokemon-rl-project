"""Rolling, GPU-resident storage for one PPO update's worth of transitions.

Stores fp16 latents rather than frames: 4,096 B/step against 23,040 B/step,
and the frozen CNN makes latents deterministic so there is nothing to
recompute. At 64 envs x 2048 slots x 2048 dims that is 537 MB.

Layout per env, in slot indices:

    [0, burn_in)                    burn-in prefix, no loss
    [burn_in, burn_in + n_steps)    trained region
    burn_in + n_steps               bootstrap slot, supplies V(s_T) only

shift() advances by exactly n_steps, so the previous update's bootstrap
observation becomes the new region's first trained slot and every observation
is trained exactly once."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ppo.config import PPOConfig
from sequence_model.config import PolicyConfig


@dataclass(frozen=True)
class ChunkInputs:
    """Exactly the arguments forward_chunk takes, plus the fields the loss and
    GAE need. Grouped so the update never assembles them by hand twice."""

    latent: torch.Tensor  # (B, L, latent_dim) float32
    aux_state: torch.Tensor  # (B, L, aux_state_dim)
    prev_action: torch.Tensor  # (B, L) int64
    prev_reward: torch.Tensor  # (B, L)
    abs_pos: torch.Tensor  # (B, L) int64
    episode_id: torch.Tensor  # (B, L) int64
    action: torch.Tensor  # (B, L) int64
    reward: torch.Tensor  # (B, L)
    done: torch.Tensor  # (B, L) bool
    rollout_logprob: torch.Tensor  # (B, L) -- diagnostic only, never the ratio
    rollout_value: torch.Tensor  # (B, L) -- diagnostic only, never the baseline


class RolloutBuffer:
    def __init__(
        self,
        config: PPOConfig,
        policy_config: PolicyConfig,
        n_envs: int,
        device: torch.device,
    ) -> None:
        self._config = config
        self.burn_in = config.burn_in(policy_config.context_len)
        self.capacity = config.buffer_capacity(policy_config.context_len)
        self.n_envs = n_envs

        shape = (n_envs, self.capacity)
        # fp16, not bf16: latents are bounded encoder outputs and fp16's extra
        # mantissa costs nothing here, while the buffer is the single largest
        # allocation in the trainer.
        self._latent = torch.zeros(*shape, policy_config.latent_dim, dtype=torch.float16, device=device)
        self._aux = torch.zeros(*shape, policy_config.aux_state_dim, dtype=torch.float32, device=device)
        self._action = torch.zeros(*shape, dtype=torch.int64, device=device)
        self._prev_action = torch.zeros(*shape, dtype=torch.int64, device=device)
        self._prev_reward = torch.zeros(*shape, dtype=torch.float32, device=device)
        self._reward = torch.zeros(*shape, dtype=torch.float32, device=device)
        self._done = torch.zeros(*shape, dtype=torch.bool, device=device)
        self._episode_id = torch.zeros(*shape, dtype=torch.int64, device=device)
        self._abs_pos = torch.zeros(*shape, dtype=torch.int64, device=device)
        self._logprob = torch.zeros(*shape, dtype=torch.float32, device=device)
        self._value = torch.zeros(*shape, dtype=torch.float32, device=device)

        self.write_cursor = 0

    @property
    def trained_slice(self) -> slice:
        return slice(self.burn_in, self.burn_in + self._config.n_steps)

    def write(
        self,
        slot: int,
        latent: torch.Tensor,
        aux: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        episode_id: torch.Tensor,
        abs_pos: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._latent[:, slot] = latent.to(torch.float16)
        self._aux[:, slot] = aux
        self._action[:, slot] = action
        self._prev_action[:, slot] = prev_action
        self._prev_reward[:, slot] = prev_reward
        self._reward[:, slot] = reward
        self._done[:, slot] = done
        self._episode_id[:, slot] = episode_id
        self._abs_pos[:, slot] = abs_pos
        self._logprob[:, slot] = logprob
        self._value[:, slot] = value
        self.write_cursor = slot + 1

    def shift(self) -> None:
        """Drop the oldest n_steps slots. What remains is the new burn-in plus
        the observation that becomes the new region's first trained slot, so
        the next rollout collects exactly n_steps observations."""
        keep = self.capacity - self._config.n_steps
        for tensor in self._tensors():
            tensor[:, :keep] = tensor[:, self._config.n_steps :].clone()
        self.write_cursor = keep

    def chunk(self, env_indices: torch.Tensor) -> ChunkInputs:
        """One minibatch: a subset of ENVS at full length. Never a slice of
        time -- the burn-in binds the time axis."""
        return ChunkInputs(
            latent=self._latent[env_indices].float(),
            aux_state=self._aux[env_indices],
            prev_action=self._prev_action[env_indices],
            prev_reward=self._prev_reward[env_indices],
            abs_pos=self._abs_pos[env_indices],
            episode_id=self._episode_id[env_indices],
            action=self._action[env_indices],
            reward=self._reward[env_indices],
            done=self._done[env_indices],
            rollout_logprob=self._logprob[env_indices],
            rollout_value=self._value[env_indices],
        )

    def field(self, name: str, env_indices: torch.Tensor) -> torch.Tensor:
        """One ChunkInputs field for a subset of envs, without building the
        whole chunk. `chunk()` copies the fp16 latents up to fp32 -- 134 MB at
        production shapes -- which is pure waste for a caller that only wants
        `reward` or `episode_id`.

        Values come back in their STORED dtype, so `latent` is fp16 here where
        `chunk().latent` is fp32."""
        return {
            "latent": self._latent,
            "aux_state": self._aux,
            "prev_action": self._prev_action,
            "prev_reward": self._prev_reward,
            "abs_pos": self._abs_pos,
            "episode_id": self._episode_id,
            "action": self._action,
            "reward": self._reward,
            "done": self._done,
            "rollout_logprob": self._logprob,
            "rollout_value": self._value,
        }[name][env_indices]

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self._latent, self._aux, self._action, self._prev_action,
            self._prev_reward, self._reward, self._done, self._episode_id,
            self._abs_pos, self._logprob, self._value,
        )
