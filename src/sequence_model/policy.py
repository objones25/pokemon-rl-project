"""The PPO-facing policy: a decoder-only transformer over frozen-CNN frame
latents, with an actor head and a critic head.

Two entry points, because PPO needs two things that a single forward
cannot serve:

  step()          KV-cached incremental decode, one token per env step.
  forward_chunk() burn-in-prefixed forward for the update.

step() is @torch.no_grad(), NOT @torch.inference_mode(). Rollout latents
become inputs to forward_chunk during the update, and a tensor created
under inference_mode raises "Inference tensors cannot be saved for
backward" the moment it enters an autograd graph -- at the first update,
not at rollout. no_grad tensors are ordinary tensors. The module has no
dropout and no BatchNorm, so train/eval mode is behaviorally irrelevant
here.

forward_chunk lets gradients flow THROUGH the burn-in prefix rather than
detaching it, departing from R2D2 deliberately: R2D2 detaches because an
RNN's alternative is unbounded backprop, while a 1024-window transformer's
backprop is already bounded. Detaching here would truncate a path that is
genuinely part of d(loss)/d(theta) -- a biased gradient, not a cheaper
equal one. The cost is 0.07 s/epoch against an 8.0 s rollout."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from sequence_model.adapter import InputAdapter
from sequence_model.block import TransformerBlock
from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig
from sequence_model.layers import RMSNorm
from sequence_model.masks import build_chunk_mask
from sequence_model.rope import rope_tables


@dataclass(frozen=True)
class StepOutput:
    logits: torch.Tensor  # (N, action_dim), raw
    value: torch.Tensor  # (N,)


@dataclass(frozen=True)
class ChunkOutput:
    logits: torch.Tensor  # (B, T, action_dim), raw -- gradient region only
    value: torch.Tensor  # (B, T)


class RecurrentTransformerPolicy(nn.Module):
    def __init__(
        self, config: PolicyConfig, latent_mean: torch.Tensor, latent_std: torch.Tensor
    ) -> None:
        super().__init__()
        self.config = config
        self.adapter = InputAdapter(config, latent_mean, latent_std)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.actor = nn.Linear(config.d_model, config.action_dim, bias=False)
        self.critic = nn.Linear(config.d_model, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        """N(0, 0.02) everywhere, then residual output projections scaled
        by 1/sqrt(2 * n_layers) to keep the residual stream's variance
        flat with depth. The heads are initialised last so the generic
        pass cannot clobber them: the actor's near-zero gain makes the
        initial policy close to uniform, which is standard PPO practice
        and stops the first update from chasing an arbitrary preference."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        residual_scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            with torch.no_grad():
                block.attention.o_proj.weight.mul_(residual_scale)
                block.mlp.down_proj.weight.mul_(residual_scale)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def new_cache(
        self, n_envs: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> RolloutCache:
        """`dtype` is the KV cache's storage dtype. The spec's 256 MiB
        figure for 64 envs x 1024 context is bf16 K+V; float32 doubles it
        to 512 MiB. float32 is the default because the CPU tests run
        there, and the PPO rollout loop passes torch.bfloat16 explicitly."""
        return RolloutCache.empty(self.config, n_envs, device, dtype=dtype)

    @torch.no_grad()
    def step(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        cache: RolloutCache,
    ) -> StepOutput:
        """One vectorized env step. `latent` (N, latent_dim), `aux_state`
        (N, aux_state_dim), `prev_action` (N,) int64, `prev_reward` (N,).
        Mutates `cache` in place and advances it by one position."""
        x = self.adapter(latent, aux_state, prev_action, prev_reward).unsqueeze(1)
        cos, sin = rope_tables(
            cache.abs_pos.unsqueeze(1), self.config.head_dim, self.config.rope_theta
        )
        for layer, block in enumerate(self.blocks):
            x = block.forward_step(x, cos, sin, cache, layer)
        cache.advance()

        hidden = self.final_norm(x).squeeze(1)
        return StepOutput(logits=self.actor(hidden), value=self.critic(hidden).squeeze(-1))

    def forward_chunk(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        abs_pos: torch.Tensor,
        episode_id: torch.Tensor,
        burn_in: int,
    ) -> ChunkOutput:
        """`L = burn_in + T` tokens in, the last T out. `abs_pos` carries
        the positions recorded during rollout so RoPE matches exactly."""
        x = self.adapter(latent, aux_state, prev_action, prev_reward)
        cos, sin = rope_tables(abs_pos, self.config.head_dim, self.config.rope_theta)
        mask = build_chunk_mask(abs_pos, episode_id, self.config.context_len)
        for block in self.blocks:
            x = block.forward_chunk(x, cos, sin, mask)

        hidden = self.final_norm(x)[:, burn_in:]
        return ChunkOutput(logits=self.actor(hidden), value=self.critic(hidden).squeeze(-1))
