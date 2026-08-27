"""One rollout segment: env -> frozen encoder -> policy -> buffer.

Ordering here is load-bearing three times over, and every one of the three
produces correctly-shaped tensors when wrong:

  1. cache.reset(done) runs AFTER the step whose transition ended the episode,
     never before the next one -- reset it first and the terminal observation
     of every episode attends to a cleared cache.
  2. abs_pos is snapshotted BEFORE policy.step, which calls cache.advance()
     internally. Read it after and the buffer records the position of the
     NEXT token, and RoPE during the update silently disagrees with RoPE
     during the rollout.
  3. prev_action becomes episode_start_action and prev_reward becomes 0.0
     after a done, because autoreset is next-step: the action taken at the
     terminal step is meaningless as context for the fresh episode that
     arrives at t+1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from pokemon_env.vec_env import VecStep
from ppo.buffer import RolloutBuffer
from sequence_model.policy import StepOutput


class VecEnvProtocol(Protocol):
    """What collect_rollout needs from VecPokemonEnv."""

    def step(self, actions: np.ndarray) -> VecStep: ...


class EncoderProtocol(Protocol):
    """What collect_rollout needs from LatentEncoder."""

    def encode(self, frames: np.ndarray) -> torch.Tensor: ...


class CacheProtocol(Protocol):
    """What collect_rollout needs from RolloutCache."""

    abs_pos: torch.Tensor

    def reset(self, done: torch.Tensor) -> None: ...


class PolicyProtocol(Protocol):
    """What collect_rollout needs from RecurrentTransformerPolicy."""

    def step(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        cache: CacheProtocol,
    ) -> StepOutput: ...


@dataclass
class RolloutState:
    """What carries across a rollout boundary: the policy's context inputs
    for the next call to collect_rollout."""

    prev_action: torch.Tensor
    prev_reward: torch.Tensor


def collect_rollout(
    vec_env: VecEnvProtocol,
    encoder: EncoderProtocol,
    policy: PolicyProtocol,
    cache: CacheProtocol,
    buffer: RolloutBuffer,
    state: RolloutState,
    n_steps: int,
    generator: torch.Generator,
    device: torch.device,
    episode_start_action: int,
    autocast_dtype: torch.dtype,
) -> RolloutState:
    """Advances the env `n_steps` times, writing one buffer slot per step."""
    for _ in range(n_steps):
        step = vec_env.step(state.prev_action.detach().cpu().numpy().astype(np.int64))

        latent = encoder.encode(step.frames)
        aux = torch.from_numpy(step.aux).to(device)
        reward = torch.from_numpy(step.reward).to(device)
        done = torch.from_numpy(step.done).to(device)
        episode_id = torch.from_numpy(step.episode_id).to(device)

        # Snapshotted BEFORE policy.step: step() calls cache.advance(), so
        # reading abs_pos afterwards records the position of the NEXT token and
        # RoPE in forward_chunk would no longer match RoPE in the rollout.
        abs_pos = cache.abs_pos.clone()

        with torch.autocast(device.type, dtype=autocast_dtype):
            output = policy.step(latent, aux, state.prev_action, state.prev_reward, cache)

        log_probabilities = torch.log_softmax(output.logits.float(), dim=-1)
        action = torch.multinomial(
            log_probabilities.exp(), num_samples=1, generator=generator
        ).squeeze(-1)
        logprob = log_probabilities.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        buffer.write(
            slot=buffer.write_cursor,
            latent=latent,
            aux=aux,
            action=action,
            prev_action=state.prev_action,
            prev_reward=state.prev_reward,
            reward=reward,
            done=done,
            episode_id=episode_id,
            abs_pos=abs_pos,
            logprob=logprob,
            value=output.value.float(),
        )

        # AFTER the step whose transition ended the episode, never before the
        # next one -- the terminal observation must attend to its own episode.
        cache.reset(done)

        state.prev_action = torch.where(
            done, torch.full_like(action, episode_start_action), action
        )
        state.prev_reward = torch.where(done, torch.zeros_like(reward), reward)
    return state
