"""One PPO update over the buffer's trained region.

Structure, in order:

  1. Recompute pi_old and V_old with a no_grad forward_chunk sweep over ALL
     envs, under the same autocast context as the training step.
  2. GAE off V_old, advantages normalized once over the whole batch.
  3. n_epochs x (n_envs / minibatch_envs) minibatches, one optimizer step each.

Step 1 is deliberately NOT fused into epoch 1. Fusing is only valid if epoch 1
takes no optimizer step until every minibatch has been seen, which forces
gradient accumulation and drops the update from 24 optimizer steps to 4. The
~1.7% a separate pass costs buys back the gradient-step count and an invariant
that holds by construction: at (epoch 1, minibatch 1) the policy has not
changed, so max|ratio - 1| is exactly 0."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from ppo.gae import compute_gae
from ppo.losses import ppo_losses
from ppo.normalizer import ReturnScaler
from sequence_model.config import PolicyConfig


@dataclass(frozen=True)
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    clip_fraction: float
    approx_kl: float
    max_abs_ratio_dev_epoch1_mb1: float
    max_abs_ratio_dev: float
    explained_variance: float
    staleness_logprob_l1: float
    skipped_minibatches: int
    grad_norm: float
    # The trunk is shared between actor and critic, so one clipped global
    # gradient covers their sum. A large value-loss gradient can consume the
    # whole clip budget and shrink the policy gradient toward nothing -- which
    # presents as "the policy stopped learning" with no other symptom, and
    # grad_norm alone cannot show it. Measured once, at (epoch 1, minibatch 1).
    policy_grad_norm: float
    value_grad_norm: float


def run_update(
    policy,
    optimizer,
    scheduler,
    buffer: RolloutBuffer,
    scaler: ReturnScaler,
    config: PPOConfig,
    policy_config: PolicyConfig,
    n_envs: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> UpdateStats:
    burn_in = buffer.burn_in
    trained = buffer.trained_slice
    env_order = torch.arange(n_envs, device=device)
    minibatches = env_order.split(config.minibatch_envs)

    # logprob_old / advantage / value_target below are each built by
    # concatenating one tensor per minibatch and then indexed as `tensor[rows]`
    # with `rows` an env-id tensor. That indexing is only correct because
    # split() partitions env_order into contiguous, order-preserving pieces,
    # so concatenation order equals env-id order. Pin it explicitly: if
    # minibatching ever stops doing that, every pi_old row silently mis-pairs
    # with the wrong env, with nothing else here to catch it.
    assert torch.equal(torch.cat(list(minibatches)), env_order), (
        "minibatch concatenation order must equal env-id order, or logprob_old/"
        "advantage/value_target rows silently mis-pair with the wrong env"
    )

    logprob_old, value_old, staleness = _recompute_old(
        policy, buffer, minibatches, burn_in, trained, autocast_dtype, device
    )

    episode_id = _gather(buffer, minibatches, "episode_id")[:, trained.start : trained.stop + 1]
    reward = _gather(buffer, minibatches, "reward")[:, trained]
    advantage, returns = compute_gae(
        reward, value_old, episode_id, config.gamma, config.gae_lambda
    )

    # Once, over the whole update batch -- not per minibatch. Per-minibatch
    # normalization would make each minibatch's targets depend on which envs
    # happened to land in it.
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    scaler.update(returns)
    value_target = scaler.normalize(returns)

    explained = _explained_variance(value_old[:, :-1], value_target)

    first_dev: float | None = None
    last = None
    skipped = 0
    grad_norm = 0.0
    policy_grad_norm = 0.0
    value_grad_norm = 0.0
    for epoch in range(config.n_epochs):
        for index, envs in enumerate(minibatches):
            chunk = buffer.chunk(envs)
            rows = envs
            with torch.autocast(device.type, dtype=autocast_dtype):
                output = policy.forward_chunk(
                    chunk.latent, chunk.aux_state, chunk.prev_action, chunk.prev_reward,
                    chunk.abs_pos, chunk.episode_id, burn_in,
                )
                loss = ppo_losses(
                    output.logits[:, : config.n_steps],
                    output.value[:, : config.n_steps],
                    chunk.action[:, trained],
                    logprob_old[rows],
                    advantage[rows],
                    value_target[rows],
                    config,
                )

            if epoch == 0 and index == 0:
                first_dev = loss.max_abs_ratio_dev
            # Tracked even when skipped below: if every minibatch in an
            # update is non-finite without crossing the abort threshold, the
            # returned stats must still describe the last thing computed
            # rather than crash on a stale None.
            last = loss

            if not torch.isfinite(loss.total):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if skipped >= config.max_nan_minibatches_per_update:
                    raise RuntimeError(
                        f"non-finite loss in {skipped} minibatches of one update; "
                        "aborting rather than stepping on corrupt gradients"
                    )
                continue

            optimizer.zero_grad(set_to_none=True)

            if epoch == 0 and index == 0:
                policy_grad_norm, value_grad_norm = _split_grad_norms(policy, loss, config)

            loss.total.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    assert first_dev is not None and last is not None  # n_epochs >= 1, so the loop always runs
    return UpdateStats(
        policy_loss=float(last.policy.detach()),
        value_loss=float(last.value.detach()),
        entropy=float(last.entropy.detach()),
        total_loss=float(last.total.detach()),
        clip_fraction=last.clip_fraction,
        approx_kl=last.approx_kl,
        max_abs_ratio_dev_epoch1_mb1=first_dev,
        max_abs_ratio_dev=last.max_abs_ratio_dev,
        explained_variance=explained,
        staleness_logprob_l1=staleness,
        skipped_minibatches=skipped,
        grad_norm=grad_norm,
        policy_grad_norm=policy_grad_norm,
        value_grad_norm=value_grad_norm,
    )


def _split_grad_norms(policy, loss, config: PPOConfig) -> tuple[float, float]:
    """Two torch.autograd.grad calls, each retaining the graph so the other
    call and the real backward() afterward can still use it. Neither call
    touches .grad -- that stays untouched until the real backward() below, so
    this cannot corrupt the accumulation the optimizer step depends on."""
    params = [p for p in policy.parameters() if p.requires_grad]
    policy_grads = torch.autograd.grad(loss.policy, params, retain_graph=True, allow_unused=True)
    value_grads = torch.autograd.grad(
        config.vf_coef * loss.value, params, retain_graph=True, allow_unused=True
    )
    return _global_norm(policy_grads), _global_norm(value_grads)


def _global_norm(grads: tuple[torch.Tensor | None, ...]) -> float:
    """allow_unused=True hands back None for parameters the loss component
    never touches (e.g. the critic head under the policy loss) -- skip those
    rather than treating a None gradient as zero-norm evidence about params
    that were never in this loss's graph."""
    squared = [g.detach().float().pow(2).sum() for g in grads if g is not None]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


@torch.no_grad()
def _recompute_old(
    policy, buffer, minibatches, burn_in, trained, autocast_dtype, device
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Returns (logprob_old (N, T), value_old (N, T+1), staleness_l1).

    value_old carries T+1 entries: the trained region plus the bootstrap slot
    forward_chunk emits beyond it."""
    logprobs, values, staleness = [], [], []
    for envs in minibatches:
        chunk = buffer.chunk(envs)
        # log_softmax + gather run INSIDE the same autocast block as
        # forward_chunk, mirroring ppo_losses's own op sequence exactly
        # (it never upcasts logits before log_softmax either). Upcasting
        # to float32 before log_softmax here, while ppo_losses' training-
        # step computation stays in the autocast dtype, rounds the two
        # paths differently -- enough to push the epoch-1/minibatch-1
        # ratio off exactly 1 even though the policy has not changed.
        with torch.autocast(device.type, dtype=autocast_dtype):
            output = policy.forward_chunk(
                chunk.latent, chunk.aux_state, chunk.prev_action, chunk.prev_reward,
                chunk.abs_pos, chunk.episode_id, burn_in,
            )
            log_probabilities = torch.log_softmax(output.logits, dim=-1)
            action = chunk.action[:, burn_in:]
            gathered = log_probabilities.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        gathered = gathered[:, : trained.stop - trained.start].float()
        logprobs.append(gathered)
        values.append(output.value.float())
        staleness.append((gathered - chunk.rollout_logprob[:, trained]).abs().mean())
    return (
        torch.cat(logprobs, dim=0),
        torch.cat(values, dim=0),
        float(torch.stack(staleness).mean()),
    )


def _gather(buffer: RolloutBuffer, minibatches, field: str) -> torch.Tensor:
    """Reassembles one buffer field in minibatch order, so its rows line up
    with the concatenated pi_old tensors."""
    return torch.cat([getattr(buffer.chunk(envs), field) for envs in minibatches], dim=0)


def _explained_variance(value: torch.Tensor, target: torch.Tensor) -> float:
    """1 - Var(target - value) / Var(target). Zero means the critic is no
    better than predicting the mean; negative means worse."""
    variance = target.var()
    if float(variance) == 0.0:
        return 0.0
    return float(1.0 - (target - value).var() / variance)
