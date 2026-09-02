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

import logging
from dataclasses import dataclass

import torch

from ppo.buffer import RolloutBuffer
from ppo.config import PPOConfig
from ppo.gae import compute_gae
from ppo.losses import ppo_losses
from ppo.normalizer import ReturnScaler
from sequence_model.config import PolicyConfig

logger = logging.getLogger(__name__)


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
    # The last minibatch's norm overwrites grad_norm every iteration, so an
    # earlier spike that a later, smaller-gradient minibatch happens to
    # overwrite is invisible in grad_norm alone -- exactly the kind of
    # divergence signal train/grad_norm exists to catch. Tracked across every
    # non-skipped minibatch of the update, not just (epoch 1, minibatch 1).
    grad_norm_max: float
    # The trunk is shared between actor and critic, so one clipped global
    # gradient covers their sum. A large value-loss gradient can consume the
    # whole clip budget and shrink the policy gradient toward nothing -- which
    # presents as "the policy stopped learning" with no other symptom, and
    # grad_norm alone cannot show it. Measured once, at (epoch 1, minibatch 1).
    policy_grad_norm: float
    value_grad_norm: float
    # Whether a minibatch's own approx_kl exceeded 1.5x config.target_kl and
    # stopped the update before every epoch/minibatch ran. False whenever
    # target_kl is None (the default -- disabled).
    target_kl_triggered: bool
    # How many minibatches actually took an optimizer step, out of the
    # n_epochs * (n_envs // minibatch_envs) that a full update would run --
    # less than that whenever target_kl_triggered is True. Distinct from
    # skipped_minibatches, which counts non-finite-loss minibatches whose
    # step was skipped for a different reason (corrupt gradients, not an
    # intentional trust-region stop).
    minibatches_completed: int
    # approx_kl/clip_fraction above are the LAST-computed minibatch only --
    # which, whenever target_kl_triggered is True, can be a REJECTED step
    # never applied to the weights. These instead average every minibatch
    # that ran (finite loss, whether or not target_kl went on to reject
    # it), matching stable-baselines3's own convention exactly: SB3
    # appends each minibatch's approx_kl to its aggregate BEFORE checking
    # target_kl and breaking, so a rejected step still counts toward what
    # the update observed.
    approx_kl_mean: float
    clip_fraction_mean: float
    # Percentiles and the true max of |ratio-1|, POOLED across every
    # processed minibatch's full (B, T) tensor -- not just the last
    # minibatch's max (max_abs_ratio_dev above, unchanged for backward
    # compatibility). This is what actually distinguishes "a few outlier
    # tokens" from "broad drift across the batch": p50/p95 staying low
    # while p99/max spike is the outlier-token shape seen in every live
    # run so far; all four moving together would be broad drift instead.
    ratio_abs_dev_p50: float
    ratio_abs_dev_p95: float
    ratio_abs_dev_p99: float
    max_abs_ratio_dev_update: float
    # Mean of the per-position largest action probability, last-computed
    # minibatch -- same convention as loss/entropy itself. Entropy's
    # sharper companion: mean entropy can look moderate while most
    # individual states already have one action near-certain.
    max_action_prob: float
    # The advantage distribution BEFORE run_update's own in-place
    # normalization overwrites it (advantage = (advantage - mean) / std).
    # That normalization can make a single extreme transition survive as a
    # many-sigma outlier while simultaneously deflating everyone else's
    # std, so the raw, pre-normalization shape is what actually tests
    # whether a handful of transitions are driving an update's instability.
    raw_advantage_mean: float
    raw_advantage_std: float
    raw_advantage_abs_max: float
    # Fraction of total |advantage| mass from the single largest-magnitude
    # transition, and from the top 1% of transitions by magnitude --
    # directly answers "is this update dominated by a few outliers."
    raw_advantage_top1_frac: float
    raw_advantage_top1pct_frac: float


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
    # Shifted one slot forward, and that is not an off-by-one: the rollout
    # writes slot t as (observation o_t, action a_t sampled at o_t, reward
    # returned by the step that APPLIED a_{t-1}). So the reward paying for
    # slot t's own action is stored at slot t+1, and compute_gae's
    # delta_t = reward[t] + gamma*V[t+1] - V[t] needs exactly that one.
    # Reading buffer[:, trained] instead trains the critic to predict a reward
    # already collected before o_t -- irreducible noise it cannot infer -- and
    # enters each action's own reward into its advantage at weight gamma*lambda
    # instead of 1. Nothing crashes and explained_variance stays positive.
    reward = _gather(buffer, minibatches, "reward")[:, trained.start + 1 : trained.stop + 1]
    # value_old is in NORMALIZED units (the critic regresses onto
    # scaler.normalize(returns)) while reward is raw, so the critic's output is
    # multiplied back out before the two are mixed. scaler.update() runs after
    # this call, so scaler.scale here is still the scale the critic was
    # actually trained under -- keep that ordering. Left un-scaled, the GAE
    # fixed point degenerates to delta_t ~= r_t*(1 - 1/scale): the baseline
    # stops reducing variance and the effective value horizon collapses from
    # gamma to gamma*lambda.
    advantage, returns = compute_gae(
        reward, value_old * scaler.scale, episode_id, config.gamma, config.gae_lambda
    )

    # Captured BEFORE normalization overwrites `advantage` below: that
    # normalization can let a single extreme transition survive as a
    # many-sigma outlier while simultaneously deflating everyone else's
    # std toward it, so only the raw distribution actually shows whether a
    # handful of transitions dominate this update.
    raw_advantage_mean = float(advantage.mean())
    raw_advantage_std = float(advantage.std())
    abs_advantage = advantage.abs().flatten()
    raw_advantage_abs_max = float(abs_advantage.max())
    total_abs_advantage = float(abs_advantage.sum())
    if total_abs_advantage > 0.0:
        raw_advantage_top1_frac = raw_advantage_abs_max / total_abs_advantage
        top1pct_count = max(1, abs_advantage.numel() // 100)
        raw_advantage_top1pct_frac = (
            float(torch.topk(abs_advantage, top1pct_count).values.sum()) / total_abs_advantage
        )
    else:
        raw_advantage_top1_frac = 0.0
        raw_advantage_top1pct_frac = 0.0

    # Once, over the whole update batch -- not per minibatch. Per-minibatch
    # normalization would make each minibatch's targets depend on which envs
    # happened to land in it.
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    scaler.update(returns)
    value_target = scaler.normalize(returns)

    # Scaled-to-scaled, deliberately: value_old is what the critic emitted and
    # value_target is what it is regressed onto, so this measures the critic in
    # the units it is actually trained in.
    explained = _explained_variance(value_old[:, :-1], value_target)

    first_dev: float | None = None
    last = None
    skipped = 0
    completed = 0
    target_kl_triggered = False
    grad_norm = 0.0
    grad_norm_max = 0.0
    policy_grad_norm = 0.0
    value_grad_norm = 0.0
    # Every finite-loss minibatch's approx_kl/clip_fraction/abs_ratio_dev,
    # appended BEFORE the target_kl check below -- so a rejected minibatch
    # (never applied) still contributes to these aggregates, matching
    # SB3's own convention of recording what the update observed rather
    # than only what it applied.
    approx_kl_values: list[float] = []
    clip_fraction_values: list[float] = []
    abs_ratio_dev_chunks: list[torch.Tensor] = []
    for epoch in range(config.n_epochs):
        for index, envs in enumerate(minibatches):
            chunk = buffer.chunk(envs)
            rows = envs
            optimizer.zero_grad(set_to_none=True)
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
                # WARNING, with the reason: a silently dropped minibatch is the
                # trainer discarding real collected experience, and on an
                # unattended run the only trace of it would be a step count
                # that quietly failed to advance.
                logger.warning(
                    "nan_minibatch_skipped",
                    extra={
                        "epoch": epoch,
                        "minibatch": index,
                        "skipped_this_update": skipped,
                        "reason": "non-finite total loss; gradients would be corrupt",
                    },
                )
                if skipped >= config.max_nan_minibatches_per_update:
                    raise RuntimeError(
                        f"non-finite loss in {skipped} minibatches of one update; "
                        "aborting rather than stepping on corrupt gradients"
                    )
                continue

            approx_kl_values.append(loss.approx_kl)
            clip_fraction_values.append(loss.clip_fraction)
            abs_ratio_dev_chunks.append(loss.abs_ratio_dev)

            # Checked BEFORE backward()/step(), matching stable-baselines3:
            # a minibatch that would move the policy past the trust region
            # never has that move applied, not even partially. Stops the
            # WHOLE update (remaining minibatches of this epoch, and every
            # later epoch) rather than just this one minibatch -- an update
            # already trending this far off pi_old only gets worse over the
            # epochs/minibatches still to come, per every live run so far.
            if config.target_kl is not None and loss.approx_kl > 1.5 * config.target_kl:
                target_kl_triggered = True
                logger.warning(
                    "target_kl_early_stop",
                    extra={
                        "epoch": epoch,
                        "minibatch": index,
                        "approx_kl": loss.approx_kl,
                        "threshold": 1.5 * config.target_kl,
                        "minibatches_completed": completed,
                    },
                )
                break

            if epoch == 0 and index == 0:
                policy_grad_norm, value_grad_norm = _split_grad_norms(policy, loss, config)

            loss.total.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            )
            grad_norm_max = max(grad_norm_max, grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            completed += 1

        if target_kl_triggered:
            break

    assert first_dev is not None and last is not None  # n_epochs >= 1, so the loop always runs
    if approx_kl_values:
        approx_kl_mean = sum(approx_kl_values) / len(approx_kl_values)
        clip_fraction_mean = sum(clip_fraction_values) / len(clip_fraction_values)
        pooled_abs_ratio_dev = torch.cat([chunk.flatten() for chunk in abs_ratio_dev_chunks])
    else:
        # Every minibatch in this update was non-finite, short of the abort
        # threshold (max_nan_minibatches_per_update) -- degrade exactly the
        # way `last`/`first_dev` above already do: describe the last thing
        # actually computed rather than crash on an empty pool.
        approx_kl_mean = last.approx_kl
        clip_fraction_mean = last.clip_fraction
        pooled_abs_ratio_dev = last.abs_ratio_dev.flatten()
    return UpdateStats(
        policy_loss=float(last.policy.detach()),
        value_loss=float(last.value.detach()),
        entropy=float(last.entropy.detach()),
        total_loss=float(last.total.detach()),
        clip_fraction=last.clip_fraction,
        approx_kl=last.approx_kl,
        max_abs_ratio_dev_epoch1_mb1=first_dev,
        max_abs_ratio_dev=last.max_abs_ratio_dev,
        approx_kl_mean=approx_kl_mean,
        clip_fraction_mean=clip_fraction_mean,
        ratio_abs_dev_p50=float(torch.quantile(pooled_abs_ratio_dev, 0.50)),
        ratio_abs_dev_p95=float(torch.quantile(pooled_abs_ratio_dev, 0.95)),
        ratio_abs_dev_p99=float(torch.quantile(pooled_abs_ratio_dev, 0.99)),
        max_abs_ratio_dev_update=float(pooled_abs_ratio_dev.max()),
        max_action_prob=last.max_action_prob,
        raw_advantage_mean=raw_advantage_mean,
        raw_advantage_std=raw_advantage_std,
        raw_advantage_abs_max=raw_advantage_abs_max,
        raw_advantage_top1_frac=raw_advantage_top1_frac,
        raw_advantage_top1pct_frac=raw_advantage_top1pct_frac,
        explained_variance=explained,
        staleness_logprob_l1=staleness,
        skipped_minibatches=skipped,
        grad_norm=grad_norm,
        grad_norm_max=grad_norm_max,
        policy_grad_norm=policy_grad_norm,
        value_grad_norm=value_grad_norm,
        target_kl_triggered=target_kl_triggered,
        minibatches_completed=completed,
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
    with the concatenated pi_old tensors.

    Reads the field directly rather than through buffer.chunk(): a chunk
    materializes an fp32 copy of the latents (~134 MB per call at production
    shapes) that this function discards immediately, twice per update."""
    return torch.cat([buffer.field(field, envs) for envs in minibatches], dim=0)


def _explained_variance(value: torch.Tensor, target: torch.Tensor) -> float:
    """1 - Var(target - value) / Var(target). Zero means the critic is no
    better than predicting the mean; negative means worse."""
    variance = target.var()
    if float(variance) == 0.0:
        return 0.0
    return float(1.0 - (target - value).var() / variance)
