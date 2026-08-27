"""Checkpoint schema for the sequence-model half of a PPO run.

Scope is deliberately this package only: policy weights, the optimizer and
scheduler driving them, the per-env KV cache, and RNG. The rest of what a
resumable RL run needs -- PyBoy emulator state, the reward baselines from
the architecture plan's section 4, update counters -- belongs to the PPO
sub-project, which composes this dict into its own. See the "Handoff"
section of the temporal-sequence-model design spec.

Everything here is a pure function over injected dependencies. File I/O
(atomic write, discovery, retention) lives in checkpointing.io.

Two design points worth stating, because both were arrived at by arithmetic
rather than taste:

`cache` is optional. At production shape (8 layers x 64 envs x 2 kv_heads x
1024 context x 64 head_dim, K and V) the ring buffer is 256 MiB in bf16 --
comparable to the 284 MB of policy-plus-AdamW-moments beside it. It is also
only *meaningful* if the emulator state is saved too: a cache restored
against a freshly-booted env is memory of a game position the env no longer
occupies, which is worse than starting empty. That decision belongs to PPO,
so this module takes `RolloutCache | None` and does not make it.

Dropping the cache is not catastrophic if PPO chooses that. Each env then
runs with truncated context for at most `context_len` steps -- roughly one
update's worth of weaker value estimates. Note the cache is *already*
one-update stale in steady state, since it holds K/V computed under the
pre-update weights and is never recomputed; resuming empty is a larger
version of an error the loop tolerates continuously, not a new kind of one.
"""

from __future__ import annotations

import dataclasses

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy

_COMPILE_PREFIX = "_orig_mod."


def build_policy_checkpoint_state(
    update: int,
    global_step: int,
    policy: RecurrentTransformerPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    cache: RolloutCache | None,
    rng_state: dict | None,
) -> dict:
    """`policy` must be the raw, uncompiled module: a torch.compile wrapper's
    state_dict keys carry an `_orig_mod.` prefix that a freshly-constructed
    module on resume will not have.

    `scheduler` may be None -- PPO learning-rate annealing is a
    hyperparameter, not a structural given, and a constant-LR run should not
    have to construct a dummy scheduler to be checkpointable.

    The config travels as a plain dict, not the frozen dataclass:
    `torch.load(weights_only=True)` refuses arbitrary pickled classes, so a
    dataclass here would save without complaint and fail only on the resume
    attempt, hours into a paid run."""
    return {
        "update": update,
        "global_step": global_step,
        "config": dataclasses.asdict(policy.config),
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "cache": None if cache is None else cache_to_state(cache),
        "rng": rng_state,
    }


def restore_policy_checkpoint(
    policy: RecurrentTransformerPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    state: dict,
) -> None:
    """Restores in place. Construct `scheduler` against `optimizer` before
    calling, following PyTorch's documented order for optimizer+scheduler
    restoration.

    The latent normalization statistics need no special handling: they are
    registered buffers on the InputAdapter, so they ride in `policy` and come
    back with it. That is load-bearing rather than incidental -- an encoder's
    mean/std are part of what the policy was trained against, and silently
    resuming under different ones would shift every input the value head
    sees."""
    _reject_config_drift(state["config"], policy.config)
    _reject_compiled_keys(state["policy"])
    policy.load_state_dict(state["policy"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])


def _reject_config_drift(saved: dict, live: PolicyConfig) -> None:
    """The shape-changing mismatches (d_model, n_layers) already fail loudly
    inside load_state_dict. The dangerous ones are rope_theta and
    context_len: they change no tensor shape, so the load succeeds and the
    run silently continues as a different model."""
    current = dataclasses.asdict(live)
    differing = sorted(k for k in current if saved.get(k) != current[k])
    if differing:
        detail = ", ".join(f"{k}: {saved.get(k)!r} -> {current[k]!r}" for k in differing)
        raise ValueError(
            f"checkpoint was saved under a different PolicyConfig ({detail}). "
            "Resuming across a config change trains a different model from the "
            "one the optimizer moments describe; rebuild the policy with the "
            "checkpointed config, or start a new run."
        )


def _reject_compiled_keys(policy_state: dict) -> None:
    if any(key.startswith(_COMPILE_PREFIX) for key in policy_state):
        raise ValueError(
            f"policy state_dict keys are prefixed with '{_COMPILE_PREFIX}', so this "
            "checkpoint was saved from a torch.compile wrapper rather than the raw "
            "module. Save from the module you passed to torch.compile, not its "
            "return value."
        )


def cache_to_state(cache: RolloutCache) -> dict:
    """Flattens the ring buffer to plain tensors and ints so the checkpoint
    stays loadable under `weights_only=True`."""
    return {f.name: getattr(cache, f.name) for f in dataclasses.fields(RolloutCache)}


def rebuild_cache(state: dict, config: PolicyConfig) -> RolloutCache:
    """Validates the stored buffers against the live config before handing
    back something the rollout loop will index into every step.

    Device and dtype come from the stored tensors: dtype because the rollout
    runs bf16 under autocast and a silent float32 rebuild raises a dtype
    mismatch inside SDPA on the first post-resume step; device because
    `load_checkpoint`'s caller decides placement via `map_location`."""
    n_envs = int(state["write_pos"].shape[0])
    expected = (config.n_layers, n_envs, config.n_kv_heads, config.context_len, config.head_dim)
    if tuple(state["k"].shape) != expected:
        raise ValueError(
            f"cached K has shape {tuple(state['k'].shape)}, expected {expected} for this "
            f"config at n_envs={n_envs}. A resume that changed n_envs, n_layers, "
            "n_kv_heads, context_len, or head_dim cannot reuse the ring buffer."
        )
    # capacity is a plain int beside the buffers rather than derived from them,
    # so an internally inconsistent state is representable: write_pos would wrap
    # at the wrong index and silently overwrite live slots mid-episode. The
    # config comparison above cannot catch it -- both are checked against the
    # buffer, not against each other.
    if int(state["capacity"]) != int(state["k"].shape[3]):
        raise ValueError(
            f"cached capacity {int(state['capacity'])} disagrees with the ring buffer's "
            f"own length {int(state['k'].shape[3])}; write_pos would wrap at the wrong "
            "position and overwrite slots that are still in the attention window."
        )
    return RolloutCache(**state)


def capture_rng_state() -> dict:
    """CPU plus every visible CUDA device. Cheap (a few KB) and it is what
    makes "did the resume actually work" answerable by comparing samples
    rather than by eyeballing a loss curve."""
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict | None) -> list[str]:
    """Returns the devices whose state it applied, so a caller can log the
    gap rather than assume there was none. `None` -- the checkpoint carried
    no RNG state, since `rng_state` is optional -- returns an empty list
    rather than raising, so a resume path can pass `state["rng"]` straight
    through without branching on it.

    A CUDA state that cannot be applied is skipped rather than raised:
    inspecting a checkpoint from the GPU pod on a CPU-only laptop is a real
    workflow. It is not warned either -- `filterwarnings = error` makes a
    warning a crash, which is the opposite of the intent."""
    if state is None:
        return []
    applied = ["cpu"]
    torch.set_rng_state(state["cpu"])
    cuda_states = state["cuda"]
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
        applied.append("cuda")
    return applied
