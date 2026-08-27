"""Checkpoint/resume for the sequence-model half of a PPO run.

Every test here is CPU-only and seeded. The production concern they stand in
for is a paid unattended GPU run that must survive a pod restart: what these
assert is that a resumed policy is the *same* policy, and that the ways it
could silently not be are loud instead.
"""

import dataclasses

import pytest
import torch

from checkpointing.io import load_checkpoint, save_checkpoint
from sequence_model.cache import RolloutCache
from sequence_model.checkpoint import (
    build_policy_checkpoint_state,
    capture_rng_state,
    rebuild_cache,
    restore_policy_checkpoint,
    restore_rng_state,
)
from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        head_dim=8,
        n_kv_heads=2,
        d_ff=64,
        context_len=8,
        latent_dim=16,
        aux_state_dim=4,
        action_embed_dim=4,
        reward_feat_dim=2,
    )


@pytest.fixture
def make_policy(tiny_config: PolicyConfig):
    def _make(seed: int = 0) -> RecurrentTransformerPolicy:
        torch.manual_seed(seed)
        return RecurrentTransformerPolicy(tiny_config, torch.zeros(16), torch.ones(16))

    return _make


@pytest.fixture
def make_optimizer():
    def _make(policy: RecurrentTransformerPolicy):
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0 - step / 100
        )
        return optimizer, scheduler

    return _make


def _step_inputs(n_envs: int = 2) -> tuple[torch.Tensor, ...]:
    """Helper, not a test: one deterministic env step's worth of inputs."""
    generator = torch.Generator().manual_seed(99)
    return (
        torch.randn(n_envs, 16, generator=generator),
        torch.randn(n_envs, 4, generator=generator),
        torch.full((n_envs,), 7, dtype=torch.long),
        torch.zeros(n_envs),
    )


def _take_one_optimizer_step(policy, optimizer) -> None:
    """Helper, not a test: puts real moments in the optimizer so restoring
    them is distinguishable from constructing a fresh optimizer."""
    latent, aux, action, reward = _step_inputs()
    out = policy.forward_chunk(
        latent.unsqueeze(1),
        aux.unsqueeze(1),
        action.unsqueeze(1),
        reward.unsqueeze(1),
        abs_pos=torch.zeros(2, 1, dtype=torch.long),
        episode_id=torch.zeros(2, 1, dtype=torch.long),
        burn_in=0,
    )
    optimizer.zero_grad()
    (out.logits.sum() + out.value.sum()).backward()
    optimizer.step()


def _anneal(policy, optimizer, scheduler, steps: int) -> None:
    """Helper, not a test: drives the schedule forward `steps` times in the
    documented order. scheduler.step() ahead of optimizer.step() emits a
    UserWarning, which filterwarnings = error turns into a failure."""
    for _ in range(steps):
        _take_one_optimizer_step(policy, optimizer)
        scheduler.step()


def _run_rollout(policy, cache, steps: int) -> None:
    """Helper, not a test: advances the ring buffer `steps` positions so the
    cache under test holds real K/V rather than its zero-initialized state."""
    for _ in range(steps):
        policy.step(*_step_inputs(), cache)


# --- the load-bearing round trip ------------------------------------------


def test_restored_policy_produces_identical_logits_to_the_saved_one(
    make_policy, make_optimizer
) -> None:
    """THE test. A differently-seeded fresh policy restored from the
    checkpoint must step() to bit-identical logits. Catches a dropped
    parameter group, a buffer that never got registered, and a state_dict
    that silently loaded nothing."""
    saved = make_policy(seed=0)
    optimizer, scheduler = make_optimizer(saved)
    state = build_policy_checkpoint_state(
        update=5,
        global_step=5120,
        policy=saved,
        optimizer=optimizer,
        scheduler=scheduler,
        cache=None,
        rng_state=None,
    )
    expected = saved.step(
        *_step_inputs(), saved.new_cache(2, torch.device("cpu"))
    ).logits

    fresh = make_policy(seed=1234)
    fresh_optimizer, fresh_scheduler = make_optimizer(fresh)
    restore_policy_checkpoint(fresh, fresh_optimizer, fresh_scheduler, state)
    restored = fresh.step(
        *_step_inputs(), fresh.new_cache(2, torch.device("cpu"))
    ).logits

    assert torch.equal(restored, expected)


def test_a_differently_seeded_policy_differs_before_restore(make_policy) -> None:
    """Guards the test above: if two seeds produced the same weights, the
    round-trip assertion would pass against a restore that did nothing."""
    a = (
        make_policy(seed=0)
        .step(*_step_inputs(), make_policy(seed=0).new_cache(2, torch.device("cpu")))
        .logits
    )
    b = (
        make_policy(seed=1234)
        .step(*_step_inputs(), make_policy(seed=1234).new_cache(2, torch.device("cpu")))
        .logits
    )

    assert not torch.equal(a, b)


def test_restore_recovers_optimizer_moments(make_policy, make_optimizer) -> None:
    saved = make_policy(seed=0)
    optimizer, scheduler = make_optimizer(saved)
    _take_one_optimizer_step(saved, optimizer)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1024,
        policy=saved,
        optimizer=optimizer,
        scheduler=scheduler,
        cache=None,
        rng_state=None,
    )
    expected = optimizer.state[saved.actor.weight]["exp_avg"]

    fresh = make_policy(seed=1234)
    fresh_optimizer, fresh_scheduler = make_optimizer(fresh)
    restore_policy_checkpoint(fresh, fresh_optimizer, fresh_scheduler, state)

    assert torch.equal(fresh_optimizer.state[fresh.actor.weight]["exp_avg"], expected)


def test_restore_recovers_the_scheduler_learning_rate(
    make_policy, make_optimizer
) -> None:
    saved = make_policy(seed=0)
    optimizer, scheduler = make_optimizer(saved)
    _anneal(saved, optimizer, scheduler, steps=10)
    state = build_policy_checkpoint_state(
        update=10,
        global_step=10240,
        policy=saved,
        optimizer=optimizer,
        scheduler=scheduler,
        cache=None,
        rng_state=None,
    )

    fresh = make_policy(seed=1234)
    fresh_optimizer, fresh_scheduler = make_optimizer(fresh)
    restore_policy_checkpoint(fresh, fresh_optimizer, fresh_scheduler, state)

    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(0.0009)


def test_restore_recovers_the_scheduler_step_count(make_policy, make_optimizer) -> None:
    """The LR assertion above passes even if the scheduler state is dropped
    entirely, because optimizer.load_state_dict restores param_groups[0]["lr"]
    on its own. What only the scheduler carries is last_epoch -- so this
    asserts the LR after one FURTHER step, where a scheduler resumed at
    last_epoch=0 jumps back to 0.00099 instead of continuing to 0.00089."""
    saved = make_policy(seed=0)
    optimizer, scheduler = make_optimizer(saved)
    _anneal(saved, optimizer, scheduler, steps=10)
    state = build_policy_checkpoint_state(
        update=10, global_step=10240, policy=saved, optimizer=optimizer,
        scheduler=scheduler, cache=None, rng_state=None,
    )

    fresh = make_policy(seed=1234)
    fresh_optimizer, fresh_scheduler = make_optimizer(fresh)
    restore_policy_checkpoint(fresh, fresh_optimizer, fresh_scheduler, state)
    _anneal(fresh, fresh_optimizer, fresh_scheduler, steps=1)

    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(0.00089)


def test_scheduler_is_optional(make_policy, make_optimizer) -> None:
    """PPO LR annealing is a hyperparameter, not a structural given; a
    constant-LR run must not have to construct a dummy scheduler."""
    saved = make_policy(seed=0)
    optimizer, _ = make_optimizer(saved)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1024,
        policy=saved,
        optimizer=optimizer,
        scheduler=None,
        cache=None,
        rng_state=None,
    )

    fresh = make_policy(seed=1234)
    fresh_optimizer, _ = make_optimizer(fresh)
    restore_policy_checkpoint(fresh, fresh_optimizer, None, state)

    assert state["scheduler"] is None


def test_state_records_the_update_and_global_step(make_policy, make_optimizer) -> None:
    policy = make_policy(seed=0)
    optimizer, scheduler = make_optimizer(policy)

    state = build_policy_checkpoint_state(
        update=7,
        global_step=7168,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        cache=None,
        rng_state=None,
    )

    assert (state["update"], state["global_step"]) == (7, 7168)


# --- the silent-corruption guards -----------------------------------------


def test_restore_rejects_a_checkpoint_saved_under_a_different_config(
    make_policy, make_optimizer
) -> None:
    """rope_theta is the dangerous kind of mismatch: unlike d_model it
    changes no tensor shape, so load_state_dict accepts it happily and the
    run silently continues as a different model."""
    saved = make_policy(seed=0)
    optimizer, scheduler = make_optimizer(saved)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1024,
        policy=saved,
        optimizer=optimizer,
        scheduler=scheduler,
        cache=None,
        rng_state=None,
    )
    state["config"]["rope_theta"] = 500000.0

    fresh = make_policy(seed=1234)
    fresh_optimizer, fresh_scheduler = make_optimizer(fresh)

    with pytest.raises(ValueError, match="rope_theta"):
        restore_policy_checkpoint(fresh, fresh_optimizer, fresh_scheduler, state)


def test_config_mismatch_error_names_every_differing_field(
    make_policy, make_optimizer
) -> None:
    saved = make_policy(seed=0)
    optimizer, scheduler = make_optimizer(saved)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1024,
        policy=saved,
        optimizer=optimizer,
        scheduler=scheduler,
        cache=None,
        rng_state=None,
    )
    state["config"]["rope_theta"] = 500000.0
    state["config"]["context_len"] = 4096

    fresh = make_policy(seed=1234)
    fresh_optimizer, fresh_scheduler = make_optimizer(fresh)

    with pytest.raises(ValueError, match=r"context_len: 4096 -> 8.*rope_theta"):
        restore_policy_checkpoint(fresh, fresh_optimizer, fresh_scheduler, state)


def test_restore_rejects_a_compiled_wrapper_state_dict(
    make_policy, make_optimizer
) -> None:
    """torch.compile prefixes every key with `_orig_mod.`. Loading that into
    a raw module fails with a wall of missing/unexpected keys that reads like
    a corrupt file; say what actually happened instead."""
    saved = make_policy(seed=0)
    optimizer, scheduler = make_optimizer(saved)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1024,
        policy=saved,
        optimizer=optimizer,
        scheduler=scheduler,
        cache=None,
        rng_state=None,
    )
    state["policy"] = {f"_orig_mod.{k}": v for k, v in state["policy"].items()}

    fresh = make_policy(seed=1234)
    fresh_optimizer, fresh_scheduler = make_optimizer(fresh)

    with pytest.raises(ValueError, match="_orig_mod."):
        restore_policy_checkpoint(fresh, fresh_optimizer, fresh_scheduler, state)


def test_restore_recovers_the_latent_normalization_buffers(
    tiny_config, make_policy, make_optimizer
) -> None:
    """latent_mean/latent_std ride in the state_dict only because they are
    register_buffer. If anyone ever demotes them to plain attributes,
    normalization silently reverts to whatever the constructor was handed and
    nothing else in the suite notices."""
    torch.manual_seed(0)
    saved = RecurrentTransformerPolicy(
        tiny_config, torch.full((16,), 3.0), torch.full((16,), 7.0)
    )
    optimizer, scheduler = make_optimizer(saved)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1024,
        policy=saved,
        optimizer=optimizer,
        scheduler=scheduler,
        cache=None,
        rng_state=None,
    )

    fresh = make_policy(seed=1234)  # built with mean 0, std 1
    fresh_optimizer, fresh_scheduler = make_optimizer(fresh)
    restore_policy_checkpoint(fresh, fresh_optimizer, fresh_scheduler, state)

    assert torch.equal(fresh.adapter.latent_mean, torch.full((16,), 3.0))
    assert torch.equal(fresh.adapter.latent_std, torch.full((16,), 7.0))


def test_a_saved_state_loads_back_under_weights_only(
    tmp_path, make_policy, make_optimizer
) -> None:
    """torch.load(weights_only=True) refuses arbitrary pickled classes, so a
    checkpoint carrying a frozen PolicyConfig dataclass would save fine and
    fail only on the resume attempt -- hours into a paid run."""
    policy = make_policy(seed=0)
    optimizer, scheduler = make_optimizer(policy)
    cache = policy.new_cache(2, torch.device("cpu"))
    policy.step(*_step_inputs(), cache)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1024,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        cache=cache,
        rng_state=capture_rng_state(),
    )
    path = tmp_path / "policy_update00000001.pt"

    save_checkpoint(path, state)
    loaded = load_checkpoint(path)

    assert loaded["config"] == dataclasses.asdict(policy.config)


# --- the KV cache ---------------------------------------------------------


def test_a_restored_cache_continues_the_rollout_identically(
    make_policy, make_optimizer
) -> None:
    """The claim that makes saving the cache worth 256 MiB: step N+1 taken
    against a restored cache must match step N+1 taken against the live one."""
    policy = make_policy(seed=0)
    optimizer, scheduler = make_optimizer(policy)
    live = policy.new_cache(2, torch.device("cpu"))
    _run_rollout(policy, live, steps=3)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1024,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        cache=live,
        rng_state=None,
    )

    restored = rebuild_cache(state["cache"], policy.config)
    expected = policy.step(*_step_inputs(), live).logits
    actual = policy.step(*_step_inputs(), restored).logits

    assert torch.equal(actual, expected)


def test_cache_state_carries_every_field_of_the_dataclass(make_policy) -> None:
    """Drift guard. Adding a sixth field to RolloutCache without adding it
    here would silently drop it from every checkpoint, and the symptom would
    be a subtly wrong rollout after a resume -- not a crash."""
    policy = make_policy(seed=0)
    cache = policy.new_cache(2, torch.device("cpu"))
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1,
        policy=policy,
        optimizer=torch.optim.AdamW(policy.parameters()),
        scheduler=None,
        cache=cache,
        rng_state=None,
    )

    assert set(state["cache"]) == {f.name for f in dataclasses.fields(RolloutCache)}


def test_cache_is_none_when_the_caller_passes_none(make_policy) -> None:
    """A cache restored against a freshly-booted emulator is memory of a game
    position the env no longer occupies -- worse than no cache. Whether to
    save it is the PPO loop's call, so None must round-trip."""
    policy = make_policy(seed=0)

    state = build_policy_checkpoint_state(
        update=1,
        global_step=1,
        policy=policy,
        optimizer=torch.optim.AdamW(policy.parameters()),
        scheduler=None,
        cache=None,
        rng_state=None,
    )

    assert state["cache"] is None


def test_rebuild_cache_rejects_a_mismatched_env_count(make_policy) -> None:
    """n_envs is a launch flag. Resuming a 64-env checkpoint into a 32-env
    run would index a (64, ...) buffer with 32 env indices and silently
    train on the first half."""
    policy = make_policy(seed=0)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1,
        policy=policy,
        optimizer=torch.optim.AdamW(policy.parameters()),
        scheduler=None,
        cache=policy.new_cache(2, torch.device("cpu")),
        rng_state=None,
    )
    state["cache"]["write_pos"] = torch.zeros(4, dtype=torch.long)

    with pytest.raises(ValueError, match="n_envs"):
        rebuild_cache(state["cache"], policy.config)


def test_rebuild_cache_rejects_a_mismatched_context_len(
    make_policy, tiny_config
) -> None:
    policy = make_policy(seed=0)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1,
        policy=policy,
        optimizer=torch.optim.AdamW(policy.parameters()),
        scheduler=None,
        cache=policy.new_cache(2, torch.device("cpu")),
        rng_state=None,
    )
    wider = dataclasses.replace(tiny_config, context_len=16)

    with pytest.raises(ValueError, match="cannot reuse the ring buffer"):
        rebuild_cache(state["cache"], wider)


def test_rebuild_cache_rejects_a_capacity_that_disagrees_with_its_own_buffer(
    make_policy,
) -> None:
    """capacity is stored as a plain int beside the buffers rather than derived
    from them, so an internally inconsistent state is representable. The config
    check cannot catch it: both capacity and context_len are compared against
    the buffer, never against each other."""
    policy = make_policy(seed=0)
    state = build_policy_checkpoint_state(
        update=1, global_step=1, policy=policy,
        optimizer=torch.optim.AdamW(policy.parameters()),
        scheduler=None, cache=policy.new_cache(2, torch.device("cpu")), rng_state=None,
    )
    state["cache"]["capacity"] = 4

    with pytest.raises(ValueError, match="disagrees with the ring buffer"):
        rebuild_cache(state["cache"], policy.config)


def test_rebuild_cache_preserves_the_stored_dtype(make_policy) -> None:
    """The rollout runs the cache in bf16 under autocast. Silently rebuilding
    it as float32 would raise a dtype mismatch inside SDPA on the first
    post-resume step, on a paid GPU."""
    policy = make_policy(seed=0)
    state = build_policy_checkpoint_state(
        update=1,
        global_step=1,
        policy=policy,
        optimizer=torch.optim.AdamW(policy.parameters()),
        scheduler=None,
        cache=policy.new_cache(2, torch.device("cpu"), dtype=torch.bfloat16),
        rng_state=None,
    )

    restored = rebuild_cache(state["cache"], policy.config)

    assert restored.k.dtype == torch.bfloat16


# --- RNG ------------------------------------------------------------------


def test_restoring_rng_state_reproduces_the_next_sample() -> None:
    state = capture_rng_state()
    expected = torch.randn(4)

    restore_rng_state(state)

    assert torch.equal(torch.randn(4), expected)


def test_restore_rng_state_reports_the_devices_it_applied() -> None:
    """Inspecting a CUDA checkpoint on a CPU-only laptop is a real workflow,
    so an unappliable device state is reported, not raised and not warned
    (filterwarnings = error makes a warning the wrong signal here)."""
    state = capture_rng_state()

    applied = restore_rng_state(state)

    assert applied == ["cpu"]


def test_restore_rng_state_skips_cuda_state_on_a_cpu_only_host() -> None:
    state = capture_rng_state()
    state["cuda"] = [torch.zeros(16, dtype=torch.uint8)]

    applied = restore_rng_state(state)

    assert applied == ["cpu"]


def test_restore_rng_state_accepts_none_from_a_checkpoint_without_rng() -> None:
    """rng_state is optional, so state["rng"] can be None. A resume path
    should be able to pass it straight through without branching."""
    assert restore_rng_state(None) == []


def test_capture_rng_state_records_every_cuda_device(monkeypatch) -> None:
    """The dev machine has no CUDA but the training pod is CUDA-only, so this
    branch is the one that actually runs in production and would otherwise
    never be executed by any test."""
    device_states = [torch.full((16,), 5, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: device_states)

    state = capture_rng_state()

    assert state["cuda"] == device_states


def test_restore_rng_state_applies_and_reports_cuda(monkeypatch) -> None:
    restored: list = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)
    saved = [torch.full((16,), 5, dtype=torch.uint8)]

    applied = restore_rng_state({"cpu": torch.get_rng_state(), "cuda": saved})

    assert applied == ["cpu", "cuda"]
    assert restored == [saved]


def test_capture_rng_state_records_the_cpu_generator() -> None:
    torch.manual_seed(0)
    expected = torch.get_rng_state()

    state = capture_rng_state()

    assert torch.equal(state["cpu"], expected)
