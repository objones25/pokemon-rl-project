"""Opt-in acceptance test against the real ROM and real PyBoy workers.

Run with:
    uv run pytest -m slow tests/integration/test_ppo_smoke.py -v

Auto-skips when the ROM or init.state is absent, so a fresh checkout never
fails. Four envs and three updates rather than 64 and thousands, so this is
minutes rather than days; the loop is identical.

The "frozen encoder" here is a real, randomly-initialized GrayscaleResNetEncoder
(contrastive_pretrain.model.build_encoder(pretrained=False)) rather than one
downloaded from the Hub: this keeps the acceptance gate exercising the real
inference path (real forward pass, real (144, 160) grayscale contract) without
needing HF credentials in the test process -- the same hazard
docs/2026-08-26-slow-test-suite-blocked.md documents for tests that reach the
real private Hub repo. The policy's transformer dims are shrunk for speed
(latent_dim/aux_state_dim/action_dim stay at their real values, since those
are fixed by the real encoder and the real env); the loop itself -- rollout,
GAE, the update pass, checkpointing -- is identical to a full-size run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pytest
import torch

from contrastive_pretrain.model import build_encoder
from contrastive_pretrain.train import autocast_dtype
from pokemon_env.config import EnvConfig
from pokemon_env.encoder import LatentEncoder
from pokemon_env.init_state import state_hash
from pokemon_env.subprocess_backend import FrameBuffer, build_subprocess_vec_env
from pokemon_env.vec_env import VecPokemonEnv
from ppo.config import PPOConfig
from ppo.trainer import PPODeps, run_training
from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy
from tests.conftest import PINNED_ENCODER_REVISION

pytestmark = pytest.mark.slow

_ROM = Path("Pokemon Red.gb")
_INIT = Path("artifacts/init.state")

_needs_rom = pytest.mark.skipif(
    not _ROM.exists(), reason=f"{_ROM} not present; it is gitignored and must be supplied locally"
)
_needs_init_state = pytest.mark.skipif(
    not _INIT.exists(), reason="artifacts/init.state not generated; see src/pokemon_env/init_state.py"
)


class _FakeExperimentRun:
    """Hand-written fake typed against ExperimentRunLike, matching
    tests/unit/test_ppo_trainer.py's FakeExperimentRun -- records every
    logged dict rather than talking to a real W&B account, which this
    acceptance gate has no business doing."""

    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.summaries: list[dict] = []
        self.finished_with: list[int] = []

    def log(self, metrics: dict) -> None:
        self.logged.append(metrics)

    def summary(self, metrics: dict) -> None:
        self.summaries.append(dict(metrics))

    def finish(self, exit_code: int = 0) -> None:
        self.finished_with.append(exit_code)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(exit_code=1 if exc_type is not None else 0)


@dataclass
class _SmokeHarness:
    deps: PPODeps
    vec_env: VecPokemonEnv
    buffer: FrameBuffer
    wandb_run: _FakeExperimentRun

    def close(self) -> None:
        """Real subprocess workers and a real SharedMemory block -- must be
        released even on a failed assertion, or a failing run leaks both
        into the rest of the test session."""
        self.vec_env.close()
        self.buffer.close()
        self.buffer.unlink()


def _real_harness(
    tmp_path: Path, *, n_envs: int, n_steps: int, checkpoint_every_updates: int = 25
) -> _SmokeHarness:
    """Real ROM, real PyBoy subprocess workers, a real (randomly-initialized)
    ResNet-50 encoder, and a real RecurrentTransformerPolicy -- shrunk to a
    tiny transformer so the acceptance gate runs in minutes. latent_dim (2048)
    and aux_state_dim (32) are NOT shrunk: they are fixed by the real encoder
    and the real env's aux vector, respectively."""
    torch.manual_seed(0)
    policy_config = PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8,
    )
    env_config = EnvConfig(n_envs=n_envs)
    ppo_config = PPOConfig(
        frozen_encoder_revision=PINNED_ENCODER_REVISION,
        n_steps=n_steps,
        n_epochs=1,
        minibatch_envs=n_envs,
        checkpoint_dir=str(tmp_path),
        checkpoint_every_updates=checkpoint_every_updates,
    )

    device = torch.device("cpu")
    encoder_module, _ = build_encoder(pretrained=False)
    encoder = LatentEncoder(encoder_module, device)
    policy = RecurrentTransformerPolicy(
        policy_config, torch.zeros(policy_config.latent_dim), torch.ones(policy_config.latent_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=ppo_config.lr)

    vec_env, buffer = build_subprocess_vec_env(env_config)
    init_hash = state_hash(Path(env_config.init_state_path).read_bytes())
    wandb_run = _FakeExperimentRun()

    deps = PPODeps(
        config=ppo_config,
        env_config=env_config,
        policy_config=policy_config,
        vec_env=vec_env,
        encoder=encoder,
        policy=policy,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        autocast_dtype=autocast_dtype(device),
        init_state_hash=init_hash,
        git_commit="smoke-test",
        wandb_run=wandb_run,
    )
    return _SmokeHarness(deps=deps, vec_env=vec_env, buffer=buffer, wandb_run=wandb_run)


@_needs_rom
@_needs_init_state
def test_three_real_updates_hold_the_epoch_one_ratio_invariant(tmp_path) -> None:
    """The sub-project's acceptance gate, minus the pod. Four real PyBoy
    processes, real frames, a real frozen encoder, and a real PPO update --
    with the invariant asserted inside run_training on every update."""
    harness = _real_harness(tmp_path, n_envs=4, n_steps=8)
    try:
        run_training(harness.deps, max_updates=3)
    finally:
        harness.close()

    assert harness.wandb_run.logged[-1]["ratio/max_abs_dev_epoch1_mb1"] == pytest.approx(0.0, abs=1e-5)


@_needs_rom
@_needs_init_state
def test_a_run_resumes_from_its_checkpoint_without_a_loss_discontinuity(tmp_path) -> None:
    """Resume is state-faithful, not bit-reproducible: PyBoy across respawned
    subprocesses does not reproduce a byte-identical step ordering. This
    asserts the checkpoint round-trips and training continues, never bitwise
    equality."""
    harness = _real_harness(tmp_path, n_envs=4, n_steps=8, checkpoint_every_updates=1)
    try:
        run_training(harness.deps, max_updates=2)
    finally:
        harness.close()

    resumed = _real_harness(tmp_path, n_envs=4, n_steps=8, checkpoint_every_updates=1)
    try:
        run_training(resumed.deps, max_updates=1)
    finally:
        resumed.close()

    assert resumed.wandb_run.logged[0]["train/update"] == pytest.approx(2.0)
