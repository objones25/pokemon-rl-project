"""Checkpoint pairing. save_checkpoint is already atomic per file; the failure
it cannot see is one of the two files landing."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from ppo.checkpoint import (
    ENV_PATTERN,
    MANIFEST_PATTERN,
    POLICY_PATTERN,
    resume,
    write_checkpoint,
)
from ppo.config import PPOConfig
from ppo.normalizer import ReturnScaler
from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy
from tests.conftest import PINNED_ENCODER_REVISION

from .fakes import FakeVecEnv

_N_ENVS = 2


@dataclass
class _CheckpointHarness:
    """Bundles everything write_checkpoint/resume need, plus the two
    keyword-dict builders the tests call so each test only states what it
    is actually varying (the update number, or a config override)."""

    directory: Path
    policy: RecurrentTransformerPolicy
    optimizer: torch.optim.Optimizer
    cache: RolloutCache
    vec_env: FakeVecEnv
    scaler: ReturnScaler
    policy_config: PolicyConfig
    config: PPOConfig
    init_state_hash: str

    def kwargs(self, update: int) -> dict:
        return {
            "directory": self.directory,
            "update": update,
            "global_step": update * self.config.n_steps,
            "policy": self.policy,
            "optimizer": self.optimizer,
            "scheduler": None,
            "cache": self.cache,
            "vec_env": self.vec_env,
            "scaler": self.scaler,
            "config": self.config,
            "init_state_hash": self.init_state_hash,
            "wandb_run_id": "run-abc123",
            "git_commit": "deadbeef",
        }

    def resume_kwargs(
        self, context_len: int | None = None, frozen_encoder_revision: str | None = None
    ) -> dict:
        policy_config = self.policy_config
        if context_len is not None:
            policy_config = dataclasses.replace(policy_config, context_len=context_len)
        config = self.config
        if frozen_encoder_revision is not None:
            config = dataclasses.replace(
                config, frozen_encoder_revision=frozen_encoder_revision
            )
        return {
            "directory": self.directory,
            "policy": self.policy,
            "optimizer": self.optimizer,
            "scheduler": None,
            "vec_env": self.vec_env,
            "scaler": self.scaler,
            "policy_config": policy_config,
            "config": config,
            "init_state_hash": self.init_state_hash,
        }


def _checkpoint_harness(tmp_path: Path) -> _CheckpointHarness:
    """Helper, not a test: a tiny policy/optimizer/cache/env/scaler wired the
    way ppo/update.py wires the real ones, small enough to run on CPU in
    milliseconds."""
    torch.manual_seed(0)
    policy_config = PolicyConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        n_kv_heads=1,
        d_ff=64,
        context_len=4,
        latent_dim=8,
        aux_state_dim=4,
    )
    policy = RecurrentTransformerPolicy(policy_config, torch.zeros(8), torch.ones(8))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    cache = policy.new_cache(_N_ENVS, torch.device("cpu"))
    vec_env = FakeVecEnv(n_envs=_N_ENVS, aux_dim=policy_config.aux_state_dim, done_at_step=None)
    return _CheckpointHarness(
        directory=tmp_path,
        policy=policy,
        optimizer=optimizer,
        cache=cache,
        vec_env=vec_env,
        scaler=ReturnScaler(),
        policy_config=policy_config,
        config=PPOConfig(frozen_encoder_revision=PINNED_ENCODER_REVISION, n_steps=4),
        init_state_hash="deadbeef",
    )


def test_write_checkpoint_returns_a_path_the_manifest_pattern_discovers(tmp_path) -> None:
    """POLICY_PATTERN/MANIFEST_PATTERN are exported so Task 15's resume loop
    and retention glob agree with what write_checkpoint actually names files --
    a hard-coded filename in one place and the pattern in another is exactly
    the kind of drift that silently stops resume from finding anything."""
    harness = _checkpoint_harness(tmp_path)

    manifest_path = write_checkpoint(**harness.kwargs(update=1))

    assert manifest_path in set(tmp_path.glob(MANIFEST_PATTERN))


def test_resume_reports_no_cache_when_none_was_checkpointed(tmp_path) -> None:
    """cache is optional in sequence_model.checkpoint's schema -- a
    checkpoint written before the first rollout, or one that deliberately
    dropped the cache, must resume with cache=None rather than raising on a
    missing "cache" state."""
    harness = _checkpoint_harness(tmp_path)
    kwargs = harness.kwargs(update=1)
    kwargs["cache"] = None

    write_checkpoint(**kwargs)
    result = resume(**harness.resume_kwargs())

    assert result.cache is None


def test_the_manifest_names_both_files_and_their_sizes(tmp_path) -> None:
    harness = _checkpoint_harness(tmp_path)

    write_checkpoint(**harness.kwargs(update=3))
    manifest = json.loads((tmp_path / "manifest_update000003.json").read_text())

    assert set(manifest) >= {"update", "global_step", "policy_file", "env_file", "sizes"}


def test_resume_returns_none_when_the_directory_is_empty(tmp_path) -> None:
    harness = _checkpoint_harness(tmp_path)

    assert resume(**harness.resume_kwargs()) is None


def test_resume_skips_a_checkpoint_whose_manifest_was_never_written(tmp_path) -> None:
    """A crash between the two .pt writes and the manifest write leaves an
    incoherent pair. Taking the newest .pt file regardless would resume a
    policy against an env from a different update."""
    harness = _checkpoint_harness(tmp_path)
    write_checkpoint(**harness.kwargs(update=1))
    write_checkpoint(**harness.kwargs(update=2))
    (tmp_path / "manifest_update000002.json").unlink()

    result = resume(**harness.resume_kwargs())

    assert result.update == 1


def test_resume_skips_a_checkpoint_whose_env_file_is_truncated(tmp_path) -> None:
    harness = _checkpoint_harness(tmp_path)
    write_checkpoint(**harness.kwargs(update=1))
    write_checkpoint(**harness.kwargs(update=2))
    (tmp_path / "env_update000002.pt").write_bytes(b"short")

    result = resume(**harness.resume_kwargs())

    assert result.update == 1


def test_resume_restores_the_return_scaler_state(tmp_path) -> None:
    harness = _checkpoint_harness(tmp_path)
    harness.scaler.update(torch.tensor([[-10.0, 10.0]]))
    write_checkpoint(**harness.kwargs(update=1))
    harness.scaler.load_state_dict({"count": 0.0, "mean": 0.0, "m2": 0.0})

    resume(**harness.resume_kwargs())

    assert harness.scaler.scale == pytest.approx(10.0, rel=0.05)


def test_resume_drops_the_cache_when_the_context_length_changed(tmp_path) -> None:
    """A curriculum stage that raises context_len cannot reuse the ring
    buffer. That is reported and the run warms up again, rather than raising."""
    harness = _checkpoint_harness(tmp_path)
    write_checkpoint(**harness.kwargs(update=1))

    result = resume(**harness.resume_kwargs(context_len=8))

    assert result.cache is None


def test_resume_restores_the_rng_state(tmp_path) -> None:
    """capture_rng_state()/restore_rng_state() round-trip a dict keyed
    "cpu"/"cuda", stored under build_policy_checkpoint_state's "rng" key --
    not "rng_state". Wiring resume() to the wrong key would look fine (no
    KeyError, restore_rng_state(None) just returns []) while silently never
    restoring anything."""
    harness = _checkpoint_harness(tmp_path)
    torch.manual_seed(123)
    write_checkpoint(**harness.kwargs(update=1))
    expected = torch.rand(3)
    torch.manual_seed(999)  # perturb the generator so a real restore is observable

    resume(**harness.resume_kwargs())

    assert torch.equal(torch.rand(3), expected)


def test_resume_selects_update_eleven_over_nine_and_ten(tmp_path) -> None:
    """checkpointing.io.find_latest_checkpoint/prune_checkpoints both sort by
    filename and document that callers must zero-pad ("step900 sorts after
    step1300 as a string"). Unpadded, "manifest_update10.json" sorts BEFORE
    "manifest_update9.json" lexically -- resume would pick a stale checkpoint,
    and prune_checkpoints (out of this module's control) would delete the
    newest files and keep old ones. Pins the fix past the two-digit boundary
    where the bug first appears."""
    harness = _checkpoint_harness(tmp_path)
    write_checkpoint(**harness.kwargs(update=9))
    write_checkpoint(**harness.kwargs(update=10))
    write_checkpoint(**harness.kwargs(update=11))

    result = resume(**harness.resume_kwargs())

    assert result.update == 11


def test_resume_refuses_a_checkpoint_written_against_a_different_encoder_revision(
    tmp_path,
) -> None:
    """The manifest records frozen_encoder_revision precisely so a mid-run
    encoder change is detectable. Accepting it anyway would feed a policy
    trained entirely on one encoder's latents the features of another, with
    nothing raised -- the exact failure the pin exists to prevent."""
    harness = _checkpoint_harness(tmp_path)
    write_checkpoint(**harness.kwargs(update=1))

    with pytest.raises(ValueError, match="frozen encoder revision"):
        resume(**harness.resume_kwargs(frozen_encoder_revision="f" * 40))


def test_resume_accepts_a_checkpoint_written_against_the_same_encoder_revision(
    tmp_path,
) -> None:
    """The other direction: the guard must not reject the ordinary resume it
    is wrapped around."""
    harness = _checkpoint_harness(tmp_path)
    write_checkpoint(**harness.kwargs(update=1))

    result = resume(**harness.resume_kwargs(frozen_encoder_revision=PINNED_ENCODER_REVISION))

    assert result.update == 1


def test_write_checkpoint_prunes_all_three_globs_to_keep_last_n(tmp_path) -> None:
    """Every other test in this file writes at most keep_last_n=3 updates, so
    prune_checkpoints's candidates[:-keep_last_n] is always empty and unlink()
    never runs -- a wrong pattern, a wrong keep_last_n argument, or a pruning
    call dropped entirely would all pass the rest of this suite silently. On
    the real 48-hour run the symptom is disk quietly filling until the volume
    is full, with no error until something unrelated fails. Writes
    keep_last_n + 1 = 3 checkpoints at keep_last_n=2 and checks exact
    surviving update numbers -- not just a count -- across all three globs."""
    harness = _checkpoint_harness(tmp_path)
    pruning_config = dataclasses.replace(harness.config, keep_last_n=2)

    for update in (1, 2, 3):
        kwargs = harness.kwargs(update=update)
        kwargs["config"] = pruning_config
        write_checkpoint(**kwargs)

    policy_survivors = {p.name for p in tmp_path.glob(POLICY_PATTERN)}
    env_survivors = {p.name for p in tmp_path.glob(ENV_PATTERN)}
    manifest_survivors = {p.name for p in tmp_path.glob(MANIFEST_PATTERN)}

    assert policy_survivors == {"policy_update000002.pt", "policy_update000003.pt"}
    assert env_survivors == {"env_update000002.pt", "env_update000003.pt"}
    assert manifest_survivors == {"manifest_update000002.json", "manifest_update000003.json"}
