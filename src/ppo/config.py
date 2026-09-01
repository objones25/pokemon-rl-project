"""PPO trainer configuration, loaded from configs/ppo.yaml.

Loading (unknown-field rejection, dataclass construction) is shared via
config_io.load_dataclass_config; validation is this dataclass's own
__post_init__.

Shape helpers live here rather than as constants because a later
curriculum stage raises context_len and n_steps together; nothing in the
trainer may hard-code 1024 or 2048."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config_io import load_dataclass_config

# A resolved git commit on the Hub is exactly 40 lowercase hex characters.
# Anything else -- a branch name, a tag, a short sha -- can move under a
# running agent, and the manifest would record the mutable name rather than
# what was actually downloaded.
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_BRANCH_HEADS = frozenset({"main", "master"})


@dataclass(frozen=True)
class PPOConfig:
    n_steps: int = 1024
    n_epochs: int = 3
    minibatch_envs: int = 8
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    # SB3's PPO default, deliberately not CLAUDE.md's transformer default of
    # 1.0 -- that value is for language-model pretraining.
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    warmup_steps: int = 100
    abort_approx_kl: float = 0.5
    max_nan_minibatches_per_update: int = 3
    seed: int = 0
    frozen_encoder_repo_id: str = "objones25/pokemon-contrastive-encoder"
    frozen_encoder_revision: str | None = None
    checkpoint_dir: str = "/workspace/checkpoints"
    keep_last_n: int = 3
    checkpoint_every_updates: int = 25
    artifact_every_updates: int = 25
    hub_snapshot_every_updates: int = 75
    diagnostics_layer: int = -1

    def __post_init__(self) -> None:
        if self.frozen_encoder_revision is None:
            raise ValueError(
                "frozen_encoder_revision must be pinned to a resolved commit. An "
                "unpinned revision lets a mid-run push to the encoder repo change "
                "the features underneath a running agent, with nothing raised."
            )
        if self.frozen_encoder_revision.lower() in _BRANCH_HEADS:
            raise ValueError(
                f"frozen_encoder_revision={self.frozen_encoder_revision!r} is a branch "
                "head, not a pin: it resolves at download time, so a mid-run push to "
                "the encoder repo changes the features underneath a running agent, and "
                "the checkpoint manifest records only the branch name -- so a resume "
                "cannot detect that the encoder moved either. Pass the resolved commit "
                "sha (HfApi().repo_info(repo_id, repo_type='model').sha)."
            )
        if not _COMMIT_SHA.fullmatch(self.frozen_encoder_revision):
            raise ValueError(
                f"frozen_encoder_revision={self.frozen_encoder_revision!r} is not a "
                "resolved commit sha (40 lowercase hex characters). Tags and short "
                "shas are not pins: only a full sha names one immutable tree."
            )
        if self.n_steps < 1:
            raise ValueError(f"n_steps={self.n_steps} must be at least 1")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs={self.n_epochs} must be at least 1")
        if self.minibatch_envs < 1:
            raise ValueError(f"minibatch_envs={self.minibatch_envs} must be at least 1")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma={self.gamma} must lie in (0, 1)")

    def validate_against_n_envs(self, n_envs: int) -> None:
        """n_envs lives on EnvConfig, so the divisibility check cannot run in
        __post_init__. The trainer calls this once at startup."""
        if n_envs % self.minibatch_envs:
            raise ValueError(
                f"minibatch_envs={self.minibatch_envs} does not divide n_envs={n_envs}; "
                "a ragged final minibatch would change the effective batch size of one "
                "optimizer step per epoch"
            )

    def burn_in(self, context_len: int) -> int:
        """The minimum prefix giving every trained position a full context
        window. Any smaller and the first trained positions see less context
        than the model was sized for."""
        return context_len - 1

    def buffer_capacity(self, context_len: int) -> int:
        """burn-in + trained region + one bootstrap slot for V(s_T)."""
        return self.burn_in(context_len) + self.n_steps + 1


def load_config(path: str | Path) -> PPOConfig:
    return load_dataclass_config(PPOConfig, path)
