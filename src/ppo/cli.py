"""Console entry points: `pokemon-ppo train` (run/resume PPO training on a
RunPod GPU pod) and `pokemon-ppo preflight` (gates 1-2: SDPA backend choice,
env-count throughput) -- following contrastive_pretrain.cli's structure.

`build_parser()` and the small filesystem/provenance helpers below are
unit-tested directly, with every heavy dependency (the frozen encoder, the
subprocess vec env, wandb, run_training itself) monkeypatched at its
`ppo.cli.*` name -- the same pattern contrastive_pretrain.cli's own tests
use. The real, unmocked mechanics (real ROM, real PyBoy workers, a real
policy update) are exercised by the slow acceptance tier instead
(tests/integration/test_ppo_smoke.py)."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from huggingface_hub import HfApi, get_token

from contrastive_pretrain.encoder_io import load_frozen_encoder
from hf_storage.client import RealHfClient
from observability.logging_config import configure_logging
from observability.tracking import WandbRun
from pokemon_env import config as env_config_module
from pokemon_env.encoder import LatentEncoder, load_latent_stats
from pokemon_env.init_state import state_hash
from pokemon_env.subprocess_backend import build_subprocess_vec_env
from ppo import checkpoint as ppo_checkpoint
from ppo import config as ppo_config_module
from ppo.preflight import sdpa_backend_report, throughput_report
from ppo.telemetry import STEP_METRICS, wandb_config
from ppo.trainer import PPODeps, run_training
from sequence_model import config as policy_config_module
from sequence_model.policy import RecurrentTransformerPolicy
from torch_utils import autocast_dtype

logger = logging.getLogger(__name__)

_DEFAULT_PPO_CONFIG = Path("configs/ppo.yaml")
_DEFAULT_ENV_CONFIG = Path("configs/pokemon_env.yaml")
_DEFAULT_POLICY_CONFIG = Path("configs/sequence_model.yaml")
_WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ppo-config", type=Path, default=_DEFAULT_PPO_CONFIG)
    parser.add_argument("--env-config", type=Path, default=_DEFAULT_ENV_CONFIG)
    parser.add_argument("--policy-config", type=Path, default=_DEFAULT_POLICY_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pokemon-ppo", description="PPO trainer for the Pokemon Red RL agent."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run (or resume) PPO training.")
    _add_config_arguments(train_parser)
    train_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any existing checkpoint under checkpoint_dir and start a new run "
        "(also starts a new W&B run rather than resuming the persisted one).",
    )

    preflight_parser = subparsers.add_parser(
        "preflight", help="Measure gates 1-2 (SDPA backend, env throughput) before a paid run."
    )
    _add_config_arguments(preflight_parser)
    preflight_parser.add_argument(
        "--n-envs",
        type=int,
        nargs="+",
        # The design spec's own gate-2 candidates (§8): "n_envs in {16, 32, 64}
        # on the target pod's actual vCPU count".
        default=[16, 32, 64],
        help="Candidate env counts for the throughput gate (gate 2).",
    )
    preflight_parser.add_argument(
        "--steps", type=int, default=100, help="Env steps measured per --n-envs candidate."
    )
    return parser


def _git_commit(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> str:
    """The commit a checkpoint/W&B run was produced from. "unknown" rather
    than raising: a missing git binary or a non-repo working directory (a
    stripped-down pod image) must not stop a real run over provenance
    metadata."""
    try:
        result = run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _gpu_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(0) if device.type == "cuda" else device.type


def _resolve_device() -> torch.device:
    return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")


def _warmup_then_constant(warmup_steps: int) -> Callable[[int], float]:
    """PPOConfig.warmup_steps: linear ramp from 0 to 1 over `warmup_steps`
    scheduler.step() calls, then held constant at 1.0 -- this project's only
    scheduler shape (design spec's PPOConfig table: "Linear, then
    constant")."""

    def _factor(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return _factor


def _run_id_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / _WANDB_RUN_ID_FILENAME


def _load_run_id(checkpoint_dir: Path) -> str | None:
    path = _run_id_path(checkpoint_dir)
    return path.read_text().strip() if path.exists() else None


def _persist_run_id(checkpoint_dir: Path, run_id: str) -> None:
    """Atomic write (tmp file + `Path.replace`, the same pattern
    `checkpointing.io.save_checkpoint` uses -- `replace` is atomic on POSIX,
    so a crash mid-write never leaves a truncated/corrupt run-id file that
    would fragment the W&B curve on the next resume). `save_checkpoint`
    itself isn't reused here: it `torch.save`-pickles its `state: dict`
    argument, which would turn this plain-text, human-`cat`-able sidecar
    file into an unreadable blob under the same `.txt` name."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _run_id_path(checkpoint_dir)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(run_id)
    tmp_path.replace(path)


def _clear_checkpoint_state(checkpoint_dir: Path) -> None:
    """--fresh's implementation. Without this, run_training's own resume()
    would still find the previous run's manifests and continue from them,
    and a stale wandb_run_id.txt would continue that run's W&B curve under
    what are, for --fresh, unrelated hyperparameters."""
    if not checkpoint_dir.exists():
        return
    patterns = (
        ppo_checkpoint.POLICY_PATTERN,
        ppo_checkpoint.ENV_PATTERN,
        ppo_checkpoint.MANIFEST_PATTERN,
    )
    for pattern in patterns:
        for path in checkpoint_dir.glob(pattern):
            path.unlink()
    run_id_path = _run_id_path(checkpoint_dir)
    if run_id_path.exists():
        run_id_path.unlink()


def _run_train(args: argparse.Namespace) -> None:
    if get_token() is None:
        raise SystemExit(
            "No Hugging Face credentials found. Set HF_TOKEN (e.g. in a .env file) or "
            "run `hf auth login` before using this command."
        )
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit(
            "No W&B credentials found. Set WANDB_API_KEY (e.g. in a .env file) or "
            "run `wandb login` before using this command."
        )

    ppo_config = ppo_config_module.load_config(args.ppo_config)
    env_config = env_config_module.load_config(args.env_config)
    policy_config = policy_config_module.load_config(args.policy_config)

    checkpoint_dir = Path(ppo_config.checkpoint_dir)
    if args.fresh:
        _clear_checkpoint_state(checkpoint_dir)

    device = _resolve_device()
    dtype = autocast_dtype(device)

    encoder_module = load_frozen_encoder(
        ppo_config.frozen_encoder_repo_id, ppo_config.frozen_encoder_revision
    )
    # Pinned to the SAME revision as the weights above: AtomicHfClient's own
    # docstring treats weights, config, and latent stats as one
    # atomically-committed bundle. An unpinned client here would fetch
    # latent_stats.json from the branch head while the weights stayed
    # pinned, so a mid-run push to the repo could silently swap the running
    # agent's input normalization underneath it. (load_frozen_encoder builds
    # its own internal HfApi() and has no parameter to share this one with --
    # not deduplicated across that boundary without widening its signature.)
    hf_client = RealHfClient(
        HfApi(),
        ppo_config.frozen_encoder_repo_id,
        repo_type="model",
        revision=ppo_config.frozen_encoder_revision,
    )
    latent_mean, latent_std = load_latent_stats(hf_client)
    encoder = LatentEncoder(encoder_module, device)

    policy = RecurrentTransformerPolicy(policy_config, latent_mean, latent_std).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=ppo_config.lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _warmup_then_constant(ppo_config.warmup_steps)
    )

    vec_env, frame_buffer = build_subprocess_vec_env(env_config)
    try:
        init_hash = state_hash(Path(env_config.init_state_path).read_bytes())
        git_commit = _git_commit()
        run_id = _load_run_id(checkpoint_dir)

        # run_training itself owns the WandbRun's with-block (it must call
        # finish() even on a config-validation error, since the run already
        # exists on the dashboard by the time deps reaches it) -- so this
        # constructs the run and hands it straight to PPODeps rather than
        # wrapping run_training in a second, redundant context manager.
        run = WandbRun(
            wandb,
            project="pokemon-ppo",
            name=f"train-{git_commit[:8]}",
            config=wandb_config(
                ppo_config, env_config, policy_config, {}, git_commit,
                _gpu_name(device), torch.__version__,
            ),
            step_metrics=STEP_METRICS,
            run_id=run_id,
        )
        if run_id is None:
            _persist_run_id(checkpoint_dir, run.run_id)

        deps = PPODeps(
            config=ppo_config,
            env_config=env_config,
            policy_config=policy_config,
            vec_env=vec_env,
            encoder=encoder,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            autocast_dtype=dtype,
            init_state_hash=init_hash,
            git_commit=git_commit,
            wandb_run=run,
        )
        run_training(deps)
    finally:
        vec_env.close()
        frame_buffer.close()
        frame_buffer.unlink()


def _run_preflight(args: argparse.Namespace) -> None:
    ppo_config = ppo_config_module.load_config(args.ppo_config)
    env_config = env_config_module.load_config(args.env_config)
    policy_config = policy_config_module.load_config(args.policy_config)
    device = _resolve_device()

    sdpa_results = sdpa_backend_report(
        policy_config, ppo_config.minibatch_envs, policy_config.context_len, device
    )

    def _build_env(n_envs: int) -> tuple:
        return build_subprocess_vec_env(replace(env_config, n_envs=n_envs))

    throughput_results = throughput_report(_build_env, args.n_envs, args.steps)
    # No run_gates wrapper: the two reports are merged into one flat mapping
    # right here, for wandb_config's gate_results parameter.
    gate_results = {**sdpa_results, **throughput_results}

    config = wandb_config(
        ppo_config, env_config, policy_config, gate_results,
        _git_commit(), _gpu_name(device), torch.__version__,
    )
    logger.info("preflight_gate_results", extra=gate_results)

    if not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "preflight_wandb_skipped",
            extra={"reason": "WANDB_API_KEY not set; gate results are in the log above only"},
        )
        return
    run = WandbRun(wandb, project="pokemon-ppo", name="preflight", config=config)
    run.log(gate_results)
    run.finish()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_dotenv()
    configure_logging()
    if args.command == "train":
        _run_train(args)
    else:
        _run_preflight(args)
