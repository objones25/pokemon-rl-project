"""CLI wiring. The heavy paths are exercised by the slow tier."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from ppo.cli import (
    _clear_checkpoint_state,
    _git_commit,
    _gpu_name,
    _load_run_id,
    _persist_run_id,
    _warmup_then_constant,
    build_parser,
    main,
)


def test_the_train_subcommand_defaults_to_resuming() -> None:
    args = build_parser().parse_args(["train"])

    assert args.fresh is False


def test_the_fresh_flag_opts_out_of_resuming() -> None:
    args = build_parser().parse_args(["train", "--fresh"])

    assert args.fresh is True


def test_the_preflight_subcommand_takes_the_env_counts_to_measure() -> None:
    args = build_parser().parse_args(["preflight", "--n-envs", "16", "32"])

    assert args.n_envs == [16, 32]


def test_an_unknown_subcommand_is_rejected() -> None:
    # argparse's own usage-error exit code -- SystemExit(2), never a message
    # string (that text goes to stderr via parser.error(), not sys.exit()).
    with pytest.raises(SystemExit, match=r"^2$"):
        build_parser().parse_args(["nonsense"])


# --- _git_commit -------------------------------------------------------


def test_git_commit_returns_the_stripped_stdout_on_success() -> None:
    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")

    assert _git_commit(run=_fake_run) == "abc123"


def test_git_commit_returns_unknown_when_the_git_binary_is_missing() -> None:
    def _fake_run(cmd, capture_output, text, check):
        raise FileNotFoundError

    assert _git_commit(run=_fake_run) == "unknown"


def test_git_commit_returns_unknown_when_rev_parse_fails() -> None:
    def _fake_run(cmd, capture_output, text, check):
        raise subprocess.CalledProcessError(128, cmd)

    assert _git_commit(run=_fake_run) == "unknown"


# --- _gpu_name -----------------------------------------------------------


def test_gpu_name_returns_the_device_type_for_a_non_cuda_device() -> None:
    assert _gpu_name(torch.device("cpu")) == "cpu"


def test_gpu_name_returns_the_cuda_device_name_for_a_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA A100")

    assert _gpu_name(torch.device("cuda")) == "NVIDIA A100"


# --- _warmup_then_constant -------------------------------------------------


def test_warmup_then_constant_ramps_linearly_inside_the_warmup_window() -> None:
    factor = _warmup_then_constant(4)

    assert factor(0) == pytest.approx(0.25)


def test_warmup_then_constant_holds_at_one_past_the_warmup_window() -> None:
    factor = _warmup_then_constant(4)

    assert factor(10) == pytest.approx(1.0)


def test_warmup_then_constant_with_zero_warmup_steps_is_constant_from_the_start() -> None:
    factor = _warmup_then_constant(0)

    assert factor(0) == pytest.approx(1.0)


# --- run-id persistence ------------------------------------------------


def test_load_run_id_returns_none_when_no_file_exists(tmp_path: Path) -> None:
    assert _load_run_id(tmp_path) is None


def test_persist_run_id_then_load_run_id_round_trips(tmp_path: Path) -> None:
    _persist_run_id(tmp_path, "abc123")

    assert _load_run_id(tmp_path) == "abc123"


def test_persist_run_id_creates_the_checkpoint_directory_if_missing(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"

    _persist_run_id(directory, "abc123")

    assert directory.exists()


# --- --fresh's checkpoint-clearing helper -----------------------------


def test_clear_checkpoint_state_removes_every_checkpoint_and_run_id_file(tmp_path: Path) -> None:
    (tmp_path / "policy_update000000.pt").write_bytes(b"x")
    (tmp_path / "env_update000000.pt").write_bytes(b"x")
    (tmp_path / "manifest_update000000.json").write_text("{}")
    (tmp_path / "wandb_run_id.txt").write_text("stale")

    _clear_checkpoint_state(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_clear_checkpoint_state_on_a_missing_directory_does_not_raise(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    assert _clear_checkpoint_state(missing) is None


def test_clear_checkpoint_state_with_no_run_id_file_only_removes_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "manifest_update000000.json").write_text("{}")

    _clear_checkpoint_state(tmp_path)

    assert list(tmp_path.iterdir()) == []


# --- full `train` wiring, every heavy dependency monkeypatched --------


class _FakeWandbRun:
    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.logged: list[dict] = []
        self.finished = False
        self.defined_metrics: list[tuple[str, str]] = []

    def define_metric(self, pattern: str, step_metric: str) -> None:
        self.defined_metrics.append((pattern, step_metric))

    def log(self, metrics: dict) -> None:
        self.logged.append(metrics)

    def finish(self, exit_code: int = 0) -> None:
        self.finished = True


class _FakeWandbModule:
    """`cli._run_train`/`_run_preflight` construct a real WandbRun, so this
    fakes the `wandb` module itself, matching contrastive_pretrain.cli's own
    test pattern -- otherwise a real wandb.init() fires on every test run."""

    def __init__(self, run_id: str = "fake-run-id") -> None:
        self._run_id = run_id
        self.init_calls: list[dict] = []
        self.runs: list[_FakeWandbRun] = []

    def init(self, **kwargs) -> _FakeWandbRun:
        self.init_calls.append(kwargs)
        run = _FakeWandbRun(self._run_id)
        self.runs.append(run)
        return run


class _FakeFrameBuffer:
    def __init__(self) -> None:
        self.closed = False
        self.unlinked = False

    def close(self) -> None:
        self.closed = True

    def unlink(self) -> None:
        self.unlinked = True


class _FakeVecEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _recording_real_hf_client_factory(calls: list[dict]):
    """Builds a `RealHfClient` double matching its real `__init__(self, api,
    repo_id, repo_type="dataset", revision=None)` signature exactly, that
    records every call into `calls` -- so a test can assert `revision`
    actually reached this call site. The prior double,
    `lambda *a, **k: "fake-hf-client"`, discarded every argument and so
    could not have caught the CLI passing no revision at all here."""

    class _RecordingRealHfClient:
        def __init__(
            self, api: object, repo_id: str, repo_type: str = "dataset",
            revision: str | None = None,
        ) -> None:
            calls.append(
                {"api": api, "repo_id": repo_id, "repo_type": repo_type, "revision": revision}
            )

    return _RecordingRealHfClient


def _write_ppo_config(tmp_path: Path, checkpoint_dir: Path) -> Path:
    path = tmp_path / "ppo.yaml"
    path.write_text(f"frozen_encoder_revision: x\ncheckpoint_dir: {checkpoint_dir}\n")
    return path


def _write_env_config(tmp_path: Path) -> Path:
    init_state = tmp_path / "init.state"
    init_state.write_bytes(b"fake-init-state")
    path = tmp_path / "env.yaml"
    path.write_text(f"init_state_path: {init_state}\n")
    return path


def _write_policy_config(tmp_path: Path) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(
        "d_model: 32\nn_layers: 2\nn_heads: 2\nhead_dim: 16\nn_kv_heads: 1\n"
        "d_ff: 64\ncontext_len: 4\nlatent_dim: 8\naux_state_dim: 4\n"
    )
    return path


@dataclass
class _CliTrainHarness:
    captured: dict
    fake_wandb: _FakeWandbModule
    vec_env: _FakeVecEnv
    frame_buffer: _FakeFrameBuffer
    checkpoint_dir: Path
    hf_client_calls: list[dict]


def _invoke_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_args: list[str] | None = None,
    run_training_raises: Exception | None = None,
) -> _CliTrainHarness:
    """Every heavy dependency (encoder resolution, the subprocess vec env,
    wandb, run_training itself) is monkeypatched at its `ppo.cli.*` name --
    the CLI's own wiring is what this exercises, not the real mechanics
    behind each dependency (those are covered where each dependency is
    defined, and end-to-end by the slow acceptance tier)."""
    checkpoint_dir = tmp_path / "checkpoints"
    ppo_path = _write_ppo_config(tmp_path, checkpoint_dir)
    env_path = _write_env_config(tmp_path)
    policy_path = _write_policy_config(tmp_path)
    captured: dict = {}
    fake_wandb = _FakeWandbModule()
    vec_env = _FakeVecEnv()
    frame_buffer = _FakeFrameBuffer()
    hf_client_calls: list[dict] = []

    monkeypatch.setattr("ppo.cli.load_dotenv", lambda: None)
    monkeypatch.setattr("ppo.cli.get_token", lambda: "fake-token")
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setattr("ppo.cli.wandb", fake_wandb)
    monkeypatch.setattr(
        "ppo.cli.load_frozen_encoder", lambda repo_id, revision: torch.nn.Linear(2, 2)
    )
    monkeypatch.setattr(
        "ppo.cli.RealHfClient", _recording_real_hf_client_factory(hf_client_calls)
    )
    monkeypatch.setattr("ppo.cli.HfApi", lambda: "fake-hf-api")
    monkeypatch.setattr(
        "ppo.cli.load_latent_stats", lambda client: (torch.zeros(8), torch.ones(8))
    )
    monkeypatch.setattr(
        "ppo.cli.build_subprocess_vec_env", lambda cfg: (vec_env, frame_buffer)
    )

    def _fake_run_training(deps) -> None:
        captured["deps"] = deps
        if run_training_raises is not None:
            raise run_training_raises

    monkeypatch.setattr("ppo.cli.run_training", _fake_run_training)

    args = [
        "train",
        "--ppo-config", str(ppo_path),
        "--env-config", str(env_path),
        "--policy-config", str(policy_path),
        *(extra_args or []),
    ]
    if run_training_raises is not None:
        with pytest.raises(type(run_training_raises), match=re.escape(str(run_training_raises))):
            main(args)
    else:
        main(args)

    return _CliTrainHarness(
        captured, fake_wandb, vec_env, frame_buffer, checkpoint_dir, hf_client_calls
    )


def test_train_command_builds_deps_from_the_loaded_policy_config(tmp_path, monkeypatch) -> None:
    harness = _invoke_train(tmp_path, monkeypatch)

    assert harness.captured["deps"].policy_config.d_model == 32


def test_train_command_pins_the_latent_stats_client_to_the_configured_revision(
    tmp_path, monkeypatch
) -> None:
    """AtomicHfClient's own docstring treats weights, config, and latent
    stats as one atomically-committed bundle -- an unpinned client here
    would fetch latent_stats.json from the branch head while the weights
    stayed pinned to frozen_encoder_revision, letting a mid-run push to the
    repo silently swap the running agent's input normalization."""
    harness = _invoke_train(tmp_path, monkeypatch)

    expected_revision = harness.captured["deps"].config.frozen_encoder_revision
    assert harness.hf_client_calls[-1]["revision"] == expected_revision


def test_train_command_uses_the_model_repo_type_for_the_latent_stats_client(
    tmp_path, monkeypatch
) -> None:
    harness = _invoke_train(tmp_path, monkeypatch)

    assert harness.hf_client_calls[-1]["repo_type"] == "model"


def test_train_command_persists_a_new_wandb_run_id_when_none_exists_yet(tmp_path, monkeypatch) -> None:
    harness = _invoke_train(tmp_path, monkeypatch)

    assert (harness.checkpoint_dir / "wandb_run_id.txt").read_text() == "fake-run-id"


def test_train_command_resumes_an_existing_wandb_run_id(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "wandb_run_id.txt").write_text("existing-run-id")

    harness = _invoke_train(tmp_path, monkeypatch)

    assert harness.fake_wandb.init_calls[-1]["id"] == "existing-run-id"


def test_train_command_resuming_an_existing_run_id_does_not_rewrite_the_file(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "wandb_run_id.txt").write_text("existing-run-id")

    _invoke_train(tmp_path, monkeypatch)

    assert (checkpoint_dir / "wandb_run_id.txt").read_text() == "existing-run-id"


def test_the_fresh_flag_clears_a_prior_checkpoint_before_training(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    stale_manifest = checkpoint_dir / "manifest_update000000.json"
    stale_manifest.write_text("{}")
    (checkpoint_dir / "wandb_run_id.txt").write_text("stale-id")

    _invoke_train(tmp_path, monkeypatch, extra_args=["--fresh"])

    assert stale_manifest.exists() is False


def test_the_fresh_flag_starts_a_new_wandb_run_instead_of_resuming(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "wandb_run_id.txt").write_text("stale-id")

    harness = _invoke_train(tmp_path, monkeypatch, extra_args=["--fresh"])

    assert "id" not in harness.fake_wandb.init_calls[-1]


def test_train_command_closes_the_vec_env_after_a_successful_run(tmp_path, monkeypatch) -> None:
    harness = _invoke_train(tmp_path, monkeypatch)

    assert (harness.vec_env.closed, harness.frame_buffer.closed, harness.frame_buffer.unlinked) == (
        True, True, True,
    )


def test_train_command_closes_the_vec_env_even_when_run_training_raises(tmp_path, monkeypatch) -> None:
    harness = _invoke_train(tmp_path, monkeypatch, run_training_raises=RuntimeError("boom"))

    assert harness.vec_env.closed is True


def test_train_command_fails_fast_with_no_hf_credentials(monkeypatch) -> None:
    monkeypatch.setattr("ppo.cli.load_dotenv", lambda: None)
    monkeypatch.setattr("ppo.cli.get_token", lambda: None)

    with pytest.raises(SystemExit, match="Hugging Face"):
        main(["train"])


def test_train_command_fails_fast_with_no_wandb_credentials(monkeypatch) -> None:
    monkeypatch.setattr("ppo.cli.load_dotenv", lambda: None)
    monkeypatch.setattr("ppo.cli.get_token", lambda: "fake-token")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="WANDB_API_KEY"):
        main(["train"])


# --- `preflight` wiring --------------------------------------------------


def _invoke_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    sdpa_results: dict,
    throughput_results: dict,
    extra_args: list[str] | None = None,
    with_wandb: bool = True,
) -> _FakeWandbModule:
    checkpoint_dir = tmp_path / "checkpoints"
    ppo_path = _write_ppo_config(tmp_path, checkpoint_dir)
    env_path = _write_env_config(tmp_path)
    policy_path = _write_policy_config(tmp_path)
    fake_wandb = _FakeWandbModule()

    monkeypatch.setattr("ppo.cli.load_dotenv", lambda: None)
    monkeypatch.setattr("ppo.cli.wandb", fake_wandb)
    monkeypatch.setattr("ppo.cli.sdpa_backend_report", lambda *a, **k: dict(sdpa_results))
    monkeypatch.setattr(
        "ppo.cli.throughput_report", lambda *a, **k: dict(throughput_results)
    )
    if with_wandb:
        monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    else:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

    main([
        "preflight",
        "--ppo-config", str(ppo_path),
        "--env-config", str(env_path),
        "--policy-config", str(policy_path),
        *(extra_args or []),
    ])
    return fake_wandb


def test_preflight_logs_the_merged_gate_dict_to_a_new_wandb_run(tmp_path, monkeypatch) -> None:
    fake_wandb = _invoke_preflight(
        tmp_path, monkeypatch,
        sdpa_results={"flash": True},
        throughput_results={"env_steps_per_sec_at_4": 100.0},
    )

    assert fake_wandb.runs[-1].logged == [{"flash": True, "env_steps_per_sec_at_4": 100.0}]


def test_preflight_finishes_the_wandb_run_it_opens(tmp_path, monkeypatch) -> None:
    fake_wandb = _invoke_preflight(
        tmp_path, monkeypatch, sdpa_results={}, throughput_results={},
    )

    assert fake_wandb.runs[-1].finished is True


def test_preflight_skips_wandb_entirely_with_no_wandb_api_key(tmp_path, monkeypatch) -> None:
    fake_wandb = _invoke_preflight(
        tmp_path, monkeypatch, sdpa_results={}, throughput_results={}, with_wandb=False,
    )

    assert fake_wandb.init_calls == []


def test_preflight_passes_n_envs_and_steps_through_to_throughput_report(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    checkpoint_dir = tmp_path / "checkpoints"
    ppo_path = _write_ppo_config(tmp_path, checkpoint_dir)
    env_path = _write_env_config(tmp_path)
    policy_path = _write_policy_config(tmp_path)

    monkeypatch.setattr("ppo.cli.load_dotenv", lambda: None)
    monkeypatch.setattr("ppo.cli.sdpa_backend_report", lambda *a, **k: {})
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def _fake_throughput_report(build_env, n_envs, steps):
        captured["n_envs"] = n_envs
        captured["steps"] = steps
        return {}

    monkeypatch.setattr("ppo.cli.throughput_report", _fake_throughput_report)

    main([
        "preflight",
        "--ppo-config", str(ppo_path),
        "--env-config", str(env_path),
        "--policy-config", str(policy_path),
        "--n-envs", "4", "8",
        "--steps", "5",
    ])

    assert (captured["n_envs"], captured["steps"]) == ([4, 8], 5)


def test_preflight_build_env_closure_overrides_n_envs_on_the_loaded_env_config(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    checkpoint_dir = tmp_path / "checkpoints"
    ppo_path = _write_ppo_config(tmp_path, checkpoint_dir)
    env_path = _write_env_config(tmp_path)
    policy_path = _write_policy_config(tmp_path)

    monkeypatch.setattr("ppo.cli.load_dotenv", lambda: None)
    monkeypatch.setattr("ppo.cli.sdpa_backend_report", lambda *a, **k: {})
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(
        "ppo.cli.build_subprocess_vec_env", lambda cfg: (cfg, _FakeFrameBuffer())
    )

    def _fake_throughput_report(build_env, n_envs, steps):
        captured["env_config"] = build_env(4)[0]
        return {}

    monkeypatch.setattr("ppo.cli.throughput_report", _fake_throughput_report)

    main([
        "preflight",
        "--ppo-config", str(ppo_path),
        "--env-config", str(env_path),
        "--policy-config", str(policy_path),
    ])

    assert captured["env_config"].n_envs == 4
