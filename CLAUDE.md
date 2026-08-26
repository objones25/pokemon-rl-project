# Pokemon RL Project

Vision-based RL agent for Pokemon Red/Blue: a frozen contrastive-pretrained CNN
(SimCLR/BYOL) feeds latent frame vectors into a RoPE/GQA transformer sequence
model, trained via PPO. Full architecture rationale: `Pokemon_RL_Architecture_Plan.pdf`.

Sub-projects are designed and planned independently (see `docs/superpowers/specs/`)
in the order: data collection -> CNN pretraining -> sequence model -> PPO agent.
Each gets its own spec and implementation plan before code is written.

## Conventions

- **Package management:** `uv` only. No bare `pip`/`venv`. `uv add <pkg>`, `uv run <cmd>`.
- **SOLID**: single-responsibility modules with injected dependencies (network/IO
  clients, filesystem, external APIs) so core logic is unit-testable without
  hitting the network, ffmpeg, or a GPU.
- **TDD**: write the failing test first for any new logic. Pure/algorithmic code
  gets fast unit tests with synthetic fixtures; anything touching real
  video/network/Hub APIs gets a separate, explicitly slow/opt-in integration test.
- **Karpathy guidelines**: minimal surgical diffs, no speculative abstraction,
  verify one stage works end-to-end before building the next stage on top of it.
- **Observability first**: every pipeline/training component logs structured
  (JSON-lines) progress and emits a live Weights & Biases (W&B) run. Anything
  that filters or drops data (dedup, anomaly detection, reward shaping) logs
  *why*, and produces a periodic visual artifact (contact sheets, sample grids,
  reward curves) a human can sanity-check without reading raw logs.
  (Migrated from Trackio: self-hosting Trackio's dashboard on a RunPod pod
  requires port-forwarding/tunneling to view it from outside the pod, which
  made it unusable in practice for actually monitoring a run in progress.)
- **Interface-fit over spec-compliance**: a component's job is to fit how it
  will actually be consumed, not just satisfy the literal spec text.
  Brainstorming must produce concrete, checkable interface contracts before
  implementation starts — no vague "reusable later." Verify integration
  requirements against context7 (`ctx7` CLI), official library docs, or the
  `inspect` module on the actual installed package; never guess a calling
  convention or interface shape from memory.
- **Production readiness for long/costly runs**: any component driving a
  long-running or paid unattended job must have logging, error handling,
  retries, and resume/checkpointing decided at design time (during
  brainstorming), not discovered after.

## PyTorch

Installed: torch 2.13.0 / torchvision 0.28.0. Dev machine is Apple Silicon (MPS,
no CUDA); training runs on RunPod CUDA — anything device-specific needs both
paths. Defaults move between minor releases, so never quote a signature or
default from memory (see interface-fit above).

- **Non-negotiables** — each of these trains a broken model silently rather than
  crashing: heads emit raw logits (never softmax/sigmoid before
  `CrossEntropyLoss`/`BCEWithLogitsLoss`); the step order is
  `zero_grad -> forward -> loss -> backward -> step`; every eval path is
  `model.eval()` + `@torch.inference_mode()` with `model.train()` restored after;
  non-parameter tensors go through `nn.Parameter`/`register_buffer` or they stay
  invisible to `.to()`, `state_dict()`, and the optimizer.
- **Device**: `torch.accelerator.current_accelerator(check_available=True) or
  torch.device("cpu")`. Tensors move by copy (`x = x.to(device)`), modules move
  in place (`model.to(device)`).
- **Mixed precision**: `torch.autocast(device.type, dtype=...)`, no `GradScaler`
  — bf16 on CUDA/CPU, fp16 on MPS (`train.autocast_dtype`). Validation runs under
  the *same* autocast context as the training step, or its numbers don't describe
  that step.
- **Checkpoints**: save `state_dict` plus optimizer/scheduler/step, never the
  model object; load with `weights_only=True`. Save and export from the raw
  module, not the `torch.compile` wrapper — compiled `state_dict` keys carry an
  `_orig_mod.` prefix.
- **Perf**: `torch.compile` after `.to(device)`; `channels_last` applied to both
  the conv module and every input batch; fixed batch shape so `cudnn.benchmark`
  and `torch.compile` don't re-tune or recompile.
- **Optimizer**: AdamW (`weight_decay` already defaults to `0.01` in 2.13),
  cosine per epoch or OneCycle per batch.
- **Debugging**: when training runs but doesn't learn, overfit a single batch
  (~200 steps to near-zero loss) *before* touching hyperparameters, data, or
  architecture. It collapses the search space to model/loss/step.

## Sequence model (transformer)

Not implemented yet — these bind when that sub-project starts.

- Size it with arithmetic, not prose. Run the transformer-architecture skill's
  `config_budget.py` before quoting any parameter count, FLOP, KV-cache, or VRAM
  number; hand-estimated parameter counts are the usual source of wrong configs.
- Get the four constraints first (train compute, inference target as
  batch/context/latency/VRAM, token budget, input vocabulary). GQA group count
  and often depth are decided by the KV cache at the real serving batch and
  context — a config chosen without that number is a guess.
- Decide in order: vocab -> `d_model`/`n_layers` -> `head_dim`/`n_heads` ->
  `n_kv_heads` -> `d_ff` -> context + `rope_theta` -> tying -> norm/act/init.
  Defaults needing a stated reason to change: pre-norm RMSNorm, RoPE, SwiGLU with
  `d_ff = round_to_256(8/3 * d_model)`, no linear biases, bf16 compute with fp32
  master/Adam state, grad clip 1.0, AdamW betas (0.9, 0.95). Add QK-norm above
  ~1B params.
- Inputs here are frozen-CNN frame latents, not tokens: vocab size, embedding
  share, and weight tying do not transfer from an LM config. Say what each maps
  to in this model before copying ratios from Llama/Qwen.
- Four bugs pass every shape check and must have property tests before a real
  run: RoPE applied before the head split; interleaved-vs-halves RoPE pairing
  (trains fine, silently incompatible with any external checkpoint);
  `x.repeat()` instead of expand-reshape in `repeat_kv` (a query-head
  permutation — breaks checkpoint interop and fused kernels, *not* a quality
  regression, and reporting it as one costs a day); `is_causal=True` during
  incremental decode (prefill perfect, generation garbage).
- Log leading indicators, not just loss and grad norm — both are late. Attention
  logit magnitude and final-layer output norm predict divergence well ahead of it.

## Testing

pytest 9.1.1 with strict config already in `pyproject.toml` (`strict_config`,
`strict_markers`, `strict_xfail`, `filterwarnings = error`, branch coverage floor
80%). Keep those gates and ratchet the floor up; a floor you have to lower is
worse than none.

- **Prove each new test can fail.** Break the code it covers (invert a condition,
  return a wrong constant), confirm red, revert — and say which test you verified
  this way. Highest-value step in the workflow and the easiest to skip.
- **Hard gates**: one behavior per test; long declarative names; Arrange/Act/
  Assert; no `if`/`for`/`while` in a test body (cases are what `parametrize` is
  for, and it reports each separately); every test asserts something, and asserts
  an exact expected value rather than a loose range that a broken implementation
  would also satisfy; floats via `pytest.approx`; `pytest.raises` always names a
  specific exception and passes `match=`; `skip`/`xfail` always carry `reason=`.
- **Isolation**: tests pass alone, in any order, in parallel. No unseeded
  randomness, no network in a unit test, no `time.sleep`, no `sys.path` edits;
  filesystem writes go to `tmp_path`, env changes through `monkeypatch.setenv`.
- **Doubles**: hand-written fakes typed against the Protocol the consumer
  actually uses (`FakeHfClient` in `tests/conftest.py`) over `mock.patch` — the
  suite patches in exactly one place, and there only as a `wraps=` call-count
  spy, not as a stub. If a stub is genuinely unavoidable, `autospec=True` and
  patch where the object is *used*, not where it's defined. Never call a
  bare `Mock` method that only looks like an assertion (`m.called_once_with(...)`
  auto-creates a truthy attribute and passes forever); the real names start with
  `assert_`.
- **ML tests**: seeded, CPU-only, tiny synthetic tensors by default. Anything
  needing the real Hub, ffmpeg, or real pretrained weights carries
  `@pytest.mark.slow` and stays deselected by default.
- Run the pytest-expert skill's `scripts/audit_tests.py tests/` after adding
  tests — it statically catches the gates above, and a green `pytest` over a
  suite it flags means the tests aren't testing.

## Infra

- **Compute**: RunPod (CPU pods for data pipeline work, GPU pods for training).
- **Storage/datasets**: Hugging Face Hub (private dataset repos), Parquet shards
  using the `datasets` library's `Image` feature.
- **Data source**: curated YouTube longplay URLs, human-approved per video
  (see the data-collection spec) — never fully automated scraping/search.
