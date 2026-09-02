# Pokemon RL Project

Vision-based RL agent for Pokemon Red: a frozen contrastive-pretrained CNN
(SimCLR) turns frames into latents, a RoPE/GQA transformer reads the sequence,
PPO trains it against a RAM-derived reward. Full rationale:
`Pokemon_RL_Architecture_Plan.pdf`.

Sub-projects are designed and planned independently (`docs/superpowers/specs/`,
then `docs/superpowers/plans/`) in the order: data collection -> CNN
pretraining -> sequence model -> environment -> PPO trainer. Each gets its own
spec and implementation plan before code is written. **`src/ppo/` now exists**
— rollout, GAE, the clipped update, checkpointing, telemetry, pre-flight
gates, and the `pokemon-ppo` CLI, per
`docs/superpowers/specs/2026-08-27-ppo-trainer-design.md`. The four gates in
that spec's §8 (SDPA backend, rollout throughput, a memory probe, and a
50-update live smoke run) are what stand between it and the first paid run.

## Attribution — do not remove

`src/pokemon_env/ram.py`'s addresses and decoding come from Peter Whidden's
[PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)
([talk](https://www.youtube.com/watch?v=DcYLT37ImBY)), not from a wiki, and
`configs/pokemon_env.yaml` pins `action_freq` / `max_steps` / `n_envs` to that
project's v2 values. Keep the credit in `ram.py`'s docstring, in `README.md`,
and here. If you change a RAM address, cite the reader you took it from.

## Codebase map

| Package | Owns | Never |
| --- | --- | --- |
| `src/data_collection/` | YouTube -> ffmpeg frames -> phash dedup -> Parquet shards -> HF dataset repo, with resume-by-manifest | Selects videos automatically; `curate` is human-gated |
| `src/contrastive_pretrain/` | SimCLR ResNet-50 + projector, augmentation, resize cache, training loop, frozen-encoder export | Trains the encoder past export — downstream it is frozen |
| `src/sequence_model/` | `RecurrentTransformerPolicy`, GQA+RoPE attention, ring-buffer KV cache, masks, checkpoint schema | Contains a training loop; PPO owns that |
| `src/pokemon_env/` | PyBoy wrapper, RAM readers, §4 reward, 32-d aux state, 64-way subprocess vectorization, env checkpoint | Knows about PPO, advantages, or losses |
| `src/ppo/` | Rollout, GAE, clipped losses, the update pass, checkpoint orchestration, telemetry, pre-flight gates | Knows about RAM addresses or emulator internals |
| `src/checkpointing/` | Atomic writes, newest-first discovery, retention glob | Knows what a checkpoint contains |
| `src/hf_storage/` | `HfClient` / `AtomicHfClient` Protocols + `RealHfClient`, retry with rate-limit backoff | Imports from any sub-project |
| `src/observability/` | JSON-lines logging, W&B wrapper, contact sheets | Is optional in a long-running component |
| `src/torch_utils.py` | `autocast_dtype()` — the one device->dtype policy every training loop shares | Contains anything project-specific |
| `src/config_io.py` | `load_dataclass_config()` — YAML-to-frozen-dataclass loading shared by every `configs/*.yaml` owner | Imports from any sub-project |

Entry points (`pyproject.toml`): `data-collection {curate,run}`,
`contrastive-pretrain {preview,train,build-cache,export-frozen-encoder}`, and
`pokemon-ppo {train,preflight}`. There is no `pokemon_env` CLI yet.

## Frozen contracts — changing one silently breaks a trained model

- **Action space**: `session.BUTTONS = ("down","left","right","up","a","b","start")`.
  Index order *is* the action space; reordering remaps every action a
  checkpointed policy learned. `ACTION_DIM` and `PolicyConfig.action_dim` (7)
  must agree.
- **Aux vector**: 32-d, `AUX_STATE_DIM`, stamped with `AUX_STATE_VERSION`.
  Changing the slot layout requires bumping the version — env checkpoints
  validate it in two places and must keep doing so.
- **Encoder interface**: `(N, 1, 144, 160)` uint8-scale float in, 2048-d out.
  Grayscale->3-channel replication lives *inside* the module so training and
  `load_frozen_encoder()` share one contract. Conv+BN fusion happens only at
  export, never during training.
- **Latent stats**: `latent_stats.json` ships with the encoder and is validated
  on load (shape vs `latent_dim`, `std > 0` per dimension). A dead channel with
  `std == 0` divides by the 1e-6 floor and feeds ~1e6-scale inputs to the value
  head.
- **Init state**: `artifacts/init.state`'s hash is recorded in env checkpoints
  so a resume detects that the starting state moved. `INTRO_SCRIPT` is the
  reviewable source; nothing ROM-derived enters git.
- **Autoreset ordering**: `VecPokemonEnv` does next-step autoreset because
  `RolloutCache.reset(done)` requires it. Don't move autoreset into
  `EnvSession`.

## Conventions

- **Package management:** `uv` only. No bare `pip`/`venv`. `uv add <pkg>`, `uv run <cmd>`.
- **SOLID**: single-responsibility modules with injected dependencies (network/IO
  clients, filesystem, emulators, subprocess spawn) so core logic is
  unit-testable without hitting the network, ffmpeg, a GPU, or a ROM. Every
  external boundary in this repo is already a Protocol with a hand-written fake
  — add to that pattern, don't bypass it.
- **TDD**: write the failing test first for any new logic. Pure/algorithmic code
  gets fast unit tests with synthetic fixtures; anything touching real
  video/network/Hub/ROM APIs gets a separate, explicitly slow/opt-in test.
- **Karpathy guidelines**: minimal surgical diffs, no speculative abstraction,
  verify one stage works end-to-end before building the next stage on top of it.
- **Observability first**: every pipeline/training component logs structured
  (JSON-lines) progress and emits a live Weights & Biases (W&B) run. Anything
  that filters or drops data (dedup, anomaly detection, reward shaping) logs
  *why*, and produces a periodic visual artifact (contact sheets, sample grids,
  reward curves) a human can sanity-check without reading raw logs.
  (Migrated from Trackio: self-hosting its dashboard on a RunPod pod requires
  port-forwarding to view from outside the pod, which made it unusable for
  actually monitoring a run in progress.)
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
- **Record what you deliberately did not fix.** Known gaps live in the relevant
  spec's handoff section (see the env spec's "Known gaps carried out of
  implementation"), not in a scratch workspace that gets deleted.

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
- **The two documented `inference_mode` exceptions**: `LatentEncoder.encode()`
  and `RecurrentTransformerPolicy.step()` use `@torch.no_grad()` instead,
  because their outputs enter an autograd graph later at the PPO update. An
  inference tensor raises "Inference tensors cannot be saved for backward" at
  the *first update*, on a paid GPU. `.clone()` does not fix it. Tests assert
  `is_inference()` on both — keep them.
- **Device**: `torch.accelerator.current_accelerator(check_available=True) or
  torch.device("cpu")`. Tensors move by copy (`x = x.to(device)`), modules move
  in place (`model.to(device)`).
- **Mixed precision**: `torch.autocast(device.type, dtype=...)`, no `GradScaler`
  — bf16 on CUDA/CPU, fp16 on MPS (`train.autocast_dtype`). Validation runs under
  the *same* autocast context as the training step, or its numbers don't describe
  that step.
- **Checkpoints**: save `state_dict` plus optimizer/scheduler/step, never the
  model object; load with `weights_only=True` (which is why `PolicyConfig` and
  the coord set travel as plain dicts/tensors, not dataclasses or tuple-keyed
  dicts). Save and export from the raw module, not the `torch.compile` wrapper —
  compiled `state_dict` keys carry an `_orig_mod.` prefix. All file I/O goes
  through `checkpointing.io`; each sub-project owns only its state schema.
- **Perf**: `torch.compile` after `.to(device)`; `channels_last` applied to both
  the conv module and every input batch; fixed batch shape so `cudnn.benchmark`
  and `torch.compile` don't re-tune or recompile.
- **Optimizer**: AdamW (`weight_decay` already defaults to `0.01` in 2.13),
  cosine per epoch or OneCycle per batch.
- **Debugging**: when training runs but doesn't learn, overfit a single batch
  (~200 steps to near-zero loss) *before* touching hyperparameters, data, or
  architecture. It collapses the search space to model/loss/step.
  `tests/integration/test_sequence_model_overfit.py` is that gate for the policy.

## Sequence model (transformer)

Implemented in `src/sequence_model/`. Config in `configs/sequence_model.yaml`
mirrors `PolicyConfig` defaults: `d_model` 512, 8 layers, 8 heads, `head_dim`
64, `n_kv_heads` 2, `d_ff` 1408, `context_len` 1024, `rope_theta` 10000,
QK-norm on. Inputs are frozen-CNN latents, not tokens — vocab size, embedding
share and weight tying do not transfer from an LM config.

- Size it with arithmetic, not prose. Run the transformer-architecture skill's
  `config_budget.py` before quoting any parameter count, FLOP, KV-cache, or VRAM
  number; hand-estimated parameter counts are the usual source of wrong configs.
- GQA group count and often depth are decided by the KV cache at the real
  serving batch and context — here that is 64 envs x 1024 steps per PPO update.
  A config chosen without that number is a guess.
- Defaults needing a stated reason to change: pre-norm RMSNorm, RoPE, SwiGLU
  with `d_ff = round_to_128(8/3 * d_model)`, no linear biases, bf16 compute with
  fp32 master/Adam state, grad clip 1.0, AdamW betas (0.9, 0.95).
- Four bugs pass every shape check and must keep their property tests: RoPE
  applied before the head split; interleaved-vs-halves RoPE pairing (this repo
  uses **halves** — trains fine either way, silently incompatible with an
  external checkpoint); `x.repeat()` instead of expand-reshape in `repeat_kv` (a
  query-head permutation — breaks checkpoint interop and fused kernels, *not* a
  quality regression, and reporting it as one costs a day); `is_causal=True`
  during incremental decode (prefill perfect, generation garbage).
- The chunk mask is causal AND sliding-window AND same-episode. All three, or
  the model attends across an episode boundary.
- Log leading indicators, not just loss and grad norm — both are late.
  `sequence_model.telemetry` exposes attention logit magnitude, attention
  distance mass, and residual norm; they predict divergence well ahead of it.

## Testing

pytest 9.1.1 with strict config in `pyproject.toml` (`strict_config`,
`strict_markers`, `strict_xfail`, `strict_parametrization_ids`,
`filterwarnings = error`, **branch coverage floor 93%**). Keep those gates and
ratchet the floor up; a floor you have to lower is worse than none. Current
state: 768 passing, 95.02% coverage, 14 deselected `slow`.

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
- **Never load `.env` in the test process.** A test run that picked up the
  developer's `.env` would authenticate as them against real private repos and a
  real W&B account with nothing asking for it. The live tier reads the *ambient*
  credential via `requires_hf_credentials` in `tests/conftest.py`, which skips
  legibly on a dev machine and runs on a pod.
- **Doubles**: hand-written fakes typed against the Protocol the consumer
  actually uses (`FakeHfClient`, `FakeEmulator`) over `mock.patch` — the suite
  patches in exactly one place, and there only as a `wraps=` call-count spy, not
  as a stub. If a stub is genuinely unavoidable, `autospec=True` and patch where
  the object is *used*, not where it's defined. Never call a bare `Mock` method
  that only looks like an assertion (`m.called_once_with(...)` auto-creates a
  truthy attribute and passes forever); the real names start with `assert_`.
- **ML tests**: seeded, CPU-only, tiny synthetic tensors by default. Anything
  needing the real Hub, ffmpeg, a real ROM, or real pretrained weights carries
  `@pytest.mark.slow`, stays deselected by default, and skips with a stated
  `reason=` when its dependency is absent — a fresh checkout must never fail.
- **A green suite over an untested tier is the failure mode to watch for.**
  `docs/2026-08-26-slow-test-suite-blocked.md` is the worked example: five slow
  tests failed for two unrelated reasons, and three of them could never have
  passed. Read it before adding a test that needs real credentials.
- Run the pytest-expert skill's `scripts/audit_tests.py tests/` after adding
  tests — it statically catches the gates above, and a green `pytest` over a
  suite it flags means the tests aren't testing.

## Infra

- **Compute**: RunPod (CPU pods for data pipeline and cache building, GPU pods
  for training). Long runs go in `tmux` so an SSH drop doesn't kill them.
- **Storage/datasets**: Hugging Face Hub, private repos. `objones25/pokemon-frames`
  (dataset, 367 shards / 64.3 GB, native-resolution grayscale frames as the
  `datasets` `Image` feature) and `objones25/pokemon-contrastive-encoder`
  (model, the frozen artifact).
- **Cost/quota hazards, all previously hit**: HF's hourly commit limit (that is
  what `--checkpoint-interval` guards); `hf_hub_download` never cleaning its
  cache, so `resize_cache._discard_hub_blob` deletes each blob after use — the
  raw shards are 64.3 GB against a 50 GB volume; and `checkpoint_keep_last_n`,
  without which a 100-epoch run writes ~46 GB of checkpoints. Pruning is safe
  because the best encoder is pushed to the Hub on every val-loss improvement.
- **Never commit**: ROMs (`*.gb`, `*.gbc`), `artifacts/` save states, `.env`,
  `/data/`, `/scratch/`. The repo has a public GitHub remote.
- **Data source**: curated YouTube longplay URLs, human-approved per video
  (see the data-collection spec) — never fully automated scraping/search.
