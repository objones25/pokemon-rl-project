# Pokemon RL

A vision-based reinforcement learning agent for Pokemon Red, built in five
independent stages. The end state is a policy that plays the game from pixels:
a frozen contrastive-pretrained CNN turns each Game Boy frame into a latent
vector, a decoder-only transformer reads the sequence of those latents, and PPO
trains it against a shaped reward read out of the emulator's RAM.

Repository: <https://github.com/objones25/pokemon-rl-project>

## Credit

This project would not exist without **Peter Whidden**'s work on the same
problem:

- Repository: <https://github.com/PWhiddy/PokemonRedExperiments>
- Talk: *Training AI to Play Pokemon with Reinforcement Learning* —
  <https://www.youtube.com/watch?v=DcYLT37ImBY>

The video and the repository are the direct inspiration for this project.
Concretely, `src/pokemon_env/ram.py` — every RAM address and the decoding of
each one — is built from PokemonRedExperiments' verified readers rather than
inferred from a wiki, and `configs/pokemon_env.yaml` pins `action_freq`,
`max_steps` and `n_envs` to the values PokemonRedExperiments v2 uses. Where
this project diverges (the reward shaping, the frozen-CNN + transformer
architecture, the 64-way subprocess vectorization) it says so in the design
specs under `docs/superpowers/specs/`.

## How the pieces fit

```text
 YouTube longplays ──> [1] data collection ──> HF dataset (grayscale frames)
                                                      │
                                                      v
                                            [2] contrastive pretraining
                                                (SimCLR, ResNet-50)
                                                      │
                                                      v
                                             frozen encoder on HF Hub
                                            (144x160 frame -> 2048-d latent)
                                                      │
        [4] Pokemon Red env  ──frames+RAM──>          v
        (64 PyBoy workers)   <──actions──   [3] transformer policy
                                            (RoPE / GQA, 1024-step context,
                                             actor head + critic head)
                                                      │
                                                      v
                                              [5] PPO trainer
```

Stage 2 learns a visual representation from *human* play (YouTube longplays),
so the RL agent in stage 5 never has to learn to see and to play at the same
time. The encoder is frozen before PPO ever runs.

## Status

| Stage | Package | State |
| --- | --- | --- |
| 1. Data collection | `src/data_collection/` | Built. Dataset uploaded: 367 shards, 64.3 GB. |
| 2. Contrastive pretraining | `src/contrastive_pretrain/` | Built. Trains on RunPod GPU; exports a frozen encoder to the Hub. |
| 3. Sequence model | `src/sequence_model/` | Built. Policy, KV-cached rollout step, chunked training forward, checkpoint schema. |
| 4. Pokemon Red environment | `src/pokemon_env/` | Built. 64 subprocess emulators, RAM observations, reward, checkpoint/resume. |
| 5. PPO trainer | `src/ppo/` | Built. Connects 3 and 4. Four pre-flight gates stand between it and the first paid run. |

Test suite as of this writing: **775 passing, 95.05% branch coverage** (the
floor in `pyproject.toml` is 93%), plus 14 opt-in `slow` tests that need a real
ROM, real ffmpeg, or real Hub credentials.

## Repository layout

```text
configs/                       run configuration, one YAML per sub-project
  contrastive_pretrain.yaml    training hyperparameters and pod paths
  pokemon_env.yaml             env sizing and reward weights
  ppo.yaml                     PPO hyperparameters, cadences, encoder pin
  sequence_model.yaml          transformer architecture
  video_sources.yaml           the approved-video registry (written by `curate`)
docs/
  superpowers/specs/           design rationale, one per sub-project
  superpowers/plans/           the implementation plan each spec was built from
src/
  checkpointing/               atomic checkpoint writes, discovery, retention
  contrastive_pretrain/        SimCLR encoder, augmentation, training, export
  data_collection/             video -> frames -> deduped Parquet shards -> Hub
  hf_storage/                  HfClient Protocol + real implementation, retry
  observability/               JSON logging, W&B wrapper, contact-sheet previews
  pokemon_env/                 PyBoy wrapper, RAM readers, reward, vectorization
  ppo/                         rollout, GAE, losses, update, checkpointing, CLI
  sequence_model/              RoPE/GQA transformer policy, KV cache, telemetry
tests/
  unit/                        fast, no network / ffmpeg / GPU / ROM
  integration/                 opt-in, `-m slow`
Pokemon_RL_Architecture_Plan.pdf  the rationale everything traces to
```

Two things are deliberately absent from git and must be supplied locally: a
Pokemon Red ROM at the repo root (`*.gb` is gitignored) and `artifacts/`, which
holds the generated emulator save state.

## Getting started

### Prerequisites

- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). This project uses `uv`
  exclusively — `uv sync`, `uv add`, `uv run`. Never bare `pip` or `venv`.
- **`ffmpeg`** on your `PATH`, for stage 1 only. macOS: `brew install ffmpeg`.
- A Hugging Face account, for stages 1–2. The dataset and encoder repos are
  private and must already exist; nothing here creates them for you.
- A legally obtained Pokemon Red ROM, for stage 4. Not distributed here.

### Install

```bash
uv sync
```

Create a `.env` at the repo root (already gitignored):

```text
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

Both CLIs load `.env` on every invocation. The commands that spend real time
or money — `data-collection run`, `contrastive-pretrain train`, and
`build-cache` — check for the credentials they need up front and fail with a
clear message, rather than partway through a long extraction or a paid GPU run.

## Stage 1 — collect frames

Two phases, because one of them needs a human and the other needs to run
unattended for hours.

**Phase A, `curate`** runs on your machine and never touches the Hub. It pulls
one frame from a candidate video, proposes a crop box with ffmpeg's `cropdetect`
(longplay uploads range from 360p to 4K, letterboxed in every possible way),
writes an annotated preview to `configs/templates/approved/_preview.png`, and
asks:

```bash
uv run data-collection curate "https://www.youtube.com/watch?v=<id>" --game red
```

`[a]pprove / [m]anual entry / [r]eject`. Manual entry lets you type new
`x/y/w/h` and loops back to a fresh preview. Nothing reaches
`configs/video_sources.yaml` without an explicit approval — video selection is
human-gated by design, never scraped.

**Phase B, `run`** is unattended and meant for a cheap CPU pod:

```bash
uv run data-collection run --repo-id you/pokemon-frames
```

For every approved video not already marked complete in the dataset repo's
`manifest.json`, it streams frames through ffmpeg at 2 fps (video never touches
local disk), drops near-duplicates by perceptual hash, and uploads Parquet
shards plus a contact-sheet preview PNG per batch. A crashed run resumes
mid-video from its last checkpoint. Useful flags: `--batch-size`,
`--checkpoint-interval`, `--max-concurrent-videos`.

Exits nonzero if any video failed; look for `"message": "video_failed"` in the
JSON logs on stdout, and at `manifest.json` in the dataset repo for the reason.

## Stage 2 — pretrain the encoder

SimCLR (NT-Xent) over a ResNet-50 backbone with the ImageNet stem's initial
maxpool removed, which preserves the spatial detail that HP bars and text
glyphs live in. The module takes a `(N, 1, 144, 160)` grayscale tensor and
emits a 2048-d feature; grayscale-to-3-channel replication happens inside the
module so training and inference share one contract.

Check the augmentation policy visually before spending GPU time:

```bash
uv run contrastive-pretrain preview --frames-dir <dir> --out preview.png
```

Build the resize cache once per machine, then train:

```bash
uv run contrastive-pretrain build-cache --config configs/contrastive_pretrain.yaml
uv run contrastive-pretrain train --config configs/contrastive_pretrain.yaml
```

`build-cache` downloads every shard, resizes it to 144x160, and writes ~1.3 GB
of Parquet locally. `train` would do this implicitly on first start, but only
after `torch.compile` and the memory probe — so on a GPU pod it would leave the
GPU idle through an hour of pure CPU work. It is interrupt-safe; finished
shards are skipped on a rerun.

`train` checkpoints on an interval, resumes automatically from the newest
checkpoint in `network_volume_checkpoint_dir`, and pushes the *best* encoder to
the Hub model repo on every validation-loss improvement. That Hub artifact —
fused Conv+BN weights as safetensors, a JSON config, and `latent_stats.json` —
is the only thing downstream stages consume. To re-export it from a saved
checkpoint by hand:

```bash
uv run contrastive-pretrain export-frozen-encoder --checkpoint <path>
```

Mid-epoch resume is *approximate*: the shuffle buffer is not part of
the checkpointed state, so a resumed run re-serves up to `shuffle_buffer_size`
rows per worker in a different order. Harmless for SimCLR, but it is not an
exact-resume guarantee.

## Stage 3 — the policy

A decoder-only transformer over frame latents rather than tokens. Defaults in
`configs/sequence_model.yaml`: 8 layers, `d_model` 512, 8 query heads of
`head_dim` 64, 2 KV heads (grouped-query attention), `d_ff` 1408, 1024-step
context, RoPE with `theta` 10000, QK-norm on.

Each timestep is fused into one token from four inputs: the normalized 2048-d
frame latent, the 32-d RAM-derived aux state, the previous action, and the
previous reward. Two heads come out: an actor over the 7 actions and a critic.

The module has two entry points, and PPO uses both — `step()` for the
KV-cached, one-timestep-at-a-time rollout, and `forward_chunk()` for the
chunked, masked forward at update time. There is no training loop in this
package; it is the model, not the trainer.

## Stage 4 — the environment

64 PyBoy emulators, each in its own subprocess, with frames handed back to the
parent through shared memory. Each step holds a button for 8 frames and ticks
24 total; episodes run 163,840 steps.

Observations are a `(144, 160)` uint8 frame plus a 32-d normalized vector read
straight out of RAM — party levels and HP, badges, event flags, map and
coordinates, money, battle state, episode progress. The reward is the
architecture plan's section-4 shaping (badges, healing, exploration, events,
levels), each component scored against its own historical maximum so that
depositing and re-withdrawing a Pokemon cannot farm reward. `state_dict()` /
`load_state_dict()` checkpoint all 64 emulators (10.7 MB total, measured) so a
run resumes at the exact game position it stopped at.

**One manual step is not yet automated.** Every env resets from
`artifacts/init.state`, a save state past the title, intro and naming screens.
`src/pokemon_env/init_state.py` holds the committed button script and
`generate_init_state()` returns the bytes, but no command writes them to disk
yet — the only caller today is the integration test. Until that command exists,
`artifacts/init.state` has to be produced by hand from those pieces.

## Stage 5 — the PPO trainer

Drives the stage-4 environment with the stage-3 policy. One update is 64 envs ×
1024 steps = 65,536 transitions, split into 8 env-minibatches over 3 epochs for
24 optimizer steps. A minibatch is a *subset of envs at full length*, never a
slice of time: each trained step is preceded by a 1023-step burn-in prefix —
earlier observations recomputed so the transformer sees a full context window —
and cutting the time axis would sever it.

```bash
uv run pokemon-ppo preflight --n-envs 16 32 64 --steps 100   # one run per count
uv run pokemon-ppo train           # resumes from the newest complete pair
uv run pokemon-ppo train --fresh   # discard it, start a new W&B run
```

Both subcommands read `configs/ppo.yaml`, `configs/pokemon_env.yaml` and
`configs/sequence_model.yaml`; `--ppo-config` / `--env-config` /
`--policy-config` override each path.

Two design decisions carry most of the correctness weight. **`π_old` and
`V_old` are recomputed** by a `no_grad` pass at the start of every update rather
than reusing the values recorded during rollout, because the KV cache is carried
across update boundaries and the rollout path is therefore slightly stale;
`max|ratio − 1|` at the first minibatch of the first epoch is then exactly 0,
and the trainer asserts it rather than logging it, aborting the run if it ever
differs. **The policy and environment checkpoints are committed together by a
manifest written last**, because `save_checkpoint` is atomic per file but cannot
see the failure where only one of the two lands — a policy restored against an
emulator at a different game position is worse than starting fresh. Resume
therefore scans manifests newest-first and takes the first whose two files both
exist at their recorded sizes, skipping any half-written pair.

Checkpoints go to the RunPod network volume, not the Hub: writing one every ~20
minutes for 48 hours would walk back into the Hub's hourly commit limit this
project has already hit once.

**Not yet run for real.** `docs/superpowers/specs/2026-08-27-ppo-trainer-design.md`
§8 names four gates before the first
paid run — the SDPA backend measurement, rollout throughput at 16/32/64 envs, a
memory probe, and a 50-update live smoke run. `preflight` implements the first
two. The `-m slow` acceptance tests exist but have never executed against a real
ROM, and that spec's §12 records the rest of what was deliberately left undone.

## Testing

```bash
uv run pytest             # the fast suite; `slow` is deselected by default
uv run pytest -m slow     # opt-in: real ROM, real ffmpeg, real Hub credentials
```

The fast suite needs no network, no GPU, no ffmpeg and no ROM — every external
boundary (the emulator, the Hub client, the frame source, the subprocess spawn)
is a Protocol with a hand-written fake. The `slow` tests skip themselves with a
stated reason when their dependency is missing, so a fresh checkout never fails
on a missing ROM.

`pyproject.toml` sets a 93% branch-coverage floor, treats warnings as errors,
and enables pytest's strict config/marker/xfail modes.

## Design documents

Each sub-project was specified and planned before any code was written. The
specs are the reasoning; the plans are the task breakdown they were built from.

| Sub-project | Spec |
| --- | --- |
| Data collection | `docs/superpowers/specs/2026-08-21-data-collection-pipeline-design.md` |
| Augmentation policy | `docs/superpowers/specs/2026-08-23-contrastive-augmentation-policy-design.md` |
| Contrastive model | `docs/superpowers/specs/2026-08-24-contrastive-pretrain-model-design.md` |
| Resize cache | `docs/superpowers/specs/2026-08-25-contrastive-pretrain-resize-cache-design.md` |
| Sequence model | `docs/superpowers/specs/2026-08-26-temporal-sequence-model-design.md` |
| Pokemon Red env | `docs/superpowers/specs/2026-08-26-pokemon-env-design.md` |
| PPO trainer | `docs/superpowers/specs/2026-08-27-ppo-trainer-design.md` |

`Pokemon_RL_Architecture_Plan.pdf` is the document all of them trace back to.
`docs/2026-08-26-slow-test-suite-blocked.md` is a worked example of how this
project handles a "green suite, untested tier" problem — worth reading before
adding a test that needs real credentials.

`CLAUDE.md` holds the working conventions and the invariants that are not safe
to change casually. Read it before opening a pull request.

## ROMs and legality

No ROM, save state, or game asset is distributed in this repository, and none
may be committed to it (`*.gb`, `*.gbc` and `artifacts/` are gitignored). Stage
4 requires a copy of Pokemon Red you obtained legally. Stages 1–3 do not need
one.
