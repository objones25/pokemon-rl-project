# Contrastive Pretraining — Model & Training Pipeline Design Spec

Date: 2026-08-24
Status: Approved for planning

## Purpose

Build the SimCLR-style ResNet-50 encoder and training pipeline that turns
`objones25/pokemon-frames` (296K Game Boy-resolution, 160x144 grayscale
frames) into a frozen semantic feature extractor, per
`Pokemon_RL_Architecture_Plan.pdf`'s "Visual Pre-Training Strategies"
section. This spec covers the encoder architecture, the training loop
(precision/performance settings, checkpointing, resume, retry, logging,
experiment tracking), and — critically — the interface contract the
future sequence-model/PPO stage will use to load these frozen weights.
It builds directly on the already-approved
`2026-08-23-contrastive-augmentation-policy-design.md`, whose
`contrastive_pretrain.augmentation` module (`make_pair`, `AugmentationConfig`)
is consumed as-is by the data pipeline described here.

Training runs on a single RunPod A100 pod; weights and training
checkpoints persist to a RunPod network volume attached to that pod.

## Scope boundary

In scope:

- ResNet-50-style encoder architecture (stem, backbone) and the SimCLR
  projection head + NT-Xent loss used only during pretraining.
- The training loop: data pipeline, precision/performance settings,
  checkpointing, resume-on-restart, retry/backoff for Hub calls,
  structured logging, Trackio experiment tracking.
- A documented, tested `load_frozen_encoder()` interface — this is how
  the future PPO stage consumes this stage's output, so its shape,
  weight format, and preprocessing contract are fixed here, now.
- Extracting the retry/HF-client code already flagged as reusable
  (`data_collection/retry.py`, the `RealHfClient` adapter in
  `data_collection/cli.py`) into a shared package.

Out of scope (deferred to their own specs):

- The sequence-model transformer (RoPE/GQA), and the "explicit affine
  normalization layer" the architecture plan says sits between the
  frozen encoder and the transformer — that layer is the sequence
  model's concern, not this stage's. This spec does, however, publish
  the raw-latent statistics (mean/std over a validation sample) that
  layer will need, since recomputing them later means re-running the
  frozen encoder over data again for no reason.
- PPO training itself.
- RunPod pod/network-volume provisioning automation — this spec assumes
  an A100 pod with a network volume already attached and mounted at a
  configured path; provisioning is a manual/CLI step outside this code.
- Hyperparameter tuning beyond informed starting defaults (batch size,
  LR, temperature, epoch count) — these are marked below as
  config-overridable and meant to be validated against the first real
  training run, the same way the augmentation spec left its own
  parameters open to revision once real data was available.

## Why this design, not alternatives considered

**Backbone stem — modified (no initial maxpool), from ImageNet-pretrained
weights.** Verified directly against the installed `torchvision` source
(`inspect.getsource`): `ResNet.maxpool` is `nn.MaxPool2d`, which owns zero
learnable parameters — it contributes 0 of the 320 keys in a ResNet-50
`state_dict()`. Replacing it with `nn.Identity()` after loading
ImageNet-pretrained weights is therefore a fully weight-compatible change,
not a from-scratch tradeoff. This matters because the standard stem (7x7
stride-2 conv + stride-2 maxpool, then four stride-2 stages) downsamples
160x144 input by 32x before global-average-pool, leaving a ~5x4 feature
map where each cell summarizes a 32x32 input region — coarse enough to
plausibly erase the HP-bar/text-glyph detail the augmentation spec was
explicitly protecting. Dropping just the maxpool halves that loss (final
map ~10x9) while keeping ResNet-50's actual block topology
(`[3, 4, 6, 3]` bottleneck stages) untouched.

**Pretrained init, grayscale handled by channel replication, not conv1
surgery.** `conv1` is `nn.Conv2d(3, 64, ...)` — it expects 3 input
channels. Two ways to reconcile that with 1-channel frames: replicate the
grayscale frame to 3 channels at the input transform (conv1's pretrained
filters are used completely unmodified), or average the pretrained
3-channel filters down to 1 channel (uses conv1 more efficiently but is
itself a lossy approximation of what was actually trained). This spec
uses replication — it's the choice that changes nothing about the
pretrained weights themselves, at the cost of a trivially small amount of
redundant compute in the single cheapest layer of the network.

**Frozen output = 2048-d backbone feature, not the SimCLR projector
output.** This matches SimCLR's own published finding that the
projection head's output transfers worse than its input — the projector
exists only to shape the loss, not to produce the downstream feature.
Concretely: backbone (`conv1..layer4` + global-avg-pool) produces a
2048-d vector; a 2-layer MLP projector (2048→2048→128) sits on top only
during pretraining, for the NT-Xent loss, and is discarded when exporting
the frozen artifact. Reducing 2048-d further for the sequence model's KV
cache budget is explicitly that stage's job (its own input-embedding
layer), not this one's — this is a SOLID separation-of-concerns call, not
an oversight: this module's job is producing a good semantic feature,
not picking a transformer's hidden size.

**Weight format — safetensors + JSON config, plain loader function.**
`torch.save`/pickle deserialization can execute arbitrary code embedded
in the file. That's a real risk specifically because this artifact will
be downloaded and loaded on an unattended RunPod pod as part of an
automated pipeline — exactly the kind of path this repo's "production
readiness for long/costly runs" convention already asks be designed
defensively from the start, not patched later. safetensors can only ever
deserialize into tensors, removing that risk by construction. A
`transformers`-style `PreTrainedModel`/`PretrainedConfig` wrapper was
considered and rejected: it pulls in a large framework surface
(generation config, tokenizer plumbing, model registries) for an
interface whose actual requirement is `load_frozen_encoder(repo_id) ->
nn.Module` with a fixed output shape — a plain function is the minimal
thing that satisfies that (interface segregation).

**Shared HF-Hub code — new `src/hf_storage/` package.**
`data_collection/retry.py`'s own docstring already documents two
consumers; this stage's checkpoint upload and dataset access make three,
and the future sequence-model/PPO stages will almost certainly need the
same retry/rate-limit-aware Hub access too (per this project's own
infra convention of persisting everything to Hub-backed storage). Past
two-plus real consumers of identical logic, the duplication costs more
than a shared module. Named `hf_storage`, not `common`, because the code
in question — retry/backoff, the `HfClient` Protocol, the `RealHfClient`
adapter around `HfApi` — is entirely about one thing: talking to the Hub
reliably. A `common`/`utils`-named package has a well-known failure mode
of becoming an unscoped junk drawer over a project's life, because its
name doesn't constrain what belongs in it; a purpose-named package keeps
that boundary legible for as long as the project grows. Leaving the code
in `data_collection` and cross-importing was rejected on two grounds:
the dependency direction is backwards (a training package depending on
the data-pipeline package), and it doesn't fix the actual duplication —
`RealHfClient` today only exists inside `data_collection/cli.py`, not in
`retry.py`, so contrastive_pretrain would still end up writing its own
copy of the `HfApi` adapter.

**Data loading — HF streaming (`datasets`, `streaming=True`), no local
download, no network volume for the dataset itself.** The dataset is
64.3GB across 367 parquet shards (confirmed via the Hub API). An earlier
draft of this spec proposed downloading it once to the network volume;
that was wrong, and dropped after checking the actual current state of
the tooling rather than assuming streaming meant giving up
resumability or shuffle quality:

- `datasets.IterableDataset` has native `.state_dict()` /
  `.load_state_dict()`: state is the current shard plus in-shard example
  index, and resuming skips already-read shards and fast-forwards within
  the current one. The one real gap — shuffle-buffer _contents_ aren't
  exactly restored on resume, the buffer just refills with new data — is
  an accepted approximation for this project (exact reshuffling isn't a
  requirement here).
- `torchdata.stateful_dataloader.StatefulDataLoader` is a drop-in
  replacement for `torch.utils.data.DataLoader` that wraps this
  automatically, including with `num_workers > 0` (it aggregates state
  across worker processes). Constraint worth designing around: resuming
  requires the _same_ `num_workers` as the checkpointing run, so
  `num_workers` is a fixed, checkpoint-compatible config value, not
  something to change casually between resumes.
- Our 367 shards are already fine-grained for streaming shuffle quality
  (`datasets`' own docs flag _few_-file datasets as needing `reshard()`
  before shuffling; 367 isn't that). Each shard is one full video's
  frames in original temporal order (per `hf_uploader.py`'s
  `shards/{video_id}/{shard_index}.parquet` layout), so shard-order
  shuffling — the default behavior of `IterableDataset.shuffle()` — is
  exactly what decorrelates consecutive, highly-similar frames from the
  same video.

**Augmentation randomness — per-row deterministic seed, not a shared
`torch.Generator`.** A single `torch.Generator` instance passed into the
`.map()` transform would be copied as-is into every `StatefulDataLoader`
worker process (`num_workers > 0` forks/spawns the dataset, including
whatever the transform closes over); every worker would then start from
an identical RNG state and produce correlated, not independent,
augmentation sequences across the parallel stream — a real correctness
gap the original draft of this section didn't address. Instead, each
row's augmentation seed is derived deterministically from data the row
already carries: `sha256(f"{base_seed}:{video_id}:{timestamp_s}")`,
truncated to a `torch.Generator.manual_seed()`-sized int. This sidesteps
the multi-worker correlation problem entirely (every worker computes the
same seed for the same row, independent of which worker happens to
process it), makes a given frame's augmentation exactly reproducible
from `(base_seed, video_id, timestamp_s)` alone, and needs zero
checkpoint state — resuming re-derives the identical seed rather than
restoring saved generator state. The training checkpoint dict below
therefore does **not** include an `augmentation_rng` entry.

Streaming ties every training step to Hub network throughput for the
run's duration; for 160x144x1-byte frames this is not expected to
bottleneck a GPU doing ResNet-50 forward/backward, but the training loop
logs data-loading wait time and GPU utilization from the first run (see
Observability below) so this is verified from real metrics, not assumed.

**Checkpoint storage — two tiers, split by cost and purpose.** Frequent
full-training-state checkpoints (model + optimizer + scheduler +
dataloader + augmentation RNG) go to the local network volume — fast,
no Hub commit-rate-limit exposure (this project already hit HF's
256-commits/hour limit once during data collection), no network
dependency on the hot path. The frozen, weights-only encoder artifact —
the thing PPO actually consumes — is pushed to an HF Hub model repo on a
coarser interval (end of epoch / new best validation loss), reusing
`hf_storage`'s retry/rate-limit-aware uploader, mirroring the exact
pattern `hf_uploader.py` already uses for `manifest.json`.

**Precision — bf16 autocast, no GradScaler.** A100 has native bf16
support (compute capability 8.0); bf16 shares fp32's exponent range, so
unlike fp16 it doesn't need loss scaling to avoid gradient underflow.
Confirmed against PyTorch's own AMP docs, which pair `GradScaler`
specifically with the fp16 autocast example, not bf16, and separately
document that bf16 autocast on an unsupported device raises rather than
silently falling back — i.e., bf16 is meant to be used directly. This
also simplifies the checkpoint: no scaler state to save/restore in the
correct order alongside everything else.

**torch.compile at `mode="default"`, not `"reduce-overhead"` or
`"max-autotune"`, for the first working version.** Both of the more
aggressive modes enable CUDA graphs by default, which impose real
constraints (static tensor shapes/addresses, a "capturable" optimizer,
no input mutation) that interact awkwardly with exactly the
resumable-checkpoint design this spec depends on. PyTorch's own docs
describe `reduce-overhead`'s benefit as targeting _small-batch,
Python-overhead-bound_ workloads — a ResNet-50 forward/backward is real
GPU compute, not obviously that regime. `default` mode still gets
automatic operator fusion via the Inductor backend (this is where
"operator fusion" mostly lives for the training loop — no separate
manual fusion step is needed there). `max-autotune` is left as a
phase-2 tuning experiment once the base pipeline is verified end-to-end,
per this repo's own convention of verifying one stage before building on
it — not a phase-1 requirement.

**Activation checkpointing (`torch.utils.checkpoint`) and hand-rolled
CUDA Graphs — deliberately not included.** Both trade real implementation
complexity for a benefit that isn't clearly needed at this problem's
scale: 160x144 frames are far smaller than the 224x224+ images these
techniques are usually reached for, and a single A100 has comfortable
headroom for a ResNet-50-scale model at the batch sizes SimCLR wants
here. Building either speculatively would be exactly the kind of
premature abstraction this repo's conventions warn against. If profiling
a real run shows we're memory-bound (activation checkpointing) or
launch-overhead-bound (CUDA graphs), `torch.compile` already has
lower-effort levers for both — an automatic activation-memory-budget
knob, and `mode="max-autotune"` for CUDA graphs — that are preferable to
hand-instrumenting the model.

**Batch size 1024, with a fail-fast startup memory probe.** Dropping
`maxpool` (see above) means the network's total downsampling is 16x per
dimension instead of 32x, so every stage's feature map has ~4x more
spatial positions than a standard ResNet-50 would have at the same input
size. Our 160x144 input is ~0.46x the pixel count of ImageNet's 224x224,
so net-net this backbone's activation memory per image is roughly ~1.8x
a standard ResNet-50 at 224x224 (4x from the missing downsample step,
0.46x from the smaller input) — a real, deliberate cost of the detail-
preservation choice made above, not free. Standard ResNet-50 at batch
256 in mixed precision is commonly cited around 8-12GB of activation
memory; by that rough scaling, `batch_size=512` was already plausibly in
the 30-40GB range before optimizer state, gradients, and workspace —
tight on a 40GB card. Confirmed the training pod is an 80GB A100, which
gives enough headroom to double to `batch_size=1024` (SimCLR quality
benefits directly from more implicit negatives per batch) — but this
arithmetic is a rough estimate, not a measurement, so the training loop
runs a startup memory probe: a few dummy forward+backward steps at the
configured `batch_size`, using real model/precision/memory-format
settings, _before_ touching the real streaming dataset or checkpoint
state. A `CUDA out of memory` here raises immediately with an actionable
message (lower `batch_size` in config, or add gradient accumulation) —
deliberately not an automatic halve-and-retry, because silently training
at a different batch size than the one `learning_rate`/`warmup_steps`
were set for would produce a quietly-worse run that still "succeeds"
instead of a loud, obvious, fixable failure.

Doubling batch size also isn't free of coupling to the schedule: halving
steps-per-epoch means the same `warmup_steps` value now covers twice as
much of the first epoch as intended, so it's halved alongside the batch
size (1000 → 500) to keep warmup covering roughly the same number of
examples seen. `learning_rate` is nudged up using the √2 heuristic
common for adaptive optimizers (SGD's linear scaling rule doesn't apply
to AdamW) rather than left unchanged: 3e-4 → 4e-4.

**Conv+BatchNorm fusion — applied only at frozen-artifact export, not
during training.** BatchNorm needs live batch statistics while training;
fusing it into the preceding conv only makes sense once the model is
frozen for inference. Since PPO will call this encoder on every
environment step, folding BN's affine transform into conv1's weights at
export time (`torch.nn.utils.fusion.fuse_conv_bn_eval`) is a one-time
cost at export that then benefits every downstream inference call.

## Package layout

```
src/hf_storage/                    # new — extracted, generalized shared module
    __init__.py
    retry.py                       # moved from data_collection/retry.py, unchanged
    client.py                      # HfClient Protocol + RealHfClient (moved/generalized
                                    # from data_collection/cli.py's inline RealHfClient)

src/data_collection/
    hf_uploader.py                 # updated: imports retry/HfClient from hf_storage
    cli.py                         # updated: imports RealHfClient from hf_storage

src/contrastive_pretrain/
    augmentation.py                # existing, unchanged
    model.py                       # ResNet-50 encoder (modified stem) + SimCLR projector
    dataset.py                     # streaming dataset construction, StatefulDataLoader wiring
    losses.py                      # NT-Xent / InfoNCE
    checkpoint.py                  # training-checkpoint save/load, two-tier storage orchestration
    encoder_io.py                  # load_frozen_encoder() — the PPO-facing interface
    train.py                       # training loop orchestration (Deps dataclass, resume, retry)
    cli.py                         # existing `preview` command + new `train` / `export-frozen-encoder`

configs/
    contrastive_pretrain.yaml      # new — hyperparams, paths, repo ids (see below)
```

`hf_storage` has no dependency on `data_collection` or
`contrastive_pretrain`; both depend on it. This keeps the dependency
graph a straightforward two-consumers-of-one-shared-module shape rather
than a cross-import between sibling packages.

## Model architecture

```python
def build_encoder(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Returns (backbone, embedding_dim). backbone maps a (N, 1, 160, 144)
    grayscale, channels_last-eligible tensor to a (N, 2048) feature."""
```

- `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if
pretrained else None)`, then `backbone.maxpool = nn.Identity()`, then
  drop `backbone.fc` (replace with `nn.Identity()` or slice it off —
  output is the post-`avgpool`, pre-`fc` 2048-d vector).
- Channel replication (`frame.repeat(1, 3, 1, 1)`, grayscale → pseudo-RGB)
  lives _inside_ `build_encoder`'s returned module, wrapping the
  torchvision backbone, not in an external transform. This is
  deliberate: it means both the training pipeline and
  `load_frozen_encoder()` (below) present the exact same external
  contract — raw 1-channel grayscale in, 2048-d feature out — with the
  replication detail encapsulated in one place instead of being a
  convention both call sites have to separately remember.
- Projection head (training-only, discarded at export):
  `nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(inplace=True),
nn.Linear(2048, 128))`.
- Loss: NT-Xent / InfoNCE over the projector's 128-d output, standard
  SimCLR formulation — every other example in the batch (post-projector,
  both views) is an implicit negative.

Config defaults (all overridable via `configs/contrastive_pretrain.yaml`,
all meant to be validated against the first real training run rather
than treated as final):

| Parameter        | Default           | Note                                                                                                                |
| ---------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `batch_size`     | 1024              | Sized for an 80GB A100 (see rationale above); verified at startup by a fail-fast memory probe, not assumed          |
| `optimizer`      | AdamW             | LARS (the SimCLR paper's choice) mainly earns its keep at batch sizes ~4096+; not justified at 1024                 |
| `learning_rate`  | 4e-4              | Cosine decay with linear warmup; scaled from a 3e-4/batch=512 baseline via the √2 heuristic for adaptive optimizers |
| `warmup_steps`   | 500               | Halved from a batch=512 baseline of 1000 so warmup covers the same number of examples seen                          |
| `weight_decay`   | 1e-6              | Matches SimCLR paper's own small value                                                                              |
| `temperature`    | 0.1               | NT-Xent temperature                                                                                                 |
| `max_epochs`     | 100               |                                                                                                                     |
| `projector_dims` | 2048 → 2048 → 128 | Standard SimCLR projector                                                                                           |

## Data pipeline

```python
dataset = datasets.load_dataset(
    "objones25/pokemon-frames", streaming=True, split="train"
)
dataset = dataset.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.seed)
dataset = dataset.map(_to_pair)  # contrastive_pretrain.augmentation.make_pair
dataloader = StatefulDataLoader(
    dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,       # fixed across resumes — see above
    drop_last=True,                       # fixed batch shape for cudnn.benchmark/compile
    pin_memory=True,
    snapshot_every_n_steps=config.checkpoint_interval_steps,
)
```

- `shuffle_buffer_size` default: 10,000 examples (matches the scale used
  in `datasets`' own streaming-shuffle documentation; single-channel
  160x144 frames make this a trivial ~460MB of host RAM even before
  decoding overhead).
- `drop_last=True` is required for two independent reasons: a fixed
  batch shape keeps `cudnn.benchmark` and `torch.compile` from
  re-tuning/recompiling on a smaller final batch, and it avoids a
  variable-size last batch breaking the fixed-shape assumptions those
  two settings depend on.
- Validation split: the dataset has a single `train` split with no
  held-out set. Because dedup already removes near-duplicate consecutive
  frames, a random per-frame split risks leaking near-identical frames
  from the same video across train/val. This spec holds out **whole
  videos** instead — one per game, keeping both splits balanced:
  `D1SrSFZrV7A` (red) and `YW29l3jJXr4` (blue) from
  `configs/video_sources.yaml`, filtered via the dataset's `video_id`
  column. Confirmed.
- Multi-epoch iteration: a streaming `IterableDataset` exhausts after one
  full pass over the shards; re-entering the `for batch in dataloader`
  loop for the next epoch starts a fresh pass. Call `dataset.set_epoch(epoch)`
  before each epoch (the standard `datasets` pattern for varying the
  shuffle across epochs, analogous to `DistributedSampler.set_epoch`).

## Training loop

Precision/performance settings, applied at trainer startup:

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.fp32_precision = "tf32"   # falls back to
                                                        # matmul.allow_tf32 = True
                                                        # on older torch
model = model.to(memory_format=torch.channels_last)
# model.load_state_dict() happens here if resuming, BEFORE compile —
# see exact ordering in Checkpointing & resume below.
model = torch.compile(model, mode="default")
```

Per-step:

- Forward + loss under `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- Input batches converted to `channels_last` alongside the model.
- No `GradScaler` (see rationale above).
- Metrics (loss, grad norm, etc.) accumulated as GPU tensors across
  steps; `.item()` / any host sync called only at the logging interval,
  never every step — avoids both the sync cost and the autograd-graph
  memory leak PyTorch's own docs warn about when raw loss tensors are
  accumulated directly.
- `pin_memory=True` (already set on the dataloader) + `.to(device,
non_blocking=True)` for host-to-device transfer.
- Fail-fast: a NaN/Inf loss raises immediately rather than being
  silently skipped — unlike `data_collection.pipeline`'s per-video
  retry (where one video's failure is independent of the others), a NaN
  loss here usually signals a real instability that continuing would
  only compound.

`hf_storage.retry.retry_with_backoff` wraps only the actual Hub I/O
(frozen-artifact upload, dataset access errors), matching
`data_collection`'s existing precedent — not the training step itself.

**Startup memory probe:** before constructing the streaming dataset or
touching checkpoint state, run a few dummy forward+backward steps
through the real (compiled, autocast, channels_last) model at the
configured `batch_size` using random input. A `torch.cuda.OutOfMemoryError`
here raises immediately with the actual `batch_size` value and a
suggestion to lower it or use gradient accumulation — see the batch-size
rationale above for why this fails fast instead of retrying smaller.

## Checkpointing & resume

**Full training-state checkpoint** (local network volume, every
`checkpoint_interval_steps`, default 1000):

```python
{
    "epoch": int,
    "global_step": int,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "dataloader": dataloader.state_dict(),   # StatefulDataLoader — shard + example index
    "best_val_loss": float,
}
```

Saved as a plain dict of tensors/primitives (no custom classes) so it
remains loadable with `torch.load(..., weights_only=True)` — cheap
defense-in-depth even for a same-machine, self-produced file.

Restore order matters and is enforced in `checkpoint.py`:

1. Build model, move to device, apply `channels_last`.
2. If resuming: `model.load_state_dict()` — **before** `torch.compile`
   wraps it. `torch.compile` can rename state_dict keys (an `_orig_mod.`
   prefix) depending on version/mode, so loading into the raw module
   first and compiling after sidesteps that entirely rather than relying
   on version-specific compiled-state_dict compatibility.
3. Apply `torch.compile(model, mode="default")`.
4. Build optimizer with the configured initial LR.
5. Build scheduler (attached to optimizer) — per PyTorch's documented
   gotcha, constructing a scheduler resets its optimizer's `lr`, so this
   must happen before optimizer state is restored, or the restored LR
   gets clobbered.
6. If resuming: `optimizer.load_state_dict()`, then
   `scheduler.load_state_dict()`, then `dataloader.load_state_dict()`.

`train.py` checks the network volume for the latest checkpoint on
startup and resumes automatically if one exists — the same
manifest-driven resume shape as `data_collection.pipeline`, scoped to a
single training run instead of per-video.

**Frozen encoder artifact** (HF Hub model repo, e.g.
`objones25/pokemon-contrastive-encoder`, pushed at end-of-epoch and on
new best validation loss):

- `model.safetensors` — backbone-only state dict (projector dropped),
  with Conv+BN fusion applied to this exported copy only (the live
  training model stays unfused).
- `config.json` — `{"embedding_dim": 2048, "stem": "no_maxpool",
"input_channels": 1, "input_size": [160, 144], "pretrained_init":
true}`.
- `latent_stats.json` — mean/std of the 2048-d backbone output, computed
  over the held-out validation videos, for the sequence-model stage's
  affine normalization layer (see Scope boundary).
- Upload goes through `hf_storage`'s rate-limit-aware retry, same as
  `manifest.json` in `data_collection`.

## Frozen encoder interface (the PPO-facing contract)

```python
def load_frozen_encoder(repo_id: str, revision: str | None = None) -> nn.Module:
    """Downloads config.json + model.safetensors, reconstructs the
    matching architecture, loads weights strictly, applies Conv+BN
    fusion, freezes all parameters, sets eval mode. Returns a module
    mapping (N, 1, 160, 144) uint8-or-float grayscale input to (N, 2048)
    float features."""
```

This function — not a class hierarchy, not a framework base class — is
the entire interface contract the sequence-model/PPO stage depends on.
Documented expectations that stage must honor: input is single-channel
grayscale at native 160x144 resolution (channel replication happens
inside this function, not the caller's responsibility); output is the
raw, unnormalized 2048-d backbone feature (the affine normalization
layer is the _caller's_ job, using `latent_stats.json` above); the
returned module has `requires_grad_(False)` already set and is in
`eval()` mode — the caller should not need to call either again.

## Observability / experiment tracking

- Reuses `observability.tracking.TrackioRun`/`NullTrackioRun` and
  `observability.logging_config.configure_logging()` unchanged. New
  Trackio project name: `"pokemon-contrastive-pretrain"`.
- Structured JSON-lines log events (`logger.info(event, extra={...})`,
  matching `data_collection.pipeline`'s pattern): `train_step` (loss, lr,
  grad_norm, images/sec, data_wait_s), `epoch_complete` (val NT-Xent
  loss, best_val_loss), `checkpoint_saved`, `frozen_artifact_pushed`.
- `data_wait_s` (time spent waiting on the dataloader vs. GPU compute)
  is logged every step specifically to make a streaming-throughput
  bottleneck visible immediately if one ever appears, rather than
  discovered after burning A100-hours.
- Visual sanity check: `observability.visualization.build_augmentation_contact_sheet`
  (already built for the standalone `preview` CLI command) is also
  logged as a Trackio image artifact once per epoch, from the first
  batch — the same human-in-the-loop check the augmentation spec
  requires, now running against real training-time data instead of only
  a manual preview.

## Testing strategy

Per this repo's TDD convention, fast unit tests against synthetic
fixtures (no GPU/network) plus opt-in `slow` integration tests:

Unit:

- `build_encoder()` produces the expected output shape/dim; `maxpool`
  is confirmed absent from the forward path.
- Conv+BN fusion (`encoder_io`) produces output numerically equivalent
  (within tolerance) to the unfused model in eval mode.
- NT-Xent loss: known positive-pair alignment on a small synthetic
  batch produces the expected loss value/gradient sign.
- Checkpoint save/load round-trip on a small dummy model/optimizer/
  scheduler preserves state exactly; the restore-order test specifically
  asserts the scheduler's LR is _not_ clobbered by
  `optimizer.load_state_dict()` (regression test for the documented
  PyTorch gotcha).
- `load_frozen_encoder()` against a fake `HfClient` (no real network).
- `hf_storage.retry`/`client` — moved tests from `data_collection`,
  unchanged behavior.

Integration (`slow`, opt-in, real network/Hub credentials):

- A real streaming read of a small sample from
  `objones25/pokemon-frames`.
- A real `StatefulDataLoader` checkpoint/resume round trip against live
  streaming data — asserts resumption skips already-consumed shards.
- A short real training smoke run (a handful of steps, CPU or GPU) —
  asserts loss is finite and decreasing, no NaNs, matching this repo's
  "verify one stage works end-to-end" convention before scaling up to a
  real paid A100 run.

## Out of scope

(See Scope boundary above for the full list.) Restated briefly: the
sequence-model transformer and its latent-normalization layer, PPO
itself, RunPod provisioning automation, and final hyperparameter tuning
beyond the informed defaults above.

## Open questions / future extensions

- **LARS optimizer**: revisit if a later run scales batch size well past
  512 (e.g. via gradient accumulation or a larger GPU) — AdamW is the
  simpler, sufficient choice at the batch sizes this spec targets.
- **torch.compile `mode="max-autotune"`**: a phase-2 tuning experiment
  once the `mode="default"` pipeline is verified stable end-to-end, per
  this repo's convention of not building the next stage before the
  current one is confirmed to work.
- **Temporal positive pairs / UI-region-aware augmentation**: unchanged
  from the augmentation spec's own open questions — still deferred.
