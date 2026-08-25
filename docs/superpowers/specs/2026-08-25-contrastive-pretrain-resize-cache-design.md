# Contrastive Pretraining — Local Resize Cache Design Spec

Date: 2026-08-25
Status: Approved for planning

## Purpose

The current training data pipeline (`contrastive_pretrain/dataset.py`)
streams `objones25/pokemon-frames` directly from the HF Hub on every run,
resizing each native-resolution frame (up to 2400x2160) down to the
model's 144x160 canonical input on the fly, per-row, inside each
`StatefulDataLoader` worker. A real training run (resumed at
`global_step=383`) showed this pipeline is badly data-starved: per-step
`data_wait_s` logs (already emitted per the model-design spec's
observability section) show waits from sub-millisecond up to **76
seconds**, dozens of times within an 18-minute window, correlated
one-to-one with `HTTP GET .../shards/<video_id>/NNNNN.parquet` calls to
the Hub's CDN. GPU utilization snapshots reading 0% (while VRAM stays
pinned at 96%) are a direct symptom: the GPU sits idle waiting on
network-bound shard fetches + in-line CPU resize far more than it
computes.

This spec adds a one-time preprocessing step that downloads the dataset,
resizes every frame to canonical resolution exactly once, and caches the
result as local Parquet shards on the training pod's persistent
`/workspace` volume. Training then reads from that local cache instead
of streaming from the Hub, eliminating both the per-epoch network
round-trips and the per-epoch redundant resize work.

This directly revisits a call made in
`2026-08-24-contrastive-pretrain-model-design.md` ("Data loading — HF
streaming, no local download... this is not expected to bottleneck a GPU
doing ResNet-50 forward/backward, but the training loop logs data-wait
time... so this is verified from real metrics, not assumed"). That spec
was explicit that the streaming choice was provisional pending real
metrics. The metrics are now in, and they show the opposite of what was
assumed — this spec is the correction.

## Scope boundary

In scope:

- A new one-time preprocessing script/CLI command that downloads every
  shard of `objones25/pokemon-frames`, resizes every row to canonical
  144x160 via the existing, already-tested `_resize_to_canonical`, and
  writes the result as local Parquet shards under a configured directory
  on the pod's `/workspace` volume.
- Resumability: the script must be safe to kill and rerun at any point
  (RunPod pods restart), picking up only unfinished shards.
- A new `TrainingConfig.local_cache_dir` field and the corresponding
  branch in `_load_base_stream` so `build_train_dataset` /
  `build_val_dataset` can read from the local cache instead of the Hub,
  with zero changes to anything downstream of `_load_base_stream`
  (filter, resize-map, shuffle, augmentation-pair transform, dataloader
  construction, checkpoint/resume — all untouched).
- Widening `hf_storage.retry.retry_with_backoff` to be generic over a
  return value, so shard downloads (which need the downloaded bytes, not
  just success/failure) can reuse it as-is.

Out of scope:

- Publishing the resized frames back to the Hub as a new dataset repo.
  Explicitly rejected for this iteration: it adds upload/commit logic
  and re-exposes the project to HF's per-hour commit rate limit (already
  hit once during data collection — see `hf_uploader.py`'s
  rate-limit-aware retry and the project's own incident memory) for a
  benefit — durability across a lost pod/volume — that isn't needed
  right now. If a future pod loses this volume, rerunning the script
  once against the Hub source is an accepted, bounded cost.
- Any change to `_resize_to_canonical`, the augmentation pipeline, the
  model, or the training loop itself. This spec touches only how bytes
  get from the Hub to the dataloader, not what happens to them once
  they're in it.
- Deleting or modifying the original `objones25/pokemon-frames` Hub
  repo. It remains the single source of truth; the local cache is a
  derived, disposable artifact.
- Parallelizing the resize script across multiple shards concurrently.
  A future optimization if the one-time cost matters enough to justify
  it; not required for correctness now.

## Why this design, not alternatives considered

**Local-only cache, not also pushed to the Hub.** Two options were
weighed: a pure local cache on `/workspace`, or the same cache also
committed to a new Hub dataset repo for durability across pods. Local-
only was chosen because this project's `/workspace` mount is confirmed
persistent for a training pod (already established in
`TrainingConfig.network_volume_checkpoint_dir`'s docstring), so the
common case — the same pod resuming training, possibly many times — is
fully covered without any Hub-write exposure. The uncommon case — losing
the volume entirely — costs one re-run of the script, which is
acceptable given resized frames are ~200x smaller than native-resolution
ones (per `_resize_to_canonical`'s own docstring: ~5MB native vs. ~23KB
resized) and therefore fast to re-download and re-resize from scratch.

**Local Parquet shards (via `datasets`), not a raw tensor cache.**
Verified empirically (not assumed) that `datasets.Dataset.to_parquet()`
embeds the `Image` feature's schema metadata in the Arrow file, and that
`datasets.load_dataset("parquet", data_files=..., streaming=True)`
correctly reconstructs a decoded PIL image on read — both in streaming
and non-streaming mode, and with a recursive glob
(`shards/**/*.parquet`) picking up all per-video subdirectories in one
call. This means `_load_base_stream`'s Hub-reading branch and its new
local-cache-reading branch differ by exactly one `datasets.load_dataset`
call each; everything else in the pipeline (`.filter()`,
`.map(_ResizeToCanonicalWithProgress())`, `.shuffle()`,
`to_pair_transform`) is unaware of which branch produced its input. A
raw tensor cache (`.pt`/`.npy` per shard) was rejected: it would read
back marginally faster, but requires a new, hand-rolled loader path that
bypasses `datasets.load_dataset` entirely, touching more of `dataset.py`
and needing its own new test surface for a project convention
(`datasets`-library Parquet shards) this repo has already standardized
on for the Hub-hosted dataset.

**Resize-map stays unconditional (runs again on already-resized
frames), rather than branching around it for the cache path.**
`_resize_to_canonical` resizes a 144x160 frame to 144x160 — a near-zero-
cost identity resize once the frame's decode/tensor-conversion cost is
already paid, which happens either way. Special-casing the pipeline to
skip it when reading from the cache would save a negligible amount of
CPU at the cost of a second, cache-aware `.map()` call site to test and
maintain. Not worth it.

**Explicit `local_cache_dir` config field, not auto-detection.**
Auto-detecting a fixed path and silently switching data sources based on
its presence risks training silently against a stale or partially-built
cache (e.g., a previous run's script was killed midway) without any
signal in the config that this happened. An explicit field, defaulting
to `None` (today's Hub-streaming behavior, unchanged), makes the data
source a visible, deliberate choice recorded in
`configs/contrastive_pretrain.yaml` — flipped on only once the cache
script has finished.

**Shard-level existence check as the entire resume mechanism, not a
separate manifest file.** `data_collection/hf_uploader.py` already has a
`Manifest` class (completed/failed/progress, serialized as JSON) for
video-level upload resumability. This script's unit of work is a single
shard file, and its output is itself a file at a deterministic path —
checking whether `<local_cache_dir>/shards/<video_id>/NNNNN.parquet`
exists already **is** a complete, trivially-correct manifest, provided
writes are atomic (write to a `.tmp` path, then `os.replace()` to the
final path). A killed process can only ever leave a `.tmp` file behind,
never a corrupt file that the existence check would mistake for
"done" — so no separate state file, and nothing that can drift out of
sync with what's actually on disk.

**Reusing `hf_storage.retry.retry_with_backoff` and
`hf_storage.client.HfClient`, not new download/retry code.** Both
already exist, are already unit-tested, and
`rate_limit_aware_backoff` is already tuned to the exact HF 429 error
text this project has hit in production (see the HF rate-limit incident
on `objones25/pokemon-frames` during data collection). The only
adjustment needed is widening `retry_with_backoff`'s callable signature
from `Callable[[], None]` to a generic `Callable[[], T] -> T`, since
shard downloads need the downloaded bytes back, not just success/
failure — a backward-compatible change (existing `None`-returning
callers are unaffected).

## Component design

```
src/contrastive_pretrain/
    dataset.py          # existing — one new branch in _load_base_stream
    resize_cache.py      # new — one-time preprocessing orchestration
    config.py            # existing — one new field on TrainingConfig
    cli.py                # existing — one new `resize-cache` command

src/hf_storage/
    retry.py             # existing — retry_with_backoff widened to generic T
```

### `config.py`

```python
@dataclass(frozen=True)
class TrainingConfig:
    ...
    # None (default): stream objones25/pokemon-frames directly from the
    # Hub, resizing per-row on the fly, exactly as before this spec. Set
    # to a directory populated by `contrastive-pretrain resize-cache` to
    # read already-resized local Parquet shards instead -- eliminates
    # both the per-epoch network round-trips and per-epoch resize work.
    local_cache_dir: str | None = None
```

### `dataset.py` — `_load_base_stream`

```python
def _load_base_stream(config: TrainingConfig):
    if config.local_cache_dir:
        return datasets.load_dataset(
            "parquet",
            data_files=f"{config.local_cache_dir}/shards/**/*.parquet",
            split="train",
            streaming=True,
        )
    return datasets.load_dataset(config.dataset_repo_id, streaming=True, split="train")
```

This is the entire change to `dataset.py`. Both existing fast tests that
monkeypatch `_load_base_stream` itself are unaffected (they replace the
whole function); a new fast test exercises this branch directly against
a tiny on-disk Parquet fixture.

### `hf_storage/retry.py` — generic `retry_with_backoff`

```python
def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int,
    base_delay: float,
    sleep_func: Callable[[float], None],
    backoff_seconds: BackoffFunc | None = None,
) -> T:
    backoff = backoff_seconds or exponential_backoff(base_delay)
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:
            attempt += 1
            if attempt >= max_retries:
                raise
            sleep_func(backoff(attempt, exc))
```

Behavior-identical for existing `Callable[[], None]` callers (`T =
None`); the only change is returning `func()`'s result.

### `resize_cache.py` — new module

Injected-dependency orchestration function, matching this codebase's
existing DI conventions (network/filesystem access passed in, not
imported directly, so the core loop is unit-testable without hitting
the network):

```python
def build_local_resize_cache(
    list_shard_paths: Callable[[], list[str]],
    download_shard: Callable[[str], bytes],
    local_cache_dir: Path,
    log_every_n_shards: int = 10,
) -> None:
    """Downloads and resizes every shard `list_shard_paths()` returns that
    isn't already present (as a completed file) under `local_cache_dir`,
    writing each as a local Parquet shard at the same relative path.
    Safe to interrupt and rerun: already-completed shards are skipped,
    and a shard is never considered complete until its output file has
    been atomically renamed into place."""
```

- `list_shard_paths`: production implementation calls
  `HfApi().list_repo_files(repo_id, repo_type="dataset")`, filtered to
  paths under `shards/`.
- `download_shard`: production implementation wraps
  `RealHfClient.download_bytes` (the existing `hf_storage.client`
  Protocol/adapter — no new client code) in
  `retry_with_backoff(..., backoff_seconds=rate_limit_aware_backoff(...))`.
- Per shard (`shards/<video_id>/NNNNN.parquet`):
  1. `output_path = local_cache_dir / shard_path`. If it already exists,
     skip (log `resize_cache_shard_skipped`).
  2. `raw_bytes = download_shard(shard_path)`.
  3. Write `raw_bytes` to a temp file; `datasets.Dataset.from_parquet(tmp_raw_path)`.
  4. `.map(_resize_to_canonical)` — the exact function `dataset.py`
     already uses and already unit-tests; zero duplication.
  5. `resized.to_parquet(output_path.with_suffix(".tmp"))`, then
     `os.replace(tmp_output_path, output_path)` — atomic, crash-safe.
  6. Delete the temp raw-bytes file.
  7. Log `resize_cache_shard_done` (shard path, row count, elapsed
     seconds) as a structured JSON-lines event, matching this project's
     observability convention.
- After all shards: log one `resize_cache_complete` summary event
  (shard count, total rows, skipped count) — the human-checkable
  progress artifact this stage needs; no image/contact-sheet artifact
  applies here since no filtering or data-dropping decision is being
  made, only a lossless resize.

### `cli.py` — new `resize-cache` command

Matches the existing `train`/`export-frozen-encoder` command shape:

```
contrastive-pretrain resize-cache --config configs/contrastive_pretrain.yaml --local-cache-dir /workspace/contrastive_pretrain/resized_cache
```

Checks for HF credentials up front (same pattern as `train`), calls
`load_config`, then `build_local_resize_cache` with production
`list_shard_paths`/`download_shard` implementations bound to
`config.dataset_repo_id`.

Once the script completes, the operator sets `local_cache_dir` in
`configs/contrastive_pretrain.yaml` to the same path and starts/resumes
training as normal.

## Testing strategy

Unit (fast, no network, per this repo's TDD convention):

- `build_local_resize_cache`, with a fake `list_shard_paths` and fake
  `download_shard`, against a `tmp_path` local_cache_dir:
  - Writes one resized Parquet shard per listed shard path, at the
    expected relative path, readable back with `datasets.load_dataset`
    at the expected (144, 160) shape.
  - Skips shards whose output file already exists (resumability),
    without calling `download_shard` for them.
  - A `download_shard` that raises on shard N doesn't leave a partial/
    corrupt file at shard N's output path, and doesn't prevent already-
    completed shards (N-1 and earlier) from being correctly skipped on
    a subsequent call.
- `_load_base_stream`'s new local-cache branch, against a tiny on-disk
  Parquet fixture written via `datasets.Dataset.to_parquet` (same
  pattern used to empirically verify the Image-feature round-trip
  above) — confirms `build_train_dataset`/`build_val_dataset` work
  unchanged when fed from this branch.
- `retry_with_backoff`'s widened generic signature: existing tests
  (moved/kept as-is) plus one new case confirming a `Callable[[], T]`'s
  return value passes through on success.

Integration (`@pytest.mark.slow`, opt-in, real Hub credentials):

- `build_local_resize_cache` against 1-2 real shards from
  `objones25/pokemon-frames`, confirming the real Hub schema round-trips
  through the real resize + local Parquet write correctly.

## Out of scope

(See Scope boundary above.) Restated briefly: publishing the resized
cache back to the Hub, any change to the resize/augmentation/model/
training-loop logic itself, modifying the original Hub dataset repo, and
parallelizing the resize script.

## Open questions / future extensions

- **Concurrent shard processing**: if the one-time resize cost (network
  + CPU resize across all shards) turns out to matter in practice,
  processing multiple shards concurrently (e.g. a small thread/process
  pool over `list_shard_paths()`'s output) is a natural follow-up — the
  per-shard atomic-write design already makes this safe, since shards
  don't share any state. Deferred until the sequential cost is actually
  measured.
- **Revisit "local-only" if a second training pod is ever provisioned**:
  the local-only decision assumes one pod's `/workspace` volume persists
  across that pod's own restarts. If this project ever needs to resume
  training on a genuinely different pod/volume, the cost is one re-run
  of `resize-cache` — acceptable now, but worth revisiting (pushing the
  resized set to the Hub instead) if multi-pod training becomes routine.
