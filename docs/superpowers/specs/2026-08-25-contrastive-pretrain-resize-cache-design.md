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

- A one-time preprocessing step that downloads every shard of
  `objones25/pokemon-frames`, resizes every row to canonical 144x160 via
  the existing, already-tested `_resize_to_canonical`, and writes the
  result as local Parquet shards under a configured directory on the
  pod's `/workspace` volume. Triggered automatically — not via a
  separate command — the first time `build_train_dataset` or
  `build_val_dataset` runs with `TrainingConfig.local_cache_dir` set;
  every call after that is a fast no-op (see resumability below).
- Resumability: the step must be safe to kill and rerun at any point
  (RunPod pods restart), picking up only unfinished shards.
- A new `TrainingConfig.local_cache_dir` field — the *only* new
  operator-facing surface this spec adds. Setting it in
  `configs/contrastive_pretrain.yaml` is the entire opt-in; there is no
  separate script/command to remember to run first.
- The corresponding branch in `_load_base_stream` so `build_train_dataset`
  / `build_val_dataset` can read from the local cache instead of the
  Hub. `.filter()`, `.shuffle()`, `to_pair_transform`, dataloader
  construction, and checkpoint/resume are all untouched; the one
  pipeline-structure change is that the pre-shuffle
  `_ResizeToCanonicalWithProgress` `.map()` stage is skipped entirely
  when reading from the cache, since it would otherwise run a whole
  extra no-op pass over already-canonical data (see rationale below).
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
  right now. If a future pod loses this volume, the next `train` run
  simply rebuilds the cache from the Hub source again — an accepted,
  bounded cost.
- Any change to `_resize_to_canonical`'s or `to_pair_transform`'s
  internal logic, the augmentation pipeline, the model, or the training
  loop itself. This spec changes only how bytes get from the Hub to the
  dataloader and whether the pre-shuffle resize *stage* runs at all —
  never what any transform function itself does.
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
the volume entirely — costs one more automatic rebuild pass, which is
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

**Resize-map is skipped entirely for the cache path — not just cheap,
actually absent.** Verified via `inspect.getsource` against the
installed `torchvision` (not assumed): `resize_image`'s kernel
explicitly short-circuits — `if (new_height, new_width) == (old_height,
old_width): return image` — and `to_image` on an already-`torch.Tensor`
input is a plain identity assignment (`output = inpt`) before a cheap
subclass wrap. So `to_pair_transform`'s own `to_image`+`resize` calls,
which run on every row regardless of data source, are already free
no-ops on canonical-sized input in both pipelines — there was never
duplicate decode or duplicate interpolation math to eliminate. An
earlier draft of this spec used that fact to justify leaving
`_ResizeToCanonicalWithProgress`'s `.map()` stage in unconditionally for
the cache path too ("near-zero-cost, not worth branching around") — that
was the wrong conclusion from a correct fact. The stage itself is not
free: every `.map()` in a `datasets` streaming pipeline is a full
row-wise Python generator hop, paid on all ~296K rows every epoch, and
for already-canonical cached frames that hop does nothing useful except
decode the PIL image into a tensor a step early (`to_pair_transform`'s
own `to_image` does that same decode immediately after `.shuffle()`
either way) and log a `resize_to_canonical_progress` line that would now
describe a resize that never happens — actively misleading, not just
wasteful. `build_train_dataset`/`build_val_dataset` therefore skip this
`.map()` stage entirely when `config.local_cache_dir` is set:

```python
if not config.local_cache_dir:
    ds = ds.map(_ResizeToCanonicalWithProgress())  # BEFORE shuffle -- only
    # needed for raw, native-resolution Hub frames; cached frames are
    # already canonical-sized, see _resize_to_canonical's docstring.
```

This is the one place this spec's pipeline structure differs between
the two data sources; everything else in the pipeline
(`.filter()`, `.shuffle()`, `to_pair_transform`, dataloader construction)
is identical either way. The one accepted trade-off: the cache path
loses the shuffle-buffer-fill wall-clock visibility
`_ResizeToCanonicalWithProgress`'s docstring was originally built to
provide (see its own docstring) — accepted because eliminating exactly
that multi-minute wait is this entire spec's purpose; local-disk reads
filling a shuffle buffer are not expected to reintroduce a wait long
enough to need its own progress log, and `resize_cache_shard_done`
already gives visibility into the one-time build pass itself.

**Explicit `local_cache_dir` config field, not auto-detection of a fixed
path.** Auto-detecting a fixed path and silently switching data sources
based on its presence risks training silently against a stale or
partially-built cache (e.g., a previous run was killed midway) with no
signal in the config that this happened. An explicit field, defaulting
to `None` (today's Hub-streaming behavior, unchanged), makes the data
source a visible, deliberate choice recorded in
`configs/contrastive_pretrain.yaml`.

**Cache build is automatic (inside `build_train_dataset`/
`build_val_dataset`), not a separate CLI command.** An earlier draft of
this spec had the operator run a standalone `resize-cache` command
before flipping `local_cache_dir` on — rejected as needless ceremony:
two manual steps that have to stay in sync (nothing stops the config
pointing at a directory that was never actually built) for what should
be a single operator action. Instead, `build_train_dataset`/
`build_val_dataset` call a new `resize_cache.ensure_local_cache(config)`
as their first step whenever `local_cache_dir` is set, before touching
`_load_base_stream`. This does not remove the one-time cost — the first
`contrastive-pretrain train` invocation after setting the field still
pays for the full download+resize pass — it just makes that pass
transparent and automatic instead of a prerequisite the operator has to
remember. Every subsequent call is a fast no-op, via the same per-shard
existence check that already gives the step its resumability. This
mirrors the precedent `_ResizeToCanonicalWithProgress`'s own docstring
already sets for this exact pipeline: surface a real, sometimes-long
wait through structured logs rather than hide it, but never make the
operator manually gate it.

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
    dataset.py           # existing — new _load_base_stream branch + one call
                          # to ensure_local_cache at the top of each builder
    resize_cache.py       # new — one-time preprocessing orchestration
    config.py             # existing — one new field on TrainingConfig

src/hf_storage/
    retry.py              # existing — retry_with_backoff widened to generic T
```

No CLI changes: `cli.py`'s existing `train` and `export-frozen-encoder`
commands already call `build_train_dataset`/`build_val_dataset`, so they
pick up the cache-build step automatically once `local_cache_dir` is set
in the config they load.

### `config.py`

```python
@dataclass(frozen=True)
class TrainingConfig:
    ...
    # None (default): stream objones25/pokemon-frames directly from the
    # Hub, resizing per-row on the fly, exactly as before this spec. Set
    # to a /workspace-backed directory to cache already-resized frames
    # there instead -- build_train_dataset/build_val_dataset populate it
    # automatically on first use (see resize_cache.ensure_local_cache),
    # eliminating both the per-epoch network round-trips and per-epoch
    # resize work on every call after that.
    local_cache_dir: str | None = None
```

### `dataset.py` — `_load_base_stream` and the two builder functions

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


def build_train_dataset(config: TrainingConfig):
    ensure_local_cache(config)
    ds = _load_base_stream(config)
    ds = ds.filter(lambda ex: ex["video_id"] not in config.val_video_ids)
    if not config.local_cache_dir:
        ds = ds.map(_ResizeToCanonicalWithProgress())  # BEFORE shuffle -- see its
        # docstring; skipped for the cache path, see rationale above.
    ds = ds.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.seed)
    ds = ds.map(
        functools.partial(to_pair_transform, augmentation_config=AugmentationConfig(), base_seed=config.seed),
        remove_columns=["image"],
    )
    return ds


def build_val_dataset(config: TrainingConfig):
    ensure_local_cache(config)
    ds = _load_base_stream(config)
    ds = ds.filter(lambda ex: ex["video_id"] in config.val_video_ids)
    if not config.local_cache_dir:
        ds = ds.map(_ResizeToCanonicalWithProgress())
    ds = ds.map(
        functools.partial(to_pair_transform, augmentation_config=AugmentationConfig(), base_seed=config.seed),
        remove_columns=["image"],
    )
    return ds
```

`ensure_local_cache` (from the new `resize_cache` module, imported into
`dataset.py`) is a no-op single `if not config.local_cache_dir: return`
when the field is unset — which is every existing test's default, so no
existing test needs to change. Both existing fast tests that
monkeypatch `_load_base_stream` directly are also unaffected — they
replace the whole function, and the `if not config.local_cache_dir`
guard around the resize-map is `False`-by-default-config the same way.
New fast tests: the `_load_base_stream` local-cache branch against a
tiny on-disk Parquet fixture; `build_train_dataset` calls
`ensure_local_cache` (monkeypatched) before `_load_base_stream`; and —
directly testing this section's fix — `build_train_dataset`/
`build_val_dataset` with `local_cache_dir` set do *not* invoke
`_ResizeToCanonicalWithProgress` at all (monkeypatch it as a call
counter, assert zero calls), while the default (`local_cache_dir=None`)
config still does.

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
  4. `.map(_resize_row_for_cache)`, a thin wrapper (defined in
     `resize_cache.py`, not `dataset.py`) around the exact
     `_resize_to_canonical` function `dataset.py` already uses and
     already unit-tests — zero duplication of the actual resize logic.
     The wrapper exists because `_resize_to_canonical` returns a
     channel-first `(1, H, W)` `torch.Tensor`, correct for the
     streaming pipeline (which never re-serializes it) but **silently
     corrupted** if written straight into a `datasets.Image()`-typed
     column: verified empirically that both a raw tensor and a raw
     `(H, W)` numpy array round-trip through `.map()` + `.to_parquet()`
     incorrectly — a bare tensor collapses the column to a nested-list
     type (losing the `Image` feature entirely), and even a numpy array
     under an explicitly-preserved `Image()` schema gets corrupted to
     32-bit int-mode pixels on reload, because `datasets`' internal
     write path round-trips the returned value through a Python-list
     intermediate that loses the `uint8` dtype before the image encoder
     ever sees it. Returning an actual `PIL.Image` object sidesteps
     this entirely — `datasets` recognizes it directly and encodes it
     as PNG bytes under the `Image()` feature, no explicit `features=`
     needed on `.map()`, verified to round-trip pixel-identical through
     a full write+reload cycle:
     ```python
     def _resize_row_for_cache(example: dict) -> dict:
         frame = _resize_to_canonical(example)["image"]  # (1, H, W) uint8
         return {"image": Image.fromarray(frame.squeeze(0).numpy(), mode="L")}
     ```
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

The module's second, thin public function is the one `dataset.py`
actually calls:

```python
def ensure_local_cache(config: TrainingConfig) -> None:
    """No-op if config.local_cache_dir is unset. Otherwise wires
    build_local_resize_cache to the real Hub (HfApi.list_repo_files +
    RealHfClient.download_bytes, retried with
    hf_storage.retry.rate_limit_aware_backoff) and runs it against
    config.local_cache_dir. Safe to call on every build_train_dataset/
    build_val_dataset invocation -- already-built shards are skipped, so
    a fully-populated cache makes this a handful of fast filesystem
    existence checks, not a re-download."""
    if not config.local_cache_dir:
        return
    api = HfApi()
    client = RealHfClient(api, config.dataset_repo_id, repo_type="dataset")
    build_local_resize_cache(
        list_shard_paths=lambda: [
            p for p in api.list_repo_files(config.dataset_repo_id, repo_type="dataset")
            if p.startswith("shards/")
        ],
        download_shard=lambda path: retry_with_backoff(
            lambda: client.download_bytes(path),
            max_retries=5,
            base_delay=2.0,
            sleep_func=time.sleep,
            backoff_seconds=rate_limit_aware_backoff(base_delay=2.0, rate_limit_delay=3600.0),
        ),
        local_cache_dir=Path(config.local_cache_dir),
    )
```

This is the only place `resize_cache.py` talks to the real Hub;
`build_local_resize_cache` itself stays dependency-injected and
network-free for testing, per this codebase's existing DI conventions.

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
- `ensure_local_cache`: a no-op (doesn't call `build_local_resize_cache`)
  when `config.local_cache_dir` is `None`; calls it with the expected
  `local_cache_dir` when set (monkeypatching `build_local_resize_cache`
  itself — no real network).
- `build_train_dataset`/`build_val_dataset` call `ensure_local_cache`
  before `_load_base_stream` (monkeypatch both, assert call order).
- `build_train_dataset`/`build_val_dataset` skip
  `_ResizeToCanonicalWithProgress` entirely when `local_cache_dir` is
  set (monkeypatch it as a call counter, assert zero calls over a
  synthetic cached-style row stream), and still invoke it when
  `local_cache_dir` is `None` (regression test for the existing
  resize-before-shuffle OOM fix — confirms this change doesn't
  resurrect it for the Hub-streaming path).
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
  - CPU resize across all shards) turns out to matter in practice,
    processing multiple shards concurrently (e.g. a small thread/process
    pool over `list_shard_paths()`'s output) is a natural follow-up — the
    per-shard atomic-write design already makes this safe, since shards
    don't share any state. Deferred until the sequential cost is actually
    measured.
- **Revisit "local-only" if a second training pod is ever provisioned**:
  the local-only decision assumes one pod's `/workspace` volume persists
  across that pod's own restarts. If this project ever needs to resume
  training on a genuinely different pod/volume, the cost is one more
  automatic build pass (same `local_cache_dir`, empty directory) the
  next time `train` runs — acceptable now, but worth revisiting (pushing
  the resized set to the Hub instead) if multi-pod training becomes
  routine.
