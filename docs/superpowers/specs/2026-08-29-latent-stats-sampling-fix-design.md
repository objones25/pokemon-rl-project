# Latent Stats Sampling — Design Spec

**Amendment to `docs/superpowers/specs/2026-08-24-contrastive-pretrain-model-design.md`.**
That spec's frozen-artifact section (`latent_stats.json` — mean/std of the
2048-d backbone output, computed over the held-out validation videos) never
specifies *how* the sample is drawn from those videos. This spec closes that
gap, because the current answer — the first 2000 rows, in stream order, from
an unshuffled iterable — is what produced this project's one actual production
incident, and the code that produced it is unchanged today.

Found during a full-codebase review requested ahead of the first paid PPO run,
not from a new incident — the original incident (9 apparently-dead latent
channels, discovered on 2026-08-28) was fixed operationally, and this spec is
about the code path that produced it, not a new occurrence.

## The bug

`compute_latent_stats` (`src/contrastive_pretrain/encoder_io.py:187-208`):

```python
def compute_latent_stats(
    encoder: nn.Module, rows: Iterable[dict], device: torch.device, max_examples: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    with torch.inference_mode():
        for i, row in enumerate(rows):
            if i >= max_examples:
                break
            frame = row["original"].unsqueeze(0).to(device).float()
            features.append(encoder(frame).squeeze(0).cpu())
    ...
    stacked = torch.stack(features)
    return stacked.mean(dim=0), stacked.std(dim=0)
```

It takes the **first** `max_examples` rows of whatever `rows` yields, in
order, with no shuffling and no diversity check. Its only two call sites both
feed it `build_val_dataset(config)`
(`src/contrastive_pretrain/train.py:458-460`, `src/contrastive_pretrain/cli.py:162-163`),
which filters the base stream to `video_id in config.val_video_ids` and —
confirmed by reading `build_val_dataset`
(`src/contrastive_pretrain/dataset.py:156-175`) — **never calls `.shuffle()`**.
That absence is deliberate for a different consumer: `train.py`'s own comment
(around line 288) explains `build_val_dataset` has no shuffle specifically so
its *validation-loss* consumer (`compute_val_loss`, a separate, small,
`max_batches=20` read) can resume without the duplicate/skip accounting a
shuffle buffer's unsaved state would otherwise require. `compute_latent_stats`
inherits that same unshuffled order as an unrelated side effect, for a
consumer with a completely different requirement: a *representative* sample of
the full held-out set, not a resumable one.

The result is that every call — and `train.py` calls this on **every epoch
that improves validation loss** (line 456-460), not just once — samples the
same narrow, front-of-stream slice of the held-out video(s). This is exactly
what the incident diagnosis found: 9 of 2048 channels measured `std == 0`
because they are real feature detectors that fire at 0.003%–0.5% of frames,
rates low enough that a fixed first-2000-frames sample can miss every
activation. **`push_frozen_encoder` (`encoder_io.py:94-127`) then uploads
whatever `compute_latent_stats` returns, unvalidated** — there is no
`std > 0` check anywhere in `contrastive_pretrain` before publish. The
matching guard exists only downstream, in `src/pokemon_env/encoder.py:58-65`
(`load_latent_stats`), which is where the incident was actually caught: on a
running PPO pod, at env construction, not at export time on the machine that
produced the bad artifact.

### Why the shipped fix didn't close this

The incident's actual resolution (`configs/ppo.yaml`, commit `e06eb1c`)
recomputed `latent_stats.json` by hand over 36,040 frames spread across all 7
source videos and republished it — a one-time data fix. `compute_latent_stats`
itself, and both of its call sites, are byte-for-byte unchanged from before the
incident. The next training run that reaches a new best validation loss and
calls `push_frozen_encoder` will sample the same way and can reproduce the
same failure mode, silently, with no test catching it before publish.

## Fix

Two independent changes — a root-cause fix (diversify the sample) and a
safety net (refuse to publish a broken artifact) — because either one alone
leaves a real gap. Diversifying the sample without a publish-time guard still
allows a genuinely rare channel to occasionally land on `std == 0` by chance,
undetected until a paid pod fails to load it; a guard without diversifying the
sample would just make the *known-narrow* sample fail loudly on the training
machine instead of the failure being fixed, converting a silent-corruption bug
into a "training can never publish" bug without addressing why.

### 1. Reservoir-sample `compute_latent_stats`, don't take-first-N

Replace the early-break loop with reservoir sampling (Algorithm R), which
visits the full `rows` iterable once and produces a uniform random sample of
size `max_examples` regardless of the iterable's order:

```python
generator = torch.Generator().manual_seed(seed)
reservoir: list[torch.Tensor] = []
with torch.inference_mode():
    for i, row in enumerate(rows):
        frame = row["original"].unsqueeze(0).to(device).float()
        feature = encoder(frame).squeeze(0).cpu()
        if len(reservoir) < max_examples:
            reservoir.append(feature)
        else:
            j = torch.randint(0, i + 1, (1,), generator=generator).item()
            if j < max_examples:
                reservoir[j] = feature
```

This is a **drop-in change entirely inside `compute_latent_stats`** — it
consumes the same `rows: Iterable[dict]` signature, does not touch
`build_val_dataset`, and therefore does not disturb the resume-without-shuffle
invariant `compute_val_loss` depends on. Add a `seed` parameter (default from
`config.seed`, matching how `build_train_dataset` already threads a seed
through) so the sample is reproducible across a resumed run, per this
project's reproducibility conventions.

**Real cost tradeoff, stated plainly**: reservoir sampling requires iterating
the *entire* `rows` stream once, not stopping at `max_examples` — unlike
today's early break. If the held-out validation set is very large this is
slower per call, and this function is called on every val-loss improvement,
not once. Measure the actual held-out set size (`sum of frame counts for
config.val_video_ids`) before accepting this cost blindly; if it turns out to
be prohibitively large, the fallback is capping the full pass at some multiple
of `max_examples` (e.g., reservoir-sample over the first `10 * max_examples`
rows rather than the entire stream) — a real compromise between representativeness
and cost, not proposed here as the default because the actual held-out set
size hasn't been measured as part of this review.

**`max_examples`'s default (2000) should be re-derived, not left as-is.** The
incident's manual fix used ~36,040 frames and it worked, but that was a
one-off "make the immediate failure go away" number, not a validated minimum —
treat it as a starting point for measurement (e.g., does `std` stabilize
change meaningfully between 2,000/10,000/36,000/full-set samples for the
specific channels that measured `std == 0`?), not a value to copy verbatim.

### 2. Validate before publish, not just before load

Add a validation check to `push_frozen_encoder`
(`src/contrastive_pretrain/encoder_io.py:94-127`) that refuses to upload if
the computed stats are broken, mirroring — deliberately duplicating, not
importing — the check `src/pokemon_env/encoder.py:58-65` already applies on
load:

```python
def _validate_latent_stats(mean: torch.Tensor, std: torch.Tensor) -> None:
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("latent_mean/latent_std contain non-finite values; refusing to publish")
    non_positive = int((std <= 0).sum())
    if non_positive:
        raise ValueError(
            f"latent_std has {non_positive} non-positive entries; refusing to publish. "
            "A dead or under-sampled encoder channel would divide by the 1e-6 floor "
            "downstream and feed ~1e6-scale inputs to the policy's value head."
        )
```

called at the top of `push_frozen_encoder`, before `export_frozen_encoder` or
any upload work. This is a deliberate duplication across the
`contrastive_pretrain` / `pokemon_env` sub-project boundary, not a shared
helper — CLAUDE.md's codebase map keeps these two sub-projects independent by
design ("Sub-projects are designed and planned independently"), and importing
one's internals into the other to save five lines would create exactly the
cross-sub-project coupling that boundary exists to prevent. Each guard stays
owned by the module that needs it: `contrastive_pretrain` fails fast at
export, `pokemon_env` fails fast at load, and the two checks happening to be
identical is a coincidence of both wanting the same invariant, not a reason to
couple them.

**Open question this spec does not resolve**: if a genuinely rare-but-real
channel (the incident found 7 of the 9 flagged channels were exactly this)
still measures `std == 0` even after reservoir-sampling over a much larger
pool, should `push_frozen_encoder` hard-fail every time until a human
intervenes (current proposal, consistent with this project's "fail fast,
never mid-run" convention and its human-gated pattern elsewhere, e.g.
`data_collection`'s `curate`), or should there be an escape hatch (log a
named list of the affected channel indices and let a human explicitly
override)? This spec recommends the hard-fail-only default and leaves adding
an override for a follow-up decision if it turns out to fire often — adding an
override now, before knowing whether it is ever actually needed, would be the
kind of speculative escape hatch CLAUDE.md's conventions caution against.

## What changes

- `src/contrastive_pretrain/encoder_io.py`: `compute_latent_stats` gains
  reservoir sampling and a `seed` parameter; `push_frozen_encoder` gains the
  `_validate_latent_stats` call before any export/upload work.
- `src/contrastive_pretrain/train.py:458-460`, `src/contrastive_pretrain/cli.py:162-163`:
  thread `config.seed` through the new `compute_latent_stats` call, and
  reconsider `max_examples` per the measurement above.
- No change to `src/contrastive_pretrain/dataset.py` — `build_val_dataset`'s
  shuffle-free property is preserved exactly, for `compute_val_loss`'s sake.
- No change to `src/pokemon_env/encoder.py` — its existing `std > 0` guard on
  load stays as the last line of defense; this spec adds an earlier one, it
  does not replace or touch the existing one.
- `docs/superpowers/specs/2026-08-24-contrastive-pretrain-model-design.md`'s
  frozen-artifact section should get a line noting the sampling method and the
  publish-time validation, so a future reader doesn't have to rediscover this
  from the incident history.

## Testing

- **Prove the current bug would have been caught**: construct a fake `rows`
  iterable where one feature dimension is exactly zero for the first
  `max_examples` rows and nonzero afterward (mirroring the real incident's
  shape — a channel invisible in an early slice, present later). Assert the
  *old* take-first-N implementation reports `std == 0` for that dimension
  (documents the bug precisely) and the *new* reservoir-sampled implementation
  reports a nonzero `std` for it, with the same total row count consumed.
- **`compute_latent_stats` is deterministic given a seed**: same `rows`, same
  seed, same `max_examples` → bit-identical `mean`/`std` across two calls —
  needed for the resume-reproducibility this project already holds elsewhere
  (e.g., `dataset.py`'s per-row seeding).
- **`push_frozen_encoder` refuses to upload broken stats**: a test asserting
  that when `latent_std` contains a zero (or a NaN), `push_frozen_encoder`
  raises `ValueError` *before* `client.upload_many_bytes` is ever called — use
  the existing hand-written `FakeHfClient`/`AtomicHfClient` fake and assert its
  call count is zero on the raising path, not just that an exception was
  raised (a test that only checks the raise could still pass if the upload
  happened first and the raise came from somewhere unrelated afterward).
- **`push_frozen_encoder` still succeeds on valid stats**: a regression test
  with all-positive, all-finite `std` confirming the new validation call does
  not block the existing happy path — `tests/unit/test_contrastive_pretrain_encoder_io.py`
  already has fixtures for this.
- Prove each new/changed test can fail: temporarily skip the validation call,
  confirm the "refuses to upload" test goes red, then restore it — the
  project's own stated gate for any new test.

## Non-goals

- Does not change `latent_dim` (2048), the encoder architecture, or the
  affine-normalization contract downstream — purely a sampling and
  validation-timing fix.
- Does not decide the escape-hatch question above; ships with hard-fail-only.
- Does not address any of the other findings from the same review pass
  (the PPO NaN-abort logging gap has its own sibling spec,
  `docs/superpowers/specs/2026-08-29-ppo-abort-logging-fix-design.md`; the
  `video_id` parsing bug, the PPO manifest atomicity gap, and the unguarded
  `wandb.init()` call are separate, un-specced findings from the same review).

## Known gaps carried out of implementation

- **`max_examples`'s default (2000) was not re-derived.** This spec calls
  for measuring whether `std` stabilizes meaningfully between
  2,000/10,000/36,000/full-set samples for the channels that measured
  `std == 0` in the incident, and sizing `max_examples` from that
  measurement rather than copying the incident's one-off manual fix
  verbatim. That measurement needs the real held-out video set and was not
  done as part of this implementation pass — `max_examples` stays at its
  pre-existing default of 2000. Reservoir sampling still visits every row
  in the stream regardless of this default, so the sampling-order bug this
  spec targets is fixed either way; an unmeasured `max_examples` only
  means the *size* of the sample, not its representativeness, is still a
  guess.
- **The full-stream traversal's CPU cost is not eliminated, only the GPU
  side of it.** A final-review fix pass batched `compute_latent_stats` so
  only the retained `max_examples` reservoir is ever encoded, in one
  forward pass, instead of the original reservoir-sampling implementation's
  one `encoder(frame)` call per row of the (100k+ row) held-out stream —
  removing the dominant, per-row GPU-forward cost. The stream traversal
  itself still runs `to_pair_transform`'s augmentation pipeline on every
  row it visits (`src/contrastive_pretrain/dataset.py` is out of scope for
  this branch and was not touched), so a full pass's CPU-side cost is
  unchanged. Separately, if `TrainingConfig.local_cache_dir` is left at its
  default (`None`), each publish still re-streams the dataset from the Hub
  rather than reading a local cache — a cost hazard CLAUDE.md's Infra
  section already names elsewhere in this project.
- **The currently-published encoder will hard-fail `_validate_latent_stats`
  on its next re-export.** Per `configs/ppo.yaml`'s comment on
  `frozen_encoder_revision`, dims 1773 and 1994 of the currently-published
  `objones25/pokemon-contrastive-encoder` artifact are genuinely constant
  channels that were fixed by manually flooring their std to 1.0 outside
  the codebase, not by anything `compute_latent_stats` or
  `push_frozen_encoder` does. A future re-export of that specific
  checkpoint through `push_frozen_encoder` will therefore hard-fail on
  `_validate_latent_stats`, by design, and will need the same manual
  intervention documented in `configs/ppo.yaml` — or the escape-hatch
  design decision this spec deliberately leaves open above — not a bug in
  this fix.
