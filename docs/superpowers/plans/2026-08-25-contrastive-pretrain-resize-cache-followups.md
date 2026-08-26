# Contrastive Pretraining Resize Cache — Deferred Follow-Ups

Date: 2026-08-25
Status: merged to `main` (commit range `3263ae8..3b4f7ee`); these items are
intentionally NOT blockers — recorded here so they don't get lost, per the
final whole-branch review and its scoped re-review. None is a correctness
bug in the merged code; each is a real, verified gap or risk worth closing
in a future pass.

## 1. `output_tmp_path.unlink` is untested for the scenario it exists for

**Where:** `src/contrastive_pretrain/resize_cache.py:98-104`, inside
`build_local_resize_cache`'s `finally` block.

```python
finally:
    raw_tmp_path.unlink(missing_ok=True)
    output_tmp_path.unlink(missing_ok=True)
```

**Gap:** `tests/unit/test_contrastive_pretrain_resize_cache.py`'s existing
failure-path test raises inside `download_shard`, which runs *before* the
`try` block starts — so it never exercises the `finally`'s
`output_tmp_path.unlink` line at all. The scenario that line was added for
(a crash inside `.map()`/`.to_parquet()`, i.e. *after* the temp files exist
but *before* `os.replace` completes) has no regression test. If a future
change breaks this cleanup, nothing will catch it.

**Suggested fix:** add a test where a fake/monkeypatched `.to_parquet()` (or
the `_resize_row_for_cache` map function) raises after `raw_tmp_path` and
`output_tmp_path` would exist, and assert neither survives.

## 2. `HF_HUB_CACHE` redirect only checks that one env var, not `HF_HOME` or the legacy name

**Where:** `src/contrastive_pretrain/resize_cache.py:140-158`, inside
`ensure_local_cache`.

```python
os.environ.setdefault(
    "HF_HUB_CACHE", str(Path(config.local_cache_dir) / ".hf_hub_cache")
)
hf_constants.HF_HUB_CACHE = os.environ["HF_HUB_CACHE"]
```

**Gap:** `huggingface_hub`'s own default resolution for `HF_HUB_CACHE` also
considers `HF_HOME` and the legacy `HUGGINGFACE_HUB_CACHE` env var (see
`huggingface_hub/constants.py`). RunPod pod templates commonly set
`HF_HOME`. An operator relying on `HF_HOME` (not `HF_HUB_CACHE` directly)
would have it silently overridden by this `setdefault`, since we only check
for `HF_HUB_CACHE` being already set — not whether the equivalent behavior
was already configured via `HF_HOME`. Consequence is cache fragmentation
(two cache locations in play), not data loss or a correctness bug.

**Suggested fix:** also respect `HF_HOME` (and possibly the legacy var) when
deciding whether to redirect, or document the precedence explicitly in the
docstring so an operator knows to use `HF_HUB_CACHE` specifically here.

**Related, same root cause:** the redirect only relocates
`hf_hub_download`'s persistent blob onto the volume that has room — it does
not eliminate the duplication with the cache's own resized output. Full
elimination would mean threading a `cache_dir` parameter through the shared
`hf_storage/client.py` (`RealHfClient.download_bytes`), which serves other
packages (`data_collection`) and was ruled out of scope for this fix wave.
Revisit if disk usage on `/workspace` becomes a real constraint.

**Also related:** the write-through (`hf_constants.HF_HUB_CACHE = ...`)
doesn't apply `expanduser`/`expandvars` the way `huggingface_hub`'s own
constant resolution does. Verified directly against the installed package
(`huggingface_hub/constants.py`):

```python
HF_HUB_CACHE = os.path.expandvars(os.path.expanduser(os.getenv("HF_HUB_CACHE", HUGGINGFACE_HUB_CACHE)))
```

— both `expanduser` (`~`) and `expandvars` (`$VAR`) are applied at import
time, and `HUGGINGFACE_HUB_CACHE`'s own default is derived from `HF_HOME`
(confirming the `HF_HOME` gap above). Harmless for `hf_hub_download` itself
(it re-expands at call time in `file_download.py`), but any other code
reading `hf_constants.HF_HUB_CACHE` directly could see an unexpanded
literal (e.g. a literal `~` if an operator's `HF_HUB_CACHE` used one).

## 3. The `slow` marker now bundles two different reasons to skip a test

**Where:** `tests/unit/test_contrastive_pretrain_train.py` — 7 tests marked
`@pytest.mark.slow` at lines 142, 201, 423, 450, 475, 531, 567 (all drive a
real `run_training` call, ~60s each on CPU due to a real `torch.compile` of
a ResNet-50), plus 6 pre-existing `slow` tests elsewhere in the suite that
require real network/Hub credentials (line 628 in this same file, plus
tests in `test_contrastive_pretrain_dataset.py` and
`test_contrastive_pretrain_resize_cache.py`).

**Gap:** `pyproject.toml`'s `slow` marker description was broadened to cover
both reasons ("requires real network/ffmpeg/HF credentials, OR is too slow
for the default suite"), but pytest's `-m slow` / `-m "not slow"` selection
can't distinguish the two groups. You can't currently run "just the slow
CPU tests, skip the ones needing credentials" or vice versa.

**Suggested fix:** split into two markers, e.g. `slow` (credentials/network)
and `expensive` (CPU-bound but credential-free), so CI or a local dev loop
can select either independently. This was raised mid-implementation and
explicitly deferred to a future test-organization pass rather than done as
part of this branch.

## 4. `list_repo_files`'s retry policy can silently block training start for hours on a real rate limit

**Where:** `src/contrastive_pretrain/resize_cache.py:163-182`
(`_list_shard_paths`, inside `ensure_local_cache`), using the same
`rate_limit_aware_backoff(base_delay=2.0, rate_limit_delay=3600.0)` policy
`_download_shard` already used (`resize_cache.py:184-200`).

**Gap:** this is the *correct* fix for the finding it addressed (an
un-retried Hub call), and applying the existing policy consistently is the
right call — but the policy itself means a real HF 429 on `list_repo_files`
now costs up to `max_retries - 1` × 3600s of essentially silent blocking
(no log line between attempts) before either succeeding or raising. Since
`ensure_local_cache` runs on *every* `build_train_dataset`/
`build_val_dataset` call (even against a fully-built cache), this sits on
the critical path of every training start, not just the one-time build.

**Suggested fix:** not a bug, but worth revisiting `rate_limit_delay=3600.0`
independently of this branch, and/or adding a log line before each retry
sleep so a stuck training start is diagnosable from the logs rather than
looking like a hang.

## 5. A permanently-missing shard is retried ~30s before failing, instead of failing fast

**Where:** `src/contrastive_pretrain/resize_cache.py:184-191`
(`_download_shard`'s inner `_fetch`, which now raises
`FileNotFoundError(f"shard listed but missing on Hub: {path}")` instead of
the old bare `assert`), wrapped in `retry_with_backoff`
(`src/hf_storage/retry.py`, which catches bare `Exception`).

**Gap:** replacing the `assert` with an explicit `FileNotFoundError` was the
right fix for the original finding (asserts are stripped under `python
-O`, and the un-guarded `write_bytes(None)` failure was confusing) — but
`retry_with_backoff`'s generic `except Exception` still retries this
exception like any transient failure, costing ~30s of exponential backoff
(2+4+8+16s) before it propagates. This is **not a regression**: the old
`AssertionError` was caught and retried identically under the same bare
`except Exception`. It's a pre-existing wart in `retry_with_backoff` that
this fix didn't introduce and wasn't asked to fix.

**Suggested fix:** if `retry_with_backoff` ever gains the ability to
special-case genuinely non-retryable errors (the way it already
special-cases rate-limit errors via `rate_limit_aware_backoff`), route
`FileNotFoundError` through that path to fail immediately instead of
retrying.

---

## Not deferred — already fixed

For context, `configs/contrastive_pretrain.yaml` not setting
`local_cache_dir` was flagged as a gap and has already been fixed (it's now
set to `/workspace/contrastive_pretrain/resized_cache`, matching the
sibling `network_volume_checkpoint_dir` convention). It is not part of this
follow-up list.
