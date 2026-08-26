# Slow test suite is unrunnable on the dev machine

Date: 2026-08-26
Status: **resolved** — see "Resolution" at the bottom. The diagnosis below was
right about the mechanism (no credentials in the pytest process, dead local
trust store) but wrong about the conclusion: only one of the five failures was
purely environmental. Three of the tests were redundant with existing offline
coverage, and two asserted guarantees the implementation does not provide.

## Symptom

`pytest -m slow` fails 5 of 6 tests, in ~4 seconds — far too fast for tests
that are supposed to download real Hub shards and real pretrained weights.
They fail at connection/auth time, before doing any real work.

```
FAILED tests/unit/test_contrastive_pretrain_dataset.py::test_build_train_dataset_excludes_val_videos
FAILED tests/unit/test_contrastive_pretrain_dataset.py::test_build_val_dataset_only_yields_held_out_videos
FAILED tests/unit/test_contrastive_pretrain_dataset.py::test_build_dataloader_resumes_against_real_streaming_data
FAILED tests/unit/test_contrastive_pretrain_resize_cache.py::test_build_local_resize_cache_against_real_hub_shard
FAILED tests/unit/test_contrastive_pretrain_train.py::test_run_training_completes_a_few_steps_without_nan
5 failed, 1 skipped, 249 deselected
```

(The 6th, `tests/integration/test_extraction_smoke.py`, skips cleanly on
`POKEMON_RL_TEST_CLIP` and is working as designed.)

## Two independent root causes

### 1. Four failures: no HF credentials inside the pytest process

```
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error.
  Repository Not Found for url: https://huggingface.co/api/datasets/objones25/pokemon-frames/tree/main
  Invalid username or password.
datasets.exceptions.DatasetNotFoundError: Dataset 'objones25/pokemon-frames' doesn't exist or cannot be accessed.
```

`objones25/pokemon-frames` is a **private** repo. The token lives in `.env`,
and `.env` is loaded by `contrastive_pretrain.cli.main()` via `load_dotenv()` —
which is a Click entry point that **pytest never invokes**. So the slow tests
run tokenless and get a 401, which the Hub reports as "repo doesn't exist".

Confirmed the token itself is fine: the same shard downloads correctly from a
plain script that calls `load_dotenv()` first (113.1 MB, verified during the
resize-cache work).

**Fix direction (not applied):** load `.env` in `tests/conftest.py`, or add an
autouse fixture that skips `slow` tests with a clear reason when
`huggingface_hub.get_token()` is None. The current failure mode is actively
misleading — a missing token reports as a missing dataset.

### 2. One failure: local SSL trust store can't verify download.pytorch.org

```
Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
  certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)>
```

`test_run_training_completes_a_few_steps_without_nan` is the one test that
deliberately uses the *real* ResNet-50 with `pretrained=True`. torchvision
fetches the weights over `urllib`, which uses the system trust store rather
than `certifi` — a known python.org-installer gap on macOS (this interpreter is
`/Library/Frameworks/Python.framework/Versions/3.12`).

Unrelated to the Hub credentials: this one would still fail with a valid token.

**Fix direction (not applied):** run `/Applications/Python 3.12/Install Certificates.command`,
or point `SSL_CERT_FILE` at `certifi.where()`. Note this is exactly the class of
environment breakage the RunPod runbook already documents for IPv6 — worth a
`scripts/check_env.py`-style preflight rather than rediscovering it each time.

## Why this matters beyond the dev loop

These 5 tests are the only automated coverage that touches the real Hub schema
and real pretrained weights. Right now `pytest` is green (249 passed) while
that entire tier is silently unexercised — and the failure text for cause #1
points at the wrong thing ("dataset doesn't exist"), so it reads like a data
problem rather than an auth problem.

Nothing about the RunPod run is blocked by this: a pod has `HF_TOKEN` exported
per the runbook and a working trust store. This is a dev-machine gap.

## What *was* verified without the slow suite

The real-Hub paths the slow tests would have covered were checked directly
instead, with `.env` loaded:

- One real shard (`shards/D1SrSFZrV7A/00000.parquet`) downloaded and resized:
  113.1 MB → 2.35 MB, output is a valid `Image` feature, 160×144 mode `L`, with
  `video_id` / `timestamp_s` / `game` preserved.
- `_discard_hub_blob` against a real `hf_hub_download` cache: 113.1 MB
  reclaimed, caller still receives its bytes.
- Dataset totals read from the Hub API: 367 shards, 64.27 GB.

## Resolution (2026-08-26, follow-up pass)

`.env` is deliberately **not** loaded in the test process. A test run that
silently picked up the developer's `.env` would authenticate as them against
real private repos and a real W&B account with nothing in the test asking for
it — the hazard `test_train_command_fails_fast_with_no_wandb_credentials`
already neutralizes `load_dotenv` for. The live tier reads the *ambient*
credential instead (`HF_TOKEN`, or the `huggingface_hub` token file), via a
`requires_hf_credentials` skipif in `tests/conftest.py` that carries the real
reason. A pod exports `HF_TOKEN` per the runbook, so the live tier runs there
and skips legibly on a dev machine.

Per-test outcome:

- `test_build_train_dataset_excludes_val_videos` and
  `test_build_val_dataset_only_yields_held_out_videos` — deleted. Both were
  exact duplicates of their `_fast` twins, which cover the same filter offline.
  The val one also looped `zip(range(5), ds)`, so it passed vacuously when the
  val stream was *empty* — the one outcome that actually matters.
- `test_build_local_resize_cache_against_real_hub_shard` — deleted. It
  downloaded ~113MB to assert `num_rows > 0` and `image.size == (160, 144)`,
  both strictly weaker than the synthetic-fixture test above it, which compares
  actual pixels.
- `test_build_dataloader_resumes_against_real_streaming_data` — **could never
  have passed**, credentials or not. It asserted exact batch-for-batch resume
  through `build_train_dataset`, which ends in `.shuffle()`; the shuffle
  buffer's contents are not part of the checkpointed state, so a resumed train
  loader re-fills the buffer from the source and serves a *different* next
  batch. Measured directly (see below). Replaced by
  `test_build_val_dataloader_resumes_from_exact_position_over_streamed_parquet_shards`,
  which runs the same mechanic over real on-disk Parquet shards through the
  *val* pipeline (no shuffle), where exact resume is a true property. Offline,
  so it now runs in the default suite instead of never running.
- `test_run_training_completes_a_few_steps_without_nan` — **also could never
  have passed**: `max_epochs=1` over all 367 shards / 64GB with no step bound.
  Replaced by `test_run_training_with_real_pretrained_encoder_completes_without_nan`,
  which keeps the real ResNet-50, real ImageNet weights and real
  `torch.compile` but feeds the in-memory fake stream, finishing in ~60s. It
  needs no Hub credentials, so the SSL trust store is now its only environmental
  dependency.

Known behaviour, worth writing down: **mid-epoch train resume is approximate,
not exact.** `.shuffle(buffer_size=N)`'s buffer is not checkpointed, so a
resumed run re-serves up to N rows per worker in a different order. Benign for
SimCLR (nothing depends on epoch-exact sample coverage), but it is not the
guarantee the old test claimed, and it is what
`train.py`'s dataloader-state restore actually delivers.

Also added: `test_build_encoder_with_pretrained_true_loads_the_real_imagenet_weights`
(`@slow`), comparing `conv1` against the reference torchvision model. A
randomly-initialised ResNet-50 trains to a perfectly finite loss, so before
this an inverted `pretrained` flag would have shipped a from-scratch backbone
with the whole suite green.

Cause #2 (SSL) is unchanged and is genuinely environment-only:
`/Applications/Python 3.12/Install Certificates.command`, or
`SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')`.
