# Slow test suite is unrunnable on the dev machine

Date: 2026-08-26
Status: **not fixed** — recorded for a follow-up pass, per instruction. None
of this was introduced by the resize-cache / checkpoint-retention / build-cache
work landed the same day; all five failures are environmental.

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
