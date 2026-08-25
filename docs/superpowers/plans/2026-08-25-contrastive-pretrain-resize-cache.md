# Contrastive Pretraining — Local Resize Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the data-starved streaming pipeline (per-step waits up to 76s, GPU reading 0% utilization) by caching resized frames locally on `/workspace`, built automatically on first use — no new CLI command, one config field.

**Architecture:** A new `contrastive_pretrain/resize_cache.py` module downloads each Hub shard, resizes every row via the existing `_resize_to_canonical`, and writes it back as a local Parquet shard (atomic write, per-shard resumability). `dataset.py`'s `_load_base_stream` gets a branch to read from that local cache instead of the Hub, and `build_train_dataset`/`build_val_dataset` call the new `ensure_local_cache()` on every invocation (a fast no-op once built) and skip the now-redundant pre-shuffle resize-map stage when reading from the cache.

**Tech Stack:** Python 3.12, `datasets` (streaming + Parquet I/O), `huggingface_hub` (`HfApi`, existing `RealHfClient`), `torchvision.transforms.v2` (existing `_resize_to_canonical`), `pytest` + `monkeypatch` for fakes, `uv` for all execution.

**Spec:** `docs/superpowers/specs/2026-08-25-contrastive-pretrain-resize-cache-design.md`

## Global Constraints

- Package management: `uv` only — every command below is `uv run <cmd>`, never bare `python`/`pytest`/`pip`.
- TDD: write the failing test before the implementation in every task; run it and confirm the failure reason before writing code.
- No placeholders, no speculative abstraction — surgical diffs only, per the spec's explicit scope boundary (this plan touches only `hf_storage/retry.py`, `contrastive_pretrain/config.py`, a new `contrastive_pretrain/resize_cache.py`, and `contrastive_pretrain/dataset.py`).
- Structured logging: any new `logger.info(...)` call uses the existing `extra={...}` JSON-lines convention already used throughout `dataset.py`.
- Fast tests (the default `pytest` marker selection, `-m "not slow"`) must never touch the network or real Hub credentials. The one real-Hub test in this plan (Task 6) is marked `@pytest.mark.slow`.
- Commit after every task with tests passing.

---

## File Structure

```
src/hf_storage/retry.py                              # MODIFY — generic return type
src/contrastive_pretrain/config.py                   # MODIFY — new local_cache_dir field
src/contrastive_pretrain/resize_cache.py             # NEW — cache-build core + Hub wiring
src/contrastive_pretrain/dataset.py                  # MODIFY — local-cache read branch + wiring

tests/unit/test_hf_storage_retry.py                  # MODIFY
tests/unit/test_contrastive_pretrain_config.py       # MODIFY
tests/unit/test_contrastive_pretrain_resize_cache.py # NEW
tests/unit/test_contrastive_pretrain_dataset.py      # MODIFY
```

---

### Task 1: Widen `retry_with_backoff` to a generic return type

**Files:**
- Modify: `src/hf_storage/retry.py:45-62`
- Test: `tests/unit/test_hf_storage_retry.py`

**Interfaces:**
- Consumes: nothing new (pure refactor of existing `retry_with_backoff`).
- Produces: `retry_with_backoff(func: Callable[[], T], max_retries: int, base_delay: float, sleep_func: Callable[[float], None], backoff_seconds: BackoffFunc | None = None) -> T` — Task 4's `ensure_local_cache` relies on the return value being passed through.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_hf_storage_retry.py`:

```python
def test_retry_with_backoff_returns_the_wrapped_callables_result() -> None:
    def make_greeting() -> str:
        return "hello"

    result = retry_with_backoff(make_greeting, max_retries=1, base_delay=1.0, sleep_func=lambda _: None)

    assert result == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_hf_storage_retry.py::test_retry_with_backoff_returns_the_wrapped_callables_result -v`
Expected: FAIL — `assert None == "hello"` (current code calls `func()` and discards the result, then hits a bare `return`).

- [ ] **Step 3: Write minimal implementation**

Replace `src/hf_storage/retry.py:45-62` (the `retry_with_backoff` function) with:

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

And add the `TypeVar` import/definition near the top of the file (after the existing `from collections.abc import Callable` line):

```python
from typing import TypeVar

T = TypeVar("T")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_hf_storage_retry.py -v`
Expected: PASS — all existing tests in this file plus the new one.

- [ ] **Step 5: Commit**

```bash
git add src/hf_storage/retry.py tests/unit/test_hf_storage_retry.py
git commit -m "feat: widen retry_with_backoff to a generic return type"
```

---

### Task 2: Add `TrainingConfig.local_cache_dir`

**Files:**
- Modify: `src/contrastive_pretrain/config.py:11-32`
- Test: `tests/unit/test_contrastive_pretrain_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TrainingConfig.local_cache_dir: str | None` (default `None`) — Task 4 (`ensure_local_cache`) and Task 5 (`_load_base_stream`, `build_train_dataset`, `build_val_dataset`) branch on this field.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_contrastive_pretrain_config.py`:

```python
def test_training_config_local_cache_dir_defaults_to_none() -> None:
    assert TrainingConfig().local_cache_dir is None


def test_load_config_applies_local_cache_dir_override(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("local_cache_dir: /workspace/contrastive_pretrain/resized_cache\n")

    config = load_config(path)

    assert config.local_cache_dir == "/workspace/contrastive_pretrain/resized_cache"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_config.py -v`
Expected: FAIL — `test_training_config_local_cache_dir_defaults_to_none` fails with `AttributeError: 'TrainingConfig' object has no attribute 'local_cache_dir'`; `test_load_config_applies_local_cache_dir_override` fails with `ValueError: unknown config field(s): {'local_cache_dir'}`.

- [ ] **Step 3: Write minimal implementation**

In `src/contrastive_pretrain/config.py`, add this field to the `TrainingConfig` dataclass (after `network_volume_checkpoint_dir`):

```python
    # None (default): stream objones25/pokemon-frames directly from the
    # Hub, resizing per-row on the fly, exactly as before this field
    # existed. Set to a /workspace-backed directory to cache already-
    # resized frames there instead -- build_train_dataset/build_val_dataset
    # populate it automatically on first use (see
    # contrastive_pretrain.resize_cache.ensure_local_cache), eliminating
    # both the per-epoch network round-trips and per-epoch resize work on
    # every call after that.
    local_cache_dir: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_config.py -v`
Expected: PASS — all tests in this file.

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/config.py tests/unit/test_contrastive_pretrain_config.py
git commit -m "feat: add TrainingConfig.local_cache_dir"
```

---

### Task 3: `resize_cache.py` core — `build_local_resize_cache`

**Files:**
- Create: `src/contrastive_pretrain/resize_cache.py`
- Test: Create `tests/unit/test_contrastive_pretrain_resize_cache.py`

**Interfaces:**
- Consumes: `contrastive_pretrain.dataset._resize_to_canonical(example: dict) -> dict` (existing, returns `{"image": <(1,H,W) uint8 torch.Tensor>}`).
- Produces: `build_local_resize_cache(list_shard_paths: Callable[[], list[str]], download_shard: Callable[[str], bytes], local_cache_dir: Path) -> None` — Task 4's `ensure_local_cache` calls this with production Hub-backed callables.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_contrastive_pretrain_resize_cache.py`:

```python
import io

import datasets
import numpy as np
import pytest
from PIL import Image

from contrastive_pretrain.resize_cache import build_local_resize_cache


def _native_res_shard_bytes(video_id: str) -> bytes:
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "video_id": datasets.Value("string"),
            "timestamp_s": datasets.Value("float64"),
        }
    )
    pixels = np.random.default_rng(0).integers(0, 256, (2160, 2400), dtype=np.uint8)
    rows = [{"image": Image.fromarray(pixels, mode="L"), "video_id": video_id, "timestamp_s": 0.0}]
    ds = datasets.Dataset.from_list(rows, features=features)
    buf = io.BytesIO()
    ds.to_parquet(buf)
    return buf.getvalue()


def test_build_local_resize_cache_writes_resized_shard_for_each_listed_path(tmp_path) -> None:
    shard_bytes = {
        "shards/vidA/00000.parquet": _native_res_shard_bytes("vidA"),
        "shards/vidB/00000.parquet": _native_res_shard_bytes("vidB"),
    }

    build_local_resize_cache(
        list_shard_paths=lambda: list(shard_bytes),
        download_shard=lambda path: shard_bytes[path],
        local_cache_dir=tmp_path,
    )

    for shard_path in shard_bytes:
        output_path = tmp_path / shard_path
        assert output_path.exists()
        reloaded = datasets.Dataset.from_parquet(str(output_path))
        assert reloaded.num_rows == 1
        assert reloaded[0]["image"].size == (160, 144)  # PIL size is (width, height)
        assert reloaded[0]["image"].mode == "L"


def test_build_local_resize_cache_skips_shards_whose_output_already_exists(tmp_path) -> None:
    existing_shard = "shards/vidA/00000.parquet"
    new_shard = "shards/vidB/00000.parquet"
    output_path = tmp_path / existing_shard
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(_native_res_shard_bytes("vidA"))  # any prior content marks it done

    download_calls: list[str] = []

    def download_shard(path: str) -> bytes:
        download_calls.append(path)
        return _native_res_shard_bytes("vidB")

    build_local_resize_cache(
        list_shard_paths=lambda: [existing_shard, new_shard],
        download_shard=download_shard,
        local_cache_dir=tmp_path,
    )

    assert download_calls == [new_shard]
    assert (tmp_path / new_shard).exists()


def test_build_local_resize_cache_leaves_no_partial_file_on_failure_and_resumes_correctly(tmp_path) -> None:
    ok_shard = "shards/vidA/00000.parquet"
    failing_shard = "shards/vidB/00000.parquet"

    def failing_download(path: str) -> bytes:
        if path == failing_shard:
            raise RuntimeError("simulated network failure")
        return _native_res_shard_bytes("vidA")

    with pytest.raises(RuntimeError, match="simulated network failure"):
        build_local_resize_cache(
            list_shard_paths=lambda: [ok_shard, failing_shard],
            download_shard=failing_download,
            local_cache_dir=tmp_path,
        )

    assert (tmp_path / ok_shard).exists()
    assert not (tmp_path / failing_shard).exists()

    download_calls: list[str] = []

    def succeeding_download(path: str) -> bytes:
        download_calls.append(path)
        return _native_res_shard_bytes("vidB")

    build_local_resize_cache(
        list_shard_paths=lambda: [ok_shard, failing_shard],
        download_shard=succeeding_download,
        local_cache_dir=tmp_path,
    )

    assert download_calls == [failing_shard]  # ok_shard skipped, not re-downloaded
    assert (tmp_path / failing_shard).exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_resize_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'contrastive_pretrain.resize_cache'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/contrastive_pretrain/resize_cache.py`:

```python
"""One-time local Parquet cache of resized frames, built from
objones25/pokemon-frames' native-resolution shards. See
docs/superpowers/specs/2026-08-25-contrastive-pretrain-resize-cache-design.md
for the full design rationale.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import datasets
from huggingface_hub import HfApi
from PIL import Image

from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.dataset import _resize_to_canonical
from hf_storage.client import RealHfClient
from hf_storage.retry import rate_limit_aware_backoff, retry_with_backoff

logger = logging.getLogger(__name__)


def _resize_row_for_cache(example: dict) -> dict:
    """_resize_to_canonical returns a channel-first (1, H, W) uint8 tensor --
    correct for the streaming pipeline (which never re-serializes it), but
    writing that tensor (or a raw (H, W) numpy array) straight into a
    datasets.Image()-typed column silently corrupts it on parquet
    round-trip: verified empirically that it either collapses the column
    to a nested-list type (losing the Image feature entirely) or, with the
    schema preserved explicitly, downcasts pixels to 32-bit int mode.
    Returning an actual PIL.Image sidesteps this -- datasets recognizes it
    directly and PNG-encodes it under the Image() feature, no explicit
    features= needed on .map()."""
    frame = _resize_to_canonical(example)["image"]  # (1, H, W) uint8
    return {"image": Image.fromarray(frame.squeeze(0).numpy(), mode="L")}


def build_local_resize_cache(
    list_shard_paths: Callable[[], list[str]],
    download_shard: Callable[[str], bytes],
    local_cache_dir: Path,
) -> None:
    """Downloads and resizes every shard `list_shard_paths()` returns that
    isn't already present under `local_cache_dir`, writing each as a local
    Parquet shard at the same relative path. Safe to interrupt and rerun:
    already-completed shards are skipped, and a shard is never considered
    complete until its output file has been atomically renamed into
    place."""
    shard_paths = list_shard_paths()
    completed = 0
    skipped = 0
    total_rows = 0
    for shard_path in shard_paths:
        output_path = local_cache_dir / shard_path
        if output_path.exists():
            skipped += 1
            logger.info("resize_cache_shard_skipped", extra={"shard_path": shard_path})
            continue

        start = time.monotonic()
        raw_bytes = download_shard(shard_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_tmp_path = output_path.parent / f"{output_path.name}.raw.tmp"
        output_tmp_path = output_path.parent / f"{output_path.name}.tmp"
        try:
            raw_tmp_path.write_bytes(raw_bytes)
            raw_dataset = datasets.Dataset.from_parquet(str(raw_tmp_path))
            resized_dataset = raw_dataset.map(_resize_row_for_cache)
            resized_dataset.to_parquet(str(output_tmp_path))
            os.replace(output_tmp_path, output_path)
        finally:
            raw_tmp_path.unlink(missing_ok=True)

        elapsed = time.monotonic() - start
        completed += 1
        total_rows += resized_dataset.num_rows
        logger.info(
            "resize_cache_shard_done",
            extra={"shard_path": shard_path, "rows": resized_dataset.num_rows, "elapsed_s": elapsed},
        )

    logger.info(
        "resize_cache_complete",
        extra={"shard_count": completed, "skipped_count": skipped, "total_rows": total_rows},
    )
```

(`ensure_local_cache` — the second public function this module needs — is added in Task 4; leaving it out for now keeps this task's diff focused on the one thing it tests.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_resize_cache.py -v`
Expected: PASS — all three tests.

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/resize_cache.py tests/unit/test_contrastive_pretrain_resize_cache.py
git commit -m "feat: add build_local_resize_cache for one-time frame resizing"
```

---

### Task 4: `resize_cache.py` wiring — `ensure_local_cache`

**Files:**
- Modify: `src/contrastive_pretrain/resize_cache.py` (append)
- Test: Modify `tests/unit/test_contrastive_pretrain_resize_cache.py`

**Interfaces:**
- Consumes: `build_local_resize_cache` (Task 3, same module), `TrainingConfig.local_cache_dir`/`.dataset_repo_id` (Task 2 / existing), `hf_storage.client.RealHfClient`, `hf_storage.retry.{retry_with_backoff, rate_limit_aware_backoff}` (Task 1).
- Produces: `ensure_local_cache(config: TrainingConfig) -> None` — Task 5's `dataset.py` calls this from `build_train_dataset`/`build_val_dataset`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_contrastive_pretrain_resize_cache.py`:

```python
from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.resize_cache import ensure_local_cache


def test_ensure_local_cache_is_a_noop_when_local_cache_dir_is_unset(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.build_local_resize_cache",
        lambda **kwargs: calls.append(kwargs),
    )
    config = TrainingConfig(local_cache_dir=None)

    ensure_local_cache(config)

    assert calls == []


def test_ensure_local_cache_wires_build_local_resize_cache_when_set(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_build_local_resize_cache(*, list_shard_paths, download_shard, local_cache_dir):
        captured["list_shard_paths"] = list_shard_paths
        captured["download_shard"] = download_shard
        captured["local_cache_dir"] = local_cache_dir

    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.build_local_resize_cache",
        fake_build_local_resize_cache,
    )

    class _FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            assert repo_id == "objones25/pokemon-frames"
            assert repo_type == "dataset"
            return ["shards/vidA/00000.parquet", "README.md", "shards/vidB/00000.parquet"]

    monkeypatch.setattr("contrastive_pretrain.resize_cache.HfApi", _FakeApi)

    class _FakeClient:
        def __init__(self, api, repo_id, repo_type):
            pass

        def download_bytes(self, path):
            return b"raw-bytes-for-" + path.encode()

    monkeypatch.setattr("contrastive_pretrain.resize_cache.RealHfClient", _FakeClient)

    config = TrainingConfig(local_cache_dir=str(tmp_path / "cache"))

    ensure_local_cache(config)

    assert captured["local_cache_dir"] == tmp_path / "cache"
    assert captured["list_shard_paths"]() == ["shards/vidA/00000.parquet", "shards/vidB/00000.parquet"]
    assert captured["download_shard"]("shards/vidA/00000.parquet") == b"raw-bytes-for-shards/vidA/00000.parquet"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_resize_cache.py -v`
Expected: FAIL — `ImportError: cannot import name 'ensure_local_cache' from 'contrastive_pretrain.resize_cache'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/contrastive_pretrain/resize_cache.py`:

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

    def _list_shard_paths() -> list[str]:
        return [
            p
            for p in api.list_repo_files(config.dataset_repo_id, repo_type="dataset")
            if p.startswith("shards/")
        ]

    def _download_shard(path: str) -> bytes:
        def _fetch() -> bytes:
            data = client.download_bytes(path)
            assert data is not None, f"shard listed but missing on Hub: {path}"
            return data

        return retry_with_backoff(
            _fetch,
            max_retries=5,
            base_delay=2.0,
            sleep_func=time.sleep,
            backoff_seconds=rate_limit_aware_backoff(base_delay=2.0, rate_limit_delay=3600.0),
        )

    build_local_resize_cache(
        list_shard_paths=_list_shard_paths,
        download_shard=_download_shard,
        local_cache_dir=Path(config.local_cache_dir),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_resize_cache.py -v`
Expected: PASS — all five tests in the file.

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/resize_cache.py tests/unit/test_contrastive_pretrain_resize_cache.py
git commit -m "feat: add ensure_local_cache to wire the resize cache to the real Hub"
```

---

### Task 5: Wire `dataset.py` to read from and populate the local cache

**Files:**
- Modify: `src/contrastive_pretrain/dataset.py:116-152`
- Test: Modify `tests/unit/test_contrastive_pretrain_dataset.py`

**Interfaces:**
- Consumes: `contrastive_pretrain.resize_cache.ensure_local_cache(config: TrainingConfig) -> None` (Task 4), `TrainingConfig.local_cache_dir` (Task 2).
- Produces: no new public interface — `build_train_dataset`/`build_val_dataset` keep their existing signatures; this task only changes their internal behavior.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_contrastive_pretrain_dataset.py` (near the other `build_train_dataset`/`build_val_dataset` tests):

```python
def test_load_base_stream_reads_from_local_cache_when_configured(tmp_path) -> None:
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "video_id": datasets.Value("string"),
            "timestamp_s": datasets.Value("float64"),
        }
    )
    pixels = np.random.default_rng(0).integers(0, 256, (144, 160), dtype=np.uint8)
    rows = [{"image": Image.fromarray(pixels, mode="L"), "video_id": "cachedA", "timestamp_s": 0.0}]
    ds = datasets.Dataset.from_list(rows, features=features)
    shard_path = tmp_path / "shards" / "cachedA" / "00000.parquet"
    shard_path.parent.mkdir(parents=True)
    ds.to_parquet(str(shard_path))

    config = TrainingConfig(local_cache_dir=str(tmp_path))

    stream = _load_base_stream(config)

    assert [row["video_id"] for row in stream] == ["cachedA"]


def test_build_train_dataset_calls_ensure_local_cache_before_loading(monkeypatch) -> None:
    call_order: list[str] = []
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.ensure_local_cache",
        lambda config: call_order.append("ensure_local_cache"),
    )

    def fake_load_base_stream(config):
        call_order.append("load_base_stream")
        return _synthetic_frame_stream(["train_a"])

    monkeypatch.setattr("contrastive_pretrain.dataset._load_base_stream", fake_load_base_stream)
    config = TrainingConfig(val_video_ids=("val_a",))

    list(build_train_dataset(config))

    assert call_order == ["ensure_local_cache", "load_base_stream"]


def test_build_train_dataset_skips_resize_map_when_local_cache_dir_is_set(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("contrastive_pretrain.resize_cache.ensure_local_cache", lambda config: None)
    resize_instantiations: list[int] = []

    class _CountingResize:
        def __init__(self) -> None:
            resize_instantiations.append(1)

        def __call__(self, example):
            return example

    monkeypatch.setattr("contrastive_pretrain.dataset._ResizeToCanonicalWithProgress", _CountingResize)
    monkeypatch.setattr(
        "contrastive_pretrain.dataset._load_base_stream",
        lambda config: _synthetic_frame_stream(["train_a"]),
    )
    config = TrainingConfig(val_video_ids=("val_a",), local_cache_dir=str(tmp_path))

    list(build_train_dataset(config))

    assert resize_instantiations == []


def test_build_train_dataset_still_uses_resize_map_when_local_cache_dir_is_unset(monkeypatch) -> None:
    resize_instantiations: list[int] = []

    class _CountingResize:
        def __init__(self) -> None:
            resize_instantiations.append(1)

        def __call__(self, example):
            return example

    monkeypatch.setattr("contrastive_pretrain.dataset._ResizeToCanonicalWithProgress", _CountingResize)
    monkeypatch.setattr(
        "contrastive_pretrain.dataset._load_base_stream",
        lambda config: _synthetic_frame_stream(["train_a"]),
    )
    config = TrainingConfig(val_video_ids=("val_a",))

    list(build_train_dataset(config))

    assert resize_instantiations == [1]


def test_build_val_dataset_calls_ensure_local_cache_before_loading(monkeypatch) -> None:
    call_order: list[str] = []
    monkeypatch.setattr(
        "contrastive_pretrain.resize_cache.ensure_local_cache",
        lambda config: call_order.append("ensure_local_cache"),
    )

    def fake_load_base_stream(config):
        call_order.append("load_base_stream")
        return _synthetic_frame_stream(["val_a"])

    monkeypatch.setattr("contrastive_pretrain.dataset._load_base_stream", fake_load_base_stream)
    config = TrainingConfig(val_video_ids=("val_a",))

    list(build_val_dataset(config))

    assert call_order == ["ensure_local_cache", "load_base_stream"]


def test_build_val_dataset_skips_resize_map_when_local_cache_dir_is_set(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("contrastive_pretrain.resize_cache.ensure_local_cache", lambda config: None)
    resize_instantiations: list[int] = []

    class _CountingResize:
        def __init__(self) -> None:
            resize_instantiations.append(1)

        def __call__(self, example):
            return example

    monkeypatch.setattr("contrastive_pretrain.dataset._ResizeToCanonicalWithProgress", _CountingResize)
    monkeypatch.setattr(
        "contrastive_pretrain.dataset._load_base_stream",
        lambda config: _synthetic_frame_stream(["val_a"]),
    )
    config = TrainingConfig(val_video_ids=("val_a",), local_cache_dir=str(tmp_path))

    list(build_val_dataset(config))

    assert resize_instantiations == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_dataset.py -v`
Expected: FAIL —
- `test_load_base_stream_reads_from_local_cache_when_configured` fails because `_load_base_stream` ignores `config.local_cache_dir` and tries to stream `objones25/pokemon-frames` from the real Hub.
- `test_build_train_dataset_calls_ensure_local_cache_before_loading` fails with `AttributeError: <module 'contrastive_pretrain.resize_cache'> does not have the attribute 'ensure_local_cache'` — wait, it exists after Task 4; it actually fails because `build_train_dataset` never calls it, so `call_order == ["load_base_stream"]`, not `["ensure_local_cache", "load_base_stream"]`.
- `test_build_train_dataset_skips_resize_map_when_local_cache_dir_is_set` fails: `resize_instantiations == [1]`, not `[]`.
- `test_build_train_dataset_still_uses_resize_map_when_local_cache_dir_is_unset` already passes (no regression expected there, but run it anyway to confirm nothing else broke).
- `test_build_val_dataset_calls_ensure_local_cache_before_loading` and `test_build_val_dataset_skips_resize_map_when_local_cache_dir_is_set` fail the same way as their `build_train_dataset` counterparts, for the same reason.

- [ ] **Step 3: Write minimal implementation**

Replace `src/contrastive_pretrain/dataset.py:116-152` (`_load_base_stream`, `build_train_dataset`, `build_val_dataset`) with:

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
    # Local import: resize_cache.py imports _resize_to_canonical from this
    # module at module level, so importing resize_cache back at this
    # module's top level would be circular.
    from contrastive_pretrain.resize_cache import ensure_local_cache

    ensure_local_cache(config)
    logger.info(
        "build_train_dataset",
        extra={
            "dataset_repo_id": config.dataset_repo_id,
            "shuffle_buffer_size": config.shuffle_buffer_size,
            "val_video_ids": config.val_video_ids,
        },
    )
    ds = _load_base_stream(config)
    ds = ds.filter(lambda ex: ex["video_id"] not in config.val_video_ids)
    if not config.local_cache_dir:
        ds = ds.map(_ResizeToCanonicalWithProgress())  # BEFORE shuffle -- see its
        # docstring; skipped for the local-cache path, whose frames are
        # already canonical-sized (see resize_cache.py's design rationale).
    ds = ds.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.seed)
    ds = ds.map(
        functools.partial(to_pair_transform, augmentation_config=AugmentationConfig(), base_seed=config.seed),
        remove_columns=["image"],
    )
    return ds


def build_val_dataset(config: TrainingConfig):
    from contrastive_pretrain.resize_cache import ensure_local_cache

    ensure_local_cache(config)
    logger.info(
        "build_val_dataset",
        extra={"dataset_repo_id": config.dataset_repo_id, "val_video_ids": config.val_video_ids},
    )
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_dataset.py -v`
Expected: PASS — all tests in the file, including the six new ones and every pre-existing test (in particular `test_build_train_dataset_resizes_native_resolution_frames_before_shuffling`, the OOM regression test, must still pass since `local_cache_dir` is unset there).

Then run the full fast suite to confirm no cross-module regressions:

Run: `uv run pytest -v`
Expected: PASS (all non-`slow` tests).

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/dataset.py tests/unit/test_contrastive_pretrain_dataset.py
git commit -m "feat: read from local resize cache when configured, skip redundant resize-map"
```

---

### Task 6: Real-Hub integration test (slow, opt-in)

**Files:**
- Modify: `tests/unit/test_contrastive_pretrain_resize_cache.py` (append)

**Interfaces:**
- Consumes: `build_local_resize_cache` (Task 3), real `HfApi`/`RealHfClient` against `objones25/pokemon-frames`.
- Produces: nothing new — this is a verification-only test.

- [ ] **Step 1: Write the test**

Append to `tests/unit/test_contrastive_pretrain_resize_cache.py`:

```python
@pytest.mark.slow
def test_build_local_resize_cache_against_real_hub_shard(tmp_path) -> None:
    """Confirms the real Hub schema round-trips through the real resize +
    local Parquet write correctly -- every other test in this file uses
    synthetic fixtures; this is the one check against the real
    objones25/pokemon-frames dataset."""
    from huggingface_hub import HfApi

    from hf_storage.client import RealHfClient

    api = HfApi()
    client = RealHfClient(api, "objones25/pokemon-frames", repo_type="dataset")
    shard_paths = [
        p
        for p in api.list_repo_files("objones25/pokemon-frames", repo_type="dataset")
        if p.startswith("shards/")
    ][:1]
    assert shard_paths, "expected at least one shard under shards/ in objones25/pokemon-frames"

    build_local_resize_cache(
        list_shard_paths=lambda: shard_paths,
        download_shard=lambda path: client.download_bytes(path),
        local_cache_dir=tmp_path,
    )

    output_path = tmp_path / shard_paths[0]
    assert output_path.exists()
    reloaded = datasets.Dataset.from_parquet(str(output_path))
    assert reloaded.num_rows > 0
    assert reloaded[0]["image"].size == (160, 144)
```

- [ ] **Step 2: Run it explicitly (it's excluded from the default suite)**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_resize_cache.py -m slow -v`
Expected: PASS (requires `HF_TOKEN`/`hf auth login` credentials with access to `objones25/pokemon-frames`, per this project's existing `slow`-marker convention).

- [ ] **Step 3: Confirm the default (fast) suite still excludes it**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_resize_cache.py -v`
Expected: PASS, and the slow test does not appear in the run output (excluded by `addopts = "-m \"not slow\""` in `pyproject.toml`).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_contrastive_pretrain_resize_cache.py
git commit -m "test: add slow integration test for build_local_resize_cache against the real Hub"
```

---

## After Task 6

At this point, setting `local_cache_dir: /workspace/contrastive_pretrain/resized_cache` in `configs/contrastive_pretrain.yaml` and running `contrastive-pretrain train` (or `export-frozen-encoder`) is the entire operator-facing change: the first invocation builds the cache automatically (visible via `resize_cache_shard_done`/`resize_cache_complete` logs), and every invocation after that is a fast no-op that reads local Parquet instead of streaming from the Hub.
