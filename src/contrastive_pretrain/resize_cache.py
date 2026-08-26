"""One-time local Parquet cache of resized frames, built from
objones25/pokemon-frames' native-resolution shards. See
docs/superpowers/specs/2026-08-25-contrastive-pretrain-resize-cache-design.md
for the full design rationale.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import datasets
from huggingface_hub import HfApi, try_to_load_from_cache
from huggingface_hub import constants as hf_constants
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


def _discard_hub_blob(repo_id: str, filename: str, cache_dir: str) -> None:
    """Deletes one file from huggingface_hub's download cache, both the
    blob holding the content and the snapshot symlink pointing at it.

    Called once a shard has been resized, because nothing else ever cleans
    that cache up: hf_hub_download persists every file it fetches, and for
    objones25/pokemon-frames that is 64.3GB of raw shards (measured) piling
    up on a 50GB volume, none of it read again after its shard is resized.

    Resolving BEFORE unlinking matters: the snapshot entry is a symlink to
    the blob, so unlinking it first would leave the blob (the large part)
    unreachable and undeletable. Where the two are the same file -- a cache
    populated without symlinks -- the second unlink is a no-op, which is
    what missing_ok covers.

    A file that was never cached is not an error: download_bytes may be
    served by something that never touches the hub cache at all."""
    cached = try_to_load_from_cache(
        repo_id=repo_id, filename=filename, cache_dir=cache_dir, repo_type="dataset"
    )
    # try_to_load_from_cache returns a str path, None (not cached), or the
    # _CACHED_NO_EXIST sentinel object -- only the str case is a real file.
    if not isinstance(cached, str):
        return
    snapshot_link = Path(cached)
    blob = snapshot_link.resolve()
    snapshot_link.unlink(missing_ok=True)
    blob.unlink(missing_ok=True)


def build_local_resize_cache(
    list_shard_paths: Callable[[], list[str]],
    download_shard: Callable[[str], bytes],
    local_cache_dir: Path,
    num_proc: int | None = None,
) -> None:
    """Downloads and resizes every shard `list_shard_paths()` returns that
    isn't already present under `local_cache_dir`, writing each as a local
    Parquet shard at the same relative path. Safe to interrupt and rerun:
    already-completed shards are skipped, and a shard is never considered
    complete until its output file has been atomically renamed into
    place.

    `num_proc` is forwarded to .map(); None (the default) keeps the resize
    single-threaded. Parallelizing is safe with the keep_in_memory=True
    dataset below specifically because an in-memory dataset has no
    cache_files, so .map() writes no per-process cache shards anywhere
    (verified empirically) -- the whole reason keep_in_memory is set here."""
    shard_paths = list_shard_paths()
    completed = 0
    skipped = 0
    total_rows = 0
    for shard_path in shard_paths:
        output_path = local_cache_dir / shard_path
        if output_path.exists():
            # Deliberately not logged per shard: once the cache is fully built
            # this branch is every shard, on every training start, saying
            # nothing the resize_cache_complete summary below doesn't already
            # report as skipped_count.
            skipped += 1
            continue

        start = time.monotonic()
        raw_bytes = download_shard(shard_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_tmp_path = output_path.parent / f"{output_path.name}.raw.tmp"
        output_tmp_path = output_path.parent / f"{output_path.name}.tmp"
        arrow_tmp_dir = output_path.parent / f"{output_path.name}.arrow.tmp"
        try:
            raw_tmp_path.write_bytes(raw_bytes)
            # Left to its defaults, from_parquet spills a full Arrow copy of
            # every shard into ~/.cache/huggingface/datasets and never cleans
            # it up -- on a RunPod pod that is the container's small overlay
            # disk, not the /workspace volume this whole cache targets, so a
            # full build (~130GB of raw shards) would ENOSPC there.
            #
            # keep_in_memory=True is necessary but NOT sufficient (measured):
            # it drops the .map() cache file and detaches the returned dataset
            # from disk, but the parquet *builder* still writes its
            # parquet-train.arrow copy. cache_dir puts that copy on the volume,
            # per shard, so the `finally` below can delete it -- keep_in_memory
            # is what makes that deletion safe, since neither raw_dataset nor
            # resized_dataset reads from it afterwards. A raw shard is ~175MB,
            # comfortably in RAM on a training pod.
            raw_dataset = datasets.Dataset.from_parquet(
                str(raw_tmp_path), keep_in_memory=True, cache_dir=str(arrow_tmp_dir)
            )
            resized_dataset = raw_dataset.map(_resize_row_for_cache, num_proc=num_proc)
            resized_dataset.to_parquet(str(output_tmp_path))
            os.replace(output_tmp_path, output_path)
        finally:
            # Every intermediate, not just the raw bytes: a crash inside .map()
            # or .to_parquet() would otherwise strand a partial <shard>.tmp.
            # After a successful os.replace, output_tmp_path is already gone,
            # which is exactly what missing_ok=True is for.
            raw_tmp_path.unlink(missing_ok=True)
            output_tmp_path.unlink(missing_ok=True)
            shutil.rmtree(arrow_tmp_dir, ignore_errors=True)

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


def ensure_local_cache(config: TrainingConfig) -> None:
    """No-op if config.local_cache_dir is unset. Otherwise wires
    build_local_resize_cache to the real Hub (HfApi.list_repo_files +
    RealHfClient.download_bytes, retried with
    hf_storage.retry.rate_limit_aware_backoff) and runs it against
    config.local_cache_dir. Safe to call on every build_train_dataset/
    build_val_dataset invocation -- already-built shards are skipped, so
    a fully-populated cache makes this a handful of fast filesystem
    existence checks, not a re-download.

    Side effect: when local_cache_dir IS set, this repoints
    huggingface_hub's own download cache onto that same volume (see the
    comment below) -- process-global state, deliberately, because the
    shared RealHfClient offers no per-call cache_dir hook."""
    if not config.local_cache_dir:
        return

    # RealHfClient.download_bytes goes through hf_hub_download, which also
    # persists every downloaded file to ~/.cache/huggingface/hub and never
    # cleans it up -- again the container's small overlay disk on a RunPod pod,
    # not the /workspace volume. Point it at the volume instead. setdefault, so
    # an operator who configured HF_HUB_CACHE themselves wins.
    #
    # The env var alone is NOT enough: huggingface_hub reads it exactly once,
    # when huggingface_hub.constants is imported (verified empirically on
    # huggingface_hub 1.28.0), which already happened at this module's import.
    # hf_hub_download does re-read constants.HF_HUB_CACHE at call time when
    # cache_dir is None, so writing the resolved value through to the constant
    # is what actually takes effect; the env var is set too so subprocesses
    # (DataLoader workers) inherit it. Redirecting rather than eliminating the
    # duplicate copy is deliberate: dropping it entirely would mean threading a
    # cache_dir through the shared hf_storage.client.RealHfClient, which serves
    # other packages -- out of scope here. The practical problem is which disk
    # fills up, and this fixes that.
    os.environ.setdefault(
        "HF_HUB_CACHE", str(Path(config.local_cache_dir) / ".hf_hub_cache")
    )
    hf_constants.HF_HUB_CACHE = os.environ["HF_HUB_CACHE"]

    api = HfApi()
    client = RealHfClient(api, config.dataset_repo_id, repo_type="dataset")

    def _list_shard_paths() -> list[str]:
        # Retried on the same terms as _download_shard below: this runs on
        # every build_train_dataset/build_val_dataset call, including when the
        # cache is fully built (build_local_resize_cache always lists to know
        # what to iterate), so an un-retried transient Hub hiccup or rate limit
        # here would kill the whole training start before a single batch.
        def _fetch() -> list[str]:
            return [
                p
                for p in api.list_repo_files(config.dataset_repo_id, repo_type="dataset")
                if p.startswith("shards/")
            ]

        return retry_with_backoff(
            _fetch,
            max_retries=5,
            base_delay=2.0,
            sleep_func=time.sleep,
            backoff_seconds=rate_limit_aware_backoff(base_delay=2.0, rate_limit_delay=3600.0),
        )

    def _download_shard(path: str) -> bytes:
        def _fetch() -> bytes:
            data = client.download_bytes(path)
            # Not an assert: asserts vanish under `python -O`, and the failure
            # would then surface as a TypeError from write_bytes(None), far
            # from the real cause.
            if data is None:
                raise FileNotFoundError(f"shard listed but missing on Hub: {path}")
            return data

        data = retry_with_backoff(
            _fetch,
            max_retries=5,
            base_delay=2.0,
            sleep_func=time.sleep,
            backoff_seconds=rate_limit_aware_backoff(base_delay=2.0, rate_limit_delay=3600.0),
        )
        # The bytes are in hand, so the cached copy has no further readers --
        # drop it now rather than letting 64.3GB of raw shards accumulate.
        # Deliberately after the retry, not inside _fetch: a retried attempt
        # that failed partway has nothing worth cleaning, and a successful one
        # must not have its blob removed before its bytes are read.
        _discard_hub_blob(config.dataset_repo_id, path, hf_constants.HF_HUB_CACHE)
        return data

    build_local_resize_cache(
        list_shard_paths=_list_shard_paths,
        download_shard=_download_shard,
        local_cache_dir=Path(config.local_cache_dir),
        num_proc=config.resize_cache_num_proc,
    )
