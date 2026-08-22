"""Opt-in integration test for the real Phase B chain.

Run explicitly with a short (~30s), locally-available test clip you have the
rights to use:

    POKEMON_RL_TEST_CLIP=/path/to/clip.mp4 uv run pytest -m slow tests/integration/test_extraction_smoke.py -v

Skipped by default (see the `addopts = -m "not slow"` in pyproject.toml) and
skipped automatically if POKEMON_RL_TEST_CLIP is unset, so this never fails
CI or a fresh checkout.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from data_collection import extract
from data_collection.batcher import FrameBatcher, FrameRecord, batch_to_parquet
from data_collection.dedup import PerceptualHashDeduper

pytestmark = pytest.mark.slow

_CLIP_ENV_VAR = "POKEMON_RL_TEST_CLIP"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
@pytest.mark.skipif(_CLIP_ENV_VAR not in os.environ, reason=f"set {_CLIP_ENV_VAR} to run this")
def test_real_extraction_chain_produces_a_local_parquet_shard(tmp_path: Path) -> None:
    clip_path = os.environ[_CLIP_ENV_VAR]

    # Grab one full frame first to let a human-equivalent crop box be chosen;
    # here we just crop the top-left 160x144 region as a deterministic smoke test.
    crop_x, crop_y, crop_w, crop_h = 0, 0, 160, 144

    frames = list(
        extract.stream_frames(clip_path, crop_x=crop_x, crop_y=crop_y, crop_w=crop_w, crop_h=crop_h, fps=2)
    )
    assert len(frames) > 0

    deduper = PerceptualHashDeduper()
    batcher = FrameBatcher(batch_size=1000)

    kept_count = 0
    for i, frame in enumerate(frames):
        if deduper.is_duplicate(frame):
            continue
        kept_count += 1
        batcher.add(
            FrameRecord(image=frame, video_id="smoke-test", timestamp_s=i / 2.0, game="red")
        )

    batch = batcher.flush()
    assert batch is not None
    assert kept_count > 0

    shard_path = tmp_path / "shard.parquet"
    batch_to_parquet(batch, shard_path)

    assert shard_path.exists()
    assert shard_path.stat().st_size > 0
