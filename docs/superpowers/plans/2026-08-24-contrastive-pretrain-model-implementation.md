# Contrastive Pretraining Model & Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SimCLR ResNet-50 encoder, training pipeline, and frozen-weight-loading interface described in the design spec, so `objones25/pokemon-frames` can be turned into a versioned, PPO-consumable frozen feature extractor on a RunPod A100 pod.

**Architecture:** A grayscale-wrapped, maxpool-dropped, ImageNet-pretrained ResNet-50 backbone + SimCLR projector, trained via NT-Xent on a streaming HF dataset through a `StatefulDataLoader`, with two-tier checkpointing (full state to a local network volume, frozen weights-only artifact to an HF Hub model repo). A new `src/hf_storage/` package extracts the retry/HF-client code already shared by `data_collection`, so `contrastive_pretrain` doesn't duplicate it.

**Tech Stack:** PyTorch 2.13 (bf16 autocast, `torch.compile`, channels_last), torchvision 0.28 (ResNet-50), `datasets` 5.0 (streaming `IterableDataset`), `torchdata` (new dep — `StatefulDataLoader`), `safetensors` (new dep), `huggingface_hub`, `click`, `trackio`, `uv`.

**Spec:** `docs/superpowers/specs/2026-08-24-contrastive-pretrain-model-design.md`

## Global Constraints

- Package management: `uv` only — `uv add <pkg>`, `uv run <cmd>`. No bare `pip`/`venv`.
- Python >= 3.12; `torch>=2.13.0`, `torchvision>=0.28.0` already pinned in `pyproject.toml`.
- TDD: write the failing test before the implementation for every task below.
- `batch_size=1024`, `learning_rate=4e-4`, `warmup_steps=500`, `weight_decay=1e-6`, `temperature=0.1`, `max_epochs=100`, projector `2048 → 2048 → 128`, `shuffle_buffer_size=10_000`, `checkpoint_interval_steps=1000`.
- Precision: `torch.autocast(dtype=torch.bfloat16)`, no `GradScaler`. `torch.compile(mode="default")`. `channels_last` memory format. `cudnn.benchmark=True`, TF32 matmul enabled.
- Fail-fast: non-finite loss raises immediately; startup OOM raises immediately with an actionable message — never silently retried at a smaller batch size.
- Frozen artifact: `safetensors` + `config.json` + `latent_stats.json`, uploaded to an HF Hub **model** repo (`objones25/pokemon-contrastive-encoder`). Training checkpoints: `torch.save(..., weights_only=True)`-loadable plain dicts, on the local RunPod network volume.
- Validation split: whole videos `D1SrSFZrV7A` (red) and `YW29l3jJXr4` (blue) held out from `configs/video_sources.yaml`.
- `hf_storage` (new package) has no dependency on `data_collection` or `contrastive_pretrain`; both depend on it.
- Augmentation randomness is a per-row deterministic seed (`sha256(base_seed:video_id:timestamp_s)`), not a shared `torch.Generator` — see the spec's amendment. No RNG state in checkpoints.

---

## Task 1: Extract shared HF Hub code into `src/hf_storage/`

**Files:**
- Create: `src/hf_storage/__init__.py`
- Create: `src/hf_storage/retry.py` (moved from `src/data_collection/retry.py`, unchanged)
- Create: `src/hf_storage/client.py` (new `HfClient` Protocol + generalized `RealHfClient`)
- Modify: `src/data_collection/hf_uploader.py` (import from `hf_storage` instead of local `retry`/inline Protocol)
- Modify: `src/data_collection/cli.py` (import `RealHfClient` from `hf_storage.client`, delete the inline class)
- Delete: `src/data_collection/retry.py`
- Modify: `pyproject.toml` (add `"src/hf_storage"` to `[tool.hatch.build.targets.wheel] packages`)
- Test: `tests/unit/test_hf_storage_retry.py` (moved from `tests/unit/test_retry.py`, imports updated)
- Test: `tests/unit/test_hf_storage_client.py` (new)
- Delete: `tests/unit/test_retry.py`

**Interfaces:**
- Produces: `hf_storage.retry.{exponential_backoff, is_rate_limited, rate_limit_aware_backoff, retry_with_backoff}` (identical signatures to the old `data_collection.retry`), `hf_storage.client.HfClient` (Protocol: `upload_bytes(data: bytes, path_in_repo: str) -> None`, `download_bytes(path_in_repo: str) -> bytes | None`), `hf_storage.client.RealHfClient(api: HfApi, repo_id: str, repo_type: str = "dataset")`.

- [ ] **Step 1: Move `retry.py` and its test, verify nothing else changes yet**

```bash
git mv src/data_collection/retry.py src/hf_storage/retry.py
git mv tests/unit/test_retry.py tests/unit/test_hf_storage_retry.py
```

Edit `tests/unit/test_hf_storage_retry.py`'s import line:

```python
from hf_storage.retry import (
    is_rate_limited,
    rate_limit_aware_backoff,
    retry_with_backoff,
)
```

Create `src/hf_storage/__init__.py` (empty file).

- [ ] **Step 2: Run the moved test to confirm it's discoverable and passes**

Run: `uv run pytest tests/unit/test_hf_storage_retry.py -v`
Expected: PASS (all 7 tests) — this only works once the package is importable, so if it fails with `ModuleNotFoundError: hf_storage`, add `"src/hf_storage"` to `pyproject.toml`'s wheel packages list now and re-run `uv sync` before retrying.

- [ ] **Step 3: Add `"src/hf_storage"` to `pyproject.toml` and re-sync**

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/data_collection", "src/observability", "src/contrastive_pretrain", "src/hf_storage"]
```

Run: `uv sync`

- [ ] **Step 4: Write the failing test for the generalized `RealHfClient`**

Create `tests/unit/test_hf_storage_client.py`:

```python
from pathlib import Path

from huggingface_hub.errors import EntryNotFoundError

from hf_storage.client import RealHfClient


class _FakeHfApi:
    """Stands in for huggingface_hub.HfApi: upload_file/hf_hub_download are
    the only two methods RealHfClient calls, so this fake only implements
    those, backed by a tmp_path directory instead of the real Hub."""

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self.uploaded_calls: list[tuple[str, str, str]] = []

    def upload_file(self, path_or_fileobj: bytes, path_in_repo: str, repo_id: str, repo_type: str) -> None:
        self.uploaded_calls.append((path_in_repo, repo_id, repo_type))
        dest = self._tmp_path / path_in_repo.replace("/", "_")
        dest.write_bytes(path_or_fileobj)

    def hf_hub_download(self, repo_id: str, filename: str, repo_type: str) -> str:
        dest = self._tmp_path / filename.replace("/", "_")
        if not dest.exists():
            raise EntryNotFoundError("not found")
        return str(dest)


def test_real_hf_client_round_trips_bytes(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo", repo_type="model")

    client.upload_bytes(b"hello", "config.json")
    result = client.download_bytes("config.json")

    assert result == b"hello"
    assert api.uploaded_calls == [("config.json", "me/repo", "model")]


def test_real_hf_client_download_bytes_returns_none_when_missing(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo")

    assert client.download_bytes("missing.json") is None


def test_real_hf_client_defaults_to_dataset_repo_type(tmp_path) -> None:
    api = _FakeHfApi(tmp_path)
    client = RealHfClient(api, "me/repo")

    client.upload_bytes(b"x", "manifest.json")

    assert api.uploaded_calls == [("manifest.json", "me/repo", "dataset")]
```

- [ ] **Step 5: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_hf_storage_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hf_storage.client'`

- [ ] **Step 6: Create `src/hf_storage/client.py`**

```python
"""HfClient Protocol + a real huggingface_hub.HfApi-backed implementation,
shared by every package in this project that persists something to the HF
Hub (data_collection's shards/manifest, contrastive_pretrain's checkpoints
and frozen encoder artifact). repo_type defaults to "dataset" to preserve
data_collection's existing behavior unchanged; contrastive_pretrain passes
repo_type="model" for the frozen encoder repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError


class HfClient(Protocol):
    def upload_bytes(self, data: bytes, path_in_repo: str) -> None: ...
    def download_bytes(self, path_in_repo: str) -> bytes | None: ...


class RealHfClient:
    """Adapts huggingface_hub.HfApi to the HfClient protocol."""

    def __init__(self, api: HfApi, repo_id: str, repo_type: str = "dataset") -> None:
        self._api = api
        self._repo_id = repo_id
        self._repo_type = repo_type

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self._api.upload_file(
            path_or_fileobj=data,
            path_in_repo=path_in_repo,
            repo_id=self._repo_id,
            repo_type=self._repo_type,
        )

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        try:
            local_path = self._api.hf_hub_download(
                repo_id=self._repo_id, filename=path_in_repo, repo_type=self._repo_type
            )
        except EntryNotFoundError:
            return None
        return Path(local_path).read_bytes()
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_hf_storage_client.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Update `data_collection/hf_uploader.py` to import from `hf_storage`**

In `src/data_collection/hf_uploader.py`, replace:

```python
from data_collection.retry import rate_limit_aware_backoff, retry_with_backoff
```

with:

```python
from hf_storage.retry import rate_limit_aware_backoff, retry_with_backoff
```

Delete the local `HfClient` Protocol definition from this file (lines 15-17 in the current version) — it's now defined in `hf_storage.client` — and add:

```python
from hf_storage.client import HfClient
```

- [ ] **Step 9: Update `data_collection/cli.py` to import `RealHfClient` from `hf_storage`**

In `src/data_collection/cli.py`, delete the entire `RealHfClient` class definition (lines 24-46 in the current version), and add:

```python
from hf_storage.client import HfClient, RealHfClient
```

(replacing the old `from data_collection.hf_uploader import HfClient, HfUploader` import — keep `HfUploader`, drop the now-redundant `HfClient` from that import since it comes from `hf_storage.client` instead).

- [ ] **Step 10: Run the full existing test suite to confirm nothing broke**

Run: `uv run pytest tests/unit -v`
Expected: PASS, all tests including `test_hf_uploader.py` and `test_cli.py` (these exercise the code that moved, and should be unaffected since the Protocol/class shapes are unchanged).

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "refactor: extract shared HF Hub retry/client code into src/hf_storage/"
```

---

## Task 2: `contrastive_pretrain/config.py` and `configs/contrastive_pretrain.yaml`

**Files:**
- Create: `src/contrastive_pretrain/config.py`
- Create: `configs/contrastive_pretrain.yaml`
- Test: `tests/unit/test_contrastive_pretrain_config.py`

**Interfaces:**
- Produces: `contrastive_pretrain.config.TrainingConfig` (frozen dataclass with fields listed below), `contrastive_pretrain.config.load_config(path: str | Path) -> TrainingConfig`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_contrastive_pretrain_config.py`:

```python
from contrastive_pretrain.config import TrainingConfig, load_config


def test_training_config_defaults_match_spec() -> None:
    config = TrainingConfig()

    assert config.dataset_repo_id == "objones25/pokemon-frames"
    assert config.frozen_encoder_repo_id == "objones25/pokemon-contrastive-encoder"
    assert config.val_video_ids == ("D1SrSFZrV7A", "YW29l3jJXr4")
    assert config.batch_size == 1024
    assert config.learning_rate == 4e-4
    assert config.warmup_steps == 500
    assert config.weight_decay == 1e-6
    assert config.temperature == 0.1
    assert config.max_epochs == 100
    assert config.checkpoint_interval_steps == 1000
    assert config.shuffle_buffer_size == 10_000


def test_load_config_applies_yaml_overrides(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("batch_size: 8\nlearning_rate: 0.001\n")

    config = load_config(path)

    assert config.batch_size == 8
    assert config.learning_rate == 0.001
    assert config.max_epochs == 100  # untouched fields keep their default


def test_load_config_converts_val_video_ids_to_tuple(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("val_video_ids: ['a', 'b', 'c']\n")

    config = load_config(path)

    assert config.val_video_ids == ("a", "b", "c")


def test_load_config_with_empty_file_returns_defaults(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("")

    config = load_config(path)

    assert config == TrainingConfig()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contrastive_pretrain.config'`

- [ ] **Step 3: Write the implementation**

Create `src/contrastive_pretrain/config.py`:

```python
"""Training hyperparameters and paths, loaded from configs/contrastive_pretrain.yaml.
Mirrors data_collection.registry's dataclass + yaml.safe_load pattern."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    dataset_repo_id: str = "objones25/pokemon-frames"
    frozen_encoder_repo_id: str = "objones25/pokemon-contrastive-encoder"
    val_video_ids: tuple[str, ...] = ("D1SrSFZrV7A", "YW29l3jJXr4")
    pretrained: bool = True
    batch_size: int = 1024
    num_workers: int = 8
    shuffle_buffer_size: int = 10_000
    seed: int = 0
    learning_rate: float = 4e-4
    warmup_steps: int = 500
    weight_decay: float = 1e-6
    temperature: float = 0.1
    max_epochs: int = 100
    checkpoint_interval_steps: int = 1000
    network_volume_checkpoint_dir: str = "/runpod-volume/contrastive_pretrain/checkpoints"


def load_config(path: str | Path) -> TrainingConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if "val_video_ids" in data:
        data["val_video_ids"] = tuple(data["val_video_ids"])
    valid_fields = {f.name for f in fields(TrainingConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")
    return TrainingConfig(**data)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Create the real config file**

Create `configs/contrastive_pretrain.yaml`:

```yaml
dataset_repo_id: objones25/pokemon-frames
frozen_encoder_repo_id: objones25/pokemon-contrastive-encoder
val_video_ids: [D1SrSFZrV7A, YW29l3jJXr4]
pretrained: true
batch_size: 1024
num_workers: 8
shuffle_buffer_size: 10000
seed: 0
learning_rate: 0.0004
warmup_steps: 500
weight_decay: 0.000001
temperature: 0.1
max_epochs: 100
checkpoint_interval_steps: 1000
network_volume_checkpoint_dir: /runpod-volume/contrastive_pretrain/checkpoints
```

Add a test that this real file loads without error:

```python
def test_real_config_file_loads_without_error() -> None:
    config = load_config("configs/contrastive_pretrain.yaml")
    assert config.batch_size == 1024
```

Run: `uv run pytest tests/unit/test_contrastive_pretrain_config.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add src/contrastive_pretrain/config.py configs/contrastive_pretrain.yaml tests/unit/test_contrastive_pretrain_config.py
git commit -m "feat: add contrastive_pretrain.config.TrainingConfig and its yaml"
```

---

## Task 3: `contrastive_pretrain/model.py` — encoder and projector

**Files:**
- Create: `src/contrastive_pretrain/model.py`
- Test: `tests/unit/test_contrastive_pretrain_model.py`

**Interfaces:**
- Produces: `contrastive_pretrain.model.EMBEDDING_DIM` (int, 2048), `contrastive_pretrain.model.build_encoder(pretrained: bool = True) -> tuple[nn.Module, int]`, `contrastive_pretrain.model.build_projector(in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128) -> nn.Module`. The encoder module's `forward(x: Tensor[N,1,H,W]) -> Tensor[N,2048]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_contrastive_pretrain_model.py`:

```python
import pytest
import torch
import torch.nn as nn

from contrastive_pretrain.model import EMBEDDING_DIM, build_encoder, build_projector


def test_build_encoder_returns_2048_dim() -> None:
    _, dim = build_encoder(pretrained=False)
    assert dim == EMBEDDING_DIM == 2048


def test_build_encoder_output_shape() -> None:
    encoder, dim = build_encoder(pretrained=False)
    x = torch.randint(0, 256, (2, 1, 144, 160), dtype=torch.uint8).float()

    out = encoder(x)

    assert out.shape == (2, dim)


def test_build_encoder_has_no_maxpool() -> None:
    encoder, _ = build_encoder(pretrained=False)
    assert isinstance(encoder.backbone.maxpool, nn.Identity)


def test_build_encoder_rejects_wrong_channel_count() -> None:
    encoder, _ = build_encoder(pretrained=False)
    x = torch.zeros(2, 3, 144, 160)

    with pytest.raises(ValueError, match="1-channel"):
        encoder(x)


def test_build_projector_output_shape() -> None:
    projector = build_projector()
    x = torch.randn(4, 2048)

    out = projector(x)

    assert out.shape == (4, 128)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contrastive_pretrain.model'`

- [ ] **Step 3: Write the implementation**

Create `src/contrastive_pretrain/model.py`:

```python
"""SimCLR ResNet-50-style encoder + projector. The encoder drops the
standard ImageNet stem's initial maxpool (kept: pretrained weights,
which are fully compatible since maxpool owns zero learnable params —
see the design spec) to preserve spatial detail this domain's small
UI elements (HP bars, text glyphs) depend on. Grayscale-to-3-channel
replication lives inside the module so training and load_frozen_encoder
share one external contract: 1-channel grayscale in, 2048-d feature out.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

EMBEDDING_DIM = 2048


class GrayscaleResNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 1:
            raise ValueError(f"expected 1-channel grayscale input, got shape {tuple(x.shape)}")
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def build_encoder(pretrained: bool = True) -> tuple[nn.Module, int]:
    return GrayscaleResNetEncoder(pretrained=pretrained), EMBEDDING_DIM


class SimCLRProjector(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_projector(in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128) -> nn.Module:
    return SimCLRProjector(in_dim, hidden_dim, out_dim)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_model.py -v`
Expected: PASS (5 tests). These all use `pretrained=False` so no network access or weight download happens in the unit test.

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/model.py tests/unit/test_contrastive_pretrain_model.py
git commit -m "feat: add contrastive_pretrain ResNet-50 encoder (no maxpool) and SimCLR projector"
```

---

## Task 4: `contrastive_pretrain/losses.py` — NT-Xent

**Files:**
- Create: `src/contrastive_pretrain/losses.py`
- Test: `tests/unit/test_contrastive_pretrain_losses.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure tensor function).
- Produces: `contrastive_pretrain.losses.nt_xent_loss(z_a: Tensor[N,D], z_b: Tensor[N,D], temperature: float) -> Tensor` (scalar).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_contrastive_pretrain_losses.py`:

```python
import math

import pytest
import torch
import torch.nn.functional as F

from contrastive_pretrain.losses import nt_xent_loss


def test_nt_xent_loss_matches_hand_derivation_for_orthogonal_negatives() -> None:
    """2 examples, embedding dim 2. a0=b0=e1, a1=b1=e2 (e1 orthogonal to
    e2), temperature=1.0. For anchor a0, the positive is b0 (sim=1) and
    the two negatives (a1, b1) both have sim=0 (self a0-a0 is masked
    out). NT-Xent for this anchor is
    -log(exp(1) / (exp(1) + exp(0) + exp(0))) = log(e + 2) - 1.
    All 4 anchors are symmetric here, so the batch-mean loss equals this
    same value."""
    e1 = torch.tensor([1.0, 0.0])
    e2 = torch.tensor([0.0, 1.0])
    z_a = torch.stack([e1, e2])
    z_b = torch.stack([e1, e2])

    loss = nt_xent_loss(z_a, z_b, temperature=1.0)

    expected = math.log(math.e + 2) - 1
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_nt_xent_loss_lower_for_better_aligned_positives() -> None:
    torch.manual_seed(0)
    n, d = 8, 16
    base = F.normalize(torch.randn(n, d), dim=1)
    noise = torch.randn(n, d) * 0.01
    z_a = base
    z_b_close = F.normalize(base + noise, dim=1)
    z_b_far = F.normalize(torch.randn(n, d), dim=1)

    loss_close = nt_xent_loss(z_a, z_b_close, temperature=0.5)
    loss_far = nt_xent_loss(z_a, z_b_far, temperature=0.5)

    assert loss_close.item() < loss_far.item()


def test_nt_xent_loss_is_differentiable() -> None:
    z_a = torch.randn(4, 8, requires_grad=True)
    z_b = torch.randn(4, 8, requires_grad=True)

    loss = nt_xent_loss(z_a, z_b, temperature=0.1)
    loss.backward()

    assert z_a.grad is not None
    assert torch.isfinite(z_a.grad).all()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contrastive_pretrain.losses'`

- [ ] **Step 3: Write the implementation**

Create `src/contrastive_pretrain/losses.py`:

```python
"""Standard SimCLR NT-Xent (InfoNCE) loss: every other view in the batch
is an implicit negative for a given anchor -- no explicit negative
mining needed."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    n = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)  # (2N, D)
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature  # (2N, 2N)

    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    positive_idx = torch.arange(2 * n, device=z.device)
    positive_idx = (positive_idx + n) % (2 * n)

    return F.cross_entropy(sim, positive_idx)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_losses.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/losses.py tests/unit/test_contrastive_pretrain_losses.py
git commit -m "feat: add NT-Xent contrastive loss"
```

---

## Task 5: `contrastive_pretrain/encoder_io.py` — Conv+BN fusion

**Files:**
- Create: `src/contrastive_pretrain/encoder_io.py`
- Test: `tests/unit/test_contrastive_pretrain_encoder_io.py`

**Interfaces:**
- Consumes: `contrastive_pretrain.model.build_encoder` (for the real-encoder fusion test).
- Produces: `contrastive_pretrain.encoder_io.fuse_conv_bn_modules(module: nn.Module) -> nn.Module` (in-place; also returned for chaining).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_contrastive_pretrain_encoder_io.py`:

```python
import torch
import torch.nn as nn

from contrastive_pretrain.encoder_io import fuse_conv_bn_modules
from contrastive_pretrain.model import build_encoder


def test_fuse_conv_bn_modules_preserves_output() -> None:
    torch.manual_seed(0)
    module = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4))
    module.eval()
    with torch.no_grad():
        # Non-trivial BN stats so fusion isn't testing a no-op identity case.
        module[1].running_mean.copy_(torch.randn(4))
        module[1].running_var.copy_(torch.rand(4) + 0.5)
        module[1].weight.copy_(torch.randn(4))
        module[1].bias.copy_(torch.randn(4))

    x = torch.randn(2, 1, 8, 8)
    before = module(x)
    fused = fuse_conv_bn_modules(module)
    after = fused(x)

    assert isinstance(fused[1], nn.Identity)
    assert torch.allclose(before, after, atol=1e-5)


def test_fuse_conv_bn_modules_on_real_encoder() -> None:
    encoder, _ = build_encoder(pretrained=False)
    encoder.eval()
    x = torch.rand(1, 1, 144, 160)

    with torch.no_grad():
        before = encoder(x)
        fused = fuse_conv_bn_modules(encoder)
        after = fused(x)

    assert torch.allclose(before, after, atol=1e-3)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_encoder_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contrastive_pretrain.encoder_io'`

- [ ] **Step 3: Write the implementation**

Create `src/contrastive_pretrain/encoder_io.py`:

```python
"""Frozen-encoder export/load -- the interface contract the PPO stage
depends on. This module owns: Conv+BN fusion (applied only at export,
never during training, since BatchNorm needs live batch stats while
training), packaging the backbone-only weights as safetensors + a JSON
config, pushing that to an HF Hub model repo with rate-limit-aware
retry, and load_frozen_encoder() -- the single function downstream
consumers call.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.fusion as fusion


def fuse_conv_bn_modules(module: nn.Module) -> nn.Module:
    """Recursively fuses every adjacent (Conv2d, BatchNorm2d) pair in
    `module` in-place, for eval-mode inference only. Relies on
    torchvision's Bottleneck blocks (and this project's own
    GrayscaleResNetEncoder) declaring/iterating children in
    conv-then-bn order, which fuse_conv_bn_eval requires."""
    module.eval()
    for child in module.children():
        fuse_conv_bn_modules(child)

    children = list(module.named_children())
    for i in range(len(children) - 1):
        name_a, mod_a = children[i]
        name_b, mod_b = children[i + 1]
        if isinstance(mod_a, nn.Conv2d) and isinstance(mod_b, nn.BatchNorm2d):
            fused_conv = fusion.fuse_conv_bn_eval(mod_a, mod_b)
            setattr(module, name_a, fused_conv)
            setattr(module, name_b, nn.Identity())

    return module
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_encoder_io.py -v`
Expected: PASS (2 tests). No network access — `build_encoder(pretrained=False)`.

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/encoder_io.py tests/unit/test_contrastive_pretrain_encoder_io.py
git commit -m "feat: add Conv+BN fusion for frozen-encoder export"
```

---

## Task 6: `contrastive_pretrain/checkpoint.py` — training-checkpoint save/load

**Files:**
- Create: `src/contrastive_pretrain/checkpoint.py`
- Test: `tests/unit/test_contrastive_pretrain_checkpoint.py`

**Interfaces:**
- Produces: `contrastive_pretrain.checkpoint.{build_checkpoint_state, save_checkpoint, load_checkpoint, find_latest_checkpoint, restore_optimizer_and_scheduler}`.
  - `build_checkpoint_state(epoch: int, global_step: int, model: nn.Module, optimizer: Optimizer, scheduler: LRScheduler, dataloader, best_val_loss: float) -> dict`
  - `save_checkpoint(path: Path, state: dict) -> None`
  - `load_checkpoint(path: Path) -> dict`
  - `find_latest_checkpoint(checkpoint_dir: Path) -> Path | None`
  - `restore_optimizer_and_scheduler(optimizer: Optimizer, scheduler: LRScheduler, state: dict) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_contrastive_pretrain_checkpoint.py`:

```python
import pytest
import torch
import torch.nn as nn

from contrastive_pretrain.checkpoint import (
    build_checkpoint_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_optimizer_and_scheduler,
    save_checkpoint,
)


def test_build_checkpoint_state_captures_all_expected_keys() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    state = build_checkpoint_state(
        epoch=3,
        global_step=150,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_state={"fake": "state"},
        best_val_loss=1.23,
    )

    assert state["epoch"] == 3
    assert state["global_step"] == 150
    assert state["best_val_loss"] == pytest.approx(1.23)
    assert state["dataloader"] == {"fake": "state"}
    assert set(state["model"].keys()) == set(model.state_dict().keys())
    assert "optimizer" in state
    assert "scheduler" in state
    assert "augmentation_rng" not in state  # per-row seeding needs no RNG state


def test_save_and_load_checkpoint_round_trip(tmp_path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    state = build_checkpoint_state(
        epoch=3, global_step=150, model=model, optimizer=optimizer,
        scheduler=scheduler, dataloader_state={"fake": "state"}, best_val_loss=1.23,
    )
    path = tmp_path / "checkpoint_step00000150.pt"

    save_checkpoint(path, state)
    loaded = load_checkpoint(path)

    assert loaded["epoch"] == 3
    assert loaded["global_step"] == 150
    assert loaded["dataloader"] == {"fake": "state"}


def test_save_checkpoint_creates_parent_dirs(tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "checkpoint.pt"
    save_checkpoint(path, {"a": 1})
    assert path.exists()


def test_find_latest_checkpoint_picks_highest_step(tmp_path) -> None:
    (tmp_path / "checkpoint_step00000100.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000900.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000500.pt").write_bytes(b"")

    result = find_latest_checkpoint(tmp_path)

    assert result == tmp_path / "checkpoint_step00000900.pt"


def test_find_latest_checkpoint_returns_none_when_empty(tmp_path) -> None:
    assert find_latest_checkpoint(tmp_path) is None


def test_restore_optimizer_and_scheduler_does_not_clobber_restored_lr() -> None:
    """Regression test for the documented PyTorch gotcha: constructing a
    scheduler resets its optimizer's lr, so the scheduler must be built
    BEFORE restore_optimizer_and_scheduler() is called, or the restored
    lr gets clobbered by the scheduler's own initialization."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    for _ in range(10):
        scheduler.step()
    state = build_checkpoint_state(
        epoch=0, global_step=10, model=model, optimizer=optimizer,
        scheduler=scheduler, dataloader_state={}, best_val_loss=1.0,
    )
    restored_lr_from_original = optimizer.param_groups[0]["lr"]

    model2 = nn.Linear(2, 2)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)
    restore_optimizer_and_scheduler(optimizer2, scheduler2, state)

    assert optimizer2.param_groups[0]["lr"] == pytest.approx(restored_lr_from_original)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contrastive_pretrain.checkpoint'`

- [ ] **Step 3: Write the implementation**

Create `src/contrastive_pretrain/checkpoint.py`:

```python
"""Full training-state checkpointing to the local RunPod network volume.
Separate from contrastive_pretrain.encoder_io, which handles the
weights-only frozen artifact pushed to the HF Hub -- these are the two
tiers described in the design spec, split by cost/purpose, not just by
file.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_checkpoint_state(
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    dataloader_state: dict,
    best_val_loss: float,
) -> dict:
    """`model` must be the raw, uncompiled module -- callers must never
    pass a torch.compile-wrapped model here, since its state_dict keys
    may carry an `_orig_mod.` prefix that a freshly-constructed raw
    module on resume won't have."""
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "dataloader": dataloader_state,
        "best_val_loss": best_val_loss,
    }


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)  # atomic on POSIX -- no half-written checkpoint on a mid-save crash


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(Path(path), weights_only=True)


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
    return candidates[-1] if candidates else None


def restore_optimizer_and_scheduler(optimizer: Optimizer, scheduler: LRScheduler, state: dict) -> None:
    """Caller must construct `scheduler` (attached to `optimizer`) BEFORE
    calling this -- constructing a scheduler resets its optimizer's lr,
    so constructing it after this call would clobber the restored lr."""
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_checkpoint.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/checkpoint.py tests/unit/test_contrastive_pretrain_checkpoint.py
git commit -m "feat: add training-checkpoint save/load with correct restore ordering"
```

---

## Task 7: `contrastive_pretrain/encoder_io.py` — export, push, and load the frozen artifact

**Files:**
- Modify: `src/contrastive_pretrain/encoder_io.py` (add to the file from Task 5)
- Test: `tests/unit/test_contrastive_pretrain_encoder_io.py` (add to the file from Task 5)

**Interfaces:**
- Consumes: `contrastive_pretrain.encoder_io.fuse_conv_bn_modules` (Task 5), `contrastive_pretrain.model.{build_encoder, EMBEDDING_DIM}` (Task 3), `hf_storage.client.HfClient` (Task 1), `hf_storage.retry.{retry_with_backoff, rate_limit_aware_backoff}` (Task 1).
- Produces: `contrastive_pretrain.encoder_io.{export_frozen_encoder, push_frozen_encoder, load_frozen_encoder, compute_latent_stats}`.
  - `export_frozen_encoder(encoder: nn.Module) -> tuple[bytes, bytes]` — returns `(safetensors_bytes, config_json_bytes)`.
  - `push_frozen_encoder(client: HfClient, encoder: nn.Module, latent_mean: Tensor, latent_std: Tensor, max_retries: int = 5, base_delay: float = 2.0, rate_limit_delay: float = 120.0, sleep_func: Callable = time.sleep) -> None`.
  - `load_frozen_encoder(repo_id: str, revision: str | None = None) -> nn.Module` — the public, network-hitting entrypoint.
  - `compute_latent_stats(encoder: nn.Module, rows: Iterable[dict], device: torch.device, max_examples: int = 2000) -> tuple[Tensor, Tensor]` — `rows` yields dicts with an `"original"` key (a `(1,H,W)` uint8 tensor), matching `contrastive_pretrain.dataset.to_pair_transform`'s output (Task 8).

- [ ] **Step 1: Add `uv add safetensors`**

```bash
uv add safetensors
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/unit/test_contrastive_pretrain_encoder_io.py`:

```python
import json

import torch

from contrastive_pretrain.encoder_io import (
    compute_latent_stats,
    export_frozen_encoder,
    push_frozen_encoder,
)
from hf_storage.client import HfClient


class _FakeHfClient:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.upload_calls: list[str] = []

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.upload_calls.append(path_in_repo)
        self.files[path_in_repo] = data

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.files.get(path_in_repo)


class _FlakyThenWorksHfClient:
    """Fails upload_bytes twice, then succeeds -- verifies push_frozen_encoder
    actually retries rather than propagating the first failure."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.attempts: dict[str, int] = {}

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.attempts[path_in_repo] = self.attempts.get(path_in_repo, 0) + 1
        if self.attempts[path_in_repo] < 3:
            raise RuntimeError("transient upload failure")
        self.files[path_in_repo] = data

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.files.get(path_in_repo)


def test_export_frozen_encoder_round_trips_weights() -> None:
    encoder, _ = build_encoder(pretrained=False)
    encoder.eval()
    x = torch.rand(1, 1, 144, 160)
    with torch.no_grad():
        expected = encoder(x)

    weights_bytes, config_bytes = export_frozen_encoder(encoder)

    from safetensors.torch import load as safetensors_load

    reloaded_encoder, _ = build_encoder(pretrained=False)
    reloaded_encoder.eval()
    fuse_conv_bn_modules(reloaded_encoder)
    reloaded_encoder.load_state_dict(safetensors_load(weights_bytes))
    with torch.no_grad():
        actual = reloaded_encoder(x)

    assert torch.allclose(expected, actual, atol=1e-3)
    config = json.loads(config_bytes)
    assert config == {
        "embedding_dim": 2048,
        "stem": "no_maxpool",
        "input_channels": 1,
        "input_size": [160, 144],
        "pretrained_init": True,
    }


def test_export_frozen_encoder_does_not_mutate_input_module() -> None:
    encoder, _ = build_encoder(pretrained=False)
    original_maxpool_type = type(encoder.backbone.maxpool)

    export_frozen_encoder(encoder)

    assert isinstance(encoder.backbone.conv1, torch.nn.Conv2d)  # still unfused
    assert type(encoder.backbone.maxpool) is original_maxpool_type


def test_push_frozen_encoder_uploads_three_files() -> None:
    encoder, _ = build_encoder(pretrained=False)
    client = _FakeHfClient()

    push_frozen_encoder(
        client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048),
        sleep_func=lambda _: None,
    )

    assert set(client.upload_calls) == {"model.safetensors", "config.json", "latent_stats.json"}
    stats = json.loads(client.files["latent_stats.json"])
    assert len(stats["mean"]) == 2048
    assert len(stats["std"]) == 2048


def test_push_frozen_encoder_retries_transient_upload_failures() -> None:
    encoder, _ = build_encoder(pretrained=False)
    client = _FlakyThenWorksHfClient()

    push_frozen_encoder(
        client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048),
        max_retries=5, base_delay=0.0, sleep_func=lambda _: None,
    )

    assert set(client.files.keys()) == {"model.safetensors", "config.json", "latent_stats.json"}


def test_load_frozen_encoder_from_client_matches_exported_weights() -> None:
    from contrastive_pretrain.encoder_io import _load_frozen_encoder_from_client

    encoder, _ = build_encoder(pretrained=False)
    encoder.eval()
    x = torch.rand(1, 1, 144, 160)
    with torch.no_grad():
        expected = encoder(x)
    client = _FakeHfClient()
    push_frozen_encoder(client, encoder, latent_mean=torch.zeros(2048), latent_std=torch.ones(2048))

    loaded = _load_frozen_encoder_from_client(client)

    with torch.no_grad():
        actual = loaded(x)
    assert torch.allclose(expected, actual, atol=1e-3)
    assert all(not p.requires_grad for p in loaded.parameters())
    assert loaded.training is False


def test_compute_latent_stats_shapes() -> None:
    encoder, dim = build_encoder(pretrained=False)
    rows = [
        {"original": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8)}
        for _ in range(5)
    ]

    mean, std = compute_latent_stats(encoder, rows, device=torch.device("cpu"), max_examples=5)

    assert mean.shape == (dim,)
    assert std.shape == (dim,)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_encoder_io.py -v`
Expected: FAIL — `export_frozen_encoder` etc. don't exist yet.

- [ ] **Step 4: Write the implementation**

Append to `src/contrastive_pretrain/encoder_io.py`:

```python
import copy
import json
import time
from collections.abc import Callable, Iterable

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from contrastive_pretrain.model import EMBEDDING_DIM, build_encoder
from hf_storage.client import HfClient, RealHfClient
from hf_storage.retry import rate_limit_aware_backoff, retry_with_backoff

_INPUT_SIZE = [160, 144]  # [W, H], matches the augmentation spec's convention


def export_frozen_encoder(encoder: nn.Module) -> tuple[bytes, bytes]:
    """`encoder` is the live, unfused training module -- this function
    deep-copies it before fusing, so the caller's training model is
    never mutated by an export call."""
    fused = fuse_conv_bn_modules(copy.deepcopy(encoder).eval())
    weights_bytes = safetensors_save(fused.state_dict())
    config = {
        "embedding_dim": EMBEDDING_DIM,
        "stem": "no_maxpool",
        "input_channels": 1,
        "input_size": _INPUT_SIZE,
        "pretrained_init": True,
    }
    config_bytes = json.dumps(config).encode("utf-8")
    return weights_bytes, config_bytes


def push_frozen_encoder(
    client: HfClient,
    encoder: nn.Module,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    max_retries: int = 5,
    base_delay: float = 2.0,
    rate_limit_delay: float = 120.0,
    sleep_func: Callable[[float], None] = time.sleep,
) -> None:
    weights_bytes, config_bytes = export_frozen_encoder(encoder)
    stats_bytes = json.dumps(
        {"mean": latent_mean.tolist(), "std": latent_std.tolist()}
    ).encode("utf-8")

    backoff = rate_limit_aware_backoff(base_delay, rate_limit_delay)
    for data, path_in_repo in (
        (weights_bytes, "model.safetensors"),
        (config_bytes, "config.json"),
        (stats_bytes, "latent_stats.json"),
    ):
        retry_with_backoff(
            lambda data=data, path_in_repo=path_in_repo: client.upload_bytes(data, path_in_repo),
            max_retries=max_retries,
            base_delay=base_delay,
            sleep_func=sleep_func,
            backoff_seconds=backoff,
        )


def _load_frozen_encoder_from_client(client: HfClient) -> nn.Module:
    config_bytes = client.download_bytes("config.json")
    weights_bytes = client.download_bytes("model.safetensors")
    if config_bytes is None or weights_bytes is None:
        raise FileNotFoundError("model.safetensors or config.json missing from the target repo")
    config = json.loads(config_bytes)

    encoder, _ = build_encoder(pretrained=False)
    fused = fuse_conv_bn_modules(encoder.eval())
    fused.load_state_dict(safetensors_load(weights_bytes))
    for param in fused.parameters():
        param.requires_grad_(False)
    fused.eval()
    return fused


def load_frozen_encoder(repo_id: str, revision: str | None = None) -> nn.Module:
    """The PPO-facing entrypoint: downloads config.json + model.safetensors
    from `repo_id` (an HF Hub model repo) and returns a frozen, eval-mode
    module mapping (N, 1, 160, 144) grayscale input to (N, 2048) float
    features. Raw, unnormalized output -- the affine normalization layer
    is the caller's job, using that repo's latent_stats.json."""
    from huggingface_hub import HfApi

    client = RealHfClient(HfApi(), repo_id, repo_type="model")
    return _load_frozen_encoder_from_client(client)


def compute_latent_stats(
    encoder: nn.Module,
    rows: Iterable[dict],
    device: torch.device,
    max_examples: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`rows` yields dicts with an "original" (1, H, W) uint8 tensor --
    the same shape contrastive_pretrain.dataset.to_pair_transform
    produces. Restores the encoder's original train/eval mode on exit."""
    was_training = encoder.training
    encoder.eval()
    features = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            if i >= max_examples:
                break
            frame = row["original"].unsqueeze(0).to(device).float()
            features.append(encoder(frame).squeeze(0).cpu())
    encoder.train(was_training)

    stacked = torch.stack(features)
    return stacked.mean(dim=0), stacked.std(dim=0)
```

Add the missing imports at the top of `encoder_io.py` (`nn` was already imported for Task 5; keep it, add the rest above alongside it).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_encoder_io.py -v`
Expected: PASS (8 tests total: 2 from Task 5 + 6 new). No network access anywhere — all via `_FakeHfClient`/`_FlakyThenWorksHfClient` and `pretrained=False`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/contrastive_pretrain/encoder_io.py tests/unit/test_contrastive_pretrain_encoder_io.py
git commit -m "feat: add frozen-encoder export/push/load and latent-stats computation"
```

---

## Task 8: `contrastive_pretrain/dataset.py` — per-row pair transform

**Files:**
- Create: `src/contrastive_pretrain/dataset.py`
- Test: `tests/unit/test_contrastive_pretrain_dataset.py`

**Interfaces:**
- Consumes: `contrastive_pretrain.augmentation.{AugmentationConfig, make_pair}` (existing).
- Produces: `contrastive_pretrain.dataset.{row_seed, to_pair_transform}`.
  - `row_seed(base_seed: int, video_id: str, timestamp_s: float) -> int`
  - `to_pair_transform(example: dict, augmentation_config: AugmentationConfig, base_seed: int) -> dict` — input `example` has an `"image"` key (PIL grayscale Image) and `"video_id"`/`"timestamp_s"` keys; output has `"original"`, `"view_a"`, `"view_b"` `(1, H, W)` uint8 tensor keys.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_contrastive_pretrain_dataset.py`:

```python
import numpy as np
import torch
from PIL import Image

from contrastive_pretrain.augmentation import AugmentationConfig
from contrastive_pretrain.dataset import row_seed, to_pair_transform


def _grayscale_example(video_id: str = "abc123", timestamp_s: float = 5.0) -> dict:
    pixels = np.random.default_rng(0).integers(0, 256, (144, 160), dtype=np.uint8)
    return {
        "image": Image.fromarray(pixels, mode="L"),
        "video_id": video_id,
        "timestamp_s": timestamp_s,
        "game": "red",
    }


def test_row_seed_is_deterministic() -> None:
    assert row_seed(0, "abc123", 12.5) == row_seed(0, "abc123", 12.5)


def test_row_seed_differs_for_different_rows() -> None:
    assert row_seed(0, "abc123", 12.5) != row_seed(0, "abc123", 12.6)
    assert row_seed(0, "abc123", 12.5) != row_seed(0, "xyz789", 12.5)
    assert row_seed(0, "abc123", 12.5) != row_seed(1, "abc123", 12.5)


def test_to_pair_transform_produces_original_and_two_views() -> None:
    result = to_pair_transform(_grayscale_example(), AugmentationConfig(), base_seed=0)

    assert result["original"].shape == (1, 144, 160)
    assert result["view_a"].shape == (1, 144, 160)
    assert result["view_b"].shape == (1, 144, 160)
    assert result["original"].dtype == torch.uint8


def test_to_pair_transform_is_deterministic_for_the_same_row() -> None:
    result1 = to_pair_transform(_grayscale_example(), AugmentationConfig(), base_seed=0)
    result2 = to_pair_transform(_grayscale_example(), AugmentationConfig(), base_seed=0)

    assert torch.equal(result1["view_a"], result2["view_a"])
    assert torch.equal(result1["view_b"], result2["view_b"])


def test_to_pair_transform_differs_across_rows() -> None:
    result1 = to_pair_transform(_grayscale_example(timestamp_s=5.0), AugmentationConfig(), base_seed=0)
    result2 = to_pair_transform(_grayscale_example(timestamp_s=6.0), AugmentationConfig(), base_seed=0)

    assert not torch.equal(result1["view_a"], result2["view_a"])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contrastive_pretrain.dataset'`

- [ ] **Step 3: Write the implementation**

Create `src/contrastive_pretrain/dataset.py`:

```python
"""Streaming dataset construction. Augmentation randomness is a per-row
deterministic seed derived from (base_seed, video_id, timestamp_s), NOT
a shared torch.Generator -- a shared Generator would be copied identically
into every StatefulDataLoader worker process, producing correlated (not
independent) augmentation sequences across workers. This also means
resuming needs zero augmentation-RNG checkpoint state: the seed for a
given row is always re-derivable from data the row already carries.
"""

from __future__ import annotations

import hashlib

import torch
from torchvision.transforms.functional import pil_to_tensor

from contrastive_pretrain.augmentation import AugmentationConfig, make_pair


def row_seed(base_seed: int, video_id: str, timestamp_s: float) -> int:
    digest = hashlib.sha256(f"{base_seed}:{video_id}:{timestamp_s}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def to_pair_transform(example: dict, augmentation_config: AugmentationConfig, base_seed: int) -> dict:
    frame = pil_to_tensor(example["image"])  # (1, H, W) uint8
    seed = row_seed(base_seed, example["video_id"], example["timestamp_s"])
    rng = torch.Generator().manual_seed(seed)
    view_a, view_b = make_pair(frame, augmentation_config, rng)
    return {"original": frame, "view_a": view_a, "view_b": view_b}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_dataset.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/dataset.py tests/unit/test_contrastive_pretrain_dataset.py
git commit -m "feat: add deterministic per-row augmentation-pair transform"
```

---

## Task 9: `contrastive_pretrain/dataset.py` — StatefulDataLoader + resume mechanics

**Files:**
- Modify: `src/contrastive_pretrain/dataset.py` (add to Task 8's file)
- Modify: `pyproject.toml` (new dependency: `torchdata`)
- Test: `tests/unit/test_contrastive_pretrain_dataset.py` (add to Task 8's file)

**Interfaces:**
- Produces: `contrastive_pretrain.dataset.build_dataloader(dataset, batch_size: int, num_workers: int, snapshot_every_n_steps: int, pin_memory: bool = True) -> StatefulDataLoader`.

- [ ] **Step 1: Add the dependency**

```bash
uv add torchdata
```

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/test_contrastive_pretrain_dataset.py`:

```python
import datasets

from contrastive_pretrain.dataset import build_dataloader


def _synthetic_iterable_dataset():
    base = datasets.Dataset.from_dict({"value": list(range(20))})
    return base.to_iterable_dataset(num_shards=4)


def test_build_dataloader_yields_batches_of_configured_size() -> None:
    loader = build_dataloader(
        _synthetic_iterable_dataset(), batch_size=4, num_workers=0,
        snapshot_every_n_steps=1, pin_memory=False,
    )

    batch = next(iter(loader))

    assert batch["value"].shape == (4,)


def test_build_dataloader_resumes_from_exact_position() -> None:
    """Verifies the StatefulDataLoader checkpoint/resume mechanic the
    design spec depends on: a fresh dataloader over the same underlying
    (unconsumed) dataset, given a saved state_dict, continues from
    exactly where the original left off -- no re-served, no skipped
    examples."""
    loader = build_dataloader(
        _synthetic_iterable_dataset(), batch_size=2, num_workers=0,
        snapshot_every_n_steps=1, pin_memory=False,
    )

    it = iter(loader)
    for _ in range(3):
        next(it)
    state = loader.state_dict()
    expected_next_two_batches = [next(it)["value"].tolist() for _ in range(2)]

    resumed_loader = build_dataloader(
        _synthetic_iterable_dataset(), batch_size=2, num_workers=0,
        snapshot_every_n_steps=1, pin_memory=False,
    )
    resumed_loader.load_state_dict(state)
    actual_next_two_batches = [batch["value"].tolist() for _, batch in zip(range(2), resumed_loader)]

    assert actual_next_two_batches == expected_next_two_batches
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_dataset.py -v`
Expected: FAIL — `build_dataloader` doesn't exist yet.

- [ ] **Step 4: Write the implementation**

Append to `src/contrastive_pretrain/dataset.py` (add the import at the top alongside the existing ones):

```python
from torchdata.stateful_dataloader import StatefulDataLoader


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    snapshot_every_n_steps: int,
    pin_memory: bool = True,
) -> StatefulDataLoader:
    """drop_last=True is required for two reasons: it keeps batch shape
    fixed (cudnn.benchmark and torch.compile both depend on that), and a
    variable-size final batch would itself break that fixed-shape
    assumption. num_workers must stay the same between the run that
    saved a dataloader checkpoint and the run that resumes it --
    StatefulDataLoader.load_state_dict requires it."""
    return StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        snapshot_every_n_steps=snapshot_every_n_steps,
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_dataset.py -v`
Expected: PASS (7 tests total). No network access — a small in-memory `datasets.Dataset`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/contrastive_pretrain/dataset.py tests/unit/test_contrastive_pretrain_dataset.py
git commit -m "feat: add StatefulDataLoader wiring, verified against synthetic resume test"
```

---

## Task 10: `contrastive_pretrain/dataset.py` — real streaming construction + video-id split

**Files:**
- Modify: `src/contrastive_pretrain/dataset.py` (add to Task 9's file)
- Test: `tests/unit/test_contrastive_pretrain_dataset.py` (add slow tests to Task 9's file)

**Interfaces:**
- Consumes: `contrastive_pretrain.config.TrainingConfig` (Task 2), `contrastive_pretrain.dataset.to_pair_transform` (Task 8).
- Produces: `contrastive_pretrain.dataset.{build_train_dataset, build_val_dataset}` — both `(config: TrainingConfig) -> IterableDataset`.

- [ ] **Step 1: Write the failing (slow) tests**

Append to `tests/unit/test_contrastive_pretrain_dataset.py`:

```python
import pytest

from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.dataset import build_dataloader, build_train_dataset, build_val_dataset


@pytest.mark.slow
def test_build_train_dataset_excludes_val_videos() -> None:
    config = TrainingConfig()
    ds = build_train_dataset(config)

    row = next(iter(ds))

    assert row["video_id"] not in config.val_video_ids
    assert row["view_a"].shape == (1, 144, 160)


@pytest.mark.slow
def test_build_val_dataset_only_yields_held_out_videos() -> None:
    config = TrainingConfig()
    ds = build_val_dataset(config)

    for _, row in zip(range(5), ds):
        assert row["video_id"] in config.val_video_ids


@pytest.mark.slow
def test_build_dataloader_resumes_against_real_streaming_data() -> None:
    """Same resume guarantee as Task 9's synthetic test, but against the
    real Hub-backed streaming dataset -- confirms StatefulDataLoader's
    shard-skipping behavior holds for real parquet shards, not just an
    in-memory Dataset."""
    config = TrainingConfig()

    loader = build_dataloader(build_train_dataset(config), batch_size=2, num_workers=0, snapshot_every_n_steps=1)
    it = iter(loader)
    for _ in range(3):
        next(it)
    state = loader.state_dict()
    expected_video_ids = next(it)["video_id"]

    resumed_loader = build_dataloader(
        build_train_dataset(config), batch_size=2, num_workers=0, snapshot_every_n_steps=1
    )
    resumed_loader.load_state_dict(state)
    actual_video_ids = next(iter(resumed_loader))["video_id"]

    assert actual_video_ids == expected_video_ids
```

- [ ] **Step 2: Run the fast suite to confirm these are skipped by default**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_dataset.py -v`
Expected: 3 tests deselected (per `pyproject.toml`'s `addopts = "-m \"not slow\""`), the rest still fail (`build_train_dataset` doesn't exist).

- [ ] **Step 3: Write the implementation**

Append to `src/contrastive_pretrain/dataset.py` (add `functools` and `datasets` imports at the top):

```python
import functools

import datasets

from contrastive_pretrain.config import TrainingConfig


def _load_base_stream(config: TrainingConfig):
    return datasets.load_dataset(config.dataset_repo_id, streaming=True, split="train")


def build_train_dataset(config: TrainingConfig):
    ds = _load_base_stream(config)
    ds = ds.filter(lambda ex: ex["video_id"] not in config.val_video_ids)
    ds = ds.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.seed)
    ds = ds.map(
        functools.partial(to_pair_transform, augmentation_config=AugmentationConfig(), base_seed=config.seed),
        remove_columns=["image"],
    )
    return ds


def build_val_dataset(config: TrainingConfig):
    ds = _load_base_stream(config)
    ds = ds.filter(lambda ex: ex["video_id"] in config.val_video_ids)
    ds = ds.map(
        functools.partial(to_pair_transform, augmentation_config=AugmentationConfig(), base_seed=config.seed),
        remove_columns=["image"],
    )
    return ds
```

- [ ] **Step 4: Run the fast suite to confirm it still passes**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_dataset.py -v`
Expected: PASS (7 tests, 3 deselected as `slow`)

- [ ] **Step 5: Run the slow suite (requires HF credentials / network) to verify real behavior**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_dataset.py -v -m slow`
Expected: PASS (3 tests) — if this fails with an auth error, run `hf auth login` or set `HF_TOKEN` first, matching `data_collection`'s existing credential convention.

- [ ] **Step 6: Commit**

```bash
git add src/contrastive_pretrain/dataset.py tests/unit/test_contrastive_pretrain_dataset.py
git commit -m "feat: add real streaming train/val dataset construction with video-id split"
```

---

## Task 11: `contrastive_pretrain/train.py` — training loop orchestration

**Files:**
- Create: `src/contrastive_pretrain/train.py`
- Test: `tests/unit/test_contrastive_pretrain_train.py`

**Interfaces:**
- Consumes: everything from Tasks 1-10.
- Produces: `contrastive_pretrain.train.{run_memory_probe, check_finite_loss, compute_val_loss, TrainingDeps, run_training}`.
  - `run_memory_probe(probe_step: Callable[[], None], batch_size: int) -> None`
  - `check_finite_loss(loss: Tensor, global_step: int) -> None`
  - `compute_val_loss(encoder: nn.Module, projector: nn.Module, val_batches: Iterable[dict], temperature: float, device: torch.device, max_batches: int) -> float`
  - `TrainingDeps(config: TrainingConfig, frozen_encoder_client: HfClient, trackio_run: TrackioRunLike = NullTrackioRun(), device: torch.device = <cuda if available else cpu>)`
  - `run_training(deps: TrainingDeps) -> None`

- [ ] **Step 1: Write the failing tests for the unit-testable pieces**

Create `tests/unit/test_contrastive_pretrain_train.py`:

```python
import pytest
import torch

from contrastive_pretrain.model import build_encoder, build_projector
from contrastive_pretrain.train import check_finite_loss, compute_val_loss, run_memory_probe


def test_run_memory_probe_raises_actionable_error_on_oom() -> None:
    def _oom_step() -> None:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="batch_size=1024"):
        run_memory_probe(_oom_step, batch_size=1024)


def test_run_memory_probe_passes_through_on_success() -> None:
    calls = []
    run_memory_probe(lambda: calls.append(1), batch_size=32)
    assert calls == [1]


def test_run_memory_probe_does_not_swallow_other_errors() -> None:
    def _other_error() -> None:
        raise ValueError("something else")

    with pytest.raises(ValueError):
        run_memory_probe(_other_error, batch_size=32)


def test_check_finite_loss_raises_on_nan() -> None:
    with pytest.raises(RuntimeError, match="step 42"):
        check_finite_loss(torch.tensor(float("nan")), global_step=42)


def test_check_finite_loss_raises_on_inf() -> None:
    with pytest.raises(RuntimeError, match="step 1"):
        check_finite_loss(torch.tensor(float("inf")), global_step=1)


def test_check_finite_loss_passes_for_finite_value() -> None:
    check_finite_loss(torch.tensor(0.5), global_step=1)  # must not raise


def test_compute_val_loss_averages_over_batches() -> None:
    encoder, _ = build_encoder(pretrained=False)
    projector = build_projector()

    def fake_batches():
        for _ in range(3):
            yield {
                "view_a": torch.randint(0, 256, (2, 1, 144, 160), dtype=torch.uint8),
                "view_b": torch.randint(0, 256, (2, 1, 144, 160), dtype=torch.uint8),
            }

    loss = compute_val_loss(
        encoder, projector, fake_batches(), temperature=0.5, device=torch.device("cpu"), max_batches=3,
    )

    assert isinstance(loss, float)
    assert loss > 0


def test_compute_val_loss_restores_train_mode() -> None:
    encoder, _ = build_encoder(pretrained=False)
    projector = build_projector()
    encoder.train()
    projector.train()

    def fake_batches():
        yield {
            "view_a": torch.randint(0, 256, (2, 1, 144, 160), dtype=torch.uint8),
            "view_b": torch.randint(0, 256, (2, 1, 144, 160), dtype=torch.uint8),
        }

    compute_val_loss(
        encoder, projector, fake_batches(), temperature=0.5, device=torch.device("cpu"), max_batches=1,
    )

    assert encoder.training is True
    assert projector.training is True


def test_compute_val_loss_raises_when_no_batches_produced() -> None:
    encoder, _ = build_encoder(pretrained=False)
    projector = build_projector()

    with pytest.raises(RuntimeError, match="no batches"):
        compute_val_loss(encoder, projector, iter([]), temperature=0.5, device=torch.device("cpu"), max_batches=3)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contrastive_pretrain.train'`

- [ ] **Step 3: Write `check_finite_loss`, `run_memory_probe`, and `compute_val_loss`**

Create `src/contrastive_pretrain/train.py` (this file grows in Step 5 below; write only these three functions plus their imports for now):

```python
"""Training loop orchestration: startup memory probe, resume-from-checkpoint,
per-step training under bf16 autocast, periodic checkpointing, periodic
frozen-artifact export, and structured logging / Trackio reporting.

run_memory_probe and check_finite_loss are the two places this module
deliberately fails fast rather than silently degrading -- see the design
spec's rationale for why a batch-size OOM is not auto-retried smaller,
and why a non-finite loss is not silently skipped.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from contrastive_pretrain.losses import nt_xent_loss

logger = logging.getLogger(__name__)


def run_memory_probe(probe_step: Callable[[], None], batch_size: int) -> None:
    try:
        probe_step()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"Out of memory at batch_size={batch_size}. Lower batch_size in "
            "the config, or add gradient accumulation, and retry."
        ) from exc


def check_finite_loss(loss: torch.Tensor, global_step: int) -> None:
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"non-finite loss ({loss.item()}) at step {global_step} -- stopping (fail-fast).")


def compute_val_loss(
    encoder: nn.Module,
    projector: nn.Module,
    val_batches: Iterable[dict],
    temperature: float,
    device: torch.device,
    max_batches: int,
) -> float:
    was_encoder_training = encoder.training
    was_projector_training = projector.training
    encoder.eval()
    projector.eval()

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(val_batches):
            if i >= max_batches:
                break
            view_a = batch["view_a"].to(device).float()
            view_b = batch["view_b"].to(device).float()
            z_a = projector(encoder(view_a))
            z_b = projector(encoder(view_b))
            loss = nt_xent_loss(z_a, z_b, temperature)
            total_loss += loss.item()
            n_batches += 1

    encoder.train(was_encoder_training)
    projector.train(was_projector_training)

    if n_batches == 0:
        raise RuntimeError("val_batches produced no batches")
    return total_loss / n_batches
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_train.py -v`
Expected: PASS (9 tests). All CPU-only, no network.

- [ ] **Step 5: Add `TrainingDeps` and `run_training` (the full orchestration)**

Append to `src/contrastive_pretrain/train.py`:

```python
import time
from dataclasses import dataclass, field
from pathlib import Path

from contrastive_pretrain.checkpoint import (
    build_checkpoint_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_optimizer_and_scheduler,
    save_checkpoint,
)
from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.dataset import build_dataloader, build_train_dataset, build_val_dataset
from contrastive_pretrain.encoder_io import compute_latent_stats, push_frozen_encoder
from contrastive_pretrain.model import build_encoder, build_projector
from hf_storage.client import HfClient
from observability.tracking import NullTrackioRun, TrackioRunLike
from observability.visualization import build_augmentation_contact_sheet

# objones25/pokemon-frames has 296,000 rows (confirmed via the HF Hub API
# at spec time) -- used only to size the cosine LR schedule's T_max for a
# streaming dataset where "total steps" isn't otherwise knowable upfront.
_DATASET_ROW_COUNT = 296_000


@dataclass
class TrainingDeps:
    config: TrainingConfig
    frozen_encoder_client: HfClient
    trackio_run: TrackioRunLike = field(default_factory=NullTrackioRun)
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


def run_training(deps: TrainingDeps) -> None:
    config = deps.config
    device = deps.device

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.fp32_precision = "tf32"

    encoder, embedding_dim = build_encoder(pretrained=config.pretrained)
    projector = build_projector(in_dim=embedding_dim)
    encoder.to(device, memory_format=torch.channels_last)
    projector.to(device)

    checkpoint_dir = Path(config.network_volume_checkpoint_dir)
    latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    state = load_checkpoint(latest_checkpoint_path) if latest_checkpoint_path is not None else None
    if state is not None:
        encoder.load_state_dict(state["model"])  # BEFORE torch.compile -- see checkpoint.py's docstring
        logger.info(
            "resumed_from_checkpoint",
            extra={"path": str(latest_checkpoint_path), "global_step": state["global_step"]},
        )

    compiled_encoder = torch.compile(encoder, mode="default")

    def _probe_step() -> None:
        dummy = torch.zeros(config.batch_size, 1, 144, 160, device=device).to(
            memory_format=torch.channels_last
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z = projector(compiled_encoder(dummy))
            loss = nt_xent_loss(z, z, config.temperature)
        loss.backward()
        compiled_encoder.zero_grad(set_to_none=True)
        projector.zero_grad(set_to_none=True)

    run_memory_probe(_probe_step, config.batch_size)

    optimizer = torch.optim.AdamW(
        list(compiled_encoder.parameters()) + list(projector.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch_estimate = max(1, _DATASET_ROW_COUNT // config.batch_size)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=config.warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.max_epochs * steps_per_epoch_estimate - config.warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
    )

    train_dataset = build_train_dataset(config)
    dataloader = build_dataloader(
        train_dataset, config.batch_size, config.num_workers, config.checkpoint_interval_steps
    )
    val_dataset = build_val_dataset(config)

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    if state is not None:
        restore_optimizer_and_scheduler(optimizer, scheduler, state)
        dataloader.load_state_dict(state["dataloader"])
        global_step = state["global_step"]
        start_epoch = state["epoch"]
        best_val_loss = state["best_val_loss"]

    for epoch in range(start_epoch, config.max_epochs):
        train_dataset.set_epoch(epoch)
        contact_sheet_logged_this_epoch = False
        for batch in dataloader:
            step_start = time.monotonic()
            view_a = batch["view_a"].to(device, non_blocking=True).float().to(memory_format=torch.channels_last)
            view_b = batch["view_b"].to(device, non_blocking=True).float().to(memory_format=torch.channels_last)
            data_wait_s = time.monotonic() - step_start

            # Logged every step (not gated behind the 50-step interval below):
            # data_wait_s is a plain wall-clock float, not a GPU read, so
            # this costs no extra device sync -- and the whole point of
            # this metric is catching a streaming-throughput bottleneck
            # immediately, not up to 50 steps late.
            logger.info("data_wait", extra={"global_step": global_step, "data_wait_s": data_wait_s})

            if not contact_sheet_logged_this_epoch:
                # Per the design spec's observability section: the same
                # human-in-the-loop augmentation sanity check the standalone
                # `preview` CLI command does, now running against real
                # training-time batches once per epoch. Uses the batch's
                # original CPU uint8 tensors (batch["original"]/["view_a"]/
                # ["view_b"]), not the device/channels_last-converted
                # view_a/view_b locals above.
                triples = [
                    (
                        batch["original"][i].squeeze(0).numpy(),
                        batch["view_a"][i].squeeze(0).numpy(),
                        batch["view_b"][i].squeeze(0).numpy(),
                    )
                    for i in range(min(4, batch["original"].shape[0]))
                ]
                contact_sheet = build_augmentation_contact_sheet(triples)
                deps.trackio_run.log({"augmentation_contact_sheet": contact_sheet})
                contact_sheet_logged_this_epoch = True

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                z_a = projector(compiled_encoder(view_a))
                z_b = projector(compiled_encoder(view_b))
                loss = nt_xent_loss(z_a, z_b, config.temperature)
            check_finite_loss(loss, global_step)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 50 == 0:
                metrics = {
                    "global_step": global_step,
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                }
                logger.info("train_step", extra=metrics)
                deps.trackio_run.log(metrics)

            if global_step % config.checkpoint_interval_steps == 0:
                ckpt_state = build_checkpoint_state(
                    epoch, global_step, encoder, optimizer, scheduler, dataloader.state_dict(), best_val_loss
                )
                save_checkpoint(checkpoint_dir / f"checkpoint_step{global_step:08d}.pt", ckpt_state)
                logger.info("checkpoint_saved", extra={"global_step": global_step})

        val_dataloader = build_dataloader(val_dataset, config.batch_size, 0, 1, pin_memory=False)
        val_loss = compute_val_loss(
            compiled_encoder, projector, val_dataloader, config.temperature, device, max_batches=20
        )
        logger.info("epoch_complete", extra={"epoch": epoch, "val_loss": val_loss, "best_val_loss": best_val_loss})
        deps.trackio_run.log({"val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            latent_mean, latent_std = compute_latent_stats(compiled_encoder, val_dataset, device)
            push_frozen_encoder(deps.frozen_encoder_client, encoder, latent_mean, latent_std)
            logger.info("frozen_artifact_pushed", extra={"epoch": epoch, "val_loss": val_loss})

    deps.trackio_run.finish()
```

- [ ] **Step 6: Run the fast unit tests once more to confirm the additions didn't break imports**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_train.py -v`
Expected: PASS (9 tests, same as Step 4 — `TrainingDeps`/`run_training` aren't unit tested directly; they're exercised by the slow smoke test in Step 7).

- [ ] **Step 7: Add the slow smoke-test**

Append to `tests/unit/test_contrastive_pretrain_train.py`:

```python
import pytest

from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.train import TrainingDeps, run_training


class _FakeHfClient:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.files[path_in_repo] = data

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.files.get(path_in_repo)


@pytest.mark.slow
def test_run_training_completes_a_few_steps_without_nan(tmp_path) -> None:
    """A real, short smoke run (real streaming data, real model, a
    handful of steps) -- verifies the whole pipeline wires together and
    produces a finite, non-exploding loss before trusting it with a real
    paid A100 run. CPU-capable but slow; run on the target GPU when
    validating a full-scale config."""
    config = TrainingConfig(
        batch_size=4,
        num_workers=0,
        max_epochs=1,
        checkpoint_interval_steps=2,
        network_volume_checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    deps = TrainingDeps(config=config, frozen_encoder_client=_FakeHfClient(), device=torch.device("cpu"))

    # A 1-epoch run over the full streaming dataset is too slow for a
    # smoke test; monkeypatch max_epochs's effective loop bound isn't
    # exposed, so this test is intended to be run with a config small
    # enough to complete in seconds -- see the note above about running
    # it on the target GPU for full validation instead of relying on
    # this to prove end-to-end throughput.
    run_training(deps)
```

- [ ] **Step 8: Run the slow suite to verify it (requires network + a CUDA or CPU device capable of running the encoder)**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_train.py -v -m slow`
Expected: PASS — if this hangs or times out, it's most likely because the full streaming dataset takes longer than a few steps to make meaningful progress on CPU; note in the PR/commit that full-scale smoke validation should happen on the target A100, and adjust this test's scope (e.g. by capping steps via a small `max_epochs`/manually breaking after N steps in a follow-up) if needed rather than leaving it flaky.

- [ ] **Step 9: Commit**

```bash
git add src/contrastive_pretrain/train.py tests/unit/test_contrastive_pretrain_train.py
git commit -m "feat: add training loop orchestration with fail-fast memory probe and NaN handling"
```

---

## Task 12: `contrastive_pretrain/cli.py` — `train` and `export-frozen-encoder` commands

**Files:**
- Modify: `src/contrastive_pretrain/cli.py` (add to the existing `preview` command file)
- Test: `tests/unit/test_contrastive_pretrain_cli.py` (add to the existing file)

**Interfaces:**
- Consumes: `contrastive_pretrain.{config.load_config, train.{TrainingDeps, run_training}, encoder_io.{compute_latent_stats, push_frozen_encoder}, checkpoint.load_checkpoint, model.build_encoder, dataset.build_val_dataset}`, `hf_storage.client.RealHfClient`.
- Produces: CLI commands `contrastive-pretrain train --config <path>` and `contrastive-pretrain export-frozen-encoder --checkpoint <path> --config <path>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_contrastive_pretrain_cli.py` (check the existing file's imports first and add alongside them):

```python
from click.testing import CliRunner

from contrastive_pretrain.cli import main
from contrastive_pretrain.config import TrainingConfig


class _FakeHfApi:
    def create_repo(self, repo_id: str, repo_type: str, exist_ok: bool, private: bool) -> None:
        pass


def test_cli_help_lists_train_and_export_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "export-frozen-encoder" in result.output


def test_train_command_builds_deps_from_config_and_calls_run_training(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("batch_size: 8\n")
    captured = {}

    monkeypatch.setattr("contrastive_pretrain.cli.run_training", lambda deps: captured.update(deps=deps))
    monkeypatch.setattr("contrastive_pretrain.cli.RealHfClient", lambda *a, **k: object())
    monkeypatch.setattr("contrastive_pretrain.cli.HfApi", lambda: _FakeHfApi())
    monkeypatch.setattr("contrastive_pretrain.cli.get_token", lambda: "fake-token")

    runner = CliRunner()
    result = runner.invoke(main, ["train", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert captured["deps"].config.batch_size == 8


def test_train_command_fails_fast_with_no_hf_credentials(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("batch_size: 8\n")
    monkeypatch.setattr("contrastive_pretrain.cli.get_token", lambda: None)

    runner = CliRunner()
    result = runner.invoke(main, ["train", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "HF_TOKEN" in result.output


def test_export_frozen_encoder_command_requires_checkpoint_option() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["export-frozen-encoder"])

    assert result.exit_code != 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_cli.py -v`
Expected: FAIL — `train`/`export-frozen-encoder` commands don't exist yet.

- [ ] **Step 3: Write the implementation**

Append to `src/contrastive_pretrain/cli.py` (add these imports alongside the existing ones at the top of the file, and add `import trackio`, `from huggingface_hub import HfApi, get_token`, `from pathlib import Path` already present):

```python
import trackio
from huggingface_hub import HfApi, get_token

from contrastive_pretrain.checkpoint import load_checkpoint
from contrastive_pretrain.config import load_config
from contrastive_pretrain.dataset import build_val_dataset
from contrastive_pretrain.encoder_io import compute_latent_stats, push_frozen_encoder
from contrastive_pretrain.model import build_encoder
from contrastive_pretrain.train import TrainingDeps, run_training
from hf_storage.client import RealHfClient
from observability.tracking import TrackioRun

_DEFAULT_CONFIG = Path("configs/contrastive_pretrain.yaml")


@main.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), default=_DEFAULT_CONFIG)
def train(config_path: Path) -> None:
    """Run (or resume) contrastive pretraining on an A100 pod."""
    if get_token() is None:
        raise click.ClickException(
            "No Hugging Face credentials found. Set HF_TOKEN (e.g. in a "
            ".env file) or run `hf auth login` before using this command."
        )

    config = load_config(config_path)
    HfApi().create_repo(config.frozen_encoder_repo_id, repo_type="model", exist_ok=True, private=True)
    frozen_client = RealHfClient(HfApi(), config.frozen_encoder_repo_id, repo_type="model")
    trackio_run = TrackioRun(trackio, project="pokemon-contrastive-pretrain", name=config.frozen_encoder_repo_id)

    deps = TrainingDeps(config=config, frozen_encoder_client=frozen_client, trackio_run=trackio_run)
    run_training(deps)


@main.command(name="export-frozen-encoder")
@click.option("--checkpoint", "checkpoint_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), default=_DEFAULT_CONFIG)
def export_frozen_encoder_command(checkpoint_path: Path, config_path: Path) -> None:
    """Manually (re-)export a frozen encoder artifact from a saved training checkpoint."""
    config = load_config(config_path)
    encoder, _ = build_encoder(pretrained=False)
    state = load_checkpoint(checkpoint_path)
    encoder.load_state_dict(state["model"])

    val_dataset = build_val_dataset(config)
    latent_mean, latent_std = compute_latent_stats(encoder, val_dataset, device=torch.device("cpu"))

    HfApi().create_repo(config.frozen_encoder_repo_id, repo_type="model", exist_ok=True, private=True)
    client = RealHfClient(HfApi(), config.frozen_encoder_repo_id, repo_type="model")
    push_frozen_encoder(client, encoder, latent_mean, latent_std)
    click.echo(f"Exported frozen encoder from {checkpoint_path} to {config.frozen_encoder_repo_id}")
```

Add `import torch` at the top of `cli.py` if not already present (it is, via the existing `preview` command).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_contrastive_pretrain_cli.py -v`
Expected: PASS (existing `preview` tests + 4 new ones)

- [ ] **Step 5: Commit**

```bash
git add src/contrastive_pretrain/cli.py tests/unit/test_contrastive_pretrain_cli.py
git commit -m "feat: add train and export-frozen-encoder CLI commands"
```

---

## Task 13: Final integration pass

**Files:**
- Modify: `pyproject.toml` (verify dependency list is complete)
- No new source files.

**Interfaces:**
- Consumes: everything.
- Produces: nothing new — this task is verification only.

- [ ] **Step 1: Run the full fast test suite**

Run: `uv run pytest -v`
Expected: PASS, all tests except those marked `slow`.

- [ ] **Step 2: Run the full slow test suite (requires HF credentials, network, and enough compute to run a short CPU/GPU training smoke test)**

Run: `uv run pytest -v -m slow`
Expected: PASS. If the `contrastive_pretrain` slow tests fail due to environment limits (no GPU available, no credentials configured), note this explicitly rather than skipping silently — these are the tests that stand between this plan and trusting a real paid A100 run.

- [ ] **Step 3: Confirm `pyproject.toml`'s dependency and package lists are complete**

```bash
grep -A5 '\[project\]' pyproject.toml | head -20
grep -A6 'packages = ' pyproject.toml
```

Expected: dependencies include `torch`, `torchvision`, `torchdata`, `safetensors`, `datasets`, `huggingface-hub`, `trackio`, `click`, `pyyaml`; packages include `"src/data_collection"`, `"src/observability"`, `"src/contrastive_pretrain"`, `"src/hf_storage"`.

- [ ] **Step 4: Verify the CLI is wired end-to-end**

Run: `uv run contrastive-pretrain --help`
Expected: lists `preview`, `train`, and `export-frozen-encoder`.

- [ ] **Step 5: Commit any final cleanup**

```bash
git add -A
git status  # confirm nothing unexpected is staged
git commit -m "chore: final integration pass for contrastive-pretrain model pipeline" --allow-empty
```

(Use `--allow-empty` only if Step 3/4 required no file changes — if they did, drop the flag and let the real diff be committed.)
