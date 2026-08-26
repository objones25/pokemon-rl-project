# Temporal Sequence Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `RecurrentTransformerPolicy` — a 23.7M-parameter decoder-only transformer over frozen-CNN frame latents that gives the PPO agent a 1024-step (~6.8 minute) memory horizon.

**Architecture:** One fused `[latent, aux_state, prev_action, prev_reward]` token per timestep feeds a pre-norm RMSNorm / GQA+RoPE / SwiGLU stack. Two entry points: `step()` does KV-cached incremental decode against a per-env ring buffer during rollout; `forward_chunk()` does a burn-in-prefixed forward with a `causal ∧ sliding_window ∧ same_episode` mask during the PPO update. Attention is `F.scaled_dot_product_attention` with `enable_gqa=True` and an explicit boolean mask — one code path on CUDA and CPU alike.

**Tech Stack:** PyTorch 2.13, pytest 9.1.1, `uv` for packaging. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-26-temporal-sequence-model-design.md`

## Global Constraints

- **Package management:** `uv` only. `uv add <pkg>`, `uv run <cmd>`. Never bare `pip`/`venv`.
- **Every module** starts with `from __future__ import annotations` and a docstring explaining *why*, matching `src/contrastive_pretrain/losses.py`.
- **All public functions and methods carry full type annotations**, including `-> None`.
- **TDD is mandatory**: the failing test is written and *observed failing* before the implementation.
- **Tests are CPU-only, seeded, and tiny.** No network, no `time.sleep`, no `sys.path` edits, filesystem writes to `tmp_path` only.
- **No `if`/`for`/`while` in any test body.** Cases go through `@pytest.mark.parametrize`.
- **Floats compare with `pytest.approx`**, never `==`. Assert exact expected values, not loose ranges.
- **`pytest.raises` always names a specific exception and passes `match=`.**
- **Coverage floor is 80% branch** and must not be lowered (`pyproject.toml`).
- **`filterwarnings = ["error", ...]`** is active — a new warning fails the suite.
- Heads emit **raw logits**. Non-parameter tensors go through `register_buffer`.
- **Tiny test config used throughout:** `d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2, d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4, action_dim=7, action_embed_dim=4, reward_feat_dim=2`.
- **Production config:** `d_model=512, n_layers=8, n_heads=8, head_dim=64, n_kv_heads=2, d_ff=1408, context_len=1024, rope_theta=1e4, latent_dim=2048, aux_state_dim=32, action_dim=7, action_embed_dim=32, reward_feat_dim=8, qk_norm=True`.

---

### Task 1: Package scaffold and config

**Files:**
- Create: `src/sequence_model/__init__.py` (empty)
- Create: `src/sequence_model/config.py`
- Create: `configs/sequence_model.yaml`
- Modify: `pyproject.toml` — add `"src/sequence_model"` to `[tool.hatch.build.targets.wheel] packages`
- Test: `tests/unit/test_sequence_model_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PolicyConfig` (frozen dataclass, fields listed in Global Constraints), `load_config(path: str | Path) -> PolicyConfig`, and `PolicyConfig.head_dim`/`n_rep` derived properties.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sequence_model_config.py
from pathlib import Path

import pytest

from sequence_model.config import PolicyConfig, load_config


def test_default_config_head_dim_times_n_heads_equals_d_model() -> None:
    config = PolicyConfig()

    assert config.n_heads * config.head_dim == config.d_model


def test_default_config_n_rep_is_query_heads_per_kv_head() -> None:
    config = PolicyConfig()

    assert config.n_rep == 4


def test_load_config_overrides_only_named_fields(tmp_path: Path) -> None:
    path = tmp_path / "seq.yaml"
    path.write_text("d_model: 64\nn_layers: 3\n")

    config = load_config(path)

    assert (config.d_model, config.n_layers, config.context_len) == (64, 3, 1024)


def test_load_config_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "seq.yaml"
    path.write_text("d_modle: 64\n")

    with pytest.raises(ValueError, match=r"unknown config field\(s\): \['d_modle'\]"):
        load_config(path)


def test_config_rejects_n_heads_not_divisible_by_n_kv_heads() -> None:
    with pytest.raises(ValueError, match="n_heads=8 is not divisible by n_kv_heads=3"):
        PolicyConfig(n_heads=8, n_kv_heads=3)


def test_config_rejects_d_model_not_equal_to_n_heads_times_head_dim() -> None:
    with pytest.raises(ValueError, match="d_model=512 != n_heads=8 x head_dim=32"):
        PolicyConfig(d_model=512, n_heads=8, head_dim=32)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_config.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence_model'`

- [ ] **Step 3: Write the implementation**

```python
# src/sequence_model/config.py
"""Architecture and sizing for the temporal sequence model, loaded from
configs/sequence_model.yaml. Mirrors contrastive_pretrain.config's
dataclass + yaml.safe_load pattern.

The defaults are the ones the design spec's arithmetic produced (22.6M
backbone parameters, 4.00 KiB/token KV cache, 256 MiB at 64 envs x 1024
context) -- changing d_model/n_layers/n_kv_heads without re-running
config_budget.py invalidates every memory number in that spec."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PolicyConfig:
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64
    n_kv_heads: int = 2
    d_ff: int = 1408
    context_len: int = 1024
    rope_theta: float = 1e4
    latent_dim: int = 2048
    aux_state_dim: int = 32
    action_dim: int = 7
    action_embed_dim: int = 32
    reward_feat_dim: int = 8
    qk_norm: bool = True
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} is not divisible by n_kv_heads={self.n_kv_heads}; "
                "GQA requires an integer number of query heads per KV head"
            )
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                f"d_model={self.d_model} != n_heads={self.n_heads} x head_dim={self.head_dim}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even; RoPE pairs it in halves")

    @property
    def n_rep(self) -> int:
        """Query heads per KV head."""
        return self.n_heads // self.n_kv_heads

    @property
    def episode_start_action(self) -> int:
        """Embedding row for 'no previous action'. Feeding a real action
        index at episode reset teaches the model a lie, so this gets its
        own row rather than reusing action 0."""
        return self.action_dim


def load_config(path: str | Path) -> PolicyConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    valid_fields = {f.name for f in fields(PolicyConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")
    return PolicyConfig(**data)
```

```yaml
# configs/sequence_model.yaml
# Defaults match PolicyConfig; this file exists so a run can be pinned and
# diffed. See docs/superpowers/specs/2026-08-26-temporal-sequence-model-design.md
d_model: 512
n_layers: 8
n_heads: 8
head_dim: 64
n_kv_heads: 2
d_ff: 1408
context_len: 1024
```

In `pyproject.toml`, change the packages line to:

```toml
packages = ["src/data_collection", "src/observability", "src/contrastive_pretrain", "src/hf_storage", "src/sequence_model"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_config.py -v --no-cov`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/sequence_model/ configs/sequence_model.yaml pyproject.toml tests/unit/test_sequence_model_config.py
git commit -m "feat(sequence-model): config dataclass with GQA and head-dim invariants"
```

---

### Task 2: RoPE

**Files:**
- Create: `src/sequence_model/rope.py`
- Test: `tests/unit/test_sequence_model_rope.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `rope_tables(positions: Tensor, head_dim: int, theta: float) -> tuple[Tensor, Tensor]` returning `(cos, sin)` each shaped `positions.shape + (head_dim // 2,)`, float32; and `apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor` where `x` is `(B, H, T, head_dim)` and `cos`/`sin` are `(B, T, head_dim // 2)`.

Two things this task pins down permanently. **Halves pairing**, not interleaved — both train fine but they are silently incompatible, so the convention is asserted against an exact tensor. And **float64 angle computation with a mod-2π reduction**: episodes run to 163,840 steps, and a naive float32 product has max error 3.0e-03 across channels versus 2.9e-08 for the float64 path. Note the error is *not* in the highest-frequency channel (163840 is exactly representable in fp32) — it is in the mid-frequency channels, so the test asserts on the max across all channels.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sequence_model_rope.py
import math

import pytest
import torch

from sequence_model.rope import apply_rope, rope_tables


def test_rope_tables_returns_cos_and_sin_of_half_head_dim() -> None:
    positions = torch.arange(5).unsqueeze(0)

    cos, sin = rope_tables(positions, head_dim=8, theta=10000.0)

    assert (cos.shape, sin.shape) == ((1, 5, 4), (1, 5, 4))


def test_rope_tables_at_position_zero_is_all_cos_one_sin_zero() -> None:
    cos, sin = rope_tables(torch.zeros(1, 1, dtype=torch.long), head_dim=8, theta=10000.0)

    assert cos.tolist() == [[[1.0, 1.0, 1.0, 1.0]]]
    assert sin.tolist() == [[[0.0, 0.0, 0.0, 0.0]]]


def test_rope_tables_are_exact_at_large_absolute_positions() -> None:
    """Episodes run to 163,840 steps. Computing t * inv_freq in float32
    loses precision in the mid-frequency channels (max error 3.0e-03);
    computing in float64 with a mod-2pi reduction keeps it at 2.9e-08.
    Asserted across ALL channels: channel 0 alone passes either way,
    because 163840 is exactly representable in float32."""
    head_dim, theta, t = 64, 10000.0, 163840
    half = head_dim // 2
    inv_freq = theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
    expected = ((torch.tensor([[t]], dtype=torch.float64).unsqueeze(-1) * inv_freq) % (2 * math.pi)).cos()

    cos, _ = rope_tables(torch.tensor([[t]]), head_dim=head_dim, theta=theta)

    assert (cos.double() - expected).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_apply_rope_uses_halves_pairing_not_interleaved() -> None:
    """head_dim=4 at position 1 with theta such that inv_freq = [1, 1].
    Halves pairing rotates (x0, x2) and (x1, x3) together. Interleaved
    would rotate (x0, x1) and (x2, x3) -- same shape, different model,
    and silently incompatible with any external checkpoint."""
    x = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    cos = torch.full((1, 1, 2), math.cos(1.0))
    sin = torch.full((1, 1, 2), math.sin(1.0))

    out = apply_rope(x, cos, sin)

    assert out.flatten().tolist() == pytest.approx(
        [math.cos(1.0), 0.0, math.sin(1.0), 0.0], abs=1e-6
    )


def test_rope_attention_score_depends_only_on_relative_distance() -> None:
    """The whole point of RoPE. A query at 100 against a key at 105 must
    score identically to a query at 0 against a key at 5."""
    torch.manual_seed(0)
    head_dim = 8
    q_vec = torch.randn(1, 1, 1, head_dim)
    k_vec = torch.randn(1, 1, 1, head_dim)

    near_cos, near_sin = rope_tables(torch.tensor([[0, 5]]), head_dim, 10000.0)
    far_cos, far_sin = rope_tables(torch.tensor([[100, 105]]), head_dim, 10000.0)
    near = (
        apply_rope(q_vec, near_cos[:, :1], near_sin[:, :1])
        * apply_rope(k_vec, near_cos[:, 1:], near_sin[:, 1:])
    ).sum()
    far = (
        apply_rope(q_vec, far_cos[:, :1], far_sin[:, :1])
        * apply_rope(k_vec, far_cos[:, 1:], far_sin[:, 1:])
    ).sum()

    assert near.item() == pytest.approx(far.item(), abs=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_rope.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence_model.rope'`

- [ ] **Step 3: Write the implementation**

```python
# src/sequence_model/rope.py
"""Rotary position embeddings, halves pairing.

Two decisions here are load-bearing and invisible in a shape check.

First, HALVES pairing: element i is rotated against element i + head_dim/2.
The interleaved convention (i against i+1) trains just as well and is
silently incompatible with any checkpoint written by the other one, so the
choice is pinned by a test against an exact tensor rather than left to a
future refactor.

Second, the angle is computed in float64 and reduced mod 2*pi before
casting. Episodes run to ~163,840 steps; at that magnitude a float32
product t * inv_freq carries max error 3.0e-03 across channels (2.9e-08
via float64). The damage is in the MID-frequency channels, not the
highest -- 163840 is exactly representable in float32 -- which is why the
test asserts on the max over all channels."""

from __future__ import annotations

import math

import torch


def rope_tables(
    positions: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`positions` is an integer tensor of absolute step indices, shape
    (B, T). Returns (cos, sin), each (B, T, head_dim // 2), float32."""
    half = head_dim // 2
    inv_freq = theta ** (
        -torch.arange(0, half, dtype=torch.float64, device=positions.device) / half
    )
    angle = (positions.to(torch.float64).unsqueeze(-1) * inv_freq) % (2 * math.pi)
    return angle.cos().to(torch.float32), angle.sin().to(torch.float32)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x` is (B, H, T, head_dim) -- ALREADY split into heads. Applying
    RoPE before the head split makes the rotation straddle head
    boundaries, which costs a few percent of loss and gets blamed on the
    data. `cos`/`sin` are (B, T, head_dim // 2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_rope.py -v --no-cov`
Expected: 5 passed

- [ ] **Step 5: Prove a test can fail**

Change `% (2 * math.pi)` to nothing and change `torch.float64` to `torch.float32` in `rope_tables`. Run `test_rope_tables_are_exact_at_large_absolute_positions`. Expected: FAIL with a max error around 3.0e-03. Revert.

- [ ] **Step 6: Commit**

```bash
git add src/sequence_model/rope.py tests/unit/test_sequence_model_rope.py
git commit -m "feat(sequence-model): RoPE with halves pairing and float64 angle reduction"
```

---

### Task 3: RMSNorm and SwiGLU

**Files:**
- Create: `src/sequence_model/layers.py`
- Test: `tests/unit/test_sequence_model_layers.py`

**Interfaces:**
- Consumes: `PolicyConfig` from Task 1.
- Produces: `RMSNorm(dim: int, eps: float = 1e-6)` with `.weight` parameter and `forward(x) -> Tensor`; `SwiGLU(d_model: int, d_ff: int)` with `.gate_proj`, `.up_proj`, `.down_proj` (all `nn.Linear(..., bias=False)`) and `forward(x) -> Tensor`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sequence_model_layers.py
import pytest
import torch

from sequence_model.layers import RMSNorm, SwiGLU


def test_rmsnorm_scales_input_to_unit_root_mean_square() -> None:
    norm = RMSNorm(4, eps=0.0)
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]])

    out = norm(x)

    assert out.pow(2).mean().item() == pytest.approx(1.0, abs=1e-5)


def test_rmsnorm_matches_hand_derivation() -> None:
    """x = [2, 2, 2, 2] has RMS 2, so the normalized value is 1.0 in
    every channel and the default unit weight leaves it there."""
    norm = RMSNorm(4, eps=0.0)

    out = norm(torch.tensor([[2.0, 2.0, 2.0, 2.0]]))

    assert out.tolist() == pytest.approx([[1.0, 1.0, 1.0, 1.0]], abs=1e-6)


def test_rmsnorm_applies_eps_inside_the_square_root() -> None:
    """eps outside the sqrt is a real bug. With x all zeros and eps=4,
    inside gives 0/sqrt(0+4)=0 and no NaN; the value that distinguishes
    the two is the scale on a nonzero input: RMS^2 = 1, eps = 3, so
    inside-sqrt gives 1/sqrt(1+3) = 0.5."""
    norm = RMSNorm(4, eps=3.0)

    out = norm(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))

    assert out.tolist() == pytest.approx([[0.5, 0.5, 0.5, 0.5]], abs=1e-6)


def test_rmsnorm_weight_is_a_learnable_parameter() -> None:
    norm = RMSNorm(4)

    assert [name for name, _ in norm.named_parameters()] == ["weight"]


def test_swiglu_output_has_d_model_width() -> None:
    torch.manual_seed(0)
    mlp = SwiGLU(d_model=8, d_ff=16)

    out = mlp(torch.randn(2, 3, 8))

    assert tuple(out.shape) == (2, 3, 8)


def test_swiglu_has_three_projections_and_no_biases() -> None:
    """d_ff = 4 * d_model with a GATED mlp is silently 50% more MLP
    parameters than intended, so the three-matrix shape is asserted."""
    mlp = SwiGLU(d_model=8, d_ff=16)

    names = sorted(name for name, _ in mlp.named_parameters())

    assert names == ["down_proj.weight", "gate_proj.weight", "up_proj.weight"]


def test_swiglu_matches_hand_derivation_for_unit_weights() -> None:
    """With gate=up=identity-like all-ones projections on x=[1], the
    gate is silu(1) = 1/(1+e^-1) and up is 1, so the product is silu(1)
    and the summing down_proj returns d_ff * silu(1)."""
    mlp = SwiGLU(d_model=1, d_ff=2)
    torch.nn.init.ones_(mlp.gate_proj.weight)
    torch.nn.init.ones_(mlp.up_proj.weight)
    torch.nn.init.ones_(mlp.down_proj.weight)

    out = mlp(torch.tensor([[1.0]]))

    expected = 2.0 * (1.0 / (1.0 + torch.tensor(-1.0).exp().item()))
    assert out.item() == pytest.approx(expected, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_layers.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence_model.layers'`

- [ ] **Step 3: Write the implementation**

```python
# src/sequence_model/layers.py
"""RMSNorm and SwiGLU -- the two non-attention pieces of the block.

RMSNorm computes the mean square in float32 regardless of the autocast
dtype, and adds eps INSIDE the square root. eps outside is a real bug
(it changes the normalizer's asymptote); the bf16 squaring question is a
0.18% norm shift per site, not the accumulation-precision story usually
told, but float32 here costs nothing.

SwiGLU is three matrices, not two. d_ff is sized round_to_128(8/3 *
d_model) precisely so a gated MLP holds parameters level against a plain
4x one -- writing d_ff = 4 * d_model here would silently add 50% more MLP
parameters than the budget accounts for."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_layers.py -v --no-cov`
Expected: 7 passed

- [ ] **Step 5: Prove a test can fail**

Move `+ self.eps` outside the `rsqrt` (i.e. `torch.rsqrt(...mean(...)) + self.eps`). Run `test_rmsnorm_applies_eps_inside_the_square_root`. Expected: FAIL (4.0 instead of 0.5). Revert.

- [ ] **Step 6: Commit**

```bash
git add src/sequence_model/layers.py tests/unit/test_sequence_model_layers.py
git commit -m "feat(sequence-model): RMSNorm and SwiGLU"
```

---

### Task 4: Chunk attention mask

**Files:**
- Create: `src/sequence_model/masks.py`
- Test: `tests/unit/test_sequence_model_masks.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `build_chunk_mask(abs_pos: Tensor, episode_id: Tensor, context_len: int) -> Tensor`, where `abs_pos` and `episode_id` are `(B, L)` int64 and the result is `(B, 1, L, L)` bool, broadcast over heads.

The composed predicate is `causal ∧ (q_pos − kv_pos < context_len) ∧ same_episode`. The sliding-window term is what makes training match rollout exactly: without it a burn-in-prefixed chunk gives early positions *more* context than they had during rollout.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sequence_model_masks.py
import torch

from sequence_model.masks import build_chunk_mask


def test_chunk_mask_shape_is_broadcastable_over_heads() -> None:
    abs_pos = torch.arange(6).unsqueeze(0)
    episode_id = torch.zeros(1, 6, dtype=torch.long)

    mask = build_chunk_mask(abs_pos, episode_id, context_len=4)

    assert tuple(mask.shape) == (1, 1, 6, 6)


def test_chunk_mask_is_causal() -> None:
    abs_pos = torch.arange(4).unsqueeze(0)
    episode_id = torch.zeros(1, 4, dtype=torch.long)

    mask = build_chunk_mask(abs_pos, episode_id, context_len=8)

    assert mask[0, 0].tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_chunk_mask_window_limits_each_row_to_context_len_positions() -> None:
    abs_pos = torch.arange(10).unsqueeze(0)
    episode_id = torch.zeros(1, 10, dtype=torch.long)

    mask = build_chunk_mask(abs_pos, episode_id, context_len=4)

    assert mask[0, 0, 9].tolist() == [False] * 6 + [True] * 4


def test_chunk_mask_blocks_attention_across_an_episode_boundary() -> None:
    abs_pos = torch.tensor([[0, 1, 2, 0, 1, 2]])
    episode_id = torch.tensor([[0, 0, 0, 1, 1, 1]])

    mask = build_chunk_mask(abs_pos, episode_id, context_len=8)

    assert mask[0, 0, 3].tolist() == [False, False, False, True, False, False]


def test_chunk_mask_diagonal_is_always_unmasked() -> None:
    """A fully-masked row makes softmax return NaN. The diagonal is the
    guarantee that never happens."""
    abs_pos = torch.tensor([[0, 1, 0, 1]])
    episode_id = torch.tensor([[0, 0, 1, 1]])

    mask = build_chunk_mask(abs_pos, episode_id, context_len=1)

    assert torch.diagonal(mask[0, 0]).tolist() == [True, True, True, True]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_masks.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence_model.masks'`

- [ ] **Step 3: Write the implementation**

```python
# src/sequence_model/masks.py
"""The chunked-forward attention mask: causal AND sliding-window AND
same-episode.

The sliding-window term is what makes a burn-in-prefixed training chunk
match the rollout exactly. Without it, a position early in the chunk
attends over the whole prefix -- MORE context than it had when the action
was actually chosen -- and PPO's importance ratio is then comparing two
different functions.

The same-episode term matters roughly once per 160 chunks (episodes run
163,840 steps, chunks are 1024), which is exactly the frequency at which a
bug here would never be noticed."""

from __future__ import annotations

import torch


def build_chunk_mask(
    abs_pos: torch.Tensor, episode_id: torch.Tensor, context_len: int
) -> torch.Tensor:
    """`abs_pos` and `episode_id` are (B, L) int64. Returns a (B, 1, L, L)
    boolean mask where True means "may attend", shaped to broadcast over
    the head dimension.

    abs_pos resets to 0 at an episode boundary, so `q_pos >= kv_pos` alone
    is not a valid causal test across one -- the same-episode term is what
    makes the conjunction correct, not merely stricter."""
    query = abs_pos.unsqueeze(-1)
    key = abs_pos.unsqueeze(-2)
    causal = query >= key
    window = (query - key) < context_len
    same_episode = episode_id.unsqueeze(-1) == episode_id.unsqueeze(-2)
    return (causal & window & same_episode).unsqueeze(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_masks.py -v --no-cov`
Expected: 5 passed

- [ ] **Step 5: Prove a test can fail**

Delete the `& window` term. Run `test_chunk_mask_window_limits_each_row_to_context_len_positions`. Expected: FAIL (row 9 is all True). Revert.

- [ ] **Step 6: Commit**

```bash
git add src/sequence_model/masks.py tests/unit/test_sequence_model_masks.py
git commit -m "feat(sequence-model): composed causal/window/episode chunk mask"
```

---

### Task 5: Rollout KV cache

**Files:**
- Create: `src/sequence_model/cache.py`
- Test: `tests/unit/test_sequence_model_cache.py`

**Interfaces:**
- Consumes: `PolicyConfig` from Task 1.
- Produces: `RolloutCache` with `RolloutCache.empty(config, n_envs, device, dtype) -> RolloutCache`; `.write(layer: int, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]` taking `(N, n_kv_heads, 1, head_dim)` and returning the full `(N, n_kv_heads, context_len, head_dim)` buffers; `.attention_mask() -> Tensor` returning `(N, 1, 1, context_len)` bool; `.advance() -> None`; `.reset(done: Tensor) -> None`; and read-only `.abs_pos` `(N,)` int64.

The scatter uses `buf[env, :, idx] = new`, which yields shape `(N, H, D)` because the two advanced indices are non-adjacent — verified empirically, not assumed. Slots are **not** in temporal order after wraparound, and that is fine: RoPE phase is baked into each key at write time and the mask keys off slot validity, not slot index.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sequence_model_cache.py
import pytest
import torch

from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )


def test_empty_cache_has_no_valid_slots(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=3, device=torch.device("cpu"), dtype=torch.float32)

    assert cache.attention_mask().sum().item() == 0


def test_write_marks_exactly_one_slot_valid_per_env(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=3, device=torch.device("cpu"), dtype=torch.float32)
    k = torch.ones(3, 2, 1, 8)

    cache.write(layer=0, k_new=k, v_new=k)

    assert cache.attention_mask().sum().item() == 3


def test_write_returns_full_capacity_buffers(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=3, device=torch.device("cpu"), dtype=torch.float32)
    k = torch.ones(3, 2, 1, 8)

    k_all, v_all = cache.write(layer=0, k_new=k, v_new=k)

    assert (tuple(k_all.shape), tuple(v_all.shape)) == ((3, 2, 8, 8), (3, 2, 8, 8))


def test_write_places_the_new_key_at_the_current_write_slot(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=1, device=torch.device("cpu"), dtype=torch.float32)
    k_all, _ = cache.write(layer=0, k_new=torch.full((1, 2, 1, 8), 7.0), v_new=torch.zeros(1, 2, 1, 8))

    assert k_all[0, 0, 0].tolist() == [7.0] * 8


def test_advance_moves_the_write_slot_and_absolute_position(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=2, device=torch.device("cpu"), dtype=torch.float32)

    cache.advance()
    cache.advance()

    assert cache.abs_pos.tolist() == [2, 2]


def test_write_slot_wraps_around_at_capacity(tiny_config: PolicyConfig) -> None:
    """context_len=8, so the 9th write must land back in slot 0."""
    cache = RolloutCache.empty(tiny_config, n_envs=1, device=torch.device("cpu"), dtype=torch.float32)
    zeros = torch.zeros(1, 2, 1, 8)
    _fill_eight_slots(cache, zeros)

    k_all, _ = cache.write(layer=0, k_new=torch.full((1, 2, 1, 8), 9.0), v_new=zeros)

    assert k_all[0, 0, 0].tolist() == [9.0] * 8


def test_all_slots_valid_after_capacity_writes(tiny_config: PolicyConfig) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=1, device=torch.device("cpu"), dtype=torch.float32)
    _fill_eight_slots(cache, torch.zeros(1, 2, 1, 8))

    assert cache.attention_mask().sum().item() == 8


def test_reset_clears_validity_and_absolute_position_for_the_named_env(
    tiny_config: PolicyConfig,
) -> None:
    cache = RolloutCache.empty(tiny_config, n_envs=2, device=torch.device("cpu"), dtype=torch.float32)
    cache.write(layer=0, k_new=torch.ones(2, 2, 1, 8), v_new=torch.ones(2, 2, 1, 8))
    cache.advance()

    cache.reset(torch.tensor([True, False]))

    assert cache.attention_mask()[:, 0, 0].sum(dim=-1).tolist() == [0, 1]
    assert cache.abs_pos.tolist() == [0, 1]


def _fill_eight_slots(cache: RolloutCache, tensor: torch.Tensor) -> None:
    """Helper, not a test: eight writes take the tiny cache to capacity.
    Unrolled because test bodies may not contain loops."""
    for _ in range(8):
        cache.write(layer=0, k_new=tensor, v_new=tensor)
        cache.advance()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_cache.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence_model.cache'`

- [ ] **Step 3: Write the implementation**

```python
# src/sequence_model/cache.py
"""Per-env ring-buffer KV cache for the rollout path.

Capacity is context_len, so memory is bounded regardless of episode length
-- episodes run 163,840 steps against a 1024-wide window, so an
append-only cache is not an option.

Slots are NOT in temporal order after the first wraparound, and nothing
downstream needs them to be: each key carries its own RoPE phase from
write time, and the attention mask keys off slot validity rather than slot
index. This is also why keys are never re-rotated on the way out -- doing
so restarts positions at 0 every decode step, which is the classic
cache bug (prefill perfect, rollout garbage).

write() and advance() are separate because every layer writes at the SAME
slot for a given timestep; the position advances once per env step, not
once per layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sequence_model.config import PolicyConfig


@dataclass
class RolloutCache:
    k: torch.Tensor  # (n_layers, N, n_kv_heads, context_len, head_dim)
    v: torch.Tensor
    slot_valid: torch.Tensor  # (N, context_len) bool
    write_pos: torch.Tensor  # (N,) int64
    abs_pos: torch.Tensor  # (N,) int64
    capacity: int

    @classmethod
    def empty(
        cls,
        config: PolicyConfig,
        n_envs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RolloutCache:
        shape = (config.n_layers, n_envs, config.n_kv_heads, config.context_len, config.head_dim)
        return cls(
            k=torch.zeros(shape, device=device, dtype=dtype),
            v=torch.zeros(shape, device=device, dtype=dtype),
            slot_valid=torch.zeros(n_envs, config.context_len, dtype=torch.bool, device=device),
            write_pos=torch.zeros(n_envs, dtype=torch.long, device=device),
            abs_pos=torch.zeros(n_envs, dtype=torch.long, device=device),
            capacity=config.context_len,
        )

    def write(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`k_new`/`v_new` are (N, n_kv_heads, 1, head_dim), already
        RoPE-rotated at their own absolute position. Returns the full
        (N, n_kv_heads, capacity, head_dim) buffers for this layer."""
        env = torch.arange(self.write_pos.shape[0], device=self.write_pos.device)
        self.k[layer][env, :, self.write_pos] = k_new[:, :, 0, :].to(self.k.dtype)
        self.v[layer][env, :, self.write_pos] = v_new[:, :, 0, :].to(self.v.dtype)
        self.slot_valid[env, self.write_pos] = True
        return self.k[layer], self.v[layer]

    def advance(self) -> None:
        """Called once per env step, after every layer has written."""
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.abs_pos = self.abs_pos + 1

    def attention_mask(self) -> torch.Tensor:
        """(N, 1, 1, capacity) bool -- one query position against every
        cached slot. Broadcasts over heads."""
        return self.slot_valid[:, None, None, :]

    def reset(self, done: torch.Tensor) -> None:
        """`done` is (N,) bool. Resetting abs_pos to 0 keeps RoPE
        positions small within an episode and makes episode masking
        automatic on the rollout path -- a reset cache holds nothing from
        the previous episode, so only a training chunk can straddle a
        boundary."""
        self.slot_valid[done] = False
        self.write_pos = torch.where(done, torch.zeros_like(self.write_pos), self.write_pos)
        self.abs_pos = torch.where(done, torch.zeros_like(self.abs_pos), self.abs_pos)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_cache.py -v --no-cov`
Expected: 8 passed

- [ ] **Step 5: Prove a test can fail**

Remove the `% self.capacity` from `advance()`. Run `test_write_slot_wraps_around_at_capacity`. Expected: FAIL with an `IndexError` on slot 8. Revert.

- [ ] **Step 6: Commit**

```bash
git add src/sequence_model/cache.py tests/unit/test_sequence_model_cache.py
git commit -m "feat(sequence-model): per-env ring-buffer KV cache"
```

---

### Task 6: Grouped-query attention

**Files:**
- Create: `src/sequence_model/attention.py`
- Test: `tests/unit/test_sequence_model_attention.py`

**Interfaces:**
- Consumes: `PolicyConfig` (Task 1), `rope_tables`/`apply_rope` (Task 2), `RMSNorm` (Task 3), `RolloutCache` (Task 5).
- Produces: `GroupedQueryAttention(config: PolicyConfig)` with `forward_chunk(x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor` for `(B, L, d_model)` input, and `forward_step(x: Tensor, cos: Tensor, sin: Tensor, cache: RolloutCache, layer: int) -> Tensor` for `(N, 1, d_model)` input.

Order is **split heads → QK-norm → RoPE**, in that order. `enable_gqa=True` on SDPA means no hand-written `repeat_kv`, which removes the `x.repeat()` query-head-permutation bug at the source rather than testing around it. `forward_step` passes `is_causal=False` — a length-1 query with `is_causal=True` attends to slot 0 only.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sequence_model_attention.py
import pytest
import torch
import torch.nn.functional as F

from sequence_model.attention import GroupedQueryAttention
from sequence_model.config import PolicyConfig
from sequence_model.masks import build_chunk_mask


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )


def test_forward_chunk_preserves_input_shape(tiny_config: PolicyConfig) -> None:
    torch.manual_seed(0)
    attn = GroupedQueryAttention(tiny_config)
    x = torch.randn(2, 5, 32)
    mask = build_chunk_mask(torch.arange(5).expand(2, 5), torch.zeros(2, 5, dtype=torch.long), 8)

    out = attn.forward_chunk(x, *_tables(5, 2, tiny_config), mask)

    assert tuple(out.shape) == (2, 5, 32)


def test_forward_chunk_is_causal(tiny_config: PolicyConfig) -> None:
    """Changing the last token must leave every earlier output untouched."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(tiny_config)
    x = torch.randn(1, 5, 32)
    mask = build_chunk_mask(torch.arange(5).unsqueeze(0), torch.zeros(1, 5, dtype=torch.long), 8)
    cos, sin = _tables(5, 1, tiny_config)
    changed = x.clone()
    changed[0, 4] = torch.randn(32)

    before = attn.forward_chunk(x, cos, sin, mask)
    after = attn.forward_chunk(changed, cos, sin, mask)

    assert (before[0, :4] - after[0, :4]).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_gqa_repeats_kv_head_j_to_query_heads_j_times_n_rep(tiny_config: PolicyConfig) -> None:
    """enable_gqa=True must map KV head j to query heads [j*n_rep,
    (j+1)*n_rep). x.repeat() gives the same shape with heads
    interleaved -- a query-head permutation that trains fine and breaks
    checkpoint interop and fused kernels."""
    torch.manual_seed(0)
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    expected = F.scaled_dot_product_attention(
        q,
        k.unsqueeze(2).expand(1, 2, 2, 3, 8).reshape(1, 4, 3, 8),
        v.unsqueeze(2).expand(1, 2, 2, 3, 8).reshape(1, 4, 3, 8),
    )

    actual = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    assert (actual - expected).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_qk_norm_parameters_exist_when_enabled(tiny_config: PolicyConfig) -> None:
    attn = GroupedQueryAttention(tiny_config)

    names = sorted(n for n, _ in attn.named_parameters() if "norm" in n)

    assert names == ["k_norm.weight", "q_norm.weight"]


def test_qk_norm_parameters_absent_when_disabled(tiny_config: PolicyConfig) -> None:
    from dataclasses import replace

    attn = GroupedQueryAttention(replace(tiny_config, qk_norm=False))

    assert [n for n, _ in attn.named_parameters() if "norm" in n] == []


def test_projections_have_no_biases(tiny_config: PolicyConfig) -> None:
    attn = GroupedQueryAttention(tiny_config)

    assert [n for n, _ in attn.named_parameters() if n.endswith("bias")] == []


def _tables(seq_len: int, batch: int, config: PolicyConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper, not a test."""
    from sequence_model.rope import rope_tables

    return rope_tables(torch.arange(seq_len).expand(batch, seq_len), config.head_dim, config.rope_theta)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_attention.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence_model.attention'`

- [ ] **Step 3: Write the implementation**

```python
# src/sequence_model/attention.py
"""Grouped-query attention with RoPE, one module serving both the chunked
training forward and the KV-cached rollout step.

Three ordering/flag decisions here each produce correctly-shaped tensors
and a broken model:

1. Heads are split BEFORE RoPE. Rotating the flat (B, T, d_model) tensor
   makes the rotation straddle head boundaries.
2. `enable_gqa=True` rather than a hand-written repeat_kv. The tempting
   `x.repeat()` implementation is exactly a query-head permutation: it
   trains fine from scratch and breaks checkpoint interop and fused
   kernels, which is a different bug class than a quality regression.
3. `is_causal=False` on the step path. The query is length 1 against
   `context_len` cached keys, so is_causal would mask everything but slot
   0 -- prefill perfect, rollout garbage.

SDPA with an explicit boolean mask is used rather than FlexAttention
because FlexAttention has no CPU backward in torch 2.13, which would make
the chunked-forward tests unrunnable under the CPU-only test rule."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig
from sequence_model.layers import RMSNorm
from sequence_model.rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        if config.qk_norm:
            self.q_norm: nn.Module = RMSNorm(config.head_dim, config.rms_norm_eps)
            self.k_norm: nn.Module = RMSNorm(config.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _project(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        cfg = self.config
        q = self.q_proj(x).view(batch, seq_len, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        return apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin), v

    def _merge_heads(self, attended: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = attended.shape
        merged = attended.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(merged)

    def forward_chunk(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """`x` is (B, L, d_model); `mask` is the (B, 1, L, L) bool mask
        from masks.build_chunk_mask."""
        q, k, v = self._project(x, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return self._merge_heads(attended)

    def forward_step(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: RolloutCache,
        layer: int,
    ) -> torch.Tensor:
        """`x` is (N, 1, d_model). Writes this step's rotated K/V into the
        cache and attends over every valid slot."""
        q, k, v = self._project(x, cos, sin)
        k_all, v_all = cache.write(layer, k, v)
        attended = F.scaled_dot_product_attention(
            q, k_all, v_all, attn_mask=cache.attention_mask(), enable_gqa=True, is_causal=False
        )
        return self._merge_heads(attended)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_attention.py -v --no-cov`
Expected: 6 passed

- [ ] **Step 5: Prove a test can fail**

In `forward_chunk`, pass `attn_mask=None, is_causal=False`. Run `test_forward_chunk_is_causal`. Expected: FAIL (earlier outputs move). Revert.

- [ ] **Step 6: Commit**

```bash
git add src/sequence_model/attention.py tests/unit/test_sequence_model_attention.py
git commit -m "feat(sequence-model): GQA attention with chunk and cached-step paths"
```

---

### Task 7: Transformer block

**Files:**
- Create: `src/sequence_model/block.py`
- Test: `tests/unit/test_sequence_model_block.py`

**Interfaces:**
- Consumes: `PolicyConfig` (Task 1), `RMSNorm`/`SwiGLU` (Task 3), `GroupedQueryAttention` (Task 6), `RolloutCache` (Task 5).
- Produces: `TransformerBlock(config: PolicyConfig)` with `forward_chunk(x, cos, sin, mask) -> Tensor` and `forward_step(x, cos, sin, cache, layer) -> Tensor`, both `(…, d_model)` in and out.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sequence_model_block.py
import pytest
import torch

from sequence_model.block import TransformerBlock
from sequence_model.config import PolicyConfig
from sequence_model.masks import build_chunk_mask
from sequence_model.rope import rope_tables


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )


def test_block_preserves_shape(tiny_config: PolicyConfig) -> None:
    torch.manual_seed(0)
    block = TransformerBlock(tiny_config)
    x = torch.randn(2, 5, 32)
    cos, sin = rope_tables(torch.arange(5).expand(2, 5), 8, 10000.0)
    mask = build_chunk_mask(torch.arange(5).expand(2, 5), torch.zeros(2, 5, dtype=torch.long), 8)

    out = block.forward_chunk(x, cos, sin, mask)

    assert tuple(out.shape) == (2, 5, 32)


def test_block_is_residual_so_zeroed_sublayers_return_the_input(
    tiny_config: PolicyConfig,
) -> None:
    """Pre-norm with both output projections zeroed must be the identity.
    A block that is not residual, or that norms the residual stream
    itself, fails this."""
    torch.manual_seed(0)
    block = TransformerBlock(tiny_config)
    torch.nn.init.zeros_(block.attention.o_proj.weight)
    torch.nn.init.zeros_(block.mlp.down_proj.weight)
    x = torch.randn(1, 5, 32)
    cos, sin = rope_tables(torch.arange(5).unsqueeze(0), 8, 10000.0)
    mask = build_chunk_mask(torch.arange(5).unsqueeze(0), torch.zeros(1, 5, dtype=torch.long), 8)

    out = block.forward_chunk(x, cos, sin, mask)

    assert (out - x).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_block_declares_its_four_norm_weights(tiny_config: PolicyConfig) -> None:
    block = TransformerBlock(tiny_config)

    names = sorted(n for n, _ in block.named_parameters() if n.endswith("_norm.weight"))

    assert names == [
        "attention.k_norm.weight",
        "attention.q_norm.weight",
        "attn_norm.weight",
        "mlp_norm.weight",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_block.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence_model.block'`

- [ ] **Step 3: Write the implementation**

```python
# src/sequence_model/block.py
"""Pre-norm transformer block: RMSNorm -> GQA -> residual, RMSNorm ->
SwiGLU -> residual.

Pre-norm, not post-norm: it trains without a warmup schedule, and this
model's optimizer never gets a stable loss surface to warm up against --
PPO's objective moves under it every update.

The residual stream itself is never normalized, only what feeds each
sublayer. The zeroed-sublayer identity test is the cheap check that this
stayed true through refactors."""

from __future__ import annotations

import torch
from torch import nn

from sequence_model.attention import GroupedQueryAttention
from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig
from sequence_model.layers import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config.d_model, config.d_ff)

    def forward_chunk(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attention.forward_chunk(self.attn_norm(x), cos, sin, mask)
        return x + self.mlp(self.mlp_norm(x))

    def forward_step(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: RolloutCache,
        layer: int,
    ) -> torch.Tensor:
        x = x + self.attention.forward_step(self.attn_norm(x), cos, sin, cache, layer)
        return x + self.mlp(self.mlp_norm(x))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_block.py -v --no-cov`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/sequence_model/block.py tests/unit/test_sequence_model_block.py
git commit -m "feat(sequence-model): pre-norm transformer block"
```

---

### Task 8: Input adapter

**Files:**
- Create: `src/sequence_model/adapter.py`
- Test: `tests/unit/test_sequence_model_adapter.py`

**Interfaces:**
- Consumes: `PolicyConfig` from Task 1.
- Produces: `InputAdapter(config: PolicyConfig, latent_mean: Tensor, latent_std: Tensor)` with `forward(latent, aux_state, prev_action, prev_reward) -> Tensor` mapping `(…, latent_dim)`, `(…, aux_state_dim)`, `(…,)` int64, `(…,)` float to `(…, d_model)`.

`latent_mean`/`latent_std` come from the frozen encoder repo's `latent_stats.json` and are **frozen buffers**, not running statistics — running stats under a shifting policy are a non-stationarity source, and §5 asks for an explicit affine layer, not an adaptive one. Registering them as buffers is also what makes them visible to `.to(device)` and `state_dict()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sequence_model_adapter.py
import pytest
import torch

from sequence_model.adapter import InputAdapter
from sequence_model.config import PolicyConfig


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )


@pytest.fixture
def adapter(tiny_config: PolicyConfig) -> InputAdapter:
    torch.manual_seed(0)
    return InputAdapter(tiny_config, torch.zeros(16), torch.ones(16))


def test_adapter_maps_a_timestep_to_d_model(adapter: InputAdapter) -> None:
    out = adapter(
        torch.randn(2, 3, 16), torch.randn(2, 3, 4),
        torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3),
    )

    assert tuple(out.shape) == (2, 3, 32)


def test_latent_stats_are_buffers_not_parameters(adapter: InputAdapter) -> None:
    """register_buffer, not a bare attribute: a bare tensor is invisible
    to .to(device) and state_dict(). And not a Parameter: these are fixed
    stats from latent_stats.json, and letting the optimizer move them
    reintroduces the non-stationarity the normalizer exists to remove."""
    buffer_names = sorted(n for n, _ in adapter.named_buffers())
    parameter_names = [n for n, _ in adapter.named_parameters()]

    assert buffer_names == ["latent_mean", "latent_std"]
    assert "latent_mean" not in parameter_names


def test_latent_stats_appear_in_state_dict(adapter: InputAdapter) -> None:
    assert "latent_mean" in adapter.state_dict()


def test_adapter_normalizes_the_latent_by_the_published_stats(
    tiny_config: PolicyConfig,
) -> None:
    """mean=2, std=4 turns a latent of 10 into (10-2)/4 = 2. Asserted by
    zeroing every other contribution and reading the projection input
    back through an identity-summing projection."""
    torch.manual_seed(0)
    adapter = InputAdapter(tiny_config, torch.full((16,), 2.0), torch.full((16,), 4.0))

    normalized = adapter.normalize_latent(torch.full((1, 1, 16), 10.0))

    assert normalized.flatten()[0].item() == pytest.approx(2.0, abs=1e-4)


def test_episode_start_action_has_its_own_embedding_row(adapter: InputAdapter) -> None:
    """Index 7 means "no previous action". Reusing action 0 (DOWN) there
    teaches the model that every episode begins with a press it never
    made."""
    zeros_latent = torch.zeros(1, 1, 16)
    zeros_aux = torch.zeros(1, 1, 4)
    zero_reward = torch.zeros(1, 1)

    as_start = adapter(zeros_latent, zeros_aux, torch.full((1, 1), 7, dtype=torch.long), zero_reward)
    as_down = adapter(zeros_latent, zeros_aux, torch.zeros(1, 1, dtype=torch.long), zero_reward)

    assert (as_start - as_down).abs().max().item() > 1e-6


def test_action_embedding_has_one_row_per_action_plus_episode_start(
    adapter: InputAdapter,
) -> None:
    assert tuple(adapter.action_embed.weight.shape) == (8, 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_adapter.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence_model.adapter'`

- [ ] **Step 3: Write the implementation**

```python
# src/sequence_model/adapter.py
"""Fuses one timestep into one token.

Layout: [normalized latent | aux_state | embed(prev_action) |
proj(prev_reward)] -> Linear -> d_model. One token per timestep, not three
interleaved: a 1024-step horizon then costs exactly 1024 KV positions
rather than 3072, and one forward yields that step's logits and value
directly.

Causality note: the token at position t carries s_t but a_{t-1} and
r_{t-1}, because that is exactly what is known when a_t must be chosen.

The latent normalizer's mean/std come from the frozen encoder repo's
latent_stats.json and are frozen buffers. PPO value networks are
hypersensitive to unscaled input variance, and raw contrastive latents
cause immediate policy collapse -- but adaptive (running) statistics
would put a second moving target under an objective that already moves."""

from __future__ import annotations

import torch
from torch import nn

from sequence_model.config import PolicyConfig


class InputAdapter(nn.Module):
    def __init__(
        self, config: PolicyConfig, latent_mean: torch.Tensor, latent_std: torch.Tensor
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("latent_mean", latent_mean)
        self.register_buffer("latent_std", latent_std)
        self.action_embed = nn.Embedding(config.action_dim + 1, config.action_embed_dim)
        self.reward_proj = nn.Linear(1, config.reward_feat_dim, bias=False)
        fused_dim = (
            config.latent_dim
            + config.aux_state_dim
            + config.action_embed_dim
            + config.reward_feat_dim
        )
        self.proj = nn.Linear(fused_dim, config.d_model, bias=False)

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / (self.latent_std + 1e-6)

    def forward(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> torch.Tensor:
        """`latent` (..., latent_dim), `aux_state` (..., aux_state_dim),
        `prev_action` (...) int64 with config.episode_start_action for
        the first step of an episode, `prev_reward` (...) float."""
        fused = torch.cat(
            [
                self.normalize_latent(latent),
                aux_state,
                self.action_embed(prev_action),
                self.reward_proj(prev_reward.unsqueeze(-1)),
            ],
            dim=-1,
        )
        return self.proj(fused)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_adapter.py -v --no-cov`
Expected: 6 passed

- [ ] **Step 5: Prove a test can fail**

Change `self.register_buffer("latent_mean", latent_mean)` to `self.latent_mean = latent_mean`. Run `test_latent_stats_are_buffers_not_parameters` and `test_latent_stats_appear_in_state_dict`. Expected: both FAIL. Revert.

- [ ] **Step 6: Commit**

```bash
git add src/sequence_model/adapter.py tests/unit/test_sequence_model_adapter.py
git commit -m "feat(sequence-model): input adapter with frozen latent normalizer"
```

---

### Task 9: RecurrentTransformerPolicy

**Files:**
- Create: `src/sequence_model/policy.py`
- Test: `tests/unit/test_sequence_model_policy.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces: `StepOutput(logits: Tensor, value: Tensor)`, `ChunkOutput(logits: Tensor, value: Tensor)`, and `RecurrentTransformerPolicy(config, latent_mean, latent_std)` with `.new_cache(n_envs, device) -> RolloutCache`, `.step(latent, aux_state, prev_action, prev_reward, cache) -> StepOutput`, and `.forward_chunk(latent, aux_state, prev_action, prev_reward, abs_pos, episode_id, burn_in) -> ChunkOutput`.

This is the task the whole sub-project exists for, and it carries the two load-bearing tests. `test_incremental_step_matches_full_chunk_forward` catches `is_causal` misuse, ring-buffer indexing errors, RoPE position drift, and mask errors *simultaneously*. `test_forward_chunk_with_burn_in_matches_recorded_rollout_outputs` is the correctness claim of the burn-in scheme itself.

`step` is decorated `@torch.no_grad()` and deliberately **not** `@torch.inference_mode()`: rollout latents become inputs to the update's forward, and an inference-mode tensor entering autograd raises `RuntimeError: Inference tensors cannot be saved for backward`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sequence_model_policy.py
import pytest
import torch

from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy


@pytest.fixture
def tiny_config() -> PolicyConfig:
    return PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )


@pytest.fixture
def policy(tiny_config: PolicyConfig) -> RecurrentTransformerPolicy:
    torch.manual_seed(0)
    return RecurrentTransformerPolicy(tiny_config, torch.zeros(16), torch.ones(16))


def test_step_returns_one_logit_per_action_and_one_value(
    policy: RecurrentTransformerPolicy,
) -> None:
    cache = policy.new_cache(n_envs=3, device=torch.device("cpu"))

    out = policy.step(
        torch.randn(3, 16), torch.randn(3, 4),
        torch.full((3,), 7, dtype=torch.long), torch.zeros(3), cache,
    )

    assert (tuple(out.logits.shape), tuple(out.value.shape)) == ((3, 7), (3,))


def test_incremental_step_matches_full_chunk_forward(
    policy: RecurrentTransformerPolicy,
) -> None:
    """THE load-bearing test. Five tokens pushed through step() one at a
    time must equal the same five through forward_chunk(). Catches
    is_causal=True during decode, ring-buffer indexing errors, RoPE
    position drift, and mask errors, all at once."""
    torch.manual_seed(1)
    latent, aux, action, reward = _episode(seq_len=5, n_envs=2)
    cache = policy.new_cache(n_envs=2, device=torch.device("cpu"))

    stepped = _run_rollout(policy, latent, aux, action, reward, cache)
    chunked = policy.forward_chunk(
        latent, aux, action, reward,
        abs_pos=torch.arange(5).expand(2, 5),
        episode_id=torch.zeros(2, 5, dtype=torch.long),
        burn_in=0,
    )

    assert (stepped - chunked.logits).abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_forward_chunk_with_burn_in_matches_recorded_rollout_outputs(
    policy: RecurrentTransformerPolicy,
) -> None:
    """The correctness claim of the burn-in scheme. context_len is 8, so
    a 15-token sequence with burn_in=7 exercises the sliding window: the
    last token attends to positions 7..14 in both paths."""
    torch.manual_seed(2)
    latent, aux, action, reward = _episode(seq_len=15, n_envs=2)
    cache = policy.new_cache(n_envs=2, device=torch.device("cpu"))

    stepped = _run_rollout(policy, latent, aux, action, reward, cache)
    chunked = policy.forward_chunk(
        latent, aux, action, reward,
        abs_pos=torch.arange(15).expand(2, 15),
        episode_id=torch.zeros(2, 15, dtype=torch.long),
        burn_in=7,
    )

    assert (stepped[:, 7:] - chunked.logits).abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_forward_chunk_returns_only_the_gradient_region(
    policy: RecurrentTransformerPolicy,
) -> None:
    latent, aux, action, reward = _episode(seq_len=15, n_envs=2)

    chunked = policy.forward_chunk(
        latent, aux, action, reward,
        abs_pos=torch.arange(15).expand(2, 15),
        episode_id=torch.zeros(2, 15, dtype=torch.long),
        burn_in=7,
    )

    assert tuple(chunked.logits.shape) == (2, 8, 7)


def test_cache_reset_makes_step_independent_of_pre_reset_history(
    policy: RecurrentTransformerPolicy,
) -> None:
    torch.manual_seed(3)
    latent, aux, action, reward = _episode(seq_len=4, n_envs=1)
    dirty = policy.new_cache(n_envs=1, device=torch.device("cpu"))
    _run_rollout(policy, latent, aux, action, reward, dirty)
    dirty.reset(torch.tensor([True]))
    fresh = policy.new_cache(n_envs=1, device=torch.device("cpu"))
    first = (latent[:, 0], aux[:, 0], action[:, 0], reward[:, 0])

    after_reset = policy.step(*first, dirty).logits
    from_fresh = policy.step(*first, fresh).logits

    assert (after_reset - from_fresh).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_episode_mask_blocks_attention_across_a_boundary(
    policy: RecurrentTransformerPolicy,
) -> None:
    """Replacing every pre-boundary token with noise must leave the
    post-boundary outputs untouched."""
    torch.manual_seed(4)
    latent, aux, action, reward = _episode(seq_len=6, n_envs=1)
    abs_pos = torch.tensor([[0, 1, 2, 0, 1, 2]])
    episode_id = torch.tensor([[0, 0, 0, 1, 1, 1]])
    noisy = latent.clone()
    noisy[:, :3] = torch.randn(1, 3, 16)

    clean_out = policy.forward_chunk(latent, aux, action, reward, abs_pos, episode_id, burn_in=3)
    noisy_out = policy.forward_chunk(noisy, aux, action, reward, abs_pos, episode_id, burn_in=3)

    assert (clean_out.logits - noisy_out.logits).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_actor_head_is_a_bare_linear_emitting_raw_logits(
    policy: RecurrentTransformerPolicy,
) -> None:
    """A softmax or sigmoid before CrossEntropyLoss trains a broken model
    silently rather than crashing."""
    assert isinstance(policy.actor, torch.nn.Linear)
    assert policy.actor.bias is None


def test_every_parameter_receives_gradient(policy: RecurrentTransformerPolicy) -> None:
    torch.manual_seed(5)
    latent, aux, action, reward = _episode(seq_len=5, n_envs=2)

    out = policy.forward_chunk(
        latent, aux, action, reward,
        abs_pos=torch.arange(5).expand(2, 5),
        episode_id=torch.zeros(2, 5, dtype=torch.long),
        burn_in=0,
    )
    (out.logits.sum() + out.value.sum()).backward()

    ungrated = sorted(n for n, p in policy.named_parameters() if p.grad is None)
    assert ungrated == []


def test_output_projections_are_scaled_by_one_over_sqrt_two_n_layers(
    tiny_config: PolicyConfig,
) -> None:
    """n_layers=2, so the target std is 0.02 / sqrt(4) = 0.01. Keeps the
    residual stream's variance flat with depth."""
    torch.manual_seed(0)
    policy = RecurrentTransformerPolicy(tiny_config, torch.zeros(16), torch.ones(16))

    observed = policy.blocks[0].attention.o_proj.weight.std().item()

    assert observed == pytest.approx(0.01, rel=0.15)


def _episode(seq_len: int, n_envs: int) -> tuple[torch.Tensor, ...]:
    """Helper, not a test: a synthetic (n_envs, seq_len) rollout."""
    return (
        torch.randn(n_envs, seq_len, 16),
        torch.randn(n_envs, seq_len, 4),
        torch.randint(0, 7, (n_envs, seq_len)),
        torch.randn(n_envs, seq_len),
    )


def _run_rollout(policy, latent, aux, action, reward, cache) -> torch.Tensor:
    """Helper, not a test: drives step() across the sequence and stacks
    the logits into (n_envs, seq_len, action_dim)."""
    collected = [
        policy.step(latent[:, t], aux[:, t], action[:, t], reward[:, t], cache).logits
        for t in range(latent.shape[1])
    ]
    return torch.stack(collected, dim=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_policy.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence_model.policy'`

- [ ] **Step 3: Write the implementation**

```python
# src/sequence_model/policy.py
"""The PPO-facing policy: a decoder-only transformer over frozen-CNN frame
latents, with an actor head and a critic head.

Two entry points, because PPO needs two things that a single forward
cannot serve:

  step()          KV-cached incremental decode, one token per env step.
  forward_chunk() burn-in-prefixed forward for the update.

step() is @torch.no_grad(), NOT @torch.inference_mode(). Rollout latents
become inputs to forward_chunk during the update, and a tensor created
under inference_mode raises "Inference tensors cannot be saved for
backward" the moment it enters an autograd graph -- at the first update,
not at rollout. no_grad tensors are ordinary tensors. The module has no
dropout and no BatchNorm, so train/eval mode is behaviorally irrelevant
here.

forward_chunk lets gradients flow THROUGH the burn-in prefix rather than
detaching it, departing from R2D2 deliberately: R2D2 detaches because an
RNN's alternative is unbounded backprop, while a 1024-window transformer's
backprop is already bounded. Detaching here would truncate a path that is
genuinely part of d(loss)/d(theta) -- a biased gradient, not a cheaper
equal one. The cost is 0.07 s/epoch against an 8.0 s rollout."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from sequence_model.adapter import InputAdapter
from sequence_model.block import TransformerBlock
from sequence_model.cache import RolloutCache
from sequence_model.config import PolicyConfig
from sequence_model.layers import RMSNorm
from sequence_model.masks import build_chunk_mask
from sequence_model.rope import rope_tables


@dataclass(frozen=True)
class StepOutput:
    logits: torch.Tensor  # (N, action_dim), raw
    value: torch.Tensor  # (N,)


@dataclass(frozen=True)
class ChunkOutput:
    logits: torch.Tensor  # (B, T, action_dim), raw -- gradient region only
    value: torch.Tensor  # (B, T)


class RecurrentTransformerPolicy(nn.Module):
    def __init__(
        self, config: PolicyConfig, latent_mean: torch.Tensor, latent_std: torch.Tensor
    ) -> None:
        super().__init__()
        self.config = config
        self.adapter = InputAdapter(config, latent_mean, latent_std)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.actor = nn.Linear(config.d_model, config.action_dim, bias=False)
        self.critic = nn.Linear(config.d_model, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        """N(0, 0.02) everywhere, then residual output projections scaled
        by 1/sqrt(2 * n_layers) to keep the residual stream's variance
        flat with depth. The heads are initialised last so the generic
        pass cannot clobber them: the actor's near-zero gain makes the
        initial policy close to uniform, which is standard PPO practice
        and stops the first update from chasing an arbitrary preference."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        residual_scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            with torch.no_grad():
                block.attention.o_proj.weight.mul_(residual_scale)
                block.mlp.down_proj.weight.mul_(residual_scale)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def new_cache(self, n_envs: int, device: torch.device) -> RolloutCache:
        return RolloutCache.empty(self.config, n_envs, device, dtype=torch.float32)

    @torch.no_grad()
    def step(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        cache: RolloutCache,
    ) -> StepOutput:
        """One vectorized env step. `latent` (N, latent_dim), `aux_state`
        (N, aux_state_dim), `prev_action` (N,) int64, `prev_reward` (N,).
        Mutates `cache` in place and advances it by one position."""
        x = self.adapter(latent, aux_state, prev_action, prev_reward).unsqueeze(1)
        cos, sin = rope_tables(
            cache.abs_pos.unsqueeze(1), self.config.head_dim, self.config.rope_theta
        )
        for layer, block in enumerate(self.blocks):
            x = block.forward_step(x, cos, sin, cache, layer)
        cache.advance()

        hidden = self.final_norm(x).squeeze(1)
        return StepOutput(logits=self.actor(hidden), value=self.critic(hidden).squeeze(-1))

    def forward_chunk(
        self,
        latent: torch.Tensor,
        aux_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        abs_pos: torch.Tensor,
        episode_id: torch.Tensor,
        burn_in: int,
    ) -> ChunkOutput:
        """`L = burn_in + T` tokens in, the last T out. `abs_pos` carries
        the positions recorded during rollout so RoPE matches exactly."""
        x = self.adapter(latent, aux_state, prev_action, prev_reward)
        cos, sin = rope_tables(abs_pos, self.config.head_dim, self.config.rope_theta)
        mask = build_chunk_mask(abs_pos, episode_id, self.config.context_len)
        for block in self.blocks:
            x = block.forward_chunk(x, cos, sin, mask)

        hidden = self.final_norm(x)[:, burn_in:]
        return ChunkOutput(logits=self.actor(hidden), value=self.critic(hidden).squeeze(-1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_policy.py -v --no-cov`
Expected: 10 passed

- [ ] **Step 5: Prove the two load-bearing tests can fail**

First: in `attention.forward_step`, change `is_causal=False` to `is_causal=True`. Run `test_incremental_step_matches_full_chunk_forward`. Expected: FAIL. Revert.

Second: in `forward_chunk`, pass `self.config.context_len * 100` to `build_chunk_mask` (disabling the sliding window). Run `test_forward_chunk_with_burn_in_matches_recorded_rollout_outputs`. Expected: FAIL — the chunk gives early positions more context than the rollout had. Revert.

Record both in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/sequence_model/policy.py tests/unit/test_sequence_model_policy.py
git commit -m "feat(sequence-model): RecurrentTransformerPolicy with step and chunk paths"
```

---

### Task 10: Telemetry

**Files:**
- Create: `src/sequence_model/telemetry.py`
- Test: `tests/unit/test_sequence_model_telemetry.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure functions over tensors).
- Produces: `DISTANCE_BUCKETS: tuple[tuple[str, int, int], ...]`; `attention_logit_max(q: Tensor, k: Tensor, mask: Tensor) -> float`; `attention_distance_mass(weights: Tensor) -> dict[str, float]`; `residual_norm(hidden: Tensor) -> float`.

Loss and grad-norm are both *late* indicators. Attention logit magnitude is the documented divergence mechanism (every run above 1e4 diverged in Wortsman et al.), and final-layer output norm led divergence by 20–30% of training in Chameleon's runs. The distance histogram is the honest check on this component's premise: if attention mass never leaves the sub-8-step buckets, the 1024-context design is dead weight and that should surface in hour one.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sequence_model_telemetry.py
import pytest
import torch

from sequence_model.telemetry import (
    attention_distance_mass,
    attention_logit_max,
    residual_norm,
)


def test_distance_mass_puts_all_weight_in_the_adjacent_bucket() -> None:
    """Weight only on distance 1 (each query attends to the token right
    before it) must land entirely in the "1" bucket."""
    weights = torch.zeros(1, 1, 4, 4)
    weights[0, 0, 1, 0] = 1.0
    weights[0, 0, 2, 1] = 1.0
    weights[0, 0, 3, 2] = 1.0

    mass = attention_distance_mass(weights)

    assert mass["1"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("distance", "expected_bucket"),
    [(1, "1"), (3, "2-8"), (20, "9-64"), (100, "65-256")],
)
def test_distance_mass_assigns_each_distance_to_its_bucket(
    distance: int, expected_bucket: str
) -> None:
    weights = torch.zeros(1, 1, 300, 300)
    weights[0, 0, 299, 299 - distance] = 1.0

    mass = attention_distance_mass(weights)

    assert mass[expected_bucket] == pytest.approx(1.0, abs=1e-6)


def test_distance_mass_buckets_sum_to_one_for_normalized_weights() -> None:
    torch.manual_seed(0)
    weights = torch.softmax(torch.randn(2, 2, 16, 16), dim=-1)

    mass = attention_distance_mass(weights)

    assert sum(mass.values()) == pytest.approx(1.0, abs=1e-5)


def test_attention_logit_max_ignores_masked_positions() -> None:
    """A huge logit at a masked position must not be reported -- it is
    never attended to, so counting it would raise false alarms every
    step."""
    q = torch.zeros(1, 1, 2, 4)
    k = torch.zeros(1, 1, 2, 4)
    q[0, 0, 0] = torch.tensor([100.0, 0.0, 0.0, 0.0])
    k[0, 0, 1] = torch.tensor([100.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([[[[True, False], [True, True]]]])

    assert attention_logit_max(q, k, mask) == pytest.approx(0.0, abs=1e-4)


def test_attention_logit_max_reports_the_scaled_score() -> None:
    """head_dim=4, so the scale is 1/sqrt(4) = 0.5. A raw dot product of
    8 is reported as 4."""
    q = torch.zeros(1, 1, 1, 4)
    k = torch.zeros(1, 1, 1, 4)
    q[0, 0, 0, 0] = 4.0
    k[0, 0, 0, 0] = 2.0
    mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)

    assert attention_logit_max(q, k, mask) == pytest.approx(4.0, abs=1e-4)


def test_residual_norm_is_the_mean_per_token_l2_norm() -> None:
    """Two tokens of norm 3 and 4 average to 3.5."""
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])

    assert residual_norm(hidden) == pytest.approx(3.5, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sequence_model_telemetry.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence_model.telemetry'`

- [ ] **Step 3: Write the implementation**

```python
# src/sequence_model/telemetry.py
"""Leading indicators for the sequence model.

Loss and grad norm are both late. These are not:

  attention_logit_max  Wortsman et al. found every run whose attention
                       logits passed 1e4 diverged. This is the mechanism
                       behind the value-loss spikes the architecture plan
                       worries about.
  residual_norm        Final-layer output norm predicted divergence 20-30%
                       of training ahead in Chameleon's runs.
  attention_distance_mass
                       The honest check on this whole component. A 1024-step
                       context is only worth its cost if attention actually
                       reaches past a few steps; if the mass never leaves
                       the "1" and "2-8" buckets, the design is wrong and
                       the histogram says so in hour one rather than week two.

These take already-computed tensors so they stay pure and CPU-testable.
SDPA does not return attention probabilities, so the caller computes them
explicitly on a sampled minibatch every N updates rather than on the hot
path."""

from __future__ import annotations

import math

import torch

# (label, inclusive lower bound, exclusive upper bound) over q_index - k_index.
DISTANCE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 2),
    ("2-8", 2, 9),
    ("9-64", 9, 65),
    ("65-256", 65, 257),
    ("257-1024", 257, 1025),
)


def attention_logit_max(
    q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor
) -> float:
    """`q` (B, H, L, D), `k` (B, H_kv, L, D), `mask` broadcastable
    (B, 1, L, L) bool. Returns the largest scaled logit at an UNMASKED
    position -- masked positions are never attended to, so including them
    would raise false alarms on every step."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    n_rep = q.shape[1] // k.shape[1]
    k_expanded = k.repeat_interleave(n_rep, dim=1)
    scores = (q.float() @ k_expanded.float().transpose(-2, -1)) * scale
    return scores.masked_fill(~mask, float("-inf")).max().item()


def attention_distance_mass(weights: torch.Tensor) -> dict[str, float]:
    """`weights` (B, H, L, L), post-softmax. Returns the fraction of total
    attention mass falling in each distance bucket."""
    seq_len = weights.shape[-1]
    query_index = torch.arange(seq_len, device=weights.device).view(-1, 1)
    key_index = torch.arange(seq_len, device=weights.device).view(1, -1)
    distance = query_index - key_index
    total = weights.sum()

    mass: dict[str, float] = {}
    for label, low, high in DISTANCE_BUCKETS:
        in_bucket = (distance >= low) & (distance < high)
        mass[label] = (weights * in_bucket).sum().div(total).item()
    return mass


def residual_norm(hidden: torch.Tensor) -> float:
    """`hidden` (B, L, d_model). Mean per-token L2 norm of the residual
    stream leaving the final block."""
    return hidden.float().norm(dim=-1).mean().item()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sequence_model_telemetry.py -v --no-cov`
Expected: 9 passed (the parametrized test reports 4)

- [ ] **Step 5: Prove a test can fail**

Remove the `.masked_fill(~mask, float("-inf"))` from `attention_logit_max`. Run `test_attention_logit_max_ignores_masked_positions`. Expected: FAIL (reports ~5000 instead of 0). Revert.

- [ ] **Step 6: Commit**

```bash
git add src/sequence_model/telemetry.py tests/unit/test_sequence_model_telemetry.py
git commit -m "feat(sequence-model): leading-indicator telemetry"
```

---

### Task 11: Overfit-one-batch gate and suite audit

**Files:**
- Create: `tests/integration/test_sequence_model_overfit.py`
- Test: the file above, plus a full-suite run

**Interfaces:**
- Consumes: `RecurrentTransformerPolicy` from Task 9.
- Produces: nothing importable — this is a gate.

CLAUDE.md's stated first move when a model trains but will not learn is to overfit a single batch before touching hyperparameters, data, or architecture. This test makes that check permanent: it collapses the search space to model/loss/step, and it is marked `slow` so it stays deselected by default.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_sequence_model_overfit.py
import pytest
import torch
import torch.nn.functional as F

from sequence_model.config import PolicyConfig
from sequence_model.policy import RecurrentTransformerPolicy


@pytest.mark.slow
def test_tiny_policy_overfits_a_single_batch_to_near_zero_loss() -> None:
    """Behaviour-cloning proxy: 200 AdamW steps on ONE fixed batch of
    random targets must drive cross-entropy near zero. If this fails, the
    defect is in the model, the loss, or the step order -- not in
    hyperparameters, data, or architecture size."""
    torch.manual_seed(0)
    config = PolicyConfig(
        d_model=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        d_ff=64, context_len=8, latent_dim=16, aux_state_dim=4,
        action_embed_dim=4, reward_feat_dim=2,
    )
    policy = RecurrentTransformerPolicy(config, torch.zeros(16), torch.ones(16))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-5)
    batch = _fixed_batch()
    targets = torch.randint(0, 7, (2, 8))

    final_loss = _train_steps(policy, optimizer, batch, targets, steps=200)

    assert final_loss < 0.05


def _fixed_batch() -> tuple[torch.Tensor, ...]:
    """Helper, not a test."""
    return (
        torch.randn(2, 8, 16),
        torch.randn(2, 8, 4),
        torch.randint(0, 7, (2, 8)),
        torch.randn(2, 8),
        torch.arange(8).expand(2, 8),
        torch.zeros(2, 8, dtype=torch.long),
    )


def _train_steps(policy, optimizer, batch, targets, steps: int) -> float:
    """Helper, not a test: zero_grad -> forward -> loss -> backward -> step."""
    latent, aux, action, reward, abs_pos, episode_id = batch
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        optimizer.zero_grad()
        out = policy.forward_chunk(latent, aux, action, reward, abs_pos, episode_id, burn_in=0)
        loss = F.cross_entropy(out.logits.reshape(-1, 7), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
    return loss.item()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/integration/test_sequence_model_overfit.py -m slow -v --no-cov`
Expected: FAIL initially only if a defect exists. If it passes immediately, verify it can fail by setting `lr=0.0` and confirming the assertion trips; then restore `lr=3e-3`.

- [ ] **Step 3: Run the static audit**

Run: `uv run python ~/.claude/skills/pytest-expert/scripts/audit_tests.py tests/`
Expected: no findings. Fix anything reported before continuing — a green `pytest` over a suite the audit flags means the tests are not testing.

One finding is expected and must be **fixed, not ignored**: this repo's `tests/conftest.py` does not seed any RNG globally, so the audit's `UNSEEDED_RANDOM` check is active. Tests that call `torch.randn` while relying on a fixture's `torch.manual_seed` are genuinely deterministic but statically indistinguishable from unseeded ones. Resolve by adding an explicit `torch.manual_seed(0)` to the Arrange block of each flagged test, not by passing `--ignore UNSEEDED_RANDOM`.

- [ ] **Step 4: Run the full suite with coverage**

Run: `uv run pytest`
Expected: all pass, `--cov-fail-under=80` satisfied.

Then confirm order-independence: `uv run pytest -p no:randomly -q`

- [ ] **Step 5: Ratchet the coverage floor**

Read the coverage percentage from the run above. Raise `fail_under` in `[tool.coverage.report]` and `--cov-fail-under` in `addopts` to the achieved figure rounded *down* to the nearest whole percent. A floor you have to lower is worse than none.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_sequence_model_overfit.py pyproject.toml
git commit -m "test(sequence-model): overfit-one-batch gate and coverage ratchet"
```

---

## Self-Review

**Spec coverage.** Walking the spec section by section: config and sizing → Task 1; RoPE with float64 reduction → Task 2; RMSNorm/SwiGLU → Task 3; composed chunk mask → Task 4; ring-buffer cache with reset semantics → Task 5; GQA with `enable_gqa` and `is_causal=False` → Task 6; pre-norm block → Task 7; input adapter with frozen buffers and `EPISODE_START` → Task 8; both APIs, heads, init, and every listed correctness test → Task 9; leading-indicator telemetry → Task 10; overfit-one-batch and the audit gate → Task 11.

Two spec items are deliberately **not** tasks here, and both are recorded in the spec's "Out of scope": the W&B/JSON-lines wiring (it belongs to the PPO training loop, which does not exist yet — Task 10 ships the pure metric functions that loop will call), and the env interface contract (binding, not implementing). The attention-distance heatmap PNG likewise needs a training loop to be logged from; `attention_distance_mass` is the data behind it.

**Placeholder scan.** No TBD/TODO, no "add appropriate error handling", no "similar to Task N", every code step carries real code. Helper functions used in tests are defined in the same file that uses them.

**Type consistency.** `PolicyConfig` field names are identical across Tasks 1, 5, 6, 8, 9. `RolloutCache.write` returns `(k_all, v_all)` in Task 5 and is unpacked that way in Task 6. `rope_tables`/`apply_rope` signatures in Task 2 match every call site in Tasks 6, 7, 9. `build_chunk_mask(abs_pos, episode_id, context_len)` in Task 4 matches Task 9's call. `forward_chunk`/`forward_step` names are consistent from Task 6 through Task 9. `StepOutput.logits`/`ChunkOutput.logits` are used under those names in every Task 9 test.
