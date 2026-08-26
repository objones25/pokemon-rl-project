# Temporal Sequence Model — Design Spec

Date: 2026-08-26
Status: Approved for planning

## Purpose

Build the memory of the agent: a decoder-only transformer over frozen-CNN frame
latents that turns a stream of single-frame observations into a policy with a
~7-minute horizon. This is §3 ("Temporal Sequence Modeling") and the sequence
half of §5 of `Pokemon_RL_Architecture_Plan.pdf`.

Concretely, this sub-project ships one module —
`RecurrentTransformerPolicy` — with two entry points that PPO calls:

- `step()`, KV-cached incremental decode, one token per vectorized env step.
- `forward_chunk()`, a chunked forward with burn-in for the PPO update.

plus the actor and critic heads, the input adapter, and the rollout cache.

## Scope boundary

**In scope**: the policy module and its state management, the property tests
that prove the state management correct, its config, and its observability
hooks.

**Out of scope, and owned by the PPO sub-project**: the PyBoy environment, the
reward function's RAM semantics, the PPO loss and advantage estimation, the
rollout buffer's storage implementation, and the vectorized-env harness. This
spec *binds interfaces* for those (see "Environment interface contract"), it
does not implement them.

**Explicitly not built**: no offline pretraining of this model. The dataset
`objones25/pokemon-frames` is YouTube longplay frames with no action or reward
labels; it exists to train the CNN and nothing else. Every sequence this model
ever sees comes from PyBoy rollouts, so it is trained from scratch inside PPO.

## Why this design, not alternatives considered

### The plan's §5 GQA premise does not survive arithmetic

§5 says to "profile the KV cache memory footprint on the A100 for a sequence
length of 1000 [and] adjust the GQA group size to guarantee the PPO rollout
buffer can hold at least 4-8 parallel environment trajectories without
triggering memory offload."

Measured with the transformer-architecture skill's `config_budget.py`, the
chosen config's KV cache is **4.00 KiB/token**, so **64** parallel trajectories
at full 1024 context cost **256 MiB** — 0.3% of an 80GB A100, not 4-8
trajectories at the edge of offload. That worry was written for an LLM-scale
model.

GQA at 4x stays in the design because it is standard, cheap, and reduces
parameters. It is **not** load-bearing for memory, and recording that here stops
a false constraint from driving future config changes.

### Compute is ~100x from binding; RL optimization is the real constraint

Measured with `torch.utils.flop_counter` on the actual built encoder, the frozen
CNN costs **14.69 GFLOP/frame** — 1.8x a stock ResNet-50 at 224², because
dropping the stem maxpool quadruples spatial resolution through all four stages.
The 22.6M-parameter transformer costs 186 MFLOP/token to train.

**The transformer is ~1% of the rollout cost.** Hardware permits a model 100x
larger. The binding constraint is optimization stability and sample efficiency
on correlated on-policy data, which argues for the small end of the range. Hence
23.7M parameters, not 74M.

### One fused token per timestep, not interleaved [s, a, r]

§3 says the sequence is "alternating (or concatenated)". Concatenated wins:

- A 1024-step memory horizon costs exactly 1024 KV positions, not 3072.
- One forward per env step yields that step's logits and value directly, with no
  question of which of three positions carries the heads.
- Decision Transformer's interleaving serves *offline*, return-conditioned RL.
  This is online PPO.

Causality is unaffected. At the moment the agent must choose `a_t` it knows
`s_t`, `a_{t-1}`, `r_{t-1}`, so the token at position `t` is
`[s_t, a_{t-1}, r_{t-1}]` — the standard recurrent-policy input (R2D2, SB3
`RecurrentPPO`). No leakage.

### Observation includes a RAM progress vector, not latent alone

At frame-skip 24, a 1024-step context is ~24,000 emulator frames ≈ **6.8 minutes
of gameplay**. Badges and story events are earned hours apart. "I already beat
Brock" is therefore not recoverable from any window this model can hold, no
matter how well trained.

`PWhiddy/PokemonRedExperiments` reached the same conclusion empirically: its v2
observation is a Dict of `screens`, `health`, `level`, `badges`, `events`,
`map`, `recent_actions`, not pixels alone.

The RAM is read regardless, because §4's delta-based rewards
(`max(0, current_total_level - max_historical_total_level)`) are computed from
it. So the marginal cost of also feeding a compact progress vector to the policy
is one input projection.

The CNN still does all the vision work. The aux vector carries only what no
6.8-minute window can hold.

### Burn-in prefix, not §5's detached KV carry

§5 proposes: "instead of zeroing out the Transformer state at the start of a new
context chunk, pass the final KV states of chunk t-1 as detached tensors to
chunk t."

The problem §5 identifies is real — chunk-boundary blindness — but its solution
buys nothing here. Costed per PPO update (64 envs x 1024 steps = 65,536
transitions) at 186 MFLOP/token training and 62 MFLOP/token forward-only, with
an assumed 120 TFLOPS effective A100 bf16 throughput. The rollout it is measured
against costs 963 TFLOP, or **8.0 s**:

| Strategy | Tokens forwarded | Cost/epoch | Correctness |
|---|---|---|---|
| Exact per-timestep window | 67.1M | 104 s | exact, 13x the rollout cost |
| §5's stored detached KV | 65.5k | 0.10 s | **stale**: KV built by θ_old, consumed by θ_k |
| Burn-in, detached prefix | 131k | 0.14 s | exact forward, truncated gradient |
| **Burn-in, gradients through prefix** | 131k | **0.20 s** | **exact** (chosen) |

The chosen row costs **0.10 s/epoch more** than §5's proposal — 1.3% of the
rollout it accompanies — and removes the staleness entirely. R2D2 (Kapturowski
et al. 2019) introduced burn-in for precisely the representational-drift failure
§5's approach invites.

And here it is genuinely exact, which an RNN cannot achieve: a transformer with
a 1024-wide sliding window has **no infinite recurrence**. Its state at step `t`
is a deterministic function of the last 1024 tokens. Prepend 1023 tokens, apply
a sliding-window causal mask, and every gradient-carrying position sees
*precisely* the window it saw at rollout, evaluated at current weights — which
is exactly what PPO's importance ratio requires.

### Gradients flow through the burn-in prefix (departing from R2D2)

R2D2 detaches its burn-in state. This design does not, and the reason is
structural: R2D2 detaches because an RNN's alternative is *unbounded* backprop.
A 1024-window transformer's backprop is already bounded, so detaching would
truncate a path that is genuinely part of ∂loss/∂θ — producing a biased
gradient, not an equal-but-cheaper one.

Cost of not detaching: **0.07 s/epoch** (0.20 s versus 0.14 s in the table
above) against an 8.0 s rollout — and one forward path in the code instead of
two.

### SDPA with an explicit mask, not FlexAttention

Verified on the installed torch 2.13: `flex_attention` raises
`NotImplementedError: FlexAttention does not support backward on CPU`. Forward
works, backward does not.

CLAUDE.md mandates CPU-only unit tests, so FlexAttention as the training path
would make the chunked-forward tests unrunnable — the single most important
tests in this sub-project.

`F.scaled_dot_product_attention(..., attn_mask=bool_mask, enable_gqa=True)` was
verified to work forward *and* backward on CPU with the composed
`causal ∧ sliding_window ∧ same_episode` mask. Mask cost is 8.4 MB at batch 2,
~34 MB at batch 8. One code path, CUDA and CPU alike, fully testable.

`enable_gqa=True` also means no hand-written `repeat_kv`, which eliminates the
`x.repeat()` query-head-permutation bug at the source rather than testing for it.

### Rollout uses `torch.no_grad()`, not `torch.inference_mode()`

This is a stated exception to CLAUDE.md's "every eval path is `model.eval()` +
`@torch.inference_mode()`".

Verified: a tensor created under `inference_mode` raises
`RuntimeError: Inference tensors cannot be saved for backward` when it later
enters an autograd graph. Rollout latents are exactly that — produced during
rollout, consumed as inputs to the PPO update's forward. The failure would
appear at the first update, not at rollout, and `.clone()`-on-write is a
silent-if-forgotten workaround.

`no_grad` tensors are ordinary tensors, safe as autograd inputs. The module has
no dropout and no BatchNorm, so train/eval mode is behaviorally irrelevant here;
`step()` is decorated `@torch.no_grad()` so the rollout loop cannot get it wrong.

## Model architecture

All parameter and memory figures below are from
`transformer-architecture/scripts/config_budget.py`, not estimated.

| field | value | rationale |
|---|---|---|
| `d_model` | 512 | aspect ratio 64, mid-range for shipping models |
| `n_layers` | 8 | |
| `n_heads` / `head_dim` | 8 / 64 | 64 is standard below 1B params |
| `n_kv_heads` | 2 (4x GQA) | 4x is modal; not memory-forced (see above) |
| `d_ff` | 1408 | `round_to_128(8/3 · d_model)`, SwiGLU |
| `context_len` | 1024 | sliding window; ≈6.8 min of gameplay |
| `rope_theta` | 1e4 | correct for a natively-trained 1k context |
| `qk_norm` | true | see deviations |

**Parameters: 22.6M backbone + 1.09M input adapter + 4.1K heads ≈ 23.7M.**
KV cache 4.00 KiB/token → **256 MiB at 64 envs x 1024 context, bf16**.

### Input adapter (§5's "Latent Space Normalization")

```
latent (2048)  ──► frozen affine norm (register_buffer) ──┐
aux_state (32) ───────────────────────────────────────────┤
action_embed(a_{t-1}) (32)  ──────────────────────────────┼─► concat (2120) ─► Linear ─► d_model
reward_feat(r_{t-1}) (8)  ─────────────────────────────────┘
```

- Normalizer mean/std come from the encoder repo's `latent_stats.json` and are
  **frozen buffers**, not running statistics. Running stats under a shifting
  policy are a non-stationarity source, and §5 asks for an explicit affine
  layer, not an adaptive one.
- `nn.Embedding(8, 32)`: 7 real actions plus index 7 = `EPISODE_START`. At
  reset there is no previous action, and feeding action 0 there teaches a lie.

### Block

Pre-norm RMSNorm → GQA + RoPE → residual; pre-norm RMSNorm → SwiGLU → residual.
No linear biases, no dropout.

Init `N(0, 0.02)`, output projections (`o_proj`, `down_proj`) scaled
`1/sqrt(2·n_layers)`. Actor head orthogonal init gain 0.01 (near-uniform initial
policy, standard PPO); critic head gain 1.0.

### Two deviations from the transformer skill's defaults

1. **QK-norm at 23.7M params**, where the default threshold is ~1B. Reason: §3's
   own stated fear — "RL value-loss spikes ... collapsing the attention weights"
   — and attention-logit growth is the documented mechanism (Wortsman et al.:
   every run above 1e4 attention logits diverged). Cost is 1,024 parameters
   total across the model.
2. **AdamW betas (0.9, 0.999), eps 1e-5, grad clip 0.5**, rather than the LM
   defaults (0.9, 0.95) / clip 1.0. Reason: this is PPO, not LM pretraining;
   these are PPO convention and RL gradients are noisier.

### RoPE precision at large absolute positions

Episodes run to 163,840 steps. At that magnitude fp32 spacing is ~0.004 rad,
which corrupts the highest-frequency RoPE channel. Compute
`angle = (t · inv_freq) mod 2π` in float64, then cast to the compute dtype.
Exact at any `t`, costs nothing.

Cached keys are **rotated once, on write, at their own absolute position**, and
never re-rotated.

### Considered and deferred

GTrXL-style GRU gating on residual connections (Parisotto et al.) — the known
fix if pre-norm proves insufficient for RL stability. Not built now: pre-norm
plus QK-norm first, gating only if the logged instability indicators demand it.

## State machine and APIs

```python
@torch.no_grad()                      # NOT inference_mode -- see rationale above
def step(self, latent, aux_state, prev_action, prev_reward, cache) -> StepOutput

def forward_chunk(self, latent, aux_state, prev_action, prev_reward,
                  abs_pos, episode_id, burn_in: int) -> ChunkOutput
```

### `step()` — rollout

One token in, `(logits (N, 7), value (N,))` out, batch = 64 envs. State is a
`RolloutCache`: per-layer ring buffers of shape
`(n_layers, N, n_kv_heads, 1024, head_dim)` for K and V, plus per-env `length`
and `abs_pos`.

Three details, each a known silent bug:

- **`is_causal=False`.** Query length is 1 against `cache_len` keys;
  `is_causal=True` makes it attend to position 0 only — prefill perfect,
  rollout garbage. An explicit slot-validity mask is passed instead.
- **Keys rotated once on write**, never on the way out of the cache.
- **Ring-buffer slots are not in temporal order** after wraparound, and that is
  fine: RoPE phase is baked into each key at write time, and the mask keys off
  slot validity, not slot index.

`cache.reset(done_mask)` clears validity and resets `abs_pos` to 0. Because the
cache is reset at episode end, episode masking during rollout is automatic —
only `forward_chunk` needs an episode mask, since only a training chunk can
straddle a boundary.

### `forward_chunk()` — PPO update

`L = burn_in + T = 1023 + 1024 = 2047` tokens in. Mask is
`causal ∧ (q_pos − kv_pos < 1024) ∧ same_episode`. Loss is computed on the last
`T` positions only; `abs_pos` comes from the recorded rollout positions so RoPE
matches exactly.

PPO minibatches are **sequences, not transitions**: 64 envs → 8 minibatches of
8 sequences.

### Rollout buffer contract

Stores **fp16 latents**, not frames and not KV states: 4,096 B/step versus
23,040 B/step for frames (5.6x smaller), and the frozen CNN makes latents
deterministic so there is nothing to recompute. ~537 MB for 64 envs x 2047
steps.

## Environment interface contract

These cease to be open questions. The env sub-project must honor them.

| Item | Bound value |
|---|---|
| Action space | `Discrete(7)`, fixed order `DOWN, LEFT, RIGHT, UP, A, B, START` |
| Action embedding rows | 8 — index 7 is `EPISODE_START` |
| Frame skip | 24 emulator ticks per agent step (≈0.4 s) |
| Parallel envs | 64 |
| Obs: frame | `(1, 144, 160)` uint8 **[0, 255]**, unscaled |
| Obs: aux_state | `(32,)` float32, every component in [-1, 1], **versioned** layout |
| Reward | float32, clipped to [-1, 1] per §4 |
| Episode | env exposes a monotonic `episode_id` per env, not just `done` |
| Rollout | `n_steps = 1024` per env per update |

The frame contract matches `load_frozen_encoder`'s published `input_shape_nchw`
and `input_scale` exactly — uint8 [0, 255] cast to float, **not** rescaled to
[0, 1]. The exported encoder has Conv+BN fused, so no BatchNorm remains to
absorb a wrong input scale and the features would be wrong with no error raised.

The aux_state *field layout* (which RAM address maps to which slot) belongs to
the env spec. This spec binds only shape, range, and the version field, so a
layout change fails loudly rather than silently retraining on shuffled
semantics.

The frozen CNN runs **batched across all 64 envs on the GPU**, not per-env. At
14.69 GFLOP/frame that is the difference between a working rollout and a 64x
serial stall.

## Testing strategy

CPU-only, seeded, tiny configs (`d_model=32, n_layers=2, n_heads=2,
n_kv_heads=1, context_len=8`) per CLAUDE.md.

### The load-bearing test

`test_incremental_step_matches_full_chunk_forward` — push T tokens through
`step()` one at a time, push the same T through `forward_chunk()`, assert
`allclose` at 1e-5. This single equivalence catches `is_causal=True` during
decode, ring-buffer indexing errors, RoPE position drift, and mask errors
simultaneously.

### The four silent bugs

- `test_rope_attention_score_depends_only_on_relative_distance` (RoPE applied
  after the head split).
- `test_rope_uses_halves_pairing_not_interleaved`, pinned to an exact tensor so
  a refactor that swaps the convention turns red.
- `test_gqa_with_two_kv_heads_matches_manual_expand_reshape`.
- `is_causal` misuse is covered by the equivalence test above.

### Design-specific

- `test_ring_buffer_evicts_oldest_when_context_exceeded`: run `C+50` steps,
  assert output equals a fresh model fed only the last `C` tokens.
- `test_cache_reset_makes_step_independent_of_pre_reset_history`.
- `test_episode_mask_blocks_attention_across_a_boundary`: replace
  pre-boundary tokens with noise, assert post-boundary outputs unchanged.
- `test_forward_chunk_with_burn_in_matches_recorded_rollout_outputs` — the
  headline correctness claim of the burn-in scheme.
- `test_rope_angle_is_exact_at_large_absolute_positions`: assert the angle at
  `t=163,840` matches a float64 reference at `rel=1e-6`. Fails without the
  mod-2π computation.

### CLAUDE.md non-negotiables, as tests

- `test_latent_normalizer_stats_are_buffers_not_parameters` — present in
  `state_dict()`, absent from `parameters()`.
- `test_actor_head_is_a_bare_linear_emitting_raw_logits`.
- `test_every_parameter_receives_gradient`.
- `test_output_projections_are_scaled_by_one_over_sqrt_two_n_layers`.

### Slow tier

`@pytest.mark.slow`:
`test_tiny_policy_overfits_a_single_batch_to_near_zero_loss` in ~200 steps —
CLAUDE.md's stated first move when a model trains but will not learn.

### Gates

Run `pytest-expert/scripts/audit_tests.py tests/` after adding tests. Break-and-
revert the two headline tests (step/chunk equivalence, burn-in exactness) and
report which were verified that way.

## Observability

Loss and grad-norm are both late indicators. Logged from day one, to W&B plus
JSON-lines:

- **Divergence predictors**: per-layer `attn/logit_max` (Wortsman et al.: every
  run above 1e4 diverged) and `resid/final_layer_output_norm` (led divergence by
  20-30% of training in Chameleon's runs).
- **PPO health**: policy entropy, approximate KL, clip fraction, value
  explained-variance.
- **Is the memory doing anything?** Attention mass histogrammed by distance
  bucket `[1, 2-8, 9-64, 65-256, 257-1024]`, plus a periodic attention-distance
  heatmap PNG to W&B.

That last metric is the honest check on this component's entire premise. If the
model never attends past 8 steps, the 1024-context design is dead weight, and
that should surface in hour one rather than week two.

## Out of scope

- The PyBoy environment, its RAM readers, and the reward function's semantics.
- The PPO loss, GAE, and the training loop that drives updates.
- Any offline or supervised pretraining of this model.
- GTrXL gating, and any context length beyond 1024.

## Open questions / future extensions

- Whether `n_steps = 1024` per env is the right rollout length is a PPO-side
  tuning question; the module supports any `T` with `burn_in = context_len - 1`.
- If the attention-distance histogram shows mass concentrated below ~64 steps
  after a real run, `context_len` should come down and the freed budget go to
  depth or width. That is a measurement to make, not a decision to pre-commit.
- Whether the aux progress vector eventually subsumes enough state that the
  latent channel can shrink is a question for after the first working run.
