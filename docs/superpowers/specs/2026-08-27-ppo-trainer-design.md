# PPO Trainer Design (Sub-project B)

Status: implemented and merged. `src/ppo/` exists; §8's gates are what remain
before the first paid run. See §12 for the gaps carried out of implementation.
Date: 2026-08-27.

Consumes the two merged sub-projects:

- `src/sequence_model/` — `RecurrentTransformerPolicy`, `RolloutCache`,
  `build_policy_checkpoint_state` / `restore_policy_checkpoint`.
- `src/pokemon_env/` — `VecPokemonEnv`, `build_subprocess_vec_env`,
  `LatentEncoder`, `build_env_checkpoint_state` / `restore_env_checkpoint`.

Their specs are
`docs/superpowers/specs/2026-08-26-temporal-sequence-model-design.md` and
`docs/superpowers/specs/2026-08-26-pokemon-env-design.md`. Every item in both
"Handoff" sections is answered here; where this spec contradicts them, this
spec wins and the contradiction is called out explicitly.

## Scope

**In:** the PPO package (rollout, GAE, losses, update, trainer, config, CLI),
checkpoint/resume orchestration across the policy and env halves, the telemetry
consumers both handoffs assign to PPO, the two observability gaps the env spec
carried out of implementation, and the four pre-flight gates measured on a real
CUDA pod.

**Out:** launching or tuning the multi-day training run. Reward-weight tuning,
the `/√k` exploration-decay constant, and whether `init.state` should advance
further are all measurements to make *from* a run, not decisions this spec
pre-commits.

**Definition of done:** the four gates in §8 pass, including a 50-update live
smoke run on a cheap pod. Launching the real run is then one command and a
config file.

## 1. Framework and verified interfaces

PPO is hand-rolled in PyTorch. `stable-baselines3` is **not** a dependency and
is not added — its `RecurrentPPO` cannot express the burn-in-prefixed chunked
update this policy requires. SB3 is used only as a source of default values,
cited below where it is.

Verified by introspection against the installed torch 2.13.0, not from memory:

- `torch.nn.attention.SDPAParams(query, key, value, attn_mask, dropout,
  is_causal, enable_gqa)` — 7 positional arguments.
- `torch.nn.attention.can_use_flash_attention(params, debug=False) -> bool` and
  `can_use_efficient_attention(params, debug=False) -> bool`. With
  `debug=True` they print the reason a backend is rejected.
- `torch.nn.attention.SDPBackend` members: `ERROR`, `MATH`, `FLASH_ATTENTION`,
  `EFFICIENT_ATTENTION`, `CUDNN_ATTENTION`, `OVERRIDEABLE`. `CUDNN_ATTENTION`
  did not exist when the sequence-model handoff was written and is a third
  candidate backend.
- `torch.nn.attention.sdpa_kernel(backends, set_priority=False)`.

Fetched from SB3's `PPO.__init__` via context7, and adopted as defaults:
`clip_range_vf=None`, `normalize_advantage=True`, `target_kl=None`,
`max_grad_norm=0.5`, `vf_coef=0.5`, `gae_lambda=0.95`.

## 2. Package layout

New package `src/ppo/`, registered in `[tool.hatch.build.targets.wheel]`.

| Module | Responsibility |
|---|---|
| `config.py` | `PPOConfig` frozen dataclass + `load_config`, from `configs/ppo.yaml` |
| `buffer.py` | `RolloutBuffer` — GPU-resident, rolling, `burn_in + n_steps + 1` slots per env |
| `rollout.py` | `collect_rollout(...)` — env → encoder → `policy.step` → buffer |
| `gae.py` | `compute_gae(...)` — pure tensor function, no modules |
| `losses.py` | `ppo_losses(...)` — pure; clipped policy, value, entropy |
| `normalizer.py` | `ReturnScaler` — running std of the discounted return |
| `update.py` | `run_update(...)` — the `π_old` pass, GAE, minibatch loop, optimizer steps |
| `trainer.py` | `PPODeps` + `run_training(deps)` — outer loop, cadence, telemetry |
| `checkpoint.py` | Paired policy+env checkpoint orchestration and the manifest |
| `telemetry.py` | Per-update scalar aggregation and artifact rendering |
| `preflight.py` | The four gates of §8, runnable standalone |
| `cli.py` | `pokemon-ppo` entrypoint in `[project.scripts]` |

Modules touched outside the new package:

- `src/pokemon_env/subprocess_backend.py` — new `Command.STATS` (§7).
- `src/pokemon_env/telemetry.py` — `exploration_heatmap` rewritten (§7),
  `rollout_metrics` extended.
- `src/pokemon_env/config.py` — delete `seed` and `frozen_encoder_repo_id` (§3).
- `src/observability/tracking.py` — `WandbRun` gains config, x-axes, resume,
  and context-manager semantics (§7).

## 3. Configuration

`PPOConfig` mirrors the existing `EnvConfig` / `TrainingConfig` pattern: frozen
dataclass, `yaml.safe_load`, unknown-field rejection, validation in
`__post_init__`.

| Field | Default | Source / reason |
|---|---|---|
| `n_steps` | 1024 | §5. A parameter, never a constant |
| `n_epochs` | 3 | 24 optimizer steps per update |
| `minibatch_envs` | 8 | ~2 GB of update activations |
| `gamma` | 0.997 | PWhiddy v2 |
| `gae_lambda` | 0.95 | SB3 default |
| `clip_range` | 0.2 | SB3 default |
| `clip_range_vf` | `None` | SB3 default; see §5 on why it stays off here |
| `ent_coef` | 0.01 | PWhiddy v2 |
| `vf_coef` | 0.5 | SB3 default |
| `max_grad_norm` | 0.5 | SB3's PPO default, **not** CLAUDE.md's transformer 1.0 |
| `lr` | 3e-4 | SB3 default |
| `warmup_steps` | 100 | Linear, then constant |
| `abort_approx_kl` | 0.5 | Divergence guard, §6 |
| `max_nan_minibatches_per_update` | 3 | §6 |
| `seed` | 0 | Action sampling; moved off `EnvConfig` |
| `frozen_encoder_repo_id` | `objones25/pokemon-contrastive-encoder` | Moved off `EnvConfig` |
| `frozen_encoder_revision` | `None`, rejected in `__post_init__` | Effectively required; §4 |
| `checkpoint_dir` | `/workspace/checkpoints` | RunPod network volume |
| `keep_last_n` | 3 | ~1.65 GB retained |
| `checkpoint_every_updates` | 25 | Provisional; gate 2's measured iteration time replaces it. §6 binds the *requirement* — ≤20 minutes of work lost — not the integer |
| `artifact_every_updates` | 25 | §7 |
| `hub_snapshot_every_updates` | 75 | Provisional, ~hourly; same treatment. §6 |

`n_envs` is deliberately **not** a `PPOConfig` field. It lives on `EnvConfig`,
where the env sub-project put it, and gate 2 sets it there.

**`max_grad_norm = 0.5` is a deliberate deviation** from CLAUDE.md's stated
transformer default of 1.0. CLAUDE.md's value is for language-model
pretraining; 0.5 is the PPO convention and SB3's default. Stated here because
CLAUDE.md requires a reason for changing a default.

**Two `EnvConfig` fields are deleted**, closing the env spec's third known gap:
`seed` (PyBoy is deterministic given `init.state` plus an action sequence; the
only randomness is action sampling, which is PPO's) and
`frozen_encoder_repo_id` (the env encodes nothing). Both move to `PPOConfig`.
Deleting rather than leaving them dead is what that handoff item asked for.

`frozen_encoder_revision` is **required and pinned**, and the resolved commit is
written into every checkpoint. A mid-run push to the encoder repo must not be
able to change the features underneath a running agent.

## 4. The rollout

Per env step, in this exact order:

1. `vec_env.step(actions)` → `VecStep` (numpy, CPU).
2. `encoder.encode(step.frames)` → `(n_envs, 2048)` on GPU. `LatentEncoder.encode`
   is `@torch.no_grad()`, deliberately not `inference_mode` — these latents are
   saved for backward at the update.
3. `policy.step(latent, aux, prev_action, prev_reward, cache)` inside
   `torch.autocast(device.type, torch.bfloat16)`. The cache is allocated bf16, and
   `new_cache`'s contract requires a matching autocast context or SDPA raises a
   dtype mismatch.
4. Sample the action from `out.logits` using a `torch.Generator` seeded from
   `PPOConfig.seed`.
5. Write into the buffer.
6. **`cache.reset(step.done)` — after this step, never before the next.** This
   is the ordering contract named in the sequence-model handoff: the terminal
   observation must attend to the episode it belongs to.
7. `prev_action, prev_reward = action, reward`, **except** where `done`:
   `prev_action ← config.episode_start_action` (index 7), `prev_reward ← 0.0`.
   Autoreset is next-step, so the action taken at the terminal step is
   meaningless as context for the fresh episode.

### Buffer contents

fp16 latents (537 MB at 64 envs × 2048 slots × 2048 dims), fp32 aux, and
`action`, `prev_action`, `prev_reward`, `reward`, `done`, `episode_id`,
`abs_pos`.

It also stores the rollout-recorded `logprob` and `value` — 262 KB — which are
**never** used in the importance ratio or as the GAE baseline. They exist
solely to compute `staleness/logprob_l1` against the recomputed `π_old` (§7).
That is the measurement which justified requirement 1 in the first place; here
it runs continuously instead of once.

### Warmup

`burn_in = policy_config.context_len − 1 = 1023`, so update 0 has no prefix.
Rather than let `L` vary — which recompiles `torch.compile` and re-tunes
`cudnn.benchmark`, both of which CLAUDE.md wants pinned — the run opens with a
1023-step warmup rollout that is burn-in for update 0 and a gradient target for
nothing. It costs 0.6% of one 163,840-step episode, once.

A resume that restores the KV cache skips the warmup. A resume that dropped the
cache (a `context_len` change at a curriculum boundary, §9) redoes it.

## 5. The update

### Indexing contract

Buffer capacity is `burn_in + n_steps + 1 = 2048` slots per env. In absolute
step numbers, update `k` starting at `n`:

- burn-in: `[n − 1023, n − 1]` (1023 slots)
- trained region: `[n, n + 1023]` (1024 slots)
- bootstrap: `n + 1024` (1 slot)

`forward_chunk` runs at `L = 2048, burn_in = 1023` and emits 1025 outputs. The
first 1024 carry the loss; the 1025th supplies `V(s_T)` for the final GAE term
and nothing else. Update `k+1` shifts by exactly 1024, so its first trained slot
*is* update `k`'s bootstrap observation, and **every observation is trained
exactly once**. Shapes are fixed across all updates.

Rollout lengths follow from that, and are not all equal: the warmup collects
`burn_in = 1023` observations, update 0's rollout collects `n_steps + 1 = 1025`
to fill the bootstrap slot, and **every later rollout collects `n_steps = 1024`**
because the previous update's bootstrap observation is retained and becomes the
new region's first trained slot. Steady-state cost is 1024 env steps per update.

The shift equals `n_steps` by necessity: `burn_in = context_len − 1` is the
minimum prefix giving every trained position a full `context_len` window, and
once burn-in is fixed, any shift other than `n_steps` trains observations twice
or skips them.

### Recomputing `π_old` and `V_old`

At update start, one `no_grad` `forward_chunk` sweep over all envs in
`minibatch_envs`-sized chunks, **under the same `torch.autocast` context as the
training step**. It yields `logprob_old` (gathered at the actions taken) and
`V_old`. GAE runs off `V_old`.

The rollout-recorded logits and values are never the ratio denominator or the
GAE baseline. The KV cache is carried across update boundaries, so the
behaviour policy is not exactly `π_θ_old`; at production depth and width the
resulting bias is the same order as the clip threshold.

**This spec deliberately does not fuse the `π_old` pass into epoch 1**, which
both handoffs suggest ("it fuses with epoch 1's forward pass"). Fusion is only
valid if epoch 1 takes no optimizer step until all of its minibatches have been
seen, which forces gradient accumulation and drops the update from 24 optimizer
steps to 4. The ~1.7% of a rollout that a separate pass costs buys back both
the gradient-step count and an invariant that is true by construction rather
than by scheduling.

### GAE and the return scaler

Standard GAE over the 1024 trained transitions, bootstrapping from the 1025th
output. A `done` inside the region truncates: no bootstrap across an episode
boundary, and `episode_id` — not `done` alone — is the authority, because a
respawned worker also produces a discontinuity.

Advantages are normalized **once per update over all `n_envs × n_steps`
transitions**, not per minibatch. Per-minibatch normalization makes each
minibatch's targets depend on which envs landed in it.

**`ReturnScaler`.** Rewards are clipped to `[0, 1]` per step and `γ = 0.997`,
so value targets can reach the tens against a critic the architecture plan
calls hypersensitive to input scale. Value targets are divided by a running std
of the discounted return, with **no mean shift**, so advantage signs are
preserved. Its state (count, running variance, the current scale) is three
floats and goes in the checkpoint. SB3's own `clip_range_vf` docstring —
*"IMPORTANT: this clipping depends on the reward scaling"* — is why value
clipping stays off until this scale is pinned and observed.

### Minibatching and the invariant

`n_epochs` × `n_envs / minibatch_envs` minibatches, one optimizer step each —
24 steps per update at the defaults. A minibatch is a **subset of envs at full
`T`**, never a slice of time; the burn-in binds the time axis.

At **(epoch 1, minibatch 1)** the policy has not yet been updated, so
`max|ratio − 1|` must be **exactly 0**. This is a hard `assert`, not a logged
metric. Its discriminating mutation is "use the rollout-recorded logprobs",
which must turn the test red.

For every later minibatch the max ratio is logged, not asserted — it is
supposed to grow. 24 optimizer steps per 65,536 transitions is 1 step per 2,731
transitions, against PWhiddy's 1 per 512; the recurrent structure forces coarser
batching, and `n_epochs` and `minibatch_envs` are the levers if learning is too
slow.

## 6. Checkpoint, resume, failure handling

### Three files, manifest last

- `policy_update{N}.pt` — `build_policy_checkpoint_state(...)` via
  `checkpointing.io.save_checkpoint`. Includes the `RolloutCache`.
- `env_update{N}.pt` — `build_env_checkpoint_state(...)`, which already covers
  emulator state, the §4 `max_historical` baselines, `needs_reset`, and the
  `init_state_hash` guard.
- `manifest_update{N}.json` — written **last**, and the commit point.

The manifest names both files with their byte sizes, plus `update`,
`global_step`, `ReturnScaler` state, torch version, git commit, the resolved
frozen-encoder revision, and the W&B run id. Resume scans manifests descending
and takes the first whose two files exist at the recorded sizes.

Per-file atomicity already exists in `save_checkpoint` (`.tmp` + `os.replace`).
The failure it cannot see is *one of two* landing; the manifest is what makes
the pair atomic.

Three globs — `policy_update*`, `env_update*`, `manifest_update*` — each go
through `prune_checkpoints(keep_last_n=3)`. This is why `checkpointing.io`
takes the glob as a parameter.

### Ruling: the KV cache is checkpointed

The sequence-model handoff left this open because a cache without emulator
state is incoherent. `build_env_checkpoint_state` saves emulator state and the
manifest makes the two atomic, so that objection is answered.

It costs 256 MiB against 10.7 MB of env state. The deciding argument is not the
65,536 env steps of recovered context — it is that **resuming with an empty
cache puts a value-loss spike in the telemetry after every restart**, and on a
preemptible multi-day run that makes the curves being read to judge reward
design harder to trust.

### Cadence

Bound as a requirement, not a number: **at most 20 minutes of work lost**.
Gate 2 measures iteration time; `checkpoint_every_updates` follows from it.

**Checkpoints live on the RunPod network volume, not the Hub.** Only the final
policy and a roughly hourly snapshot are pushed. Writing every checkpoint to the
Hub over 48 hours would walk back into the 256-commits/hour rate limit this
project has already hit on `objones25/pokemon-frames`.

### Failure handling

| Failure | Handling |
|---|---|
| Worker death | `SubprocessBackend._restart()` already respawns and bumps the episode offset. PPO logs `env/respawns`; the bumped `episode_id` is what stops `build_chunk_mask` attending across the discontinuity, and that is a test, not an assumption |
| CUDA OOM | One full-size update as a memory probe *before* the loop, following `contrastive_pretrain.run_memory_probe`. Fixed shapes mean OOM surfaces in 30 seconds or not at all |
| NaN/inf loss | Skip the minibatch, zero grads, log `train/skipped_minibatches`; **3 within one update aborts.** A bare abort loses 40 unattended hours to one bad batch; no guard corrupts the weights |
| `approx_kl > abort_approx_kl` | Abort. The checkpoint is intact |
| W&B outage | Already swallowed by `WandbRun`; JSON-lines on the volume is the durable record |
| Pod preemption | Manifest resume, automatic in `cli.py` unless `--fresh` |

**Resume is state-faithful, not bit-reproducible.** RNG state round-trips via
`capture_rng_state` / `restore_rng_state`, but PyBoy across 64 respawned
subprocesses does not reproduce a byte-identical step ordering. No test may
assert bitwise equality across a restart. The assertions are that weights,
optimizer moments, cache contents, emulator saves, reward baselines, and
`ReturnScaler` state match exactly, and that training continues without a
discontinuity in the loss curve.

## 7. Observability

Division of labour: **numbers moving over time go to W&B; events with a cause
go to the JSON-lines log.** Nothing is logged per env step — 65,536 of those
per update is a self-inflicted outage.

### Closing the env spec's two known gaps

**One new worker command, `Command.STATS`, called once per update** — not per
step. It returns, per env: the packed `coord_key` int32 array (`ram.coord_key`
packs `(map_id, x, y)` injectively), `badge_count`, `event_flag_count`, steps
since reset, and the lengths of episodes completed since the last call. About
80 KB across 64 envs, against the 10.7 MB `STATE_DICT` round trip that was
previously the only route.

That single command supplies every one of the ~4 missing `rollout_metrics`
scalars and all of the heatmap's data.

**`exploration_heatmap` is rewritten.** It no longer folds `x` and `y` mod 16 —
which collided most distinct coordinates and is why the current artifact does
not match what the env spec promised. It takes coord keys, unpacks them, and
renders true `(x, y)` within each map, tiling the top 12 maps by unique-
coordinate count into one image. Plus `explore/unique_coords_total` and
`explore/unique_maps` as scalars.

### `WandbRun` changes

The existing wrapper would have fragmented this run. It is extended to:

- accept `config` and pass it to `wandb.init` (WB002 — currently metrics arrive
  with no hyperparameters at all);
- accept `step_metrics` and declare x-axes with `define_metric`, so `log` never
  passes `step=` (WB006);
- accept a **stable run id** persisted beside the checkpoints, with
  `resume="allow"`. **Without this, every pod preemption starts a new W&B run**
  and a 48-hour curve arrives as several disconnected fragments;
- support `__enter__` / `__exit__`, marking the run failed on an exception
  rather than leaving it "running" forever (WB003);
- pass `exc_info=True` on its two failure warnings (LOG006), so a W&B death at
  hour 30 says whether it was auth, network, or disk.

Its existing guarantee is preserved: `log` and `finish` never raise, because
W&B is a live view and not a correctness dependency.

### What is logged

**W&B config:** the three merged dataclasses (`PPOConfig`, `EnvConfig`,
`PolicyConfig`), git commit, torch version, resolved encoder revision, GPU
name, and **the §8 gate results** — the chosen SDPA backend and measured
throughput become part of the run record. No credential may appear here; a test
asserts no secret-shaped key reaches it.

**W&B history, per update**, with `train/update` and `train/env_step` as
declared x-axes:

- `loss/policy`, `loss/value`, `loss/entropy`, `loss/total`
- `ratio/max_abs_dev_epoch1_mb1` (also asserted 0), `ratio/max_abs_dev`,
  `clip_fraction`, `approx_kl`
- `staleness/logprob_l1` — the recomputed-vs-recorded `π_old` diagnostic
- `value/explained_variance`, `value/return_scale`
- `train/grad_norm`, `train/lr`, `train/skipped_minibatches`
- `system/peak_vram_gb`, `perf/env_steps_per_sec`, `perf/iteration_s`
- Leading indicators from `sequence_model.telemetry`: `attn/logit_max`,
  `model/residual_norm`. The sequence-model spec is explicit that these move
  before loss and grad norm do.
- Env: `reward/mean`, the per-component breakdown from `last_components`,
  `env/clip_fire_rate`, `env/respawns`, `progress/badges`,
  `progress/event_flags`, `explore/unique_coords_total`,
  `explore/unique_maps`, `episode/length_mean`

**W&B summary**, set explicitly rather than left as last-value:
`best/badges`, `best/unique_coords`, `best/reward_mean`.

**W&B artifacts, every `artifact_every_updates`:** the exploration heatmap, a
frame contact sheet, and the attention-distance histogram — the last doubling
as the curriculum gate of §9.

**JSON-lines log — events only:** run started (with config summary and the
dashboard URL), gate results, resumed-from-update-N, checkpoint written, worker
respawned (WARNING, with env index and cause), NaN minibatch skipped (WARNING),
abort conditions (ERROR with `exc_info`), finished. `wandb_run_id` and `update`
bind into the logging context so a log record can be matched to a dashboard
point without guessing.

### Audit baseline

`audit_observability.py src/` currently reports **12 findings**
(`LOG004:5, LOG006:3, LOG007:4`). This sub-project takes it to **7**:

- `tracking.py:50,56` — LOG006, fixed with `exc_info=True`.
- `subprocess_backend.py:157` — a false positive on the heuristic (it forwards
  the error to the parent rather than swallowing it), but the traceback stays
  in the worker. Fixed by `logger.exception(...)` before the send.
- `subprocess_backend.py:270,340` — genuinely benign (closing an already-closed
  pipe; shutdown against a dead worker). Marked `# obs: allow LOG007`, no code
  change.
- The five `LOG004` prints are in `data_collection` and are out of scope.

The number 9 is stated so it cannot drift silently.

## 8. Pre-flight gates

Run in order by `preflight.py`, before any long run.

**Gate 1 — SDPA backend.** `attention.py` calls
`F.scaled_dot_product_attention(..., enable_gqa=True)`, so the gate must use
the model's real asymmetric shapes: `q` at `(8, 8, 2048, 64)`, `k` and `v` at
`(8, 2, 2048, 64)` — `n_kv_heads = 2` — with the real bool mask and
`enable_gqa=True`. `enable_gqa` is itself a `SDPAParams` field and can
disqualify a backend on its own, so a gate run with symmetric head counts would
measure something the model never executes.

Call `can_use_flash_attention(p, debug=True)` and
`can_use_efficient_attention(p, debug=True)`, logging what `debug` prints. Then
time one `forward_chunk` under each backend forced via `sdpa_kernel`, recording
peak memory. **Pass:** a non-`MATH` backend is available, or `MATH`'s measured
peak fits the pod with margin. A materialized bool mask rules out
FlashAttention, and `MATH` would materialize roughly 537 MB of scores at
`(B=8, H=8, L=2048)` in bf16; if neither alternative is usable, the restructure
decision surfaces before the money is spent. `CUDNN_ATTENTION` is a candidate
the original handoff did not know about.

**Gate 2 — rollout throughput** at `n_envs ∈ {16, 32, 64}` on the target pod's
actual vCPU count. This answers the env spec's own open question — *"whether 64
envs is right for our per-step cost"* — with a number. Its measured iteration
time sets `checkpoint_every_updates` and `hub_snapshot_every_updates`.

VRAM is not expected to bind: update activations at 8 envs × `L = 2048` are
~2 GB, the KV cache 256 MiB, the latent buffer 537 MB. vCPU is the constraint,
because 64 PyBoy workers are 64 processes.

**Gate 3 — memory probe.** One full-size update at the chosen shapes, peak VRAM
recorded.

**Gate 4 — a 50-update live smoke run** on a cheap pod (~$1). Pass conditions:
the epoch-1-minibatch-1 invariant holds every update, all losses finite,
`explained_variance > 0`, reward telemetry non-degenerate, one checkpoint
written and resumed from successfully. **This is the sub-project's acceptance
gate.**

## 9. Room for longer context later

Curriculum or transfer to a longer context is not built now, but nothing
forecloses it:

1. Nothing hard-codes 1024 or 2048. `burn_in = policy_config.context_len − 1`
   and `capacity = burn_in + n_steps + 1`; buffer, mask, and chunk shapes all
   derive. A stage-2 run changes two config numbers.
2. `restore_policy_checkpoint`'s config-drift check **already** permits
   `context_len` and `rope_theta` to change, because neither changes a
   parameter shape. Stage-1 weights load into a longer-context stage-2 run by
   design.
3. The KV cache does not survive such a change — `rebuild_cache` validates
   capacity. The loader reports this in words and resumes with an empty cache
   plus a warmup, rather than raising.
4. At `context_len = n_steps = 2048`: `L = 4096`, mask 134 MB per minibatch, KV
   cache 512 MiB, buffer 1.07 GB. Same pod.
5. Two guardrails. Raising `context_len` past ~2048 **requires** a `rope_theta`
   decision — `1e4` at long native context with no scaling plan is a named
   failure mode. And the **attention-distance histogram is the gate**: the
   sequence-model spec says that if mass concentrates below ~64 steps,
   `context_len` should come *down*, not up. Longer context is earned by that
   measurement.

## 10. Testing

CPU-only, seeded, tiny configs by default: `d_model=32, n_layers=2,
context_len=8, n_envs=4, n_steps=8`. A hand-written `FakeVecEnv` typed against
the env's Protocol and a tiny `nn.Module` encoder — hand-written fakes, not
`mock.patch` stubs, matching the existing suite.

### The load-bearing test

**Epoch-1-minibatch-1 `max|ratio − 1|` is exactly 0.** Its discriminating
mutation is "use the rollout-recorded logprobs as the denominator", which must
turn it red. A version of this test that passes under that mutation is
decorative and does not count.

### The rest

| Behavior | Discriminating mutation |
|---|---|
| GAE over a hand-computed 4-step trajectory | Wrong `γλ` ordering |
| GAE does not bootstrap across a mid-chunk `done` | Ignore `episode_id`, use `done` only |
| Buffer reconstructs a known synthetic trajectory | Off-by-one in the 1024 shift |
| Bootstrap slot supplies `V(s_T)` and carries no loss | Include slot 1025 in the loss |
| `cache.reset(done)` runs after the terminal step | Reset before the next step |
| `episode_start_action` substituted after `done` | Carry the terminal action forward |
| `ReturnScaler` state round-trips through a checkpoint | Drop it from the state dict |
| Resume skips a manifest-less half-written checkpoint | Take the newest file regardless |
| Resume rejects a mismatched `init_state_hash` | Skip the check |
| 3 NaN minibatches abort; 2 do not | Off-by-one on the counter |
| `WandbRun.log` never passes `step=` | Pass `step=update` |
| No secret-shaped key reaches the W&B config | Merge `os.environ` into config |
| Resume event carries `update` and `global_step` | Log a bare message |

Slow tier (`@pytest.mark.slow`, deselected by default): real ROM, 4 envs, a
handful of real updates end to end, and a real checkpoint/resume round trip.

### Gates

- `pytest` green with branch coverage floor at **93**, ratcheted upward, never
  lowered. No `omit`, no `pragma: no cover`.
- `scripts/audit_tests.py tests/` at or below the 11-finding pre-existing
  baseline.
- `audit_observability.py src/` at **9** (§7).
- `ruff` clean.
- **Prove-it-can-fail** on every new test, with the verified ones named in the
  report. A green suite over tests that pass equally against correct and broken
  code is the failure this project has already paid for once.

## 11. Required changes to the merged sub-projects

Every edit PPO needs outside `src/ppo/`, enumerated so the implementation plan
can sequence them and no integration surprise arrives mid-build. Verified
against the code, not inferred from the specs.

### `src/sequence_model/` — one additive change

**`policy.py`: add `RecurrentTransformerPolicy.diagnostics(...)`**, a
`@torch.no_grad()` method taking the same arguments as `forward_chunk` plus a
`layer` index, returning `attn/logit_max`, the `attention_distance_mass`
buckets, and `model/residual_norm`.

It is needed because the pieces exist but are unreachable from outside:
`GroupedQueryAttention.attention_diagnostics(x, cos, sin, mask)` already
returns `(q, k, probabilities)` — exactly what `attention_logit_max` and
`attention_distance_mass` consume — but its `x` is the *post-`attn_norm`* input
to one block's attention, and `cos`/`sin`/`mask` are built inside
`forward_chunk`. None of that is reconstructible from outside without
duplicating the stack. Separately, `residual_norm(hidden)` needs the
`(B, L, d_model)` final hidden state, and `ChunkOutput` exposes only `logits`
and `value`.

Run on a sampled minibatch every `artifact_every_updates`, never on the hot
path — it materializes the full attention matrix that SDPA deliberately never
forms.

**No change to `checkpoint.py`.** §9's "report in words rather than raise" on a
`context_len` change is **PPO-side behaviour**: PPO compares the checkpoint's
saved `context_len` against the live config *before* calling `rebuild_cache`,
and skips the cache when they differ. Catching `rebuild_cache`'s `ValueError`
for control flow would be fragile — it raises for several distinct reasons.

No change to `cache.py`, `attention.py`, `masks.py`, `adapter.py`, `block.py`,
`config.py`, or `telemetry.py`.

### `src/pokemon_env/` — seven changes

1. **`rewards.py`** — add `RewardAccumulator.coord_keys() -> list[int]`. The
   seen-coordinate set already lives in `_State` and is already serialized by
   `state_dict`; only a read accessor is missing.
2. **`session.py`** — track completed-episode lengths (append `_step_count` on
   `reset`) and add `EnvSession.stats() -> dict` returning coord keys,
   `badge_count`, `event_flag_count`, `step_count`, and the drained episode
   lengths.
3. **`subprocess_backend.py`** — add `Command.STATS`, its `handle_command`
   branch, and `SubprocessBackend.stats()`; add `logger.exception(...)` before
   the worker forwards an error to the parent, so the traceback stops dying in
   the worker; add `# obs: allow LOG007` markers to the two genuinely benign
   shutdown `except` blocks (lines 270 and 340).
4. **`vec_env.py`** — `EnvBackend` Protocol gains `stats()`, `InProcessBackend`
   implements it, and `VecPokemonEnv.stats()` aggregates across backends.
   **This is a Protocol change**: every test fake implementing `EnvBackend`
   must gain the method or the suite stops type-checking.
5. **`vec_env.py`** — bump `VEC_ENV_SCHEMA_VERSION` from 1 to 2, because the
   session's `state_dict` gains the episode-length history and
   `load_state_dict` reads its keys directly. No PPO checkpoints exist yet, so
   nothing is stranded; the bump is what keeps the existing guard honest.
6. **`config.py`** — delete `EnvConfig.seed` and
   `EnvConfig.frozen_encoder_repo_id` (§3). `configs/pokemon_env.yaml` sets
   neither, verified, so no config file changes and `load_config`'s
   unknown-field rejection is not tripped.
7. **`telemetry.py`** — `exploration_heatmap` keeps its
   `(coord_keys, height, width)` signature and its correct unpacking; only the
   *projection* changes, from a 16×16 folded cell per map to true `(x, y)`
   within per-map tiles across the top 12 maps by unique-coordinate count.
   `rollout_metrics` gains a `stats` parameter and emits the ~4 missing
   scalars.

### `src/observability/` — one change

**`tracking.py`** — `WandbRun` gains `config`, `step_metrics`, a stable run id
with `resume="allow"`, `__enter__`/`__exit__`, and `exc_info=True` on its two
failure warnings (§7).

Every addition is a keyword argument with a default, and `ExperimentRunLike`'s
new context-manager methods are implemented on `NullExperimentRun` too, so
`data_collection` and `contrastive_pretrain` call sites keep working unchanged.
That backward compatibility is a test, not an intention.

### `pyproject.toml`

Add `src/ppo` to `[tool.hatch.build.targets.wheel].packages` and
`pokemon-ppo = "ppo.cli:main"` to `[project.scripts]`.

## 12. Known gaps carried out of implementation

Surfaced by the whole-branch review and deliberately not fixed on this branch.
Recorded here because the SDD workspace that held them is scratch, and these
are real.

**Blocking the first paid run:**

- **The slow acceptance tier has never run against a real ROM.** Both tests in
  `tests/integration/test_ppo_smoke.py` skipped on the dev machine with stated
  reasons — no `Pokemon Red.gb`, no `artifacts/init.state`. Their harness logic
  was verified with a `FakeVecEnv` substitution, but §8's gate 4 is unmet until
  they run for real. That is a pod gate, not a laptop gate.
- **`torch.compile`, `channels_last`, and `cudnn.benchmark` are not applied.**
  §4's fixed-shape design — including the 1023-step warmup — exists precisely so
  these are safe, and nothing currently collects the benefit. They belong with
  gates 1–3 because they must be measured on the pod anyway.
- **§8's gate 3 (memory probe) is not implemented in `preflight.py`.** Fixed
  shapes make the first real update an implicit probe, so an OOM surfaces in the
  first iteration rather than 40 hours in. Either add it or amend §8's count.

**Deferred, not blocking:**

- **No Hub snapshot.** `PPOConfig.hub_snapshot_every_updates` exists and is read
  nowhere; §6's "roughly hourly snapshot" and "the final policy is pushed" are
  unimplemented. The trainer has no upload client or credential path, and
  checkpoints live on the network volume by design. Wire it or delete the field.
- **§6's cache-resume wording overstates what the KV cache buys.** "Resuming
  with an empty cache puts a value-loss spike in the telemetry after every
  restart" is true of the *rollout* path only. `forward_chunk` never consults the
  KV cache, so the cache alone cannot prevent an update-side spike. The trainer
  now always rebuilds the burn-in prefix on resume, which is what actually
  closes that gap.
- **Truncation at the terminal slot is treated as termination.** The final
  transition of each episode gets `reward = 0` and no bootstrap, rather than
  bootstrapping off `V(o_T)`. One slot per 163,840-step episode against a ~333-step
  effective horizon at `γ = 0.997`, so immaterial — but it is the terminal
  treatment, not a truncation bootstrap, and it would matter if episodes shortened.
- **Metric names drift from §7:** `env/worker_respawns` vs `env/respawns`,
  `progress/badges_max` / `_mean` vs `progress/badges`. Pick one and reconcile.
- **`ResumeResult.wandb_run_id` is dead** — the CLI reads the run id from the
  `wandb_run_id.txt` sidecar instead. Two recorded sources, one consumer.
- **Unused parameters:** `run_update` accepts `policy_config` and never uses it.
- **`trainer._respawns` reaches into `vec_env._backends`.** `SubprocessBackend.respawns`
  is already public; only the container access is private. A respawn count on
  `stats()` would remove the reach-in.
- **`.abs()` in the staleness computation is unproven,** because the test
  harness's offset is always positive so the difference is never negative.

## 13. Open questions

Deliberately not decided here, because each needs a run to answer:

- Whether 64 envs is right — gate 2 measures it.
- Whether `n_epochs = 3` and `minibatch_envs = 8` give enough gradient steps.
  The first run's `explained_variance` and reward curves decide.
- Whether the `/√k` exploration-decay constant (0.30) and the reward weights
  are right. The per-component breakdown and the heatmap are how they get
  tuned.
- Whether `init.state` should advance past the first rival battle.
- Whether `context_len` should move at all, in either direction (§9).
