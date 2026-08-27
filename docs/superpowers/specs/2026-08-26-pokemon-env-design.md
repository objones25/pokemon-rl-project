# Pokemon Red Environment — Design Spec

**Sub-project A of two.** This spec covers the environment: a PyBoy wrapper, the
RAM-derived observation vector, the reward system from the architecture plan's
§4, 64-way vectorization, and frozen-encoder inference. **Sub-project B**, the
PPO trainer, gets its own spec and consumes what this one produces.

The split is deliberate. This half is fully verifiable with a random agent —
reward curves, exploration heatmaps, contact sheets, RAM reads — before a line
of PPO exists. Merging them means the first training run debugs the env and the
trainer simultaneously, and from inside a loss curve a wrong RAM address looks
exactly like a bad hyperparameter. It is also CLAUDE.md's Karpathy guideline:
verify one stage end-to-end before building the next on top.

Consumers: `sequence_model.RecurrentTransformerPolicy` (merged) and the
Sub-project B trainer (not yet written).

## Verified interface facts

Every number here was read from the installed package's docs via `ctx7` or from
PWhiddy/PokemonRedExperiments' actual source. None is recalled. Re-verify before
changing anything that depends on them.

### PyBoy (v2 API)

| fact | value |
|---|---|
| headless window | `window='null'` — `'headless'` and `'dummy'` were removed in v2.0.0 |
| screen buffer | `pyboy.screen.ndarray` → `(144, 160, 4)` uint8 |
| buffer format | `pyboy.screen.raw_buffer_format` → `'RGBA'` |
| buffer dims | `pyboy.screen.raw_buffer_dims` → `(144, 160)` |
| frame advance | `pyboy.tick(count, render)` — renders **only the last frame** of the tick |
| input | `pyboy.button_press(s)` / `pyboy.button_release(s)`, `s` in `'a','b','start','select','left','right','up','down'` |
| memory | `pyboy.memory[addr]`, `pyboy.memory[bank, addr]`, slices return copies |
| save state | `pyboy.save_state(f)` — file-like object |
| load state | `pyboy.load_state(f)` — binary file object, **`f.seek(0)` first** |

`screen.ndarray` is `(H=144, W=160, RGBA)`. The Game Boy is monochrome, so
channel 0 alone is the grayscale image, and `(144, 160)` matches
`GrayscaleResNetEncoder`'s `(N, 1, 144, 160)` contract exactly — including its
transposed-input guard, which rejects `(N, 1, 160, 144)`.

`screen.ndarray` references a live backing buffer that is overwritten on the
next tick. It **must** be copied before being stored or written to shared
memory.

### PWhiddy/PokemonRedExperiments v2 (reference implementation)

| fact | value | our choice |
|---|---|---|
| actions | 7: down, left, right, up, A, B, START | same — matches `action_dim = 7` |
| frame-skip | `action_freq = 24` | same |
| episode length | `ep_length = 2048 * 80 = 163,840` | same |
| parallel envs | `num_cpu = 64` | same |
| observation | `screen.ndarray[:,:,0:1]`, downscaled `(2,2,1)` → 72×80, 3-frame stack | **differs**: full 144×160, no stack (memory is the transformer's job) |
| RL stack | SB3 `PPO("MultiInputPolicy")`, `SubprocVecEnv` | **differs**: custom loop (Sub-project B) |
| PPO config | `n_steps=2560, batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.01` | Sub-project B decides |
| press mechanics | press → `tick(8)` → release → `tick(15)` → `tick(1, render=True)` | same, totals 24 |

Their reward is a cumulative-total difference — `new_total - self.total_reward`
over monotone running maxima — which is structurally the same construction the
architecture plan's §4 calls `max_historical`. Component weights in v2:
`event × 4`, `heal × 10`, `badge × 10`, `explore × 0.1`, `stuck × -0.05`;
`level`, `op_lvl` and `dead` are present but commented out.

### RAM map (Pokemon Red/Blue)

Read from their verified readers, not inferred from constant names.

| datum | address | encoding |
|---|---|---|
| party size | `0xD163` | uint8, 0–6 |
| party levels | `0xD18C + 44i`, i∈[0,6) | uint8 |
| party current HP | `0xD16C + 44i` | **uint16 big-endian**: `256*m[a] + m[a+1]` |
| party max HP | `0xD18D + 44i` | uint16 big-endian |
| opponent levels | `0xD8C5 + 44i` | uint8 |
| badges | `0xD356` | bitfield — `popcount` |
| event flags | `0xD747 .. 0xD87E` **exclusive** | 311 bytes = **2488 flags** |
| museum ticket | `0xD754` bit 0 | excluded from the event count |
| map id | `0xD35E` | uint8 |
| x, y position | `0xD362`, `0xD361` | uint8 (**x before y**) |
| in-battle flag | `0xD057` | 0 = overworld |
| money | `0xD347 .. 0xD349` | 3-byte BCD |

The party struct stride is **44 bytes** (`0x2C`). `level` sits immediately
before `maxHP` (`0xD18C` and `0xD18D`), so a one-digit typo in an address list
lands on a neighbouring field and still reads plausible-looking small integers
rather than raising — which is why the stride test below exists. Bit indexing is
LSB-first: `read_bit(addr, n)` is `(m[addr] >> n) & 1`, matching their
`bin(256 + v)[-n-1]` formulation.

## Architecture

```
src/pokemon_env/
  config.py      EnvConfig dataclass + YAML loader (mirrors contrastive_pretrain/config.py)
  ram.py         address constants + typed readers over a MemoryReader Protocol
  aux_state.py   the 32-d vector + AUX_STATE_VERSION
  rewards.py     reward components + the max-historical accumulator
  emulator.py    Emulator Protocol + PyBoyEmulator adapter
  worker.py      subprocess entry point — owns exactly one emulator
  vec_env.py     parent side — 64 workers, shared-memory frames, batched step
  encoder.py     frozen CNN inference + latent_stats loading
  init_state.py  button-script replay producing init.state
  telemetry.py   reward decomposition, contact sheets, exploration heatmap
```

### The Emulator Protocol is the load-bearing boundary

PyBoy requires a real commercial ROM, which is gitignored and can never exist in
CI. Behind a Protocol:

```python
class Emulator(Protocol):
    def tick(self, count: int, render: bool) -> bool: ...
    def button_press(self, button: str) -> None: ...
    def button_release(self, button: str) -> None: ...
    def read_memory(self, addr: int) -> int: ...
    def screen_frame(self) -> np.ndarray: ...      # (144, 160) uint8, already copied
    def save_state(self) -> bytes: ...
    def load_state(self, state: bytes) -> None: ...
```

a `FakeEmulator` holding a synthetic 64 KB `bytearray` makes `ram.py`,
`aux_state.py` and `rewards.py` — where essentially every game-specific bug will
live — fully unit-testable with no ROM and no PyBoy. Same pattern as
`FakeHfClient` in `tests/conftest.py`.

`screen_frame()` returns `(144, 160)` uint8 and owns the copy, so no caller can
accidentally hold a reference into PyBoy's live buffer.

### Env and encoder stay separate

```python
VecPokemonEnv.step(actions) -> VecStep(
    frames: np.ndarray,      # (N, 1, 144, 160) uint8
    aux: np.ndarray,         # (N, 32) float32, in [-1, 1]
    reward: np.ndarray,      # (N,) float32, in [0, 1]
    done: np.ndarray,        # (N,) bool
    episode_id: np.ndarray,  # (N,) int64, monotonic per env
)

LatentEncoder.encode(frames) -> torch.Tensor   # (N, 2048)
```

Merging them would drag torch, CUDA, autocast and `channels_last` into the IPC
layer and make the env untestable without a GPU. Keeping them apart also leaves
raw frames reachable for video artifacts. Sub-project B composes the two.

### No `gymnasium` dependency

We write our own vectorization, so gymnasium would supply `spaces.Discrete(7)`
as documentation and an API we do not conform to. The sequence-model spec's
"`Discrete(7)`" is descriptive, not a library requirement. Revisit only if
SB3-ecosystem tooling becomes worth a dependency; nothing in this design needs
it.

## Observation contract

### Frames

`(N, 1, 144, 160)` uint8 in [0, 255], channel 0 of the RGBA buffer, **not**
downscaled. PWhiddy reduces to 72×80 because their CNN is small and they stack 3
frames for motion; our encoder was pretrained on full-resolution 144×160 frames
and temporal context is the transformer's job, so neither applies.

### The 32-d aux vector — `AUX_STATE_VERSION = 1`

Width is fixed at 32 by the merged `PolicyConfig.aux_state_dim`; changing it
changes the model. Thirty real signals, one reserved.

| idx | signal | source | normalization |
|---|---|---|---|
| 0 | party size | `0xD163` | `/6` |
| 1–6 | party levels | `0xD18C + 44i` | `/100` |
| 7–12 | per-slot HP fraction | `hp16(0xD16C+44i) / hp16(0xD18D+44i)` | ratio, 0 when max is 0 |
| 13 | badge count | `popcount(0xD356)` | `/8` |
| 14 | event flags set | `Σ popcount(0xD747..0xD87E)` | `/2488` |
| 15 | map id | `0xD35E` | `/255` |
| 16 | x position | `0xD362` | `/255` |
| 17 | y position | `0xD361` | `/255` |
| 18 | in battle | `0xD057 != 0` | 0 or 1 |
| 19 | money | BCD `0xD347..0xD349` | `/999999` |
| 20–25 | opponent levels | `0xD8C5 + 44i` | `/100` |
| 26 | aggregate HP fraction | `Σhp / max(Σmaxhp, 1)` | ratio |
| 27 | distinct coords seen | env-side counter | `log1p(n) / log1p(20000)` |
| 28 | episode progress | `step / 163840` | `[0, 1]` |
| 29 | steps since last new coord | env-side counter | `min(n, 1000) / 1000` |
| 30 | distinct maps visited | env-side counter | `/255` |
| 31 | reserved | — | held at 0 post-centering |

**Slots 0–30 are then mapped `2x - 1` into [-1, 1].** Slot 31 is exempt and
written as literal `0.0` — running a "constant 0" through the centering map
would emit `-1.0`, which is a strong constant signal rather than the absence of
one. This is interface fit
against the merged `InputAdapter`, whose `proj` is `nn.Linear(..., bias=False)`:
a block of inputs with mean 0.5 becomes a fixed offset vector the model must
absorb with real capacity. Centering costs nothing and removes it.

All values are clamped into range after normalization. RAM can hold
out-of-range garbage during transitions (a level of 255 mid-write), and an
unclamped `2x - 1` would inject a large outlier into a value network the
architecture plan explicitly warns is hypersensitive to input scale.

### Versioning

`AUX_STATE_VERSION` is recorded in every checkpoint and validated on resume.
A policy trained against layout v1 and fed v2 data is silently wrong in exactly
the way `PolicyConfig` drift is — no crash, no shape error, just a different
model. Bump it whenever any row of the table above changes meaning.

## Reward system

The architecture plan's §4 form, exactly:

```
total_t = Σ wᵢ · cᵢ(t)                     # every cᵢ monotone cumulative
r_t     = clip(max(0, total_t − M), 0, 1)
M       = max(M, total_t)                  # the max_historical baseline
```

| component | `cᵢ(t)` | weight | one unit earns |
|---|---|---|---|
| badges | `popcount(0xD356)` | 1.00 | **1.00** — at the clip cap |
| heal | `Σ (Δhp_frac)²` while party size unchanged | 0.50 | ≤ 0.50 |
| explore | `Σ_{k=1..N} 1/√k` | 0.30 | 0.30 at k=1, 0.03 at k=100 |
| events | `flags_set − base_flags − museum_ticket` | 0.10 | 0.10 |
| levels | `Σ max(level − 2, 0) − 4` | 0.05 | 0.05 |

Each constant appears in exactly one column: `cᵢ` is unit-free and `wᵢ` carries
the scale. Folding a coefficient into both is how a weight silently gets applied
twice during tuning.

`base_flags` is captured at episode reset and subtracted, so the flags already
set by `init.state` earn nothing.

**Delta-based and non-negative**, per §4. Every `cᵢ` is a running maximum or a
monotone count, and the step reward is its increase, so cycles pay nothing — the
deposit/withdraw exploit §4 names, and equally walking back and forth over
known ground.

**Exploration decays**, per §4's "decaying scalar reward". The k-th newly
discovered coordinate earns `0.30/√k`. PWhiddy's flat `0.1` does not decay.
Coordinates key on `(x, y, map_id)` and are only recorded outside battle
(`0xD057 == 0`), because in battle the position bytes are stale.

**Clipping is an outlier guard, not the normalizer.** Weights are chosen so a
normal step's reward is already well inside the range; the clip fires only on
genuine outliers. Hard-clipping raw weights of the PWhiddy scale (badge 10,
event 4) would collapse both to exactly 1.0 and leave the agent unable to
distinguish a gym badge from a door opening. Note `max(0, ·)` makes the lower
bound unreachable, so the effective range is [0, 1] — which is what the
sequence-model spec's `reward_feat` input expects.

**Log the clip-fire rate.** Beating a gym leader fires badge + several events +
level-ups in one step, sums past 1.0, and the excess is discarded. That is the
guard doing its job, but above roughly 0.1% of steps it means the weights are
miscalibrated and achievement ordering is being flattened.

### Stalling without a penalty

§4 forbids negative reward, so PWhiddy's `stuck: -0.05` is out. Three mechanisms
replace it: revisits already earn zero; `γ` plus the 163,840-step budget makes
dithering cost return; and **slot 29 hands the policy "steps since last new
coordinate" directly**, so it can perceive being stuck rather than be punished
for it. A penalty would also invite the standard failure where an agent ends its
episode early to stop accruing it.

## Vectorization

64 subprocess workers, each owning one emulator. PyBoy is GIL-bound, so threads
buy nothing.

- **Commands** over a `multiprocessing.Pipe`: `RESET`, `STEP(action)`,
  `SAVE_STATE`, `LOAD_STATE(bytes)`, `CLOSE`.
- **Frames** through one `(64, 144, 160)` uint8 `SharedMemory` block, each
  worker writing a disjoint slice. No locking: slices do not overlap, and the
  parent joins all 64 responses before reading. The parent presents it as
  `(64, 1, 144, 160)` — the channel axis is a view, not a copy, since the
  encoder's contract is 4-D.
- **Everything else** (aux, reward, done, episode_id) is small and rides the
  pipe.

The arithmetic that decides this: a frame is 23,040 B, so 64 of them is
**1.47 MB per vector step**, and at 1024 steps per rollout that is **1.5 GB of
pickling per rollout** — roughly 19% of the spec's 8.0 s rollout budget spent
serializing. Shared memory reduces it to a memcpy.

An **in-process serial backend** implements the same interface for tests and
debugging. It is roughly 64× too slow for a real run and exists only so
`vec_env` logic can be tested without spawning processes.

### Action execution

The 7 discrete actions map to `('down', 'left', 'right', 'up', 'a', 'b',
'start')`. Each step:

```
button_press(button)
tick(8,  render=False)
button_release(button)
tick(15, render=False)
tick(1,  render=True)      # only the final frame is rendered
```

totalling the 24 frames of frame-skip. Rendering only the last frame is PyBoy's
documented performance guidance and is why `tick` takes a `render` flag at all.

### Autoreset is next-step, deliberately

`done=True` at step *t* returns the **terminal** observation; the reset
observation arrives at *t+1*. This exists to satisfy the sequence-model spec's
`cache.reset(done)` ordering contract: reset must run *after* the step whose
transition ended the episode, or the final transition of every episode attends
to a cleared cache. Making the env's autoreset structural means the trainer
cannot get the ordering wrong.

## Frozen encoder and latent statistics

`encoder.py` owns two things the sequence-model spec's handoff assigned to PPO,
because they belong wherever the frozen encoder lives:

- Loading the frozen encoder from the Hub repo and running batched inference,
  `(N, 1, 144, 160)` uint8 → `(N, 2048)` float32, under `model.eval()` and
  **`@torch.no_grad()` — deliberately not `@torch.inference_mode()`**,
  `channels_last`, with a fixed batch shape so `cudnn.benchmark` and
  `torch.compile` do not re-tune.
- Fetching and **validating** `latent_stats.json`: shape against `latent_dim`,
  and `std > 0` in every dimension. `InputAdapter` already raises on both, but
  it can only do so once someone hands it the values; fetching them is this
  module's job. A dead encoder channel with `std == 0` divides by the 1e-6 floor
  and feeds ~1e6-scale inputs to a value head.

Sub-project B consumes the returned stats when constructing the policy.

### Why the encoder is `no_grad`, not `inference_mode`

CLAUDE.md's standing rule is that every eval path is `model.eval()` +
`@torch.inference_mode()`. The encoder is the documented exception, for the
identical reason `RecurrentTransformerPolicy.step()` already is: its output
**enters an autograd graph later**. Latents recorded during rollout become
inputs to `forward_chunk` at the PPO update, and a tensor created under
`inference_mode` raises "Inference tensors cannot be saved for backward" the
moment the adapter tries to save it — at the first update, on a paid GPU, not at
rollout. `no_grad` tensors are ordinary tensors. `requires_grad` is `False`
either way, so `is_inference()` is the only check that tells them apart, and a
test asserts it here exactly as it does for `step()`.

Cloning the output is not a fix: inside an `inference_mode` context, `.clone()`
returns another inference tensor.

## Initial state

`init_state.py` replays a committed, deterministic button script against the
local ROM once at setup and writes `init.state` into a gitignored `artifacts/`
directory. Every env loads it at reset.

Nothing ROM-derived enters git, the result is reproducible from the ROM you
already own, and the button sequence is reviewable in the diff. The alternative
of committing or downloading a third-party `.state` means shipping a ROM memory
dump with provenance we do not control; booting from scratch means all 64 envs
burn the intro, title and naming screens on every reset, learning a button dance
that has nothing to do with the task.

The script is authored once by stepping through interactively and recording the
presses. `init.state`'s hash is recorded in checkpoints so a resume can detect
that the starting state changed underneath it.

## Checkpoint and resume

`VecPokemonEnv.state_dict()` returns, per env:

| field | type | size |
|---|---|---|
| emulator state | `bytes` | ~50 KB |
| `M` per component | float | — |
| coord set | `int32` tensor, `map*65536 + x*256 + y` | ~80 KB at 20k coords |
| running explore sum, `N` | float, int | — |
| `last_health`, `party_size` | float, int | — |
| `step_count`, `episode_id` | int | — |

plus `AUX_STATE_VERSION`, a schema version, and the `init.state` hash. Written
through the existing `checkpointing.io`, which already provides atomic writes,
newest-first discovery and a parameterized retention glob.

Coordinates are an `int32` tensor rather than a set of tuples because
`torch.load(weights_only=True)` will not restore tuple-keyed dicts — the same
constraint that forced `PolicyConfig` to travel as a plain dict. It works out
cleaner regardless: the reward needs set membership, the running decayed sum and
`N`, never per-coordinate visit counts. Those existed for PWhiddy's stuck
penalty, which this design drops.

### This settles the cache-vs-emulator-state question

All 64 emulator states total roughly **3.2 MB**, negligible beside the
sequence model's 256 MiB KV cache. Once emulator state is saved, the KV cache is
being restored against the exact game position it remembers — the only condition
under which saving it was ever coherent. **Save both.**

This is conditional on the ~50 KB figure, which is an estimate from Game Boy
memory sizes (WRAM 8K + VRAM 8K + cart SRAM 32K + HRAM/OAM/CPU), not a
measurement. **Measuring the real `save_state` size is the first task in the
implementation plan**, because the entire decision pivots on it.

## Failure handling

A multi-day 64-process run makes worker death a certainty. PyBoy is Cython — a
segfault takes the process, not an exception.

- **Dead or hung worker**: respawn from `init.state`, force `done=True`,
  increment `episode_id`, reset that env's reward accumulator, and log it at
  WARNING with the env index and cause. One env's lost episode costs 1.6% of a
  rollout; killing the run costs hours.
- **Not** respawned from the last checkpoint's emulator state. That state pairs
  with a checkpoint-time `M` and coord set, so an env restored to a mid-game
  position with a current-time accumulator would silently re-earn rewards for
  progress it had already banked.
- **Hang detection**: 60 s timeout on pipe `recv`. A 24-frame tick takes ~1 ms,
  so 60 s is four orders of magnitude of headroom and can only fire on a real
  hang.
- **Fail fast at startup**, never mid-run: missing ROM, missing or hash-mismatched
  `init.state`, encoder weights that will not load, `latent_stats.json` failing
  validation.
- Worker respawn count is a logged metric. A rising respawn rate is a leading
  indicator of a real problem — memory pressure, a bad state — long before it
  shows in reward.

## Observability

Per CLAUDE.md: structured JSON-lines plus a live W&B run, and anything that
discards data says why.

- **Per-update scalars**: reward decomposed by component, **clip-fire rate**,
  new coordinates discovered, badges and event flags, episode lengths, worker
  respawns, steps/second.
- **Contact sheet**: one frame per env in an 8×8 grid, every N updates. The
  direct analogue of the contrastive pipeline's sheets and the fastest way to
  see that all 64 agents are stuck in the same menu.
- **Exploration heatmap**: `(x, y, map_id)` projected to global coordinates and
  accumulated across all 64 envs into one image. PWhiddy's `global_map.py`
  exists for exactly this projection. This is the single most informative
  artifact this project can produce — it answers "is it actually playing?" at a
  glance.
- **Clipped reward is logged**, since clipping is the one place this component
  discards signal.

## Testing

Per CLAUDE.md's gates: seeded, CPU-only, tiny synthetic fixtures by default;
`tmp_path` for writes; anything needing the real ROM or PyBoy marked
`@pytest.mark.slow` and deselected.

- **`ram.py`, `aux_state.py`, `rewards.py`** are pure functions over the
  `Emulator` Protocol. A `FakeEmulator` over a synthetic 64 KB `bytearray`
  covers all of it with no ROM and no PyBoy.
- **Address-stride test**: every party and opponent address list is exactly 44
  apart. A one-digit hex typo otherwise reads live game data from a garbage
  offset and presents as a bad reward, not as an error.
- **Reward property tests**: a level that rises then falls earns zero net (the
  §4 cycle exploit); a revisited coordinate earns zero; `Σ 0.30/√k` matches its
  closed form; a step crossing several components at once clips to exactly 1.0.
- **Aux range test**: every slot lands in [-1, 1] for both zeroed and
  all-`0xFF` memory — the latter is the out-of-range-garbage case the clamp
  exists for.
- **`vec_env.py`** is tested against the in-process serial backend, so the
  vectorization logic — autoreset ordering, episode_id monotonicity, respawn —
  is covered without spawning processes.
- **One `@pytest.mark.slow` integration test** drives real PyBoy against the
  real ROM for a few hundred steps, auto-skipped when the ROM is absent,
  mirroring the `POKEMON_RL_TEST_CLIP` pattern in
  `tests/integration/test_extraction_smoke.py`.
- Every new test is verified to fail when the code it covers is broken, and the
  suite is run through `audit_tests.py` before the branch is finished.

## Dependencies

`uv add pyboy`. Nothing else new — numpy, torch, wandb, pyyaml and
`huggingface-hub` are already present. No `gymnasium`, no `stable-baselines3`.

## Handoff: what Sub-project B's spec must decide

- **Recompute `π_old` and `V_old`** with one `no_grad` `forward_chunk` pass at
  update start, and use those as the importance-ratio denominator and the GAE
  baseline — never the logits and values recorded during rollout. See "The
  rollout path is still stale" in the sequence-model spec: the KV cache is
  carried across update boundaries, so the behaviour policy is not exactly
  `π_θ_old` and the epoch-1 ratio is otherwise not 1.0. Cost is ~1.7% of a
  rollout and it fuses with epoch 1's forward pass.
- **Log `max|ratio − 1|` at epoch 1 as a hard invariant.** After the above it
  must be exactly 0; anything else is a real bug.
- **Save the KV cache and the emulator state together**, per the section above,
  once task 1 confirms the state-size measurement. Neither is coherent alone.
- **`n_steps`, `n_epochs`, `γ`, `ent_coef`, clip range, GAE `λ`.** PWhiddy v2
  uses 2560 / 1 / 0.997 / 0.01; the sequence-model spec assumes `n_steps=1024`
  with `burn_in = context_len − 1`. Note `n_epochs=1` makes the epoch-1 ratio
  the *only* ratio, which raises the stakes on the item above rather than
  lowering them.
- **The CUDA SDPA backend measurement** — a materialized bool mask rules out
  FlashAttention, and the math backend would materialize ~536 MB of scores at
  (B=8, H=8, L=2047). A gate to clear before the first paid GPU hour.
- **Telemetry consumers**: the attention-distance heatmap artifact from the
  sequence-model spec's Observability section.

## Open questions

- Whether 64 envs is right for our per-step cost. PWhiddy's 64 is tuned for a
  small CNN; ours adds 14.69 GFLOP/frame of frozen encoder. The rollout budget
  says 64 fits, but it should be measured on the real pod before committing.
- Whether the `/√k` exploration decay constant (0.30) and its shape are right.
  This is the reward parameter most likely to need tuning, and the exploration
  heatmap plus the per-component reward breakdown are how it gets tuned.
- Whether `init.state` should advance further than the post-naming point — e.g.
  past the first rival battle — to skip a segment every agent must relearn. A
  measurement to make after the first run, not a decision to pre-commit.
