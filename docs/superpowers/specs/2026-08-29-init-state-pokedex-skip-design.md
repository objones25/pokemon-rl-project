# init.state Past the Parcel Quest — Design Spec

**Amendment to `docs/superpowers/specs/2026-08-26-pokemon-env-design.md`.**
That spec's Initial State section and its own Open Questions bullet — *"whether
`init.state` should advance further than the post-naming point... a
measurement to make after the first run, not a decision to pre-commit"* — are
both revisited here, deliberately before the first run, on reasoning and a
reference-implementation precedent rather than a measurement. See "Why decide
this now" below for the justification.

## Motivation

Peter Whidden's PokemonRedExperiments (this project's cited RAM/timing
reference, see CLAUDE.md's Attribution section) does not start its agents from
a fresh save. It starts from `has_pokedex.state` — past choosing a starter,
past the mandatory rival battle, past fetching Oak's Parcel from the Viridian
Poké Mart, and past delivering it back to Oak in Pallet Town to receive the
Pokédex.

The reason to mirror that, restated in terms of this project's own reward
design: §4's reward is delta-based and monotone —
`docs/superpowers/specs/2026-08-26-pokemon-env-design.md`'s Reward system
section — with an explore term that rewards *new* coordinates and an event
term that rewards *new* flags, both regardless of direction. Reaching the
Viridian Mart already fires the "got parcel" event flag and a burst of new
exploration coordinates; nothing in the reward tells the agent to turn around
and walk back to Pallet Town afterward — from the reward's perspective, the
quest already paid out. A policy with no forward model and no long-horizon
credit assignment (this project's policy is a transformer over frozen-CNN
latents trained by PPO, not a planner) has no signal pointing at the 40+ tile
backtrack through Route 1 and Pallet Town, then a specific NPC interaction at a
specific tile in Oak's lab, needed to receive the Pokédex and its own event
flags and reward. This is the concrete version of the problem Whidden's own
project ran into and solved by starting past it.

### A second, independent defect found while verifying this

Confirmed empirically against the real ROM this session (not merely a reading
of the code): `generate_init_state(emulator, INTRO_SCRIPT)` with the *current*
`INTRO_SCRIPT` does reach the exact RAM values
`tests/integration/test_pokemon_env_smoke.py::test_generated_init_state_starts_in_the_bedroom_before_the_starter`
asserts — `(x, y, map) == (3, 6, 38)`, `party_size == 0`, `money == 3000`,
`in_battle == False` — but the actual screen at that state is:

```text
My name is OAK!
People call me
```

still an open dialogue box, mid-narration. The four RAM values the existing
test checks all happen to already hold their final resting value at this point
in the intro (position, empty party, starting money), which is why that test
passes, but none of them can detect "is a textbox currently open" — exactly
the gap its own docstring calls out for the weaker `map_id != 0` check it
replaced, just one level further in. `init_state.py`'s module docstring claim
that the script "advances... to the first controllable overworld step" is
false as written today: every env reset starts by presenting a stalled
dialogue box, not free movement, and the first several actions of every
episode are effectively forced into clearing it. This is worth fixing
regardless of the Pokédex decision, and the work below fixes both at once
since the second requires re-deriving the whole intro sequence anyway.

## Decision

Extend `INTRO_SCRIPT`, not swap it for a downloaded or embedded save state.
This is "Option 1" from the choice already made: implement by mirroring
Whidden's reference path with this project's own reviewable button script,
never by importing his binary `.state` — CLAUDE.md's Initial State frozen
contract and the pokemon-env spec's own justification both require nothing
ROM-derived to enter git, and the button sequence to stay reviewable in a
diff. A downloaded `has_pokedex.state` would be a ROM memory dump with
provenance this project does not control; extending the button script keeps
`init.state` reproducible from a ROM the user already owns.

Target: reach Whidden's `has_pokedex.state`-equivalent point via ordinary
play — pick a starter, take the mandatory rival battle, walk to Viridian
City's Poké Mart, receive Oak's Parcel, walk back to Pallet Town, deliver it
to Oak, receive the Pokédex — landing in Oak's lab, fully controllable, no
dialogue open.

### Why decide this now, against the original spec's own sequencing

The pokemon-env spec explicitly deferred this to "after the first run." The
reason to override that here: the two paths have starkly different costs.
Extending the button script and regenerating `init.state` is a local,
free, roughly-one-minute-per-attempt operation with no GPU and no paid time
involved. Discovering *after* a multi-day paid PPO run that the fetch-quest
segment is effectively unreachable — because the reward structure gives no
signal to attempt it — costs the entire run's compute and wall-clock, then
requires this exact fix anyway plus discarding every checkpoint trained
against the old `init.state` (see "Operational consequences" below,
which applies unconditionally to changing `init.state` at any point in the
project, so doing it now costs nothing beyond what changing it later would
have cost, but claims fewer sunk GPU-hours against it). Given a reference
implementation already demonstrates the fetch-quest is worth skipping, waiting
for a measurement that this project would have to pay for to obtain — one the
reference project already effectively ran and reported the answer to — is not
buying real information proportionate to what it costs.

### Starter species: pick Squirtle, deliberately

No code downstream of `generate_init_state` depends on which Pokémon is
chosen — only `party_size`, `badges`, `oak_parcel`, `oak_pokedex` are
frozen-contract-relevant (aux vector slots 0, 7-12, 13 read party stats
generically; nothing keys on species). But the choice is not free of
consequence for *training dynamics*, which is a real reason to make it
deliberately rather than by whichever ball is nearest.

Every claim below is read directly from `pret/pokered` — Brock's and Misty's
actual trainer data (`data/trainers/parties.asm`), confirmation that neither
gets a custom moveset (`data/trainers/special_moves.asm`, which lists exactly
which gym leaders get a hand-picked move and does not mention Brock or
Misty), each Pokémon's level-up learnset (`data/pokemon/evos_moves.asm`), and
the real Gen 1 type-effectiveness table (`data/types/type_matchups.asm`) —
not recalled from memory.

**Brock's team is Geodude (level 12) and Onix (level 14), and neither one
knows a Rock-type move.** Geodude's learnset doesn't add Rock Throw until
level 16; Onix's not until level 19. At the levels Brock actually fields
them, their only damaging move is Tackle (Normal-type, resisted or boosted by
nothing relevant here). This means beating Brock is entirely about your own
offense, not surviving his — and by the type chart, **Water and Grass are
equally strong against him**: Water is `SUPER_EFFECTIVE` against both Rock
and Ground, and separately, so is Grass — a same-type-effective attack from
either starter hits Geodude/Onix's Rock/Ground typing twice over. Typing
alone does not separate Squirtle from Bulbasaur at this gym.

What does separate them is **when each starter actually gets that move**:
Squirtle learns Bubble (its first Water-type damaging move) at level 8;
Bulbasaur doesn't learn Vine Whip (its first Grass-type damaging move) until
level 13. Bulbasaur's own earliest tool, Leech Seed at level 7, deals no
direct damage at all — it's a multi-turn drain-and-stall move, a genuinely
strong tool in skilled play but one that asks for exactly the kind of
turn-sequencing reasoning a policy with no forward model and no long-horizon
credit assignment is least likely to discover; a policy that never learns to
use it is just fighting with Tackle for longer. Squirtle's kit is simpler
throughout: Tackle, then Bubble, then Water Gun (level 15) — always "use your
strongest single-turn damage move," never a setup move to sequence.

**Misty's team is Staryu (level 18) and Starmie (level 21)**, also with no
custom moveset, and their only damaging moves are Tackle and Water Gun.
Squirtle's Water attacks are resisted by her Water-typed Pokémon, and her
Water Gun is equally resisted by Squirtle right back (`WATER` vs `WATER` is
`NOT_VERY_EFFECTIVE` in both directions) — a mutual wash, not the "neutral
matchup" an earlier draft of this spec claimed. Squirtle is not in danger
here, but it has no offensive edge either. Bulbasaur is actually better
positioned at this specific gym — Grass is `SUPER_EFFECTIVE` against her
Water team. Charmander is unambiguously the worst choice against her: Fire is
resisted by her team, and her Water Gun hits Charmander `SUPER_EFFECTIVE` in
return (`WATER` vs `FIRE`) — the only pairing among the three starters that
is bad in both directions at once.

Given Bulbasaur is genuinely competitive on typing (as strong as Squirtle
against Brock, stronger against Misty), the case for Squirtle rests on two
things this project actually cares about, not on a typing advantage that
does not hold up: it reaches its first hard-hitting move four levels sooner
(8 vs. 13), and its entire early kit never requires the multi-turn setup
reasoning Leech Seed calls for. It is also Whidden's own choice in
`has_pokedex.state`, so picking it keeps this project's starting point
identical to the reference implementation's, not just structurally
equivalent to it.

The script should walk to and select the Squirtle ball specifically (per
`pret/pokered`'s `scripts/OaksLab.asm`, the three balls sit at fixed,
distinguishable coordinates on the starter table), not whichever ball is
nearest or fastest to reach.

## What needs building

### 1. New RAM readers, sourced and cited per CLAUDE.md's Attribution rule

`src/pokemon_env/ram.py` has no readers yet for Oak's Parcel or the Pokédex
flags. This session used `0xD74E` bit 1 (parcel) and `0xD74B` bit 5 (Pokédex)
directly against the emulator during interactive verification. A dedicated
review agent has since confirmed both addresses trace exactly to PWhiddy's own
repository — `baselines/red_gym_env.py:509` (`oak_parcel = self.read_bit(0xD74E,
1)`) and `:510` (`oak_pokedex = self.read_bit(0xD74B, 5)`) — so this project's
values are not generic Gen1 lore, they are a real match to the cited reference.

Two caveats the citation in `ram.py` must carry, because they are load-bearing
for anyone reconciling this against the v2 baseline the rest of `ram.py`
follows: **(1)** those two lines sit inside a triple-quoted, commented-out
block in `red_gym_env.py` (opens line 503, closes line 523) — dead code in v1,
not part of its executed reward path — and **(2)** neither name appears
anywhere in `v2/red_gym_env_v2.py` at all. So the citation should read
`baselines/red_gym_env.py:509-510 (v1, inside a commented-out block; absent
from v2)`, not a bare address match, so a future reader does not assume v2's
implicit vetting extends to these two. When added, extend `ram.py`'s
Attribution docstring with this exact citation rather than the bare addresses.

These readers are needed only to **verify** the generated state during
authoring (confirm `oak_parcel` and `oak_pokedex` are set before accepting the
new `init.state`) — they do not need to become part of the 32-d aux vector or
the reward system; `EVENT_FLAG_COUNT`'s existing aggregate flag reader already
covers them for reward purposes, since both are ordinary event flags within
`0xD747..0xD87E`.

### 2. The extended `INTRO_SCRIPT`

Rebuild the full button sequence from boot through delivery. Key game-logic
facts discovered this session, load-bearing for whoever authors the final
script:

- **Movement calibration**: an 8-frame button press followed by a 40-frame
  idle settle reliably produces exactly one tile of movement. A 20-frame press
  (the calibration used by the *original*, shorter script's post-control
  presses) can overshoot into two tiles of movement on the overworld, and
  reading position immediately after release without an idle wait catches
  mid-transition RAM values.
- **A direction change's first press only turns the sprite** when the
  character was not already facing that way; the *second* press in the same
  direction is what actually moves. Scripts that assume every directional
  press moves one tile will silently under-shoot by one tile at every turn.
- **The bedroom's exit stairs require pressing UP, not DOWN**, at the specific
  column that lines up with the stairwell — a Gen1 quirk where the visually
  "downstairs" transition tile is triggered by walking toward it from above,
  not by a literal "down" input semantics might suggest.
- **The mandatory rival battle in Oak's lab is triggered by a coordinate the
  player walks through, not by bumping into or interacting with the rival's
  sprite.** Facing the rival directly and pressing "A" while adjacent does
  nothing (confirmed by direct experiment this session — multiple approach
  angles, all inert); walking down through the table's approach column past a
  specific row triggers his "Wait!" interception automatically.
- **The parcel-delivery trigger in Oak's lab is Oak's own object, not the
  pokeball table.** Re-approaching the now-empty starter table after returning
  from Viridian does nothing; `pret/pokered`'s `scripts/OaksLab.asm` (fetched
  via `gh api repos/pret/pokered/contents/...` this session) identifies the
  actual trigger as `OaksLabOak1Text`, Oak's own sprite.
- **Route 1's ledges are one-directional**: freely passable walking south,
  blocked walking north except at specific gap columns. Both the Pallet
  Town↔Route 1 and Route 1↔Viridian City boundaries have this shape.
- **The Viridian Poké Mart's entrance warp** sits at a specific coordinate
  pair on the Viridian City map (confirmed at `(29, 19)` against
  `pret/pokered` warp data this session) — approaching from the wrong tile
  does not trigger the interior warp.

### 3. How to build the script reliably — a process note, not just a fact list

This session found that reconstructing a final script by reading and
concatenating the many small single-purpose scratch files produced while
originally exploring the path is **unreliable**: many of those files represent
alternative attempts applied to the *same* starting checkpoint (trial and
error at a decision point — "try pressing up here" vs. "try pressing left
twice then up") rather than a linear chain, and naively concatenating them by
filename produced a script that silently diverged partway through (landed
back in the bedroom with `party_size == 0` instead of progressing). **Do not
trust a reconstruction from historical scratch files without re-verifying it
end-to-end in one continuous replay.**

The reliable method, and the one to use when finishing this: drive the
emulator forward **live**, in one continuous session, using a harness that
loads a checkpoint, applies a small batch of presses, and prints RAM state
(`map id`, `x, y`, `party_size`, `oak_parcel` bit, `oak_pokedex` bit) after
each batch — exactly the `drive.py`-style harness used this session. Accept
each phase's presses into the final script only once its RAM state and a
screenshot (to catch "numerically correct but a dialogue box is still open,"
the exact trap in the second defect above) confirm progress. Only assemble the
final `INTRO_SCRIPT` from presses verified this way, and re-verify the
assembled whole in one uninterrupted run from boot before accepting it — a
script built from independently-verified *phases* can still drift when
concatenated, since PyBoy's tick-level timing is sensitive to exactly how many
frames preceded a given input.

This session verified, in one continuous live replay, from boot through: the
existing intro/naming sequence, the corrected bedroom exit (UP at the stairs
column), the ground-floor exit, Pallet Town navigation to Oak's "Hey! Wait!"
trigger, lab entry, starter selection (`party_size` confirmed `1`), decline-
nickname, and the full mandatory rival battle through to being controllable
again in the lab. It was interrupted before completing the Viridian Mart
round trip and the final delivery to Oak, so **the parcel-fetch and delivery
legs still need to be redone and verified live** before this script is
complete — the facts above (mart warp coordinates, Oak's-object delivery
trigger, ledge one-directionality) come from a prior pass at this same path
earlier in the session and should still hold, but must be re-confirmed by the
live method above rather than assumed.

### 4. Target validation

Before accepting a new `init.state`, verify by direct RAM read (via the
harness, not yet through `ram.py` — see point 1):

| check | expected |
| --- | --- |
| `party_size` (`0xD163`) | `1` |
| `badges` (`0xD356`) | `0` |
| `oak_parcel` (`0xD74E` bit 1) | `1` |
| `oak_pokedex` (`0xD74B` bit 5) | `1` |
| screenshot | no open dialogue box, in Oak's lab, sprite responsive |

The last row exists because of the second defect above: RAM values alone
already proved insufficient to catch "still mid-dialogue" once, and nothing
about this new endpoint makes that trap less likely to recur — Oak's parcel-
delivery text ends in another multi-box dialogue sequence with the same shape
as the "My name is OAK!" box that fooled the existing check.

## What changes

- `src/pokemon_env/init_state.py`: `INTRO_SCRIPT` extended per above; the
  module docstring's "to the first controllable overworld step" claim
  corrected to describe the actual new endpoint (past starter selection, the
  rival battle, and the parcel delivery, in Oak's lab with the Pokédex) and to
  stop claiming the *old* endpoint was ever controllable. While touching this
  file's docstrings: `state_hash`'s own docstring (currently "Will be recorded
  in checkpoints (wired in by Task 12, not yet a caller of this function)") is
  separately stale and should be corrected in the same pass — `src/ppo/cli.py`,
  `src/ppo/trainer.py`, and `tests/integration/test_ppo_smoke.py` already call
  it, so the hash-check wiring this docstring describes as unfinished has been
  live since the PPO trainer landed. Leaving it as-is would mislead a future
  reader into thinking the checkpoint-invalidation behavior "Operational
  consequences" describes below is still pending, when it is already active
  today and will apply to this exact change.
- `artifacts/init.state`: regenerated from the new script. Gitignored, so this
  is a local action each user (or pod) takes, not a commit.
- `tests/integration/test_pokemon_env_smoke.py::test_generated_init_state_starts_in_the_bedroom_before_the_starter`:
  its pinned assertion — `(3, 6, 38)`, `party_size == 0`, `money == 3000`,
  `in_battle == False` — describes the *old* endpoint and must be rewritten
  for the new one. Per the second defect above, add a check strong enough to
  catch "numerically matches but a dialogue box is open" — e.g. asserting
  against a known-good screenshot hash, or a documented "no active text
  interpreter" RAM flag if one is findable in `pret/pokered`'s source, rather
  than trusting position/party/money alone a second time. Rename the test
  once its target changes (its current name literally says "before the
  starter," which stops being true).
- No change needed to `rewards.py`: `base_event_flags` is captured fresh at
  every episode reset (`rewards.py:79`), so it automatically absorbs whatever
  event flags the new `init.state` already carries — the parcel and Pokédex
  flags earn nothing on reset, exactly as the existing "flags already set by
  `init.state` earn nothing" behavior the pokemon-env spec already documents
  for the current script.
- No change needed to `AUX_STATE_VERSION` or the aux vector's slot layout —
  nothing about the starting state changes what any slot measures, only what
  value it holds at reset.

### Operational consequences

`checkpoint.py`'s `init_state_hash` check
(`src/pokemon_env/checkpoint.py:51-57`) hard-rejects resuming a checkpoint
whose recorded hash does not match the current `init.state` — by design, since
every reward baseline in an old checkpoint describes progress from a starting
position the new run no longer shares. Regenerating `init.state` means **any
existing PPO checkpoint becomes unresumable** and training must restart from
update 0. This is a real cost, and the reason to pay it now — before the
first real paid run has produced anything to lose — is exactly the "why decide
this now" argument above: the cost only grows the longer this is deferred.

## Testing

- Update the existing `test_generated_init_state_starts_in_the_bedroom_before_the_starter`
  (rename, retarget, and strengthen per above) rather than deleting it — it is
  this project's only automated guard that `INTRO_SCRIPT` reaches a specific,
  intentional state rather than stalling somewhere plausible-looking.
- Prove the new test can fail: temporarily truncate the new `INTRO_SCRIPT` to
  stop one phase early (e.g. before the parcel pickup) and confirm the test
  fails with a value mismatch, not a crash — same "prove each new test can
  fail" gate this project's CLAUDE.md requires everywhere else, and
  particularly important here since the previous version of this exact test
  passed despite the state it validated being wrong in a way its assertions
  could not see.
- If oak_parcel/oak_pokedex readers are added to `ram.py` per point 1, they get
  the same treatment every other `ram.py` reader already has: pure functions
  over the `Emulator` Protocol, tested against `FakeEmulator`'s synthetic
  bytearray, no ROM required for the unit test.
- The existing `@pytest.mark.slow` acceptance test
  (`test_a_random_agent_drives_four_real_envs_end_to_end`) is unaffected in
  structure — it already reads `init.state` from disk rather than assuming
  its contents, so it continues to exercise whatever `init.state` is present
  without changes, real ROM required either way.

## Non-goals

- Does not decide whether to advance `init.state` *further* than Whidden's
  reference point (e.g., skipping the rival battle too, or granting a
  stronger starting party). Whidden's own point is the one precedent this
  project has actual evidence for; going further would be a new, unvalidated
  guess this spec does not make.
- Does not change the reward system's shape or weights — only what state
  `base_event_flags`/`base_flags` are captured relative to at reset, which the
  existing subtraction already handles generically.
- Does not address the `VecPokemonEnv.step()` dispatch-concurrency issue —
  see the sibling spec,
  `docs/superpowers/specs/2026-08-29-vec-env-step-concurrency-design.md`,
  filed separately because the two are unrelated beyond both being found
  during the same pre-first-run review pass.
