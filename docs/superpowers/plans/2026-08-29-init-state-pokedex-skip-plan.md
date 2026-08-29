# init.state Past the Parcel Quest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `INTRO_SCRIPT` so `artifacts/init.state` lands past starter
selection, the mandatory rival battle, the Viridian Poké Mart parcel fetch,
and delivery to Oak — mirroring PWhiddy/PokemonRedExperiments'
`has_pokedex.state` — while also fixing the independently-discovered defect
that the *current* script's endpoint still has a dialogue box open, not a
controllable overworld step.

**Architecture:** No new subsystem. `src/pokemon_env/init_state.py`'s
`ButtonPress`/`INTRO_SCRIPT`/`generate_init_state` shapes are unchanged; only
the script's contents grow. Two new pure RAM readers land in `ram.py`
alongside the existing ones. The hard part is empirical, not architectural:
the exact button/frame sequence for four new phases (bedroom→lab, lab→Mart,
Mart→lab, lab→Pokédex) can only be discovered by driving a real PyBoy
instance and reading back RAM state — no amount of reasoning from the
disassembly substitutes for that. This plan builds a small interactive
harness first, then uses it phase-by-phase exactly as
`docs/superpowers/specs/2026-08-29-init-state-pokedex-skip-design.md`
prescribes: verify each phase live via RAM + screenshot, only assemble and
re-verify the whole script continuously at the end.

**Tech Stack:** PyBoy 2.7.0 (installed; `pyproject.toml` pins `>=2.7.0`),
pytest 9.1.1, the project's existing `Emulator` Protocol / `PyBoyEmulator` /
`FakeEmulator` boundary.

**Spec:** `docs/superpowers/specs/2026-08-29-init-state-pokedex-skip-design.md`

## Global Constraints

- **Never commit anything ROM-derived.** `artifacts/init.state` is gitignored
  (confirmed: `.gitignore:29`, `git check-ignore -v artifacts/init.state`
  reports a match). The interactive harness script and any checkpoint/
  screenshot files it writes go under `scratch/`, also gitignored
  (`.gitignore:16`, `/scratch/`) — never under `src/` or `tests/`.
- **`INTRO_SCRIPT` picks Squirtle specifically** by walking to and selecting
  the Squirtle pokéball object, not the nearest one. Verified against
  `pret/pokered`'s `data/maps/objects/OaksLab.asm` this session: the three
  starter balls are separate map objects at `(6, 3)` = Charmander,
  `(7, 3)` = Squirtle, `(8, 3)` = Bulbasaur (object-event x, y in tile
  coordinates, Oak's Lab map). The player must approach and interact with the
  object at `(7, 3)`.
- **Test isolation and quality gates from `CLAUDE.md` apply throughout**:
  strict pytest config, branch coverage floor 93% (currently enforced via
  `pyproject.toml:51`, `--cov-fail-under=93`), no unseeded randomness, no
  network in a unit test, hand-written fakes (`FakeEmulator`) over
  `mock.patch`, every new test proven to fail first.
- **`ram.py` readers stay pure functions over the `Emulator` Protocol** —
  `read_memory` only, tested against `FakeEmulator`'s synthetic bytearray, no
  ROM required for any unit test.
- **The `oak_parcel`/`oak_pokedex` readers are for authoring-time
  verification only** — they do not join the 32-d aux vector or
  `AUX_STATE_VERSION` does not change (per spec's "No change needed"
  section, verified against `src/pokemon_env/aux_state.py:22-23`: version is
  currently `1`, dim `32`, neither is touched by this plan).
- **Real ROM required for every phase task below.** `Pokemon Red.gb` is
  present at the repo root of the current checkout (confirmed this session,
  1.0 MB) and is gitignored (`.gitignore` line for `*.gb`) — the isolated
  worktree this plan executes in must have its own copy; see Task 0.

---

## Task 0: Worktree setup — copy the ROM in

Git worktrees share history but not untracked/gitignored files. `Pokemon
Red.gb` is gitignored (`*.gb` in `.gitignore`) and required by every
`@pytest.mark.slow` test and every phase task below; without it, every task
from Task 3 onward is unexecutable in the new worktree.

**Files:** none (no source changes — a one-time setup step for whichever
worktree executes this plan)

- [ ] **Step 1: Confirm the worktree has no ROM yet**

Run (from inside the new worktree directory):
```bash
ls "Pokemon Red.gb" 2>&1
```
Expected: `No such file or directory` (worktrees don't inherit untracked
files).

- [ ] **Step 2: Copy the ROM from the main checkout**

```bash
cp "/Users/theelusivegerbilfish/Python_Projects/pokemon-rl-project/Pokemon Red.gb" "Pokemon Red.gb"
```

- [ ] **Step 3: Confirm PyBoy can load it**

```bash
uv run python -c "
from pokemon_env.emulator import PyBoyEmulator
e = PyBoyEmulator('Pokemon Red.gb')
e.tick(60, False)
print('ok', len(e.save_state()), 'bytes')
e.close()
"
```
Expected: `ok <some number> bytes`, no traceback.

No commit for this task — the ROM must never enter git.

---

## Task 1: `oak_parcel`/`oak_pokedex` RAM readers

**Files:**
- Modify: `src/pokemon_env/ram.py`
- Test: `tests/unit/test_pokemon_env_ram.py`

**Interfaces:**
- Produces: `ram.oak_parcel_set(mem: Emulator) -> bool`,
  `ram.oak_pokedex_set(mem: Emulator) -> bool` — used by every phase task
  below (via the harness's `report()`) and by Task 8's target-validation
  step.

Addresses (`0xD74E` bit 1 for the parcel, `0xD74B` bit 5 for the Pokédex)
were confirmed this session directly against the real ROM, and separately
traced to PWhiddy/PokemonRedExperiments'
`baselines/red_gym_env.py:509-510` — inside a commented-out block in that
file (v1; the block opens at line 503 and closes at line 523), and **absent
from `v2/red_gym_env_v2.py` entirely**. That caveat must travel with the
citation so a future reader doesn't assume v2's implicit vetting extends to
these two addresses.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_pokemon_env_ram.py`, next to the existing
`test_museum_ticket_set_reads_bit_zero_of_its_own_address` (same file,
matching its exact style — `fake_emulator.memory[ADDR] = <bit pattern>` then
assert the reader):

```python
def test_oak_parcel_set_reads_bit_one_of_0xd74e(fake_emulator) -> None:
    fake_emulator.memory[0xD74E] = 0b0000_0010

    assert ram.oak_parcel_set(fake_emulator) is True


def test_oak_parcel_set_is_false_when_bit_one_is_clear(fake_emulator) -> None:
    fake_emulator.memory[0xD74E] = 0b0000_0001

    assert ram.oak_parcel_set(fake_emulator) is False


def test_oak_pokedex_set_reads_bit_five_of_0xd74b(fake_emulator) -> None:
    fake_emulator.memory[0xD74B] = 0b0010_0000

    assert ram.oak_pokedex_set(fake_emulator) is True


def test_oak_pokedex_set_is_false_when_bit_five_is_clear(fake_emulator) -> None:
    fake_emulator.memory[0xD74B] = 0b0000_0000

    assert ram.oak_pokedex_set(fake_emulator) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_pokemon_env_ram.py -k "oak_parcel or oak_pokedex" -v`
Expected: FAIL — `AttributeError: module 'pokemon_env.ram' has no attribute 'oak_parcel_set'`.

- [ ] **Step 3: Add the constants and readers**

In `src/pokemon_env/ram.py`, add near `MUSEUM_TICKET_ADDR`/`MUSEUM_TICKET_BIT`
(lines 27-28):

```python
OAK_PARCEL_ADDR = 0xD74E
OAK_PARCEL_BIT = 1
OAK_POKEDEX_ADDR = 0xD74B
OAK_POKEDEX_BIT = 5
```

And add near `museum_ticket_set` (after line 105):

```python
def oak_parcel_set(mem: Emulator) -> bool:
    return read_bit(mem, OAK_PARCEL_ADDR, OAK_PARCEL_BIT)


def oak_pokedex_set(mem: Emulator) -> bool:
    return read_bit(mem, OAK_POKEDEX_ADDR, OAK_POKEDEX_BIT)
```

- [ ] **Step 4: Extend the module's Attribution docstring**

`src/pokemon_env/ram.py`'s current module docstring (lines 1-8):

```python
"""Typed readers over Pokemon Red/Blue's RAM map.

Addresses and decoding are read from PWhiddy/PokemonRedExperiments' verified
readers, not inferred from constant names. Source of truth:
https://datacrystal.romhacking.net/wiki/Pokémon_Red/Blue:RAM_map

Everything here is a pure function over the Emulator Protocol's read_memory,
so all of it is testable against a synthetic bytearray."""
```

Replace with:

```python
"""Typed readers over Pokemon Red/Blue's RAM map.

Addresses and decoding are read from PWhiddy/PokemonRedExperiments' verified
readers, not inferred from constant names. Source of truth:
https://datacrystal.romhacking.net/wiki/Pokémon_Red/Blue:RAM_map

oak_parcel_set/oak_pokedex_set additionally match
baselines/red_gym_env.py:509-510 -- v1, inside a commented-out block (opens
line 503, closes line 523) and absent from v2/red_gym_env_v2.py entirely, so
this is not a v2-vetted address the way the rest of this module's readers
are. Confirmed directly against the real ROM this session, not merely a
reading of that dead code.

Everything here is a pure function over the Emulator Protocol's read_memory,
so all of it is testable against a synthetic bytearray."""
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_pokemon_env_ram.py -k "oak_parcel or oak_pokedex" -v`
Expected: 4 passed.

- [ ] **Step 6: Prove one test can fail for the right reason**

Temporarily change `OAK_PARCEL_BIT = 1` to `OAK_PARCEL_BIT = 0`, rerun
`test_oak_parcel_set_reads_bit_one_of_0xd74e` — expect FAIL (reads bit 0,
which is 0 in the test's `0b0000_0010`, so it returns `False` against an
expected `True`). Revert the change.

- [ ] **Step 7: Run the full fast suite and the audit**

```bash
uv run pytest
python /Users/theelusivegerbilfish/.claude/skills/pytest-expert/scripts/audit_tests.py tests/unit/test_pokemon_env_ram.py
```
Expected: full suite green, coverage still ≥93%; audit reports no findings
for the new tests.

- [ ] **Step 8: Commit**

```bash
git add src/pokemon_env/ram.py tests/unit/test_pokemon_env_ram.py
git commit -m "feat(pokemon_env): add oak_parcel/oak_pokedex RAM readers"
```

---

## Task 2: Interactive drive harness

**Files:**
- Create: `scratch/drive_init_state.py` (gitignored — never committed; see
  Global Constraints)

**Interfaces:**
- Consumes: `pokemon_env.init_state.ButtonPress`, `generate_init_state`;
  `pokemon_env.emulator.PyBoyEmulator`; `pokemon_env.ram.{game_coords,
  party_size, badge_count, read_money, in_battle, oak_parcel_set,
  oak_pokedex_set}` (the last two from Task 1).
- Produces: a CLI every phase task below (Tasks 4-7) drives directly; no
  other code imports this module.

Per the spec's "How to build the script reliably" section: reconstructing a
script from independently-authored fragments drifts silently. The harness
below instead applies one small, explicit batch of presses to a *resumed*
checkpoint each invocation and prints ground truth immediately after —
`generate_init_state` already does exactly the press/tick/release/tick
pattern needed, so the harness reuses it rather than re-implementing it.

- [ ] **Step 1: Write the script**

```python
"""Interactive tool for authoring INTRO_SCRIPT phase by phase against a real
ROM. Not part of the package -- gitignored under /scratch/, run directly:

    uv run python scratch/drive_init_state.py \
        --load scratch/checkpoints/phase_a_03.state \
        --batch "up:8,None:40" \
        --save scratch/checkpoints/phase_a_04.state \
        --screenshot scratch/checkpoints/phase_a_04.png

Omit --load to boot fresh. Prints RAM state after applying the batch, and
writes both a resumable checkpoint and a screenshot -- open the screenshot to
catch "numerically right but a dialogue box is still open", which is exactly
the trap docs/superpowers/specs/2026-08-29-init-state-pokedex-skip-design.md's
second defect describes RAM-only checks missing.

Building INTRO_SCRIPT any other way -- concatenating independently-authored
fragments without re-running each one live -- has already been shown (same
spec) to silently diverge partway through.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from pokemon_env import ram
from pokemon_env.emulator import PyBoyEmulator
from pokemon_env.init_state import ButtonPress, generate_init_state

_ROM = "Pokemon Red.gb"
# Bounds, not guesses: the longest existing wait in INTRO_SCRIPT is 600
# frames; these catch a fat-fingered extra digit before it burns minutes of
# real emulator time, and a batch this long is never actually needed --
# one interactive step is meant to be small enough to verify by eye.
_MAX_BATCH_PRESSES = 200
_MAX_FRAMES_PER_PRESS = 2000


def parse_batch(spec: str) -> tuple[ButtonPress, ...]:
    """`"a:10,None:240,up:8"` -> a tuple of ButtonPress. Comma-separated
    button:frames pairs; button is one of pokemon_env.session.BUTTONS or the
    literal string "None" for a wait."""
    parts = spec.split(",")
    if not (1 <= len(parts) <= _MAX_BATCH_PRESSES):
        raise ValueError(f"batch has {len(parts)} presses, expected 1..{_MAX_BATCH_PRESSES}")
    presses = []
    for part in parts:
        button_str, sep, frames_str = part.partition(":")
        if not sep:
            raise ValueError(f"malformed press {part!r}, expected 'button:frames'")
        frames = int(frames_str)
        if not (1 <= frames <= _MAX_FRAMES_PER_PRESS):
            raise ValueError(f"frames={frames} outside 1..{_MAX_FRAMES_PER_PRESS} in {part!r}")
        button = None if button_str == "None" else button_str
        presses.append(ButtonPress(button=button, frames=frames))
    return tuple(presses)


def report(emulator: PyBoyEmulator) -> dict[str, object]:
    return {
        "coords (x, y, map)": ram.game_coords(emulator),
        "party_size": ram.party_size(emulator),
        "badges": ram.badge_count(emulator),
        "money": ram.read_money(emulator),
        "in_battle": ram.in_battle(emulator),
        "oak_parcel": ram.oak_parcel_set(emulator),
        "oak_pokedex": ram.oak_pokedex_set(emulator),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", type=Path, default=None, help="checkpoint to resume from; omit to boot fresh")
    parser.add_argument("--batch", required=True, help="e.g. 'a:10,None:240,up:8'")
    parser.add_argument("--save", type=Path, required=True, help="where to write the resulting checkpoint")
    parser.add_argument("--screenshot", type=Path, required=True, help="where to write the resulting PNG")
    args = parser.parse_args()

    if args.load is not None and not args.load.exists():
        raise FileNotFoundError(f"checkpoint {args.load} does not exist")

    emulator = PyBoyEmulator(_ROM)
    try:
        if args.load is not None:
            emulator.load_state(args.load.read_bytes())

        batch = parse_batch(args.batch)
        state = generate_init_state(emulator, batch)

        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_bytes(state)
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(emulator.screen_frame(), mode="L").save(args.screenshot)

        for key, value in report(emulator).items():
            print(f"{key}: {value}")
    finally:
        emulator.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke it against the real ROM**

```bash
uv run python scratch/drive_init_state.py \
    --batch "None:600,start:10" \
    --save scratch/checkpoints/smoke.state \
    --screenshot scratch/checkpoints/smoke.png
```
Expected: prints all seven `report()` lines with no traceback, and
`scratch/checkpoints/smoke.png` opens as a recognizable Game Boy title
screen (or the frame immediately after pressing start). This is the harness's
only verification — it is a throwaway interactive tool, not part of the
package under `src/`, so it is not exercised by `pytest` or
`audit_tests.py`.

- [ ] **Step 3: No commit**

`scratch/` is gitignored; nothing here enters git. Do not run `git add` on
this file.

---

## Task 3: Phase A — boot through a genuinely controllable bedroom

Fixes the spec's second defect: the *current* `INTRO_SCRIPT`'s exact
endpoint is mid-dialogue ("My name is OAK! People call me"), not the
controllable step its docstring claims. This phase must press further than
the current script does, and must prove controllability directly rather than
inferring it from RAM values a stalled dialogue can also produce.

**Files:** none yet (working state lives in `scratch/checkpoints/`; the
final accepted sequence is written to `scratch/phase_a.py` in Step 4)

- [ ] **Step 1: Iterate with the harness from boot**

Starting point (the current `INTRO_SCRIPT`'s own sequence, since its RAM
values *are* already correct — only the dialogue-closed state is missing):
boot + logo (~600 frames), `start` (title), wait, `a` (NEW GAME), wait, `a`
through Oak's opening narration beats, `a` to accept the default player
name, `a` to accept the default rival name, wait through the rest of the
cutscene, `a`. Apply each step as its own harness invocation (`--load` the
previous step's `--save` output), and after the sequence that currently ends
`INTRO_SCRIPT`, keep going: press `a` (or wait, if the box auto-advances)
until the screenshot shows the bedroom with **no bordered text box on
screen** — not just matching RAM values.

- [ ] **Step 2: Confirm controllability directly, not just visually**

Once the screenshot looks clear, run one more batch with a directional press
(e.g. `"down:8,None:40"`, the calibration below) and confirm `coords`
actually changed between the two checkpoints. A frozen dialogue box can look
static in a screenshot even mid-animation; a position that moves in response
to input is the actual thing "controllable" means. If it did not move, the
dialogue is still consuming input — keep pressing `a`/waiting and retry.

Movement calibration for this and every later phase: an 8-frame press
followed by a 40-frame idle settle reliably produces exactly one tile of
movement; a 20-frame press can overshoot to two tiles, and reading position
without the idle wait catches mid-transition values.

- [ ] **Step 3: Record acceptance**

Confirm via `report()`: `coords == (3, 6, 38)` (or wherever the one
extra confirmation-movement in Step 2 left it — recompute accordingly),
`party_size == 0`, `money == 3000`, `in_battle == False`, and the
screenshot from the batch immediately before the confirmation movement shows
no open text box.

- [ ] **Step 4: Write the verified phase to a file**

Create `scratch/phase_a.py`:

```python
"""Verified live this session (Task 3). Boot through a genuinely
controllable bedroom, no dialogue open -- fixes the stalled-dialogue defect
in the pre-existing INTRO_SCRIPT."""

from pokemon_env.init_state import ButtonPress

PHASE_A: tuple[ButtonPress, ...] = (
    # Fill in with the exact accepted sequence from Steps 1-2 above, MINUS
    # the confirmation-only movement press from Step 2 (that press proved
    # controllability; it is not part of the state phase B should resume
    # from -- phase B starts its own navigation from the resting bedroom
    # position).
)
```

- [ ] **Step 5: No commit**

`scratch/` is gitignored.

---

## Task 4: Phase B — bedroom exit through the rival battle to controllable in the lab

**Files:** `scratch/phase_b.py` (gitignored)

Facts to apply, confirmed against `pret/pokered` this session:

- The bedroom's exit stairs require pressing **UP**, not DOWN, at the column
  aligned with the stairwell — walking toward the "downstairs" tile from
  above is what triggers the transition.
- A direction change's first press only turns the sprite when it wasn't
  already facing that way; the second press in the same direction is what
  moves it.
- The mandatory rival battle triggers on a coordinate the player walks
  through in Oak's lab, not by facing/interacting with the rival's sprite —
  confirmed by direct experiment in the spec's authoring session (multiple
  approach angles with `a` were inert).
- The Squirtle ball is the map object at `(7, 3)` in Oak's Lab (see Global
  Constraints) — walk to and interact with that specific object, not
  whichever ball is nearest.

- [ ] **Step 1: Iterate with the harness from `scratch/phase_a`'s final checkpoint**

Navigate: bedroom exit (UP at the stairs column) → ground floor → out of the
house → Pallet Town → to Oak's rival-trigger coordinate (walking through it
triggers his "Wait!" interception automatically — no `a` press needed there)
→ lab entry → navigate to and select the ball at `(7, 3)` → confirm
`party_size == 1` after the selection dialogue closes → decline the
nickname prompt → the mandatory rival battle plays out (battle inputs:
`a` through move/attack prompts is sufficient — the battle is fixed,
not adversarial) → through to controllable again in the lab.

- [ ] **Step 2: Confirm controllability the same way as Task 3**

After the post-battle dialogue appears to clear, run one directional batch
and confirm `coords` changes. Do not accept the phase on RAM values alone.

- [ ] **Step 3: Record acceptance**

`party_size == 1`, `badges == 0`, `in_battle == False`, screenshot shows no
open text box, and the confirmation movement in Step 2 changed `coords`.

- [ ] **Step 4: Write `scratch/phase_b.py`**

Same shape as `scratch/phase_a.py` — a `PHASE_B: tuple[ButtonPress, ...]`
constant holding the exact accepted sequence (minus the confirmation-only
movement, same reasoning as Task 3 Step 4).

- [ ] **Step 5: No commit**

---

## Task 5: Phase C — lab to the Viridian Poké Mart, receive the parcel

**Files:** `scratch/phase_c.py` (gitignored)

Facts to apply, confirmed this session:

- Route 1's ledges are one-directional: freely passable walking south,
  blocked walking north except at specific gap columns. Both the Pallet
  Town↔Route 1 and Route 1↔Viridian City boundaries have this shape — plan
  the route south-to-north through Route 1 accordingly, or detour to a gap
  column if a direct path is blocked.
- The Viridian Poké Mart's entrance warp sits at `(29, 19)` on the Viridian
  City map — confirmed independently this session against
  `pret/pokered`'s `data/maps/objects/ViridianCity.asm:15`
  (`warp_event 29, 19, VIRIDIAN_MART, 1`), matching the spec's own claim
  exactly. Approach that exact tile; other tiles near the storefront do not
  trigger the interior warp.

- [ ] **Step 1: Iterate with the harness from `scratch/phase_b`'s final checkpoint**

Navigate: lab exit → Pallet Town → Route 1 (south-to-north, respecting the
ledge directionality above) → Viridian City → to `(29, 19)` → the interior
warp triggers automatically → inside the Mart, navigate to the clerk and
receive Oak's Parcel (dialogue: `a` through the exchange).

- [ ] **Step 2: Confirm controllability**

Same pattern as Tasks 3-4: after the parcel dialogue appears to clear, run a
directional batch and confirm `coords` changes inside the Mart.

- [ ] **Step 3: Record acceptance**

`oak_parcel_set(mem) is True`, screenshot shows no open text box, and the
confirmation movement changed `coords`.

- [ ] **Step 4: Write `scratch/phase_c.py`**

Same shape, `PHASE_C: tuple[ButtonPress, ...]`.

- [ ] **Step 5: No commit**

---

## Task 6: Phase D — back to Oak's lab, deliver the parcel, receive the Pokédex

**Files:** `scratch/phase_d.py` (gitignored)

Facts to apply, confirmed this session:

- The parcel-delivery trigger is **Oak's own object**, not a re-approach of
  the now-empty starter table. Confirmed against
  `pret/pokered`'s `scripts/OaksLab.asm` this session: entering the lab map
  while `EVENT_OAK_APPEARED_IN_PALLET` is set runs
  `OaksLabDefaultScript` → `OaksLabOakEntersLabScript`, which walks the
  `OAKSLAB_OAK2` sprite object (map object at `(5, 10)`, near the south
  exit) in via a scripted `PAD_UP, 8` movement, then
  `OaksLabToggleOaksScript` hides `OAK2` and shows `OAKSLAB_OAK1` (the same
  object at `(5, 2)` used during starter selection) for the actual
  follow/delivery dialogue. **This sequence fires automatically on walking
  into the lab** with the event flag set — it is not a "walk up and press A
  on a static NPC" interaction the way the Mart clerk was.

- [ ] **Step 1: Iterate with the harness from `scratch/phase_c`'s final checkpoint**

Navigate: Mart exit → Viridian City → Route 1 (south, the freely-passable
direction this time) → Pallet Town → walk into Oak's lab. Expect the
automatic Oak-enters/toggle/follow sequence to begin without further input;
apply waits generous enough to cover it, then `a` through the resulting
Pokédex-delivery dialogue.

- [ ] **Step 2: Confirm controllability**

Same pattern as every prior phase: after the delivery dialogue appears to
clear, run a directional batch and confirm `coords` changes.

- [ ] **Step 3: Record acceptance — the full target validation table**

| check | expected |
| --- | --- |
| `party_size` | `1` |
| `badges` | `0` |
| `oak_parcel_set` | `True` |
| `oak_pokedex_set` | `True` |
| screenshot | no open text box, in Oak's lab |
| confirmation movement | `coords` changes |

- [ ] **Step 4: Write `scratch/phase_d.py`**

Same shape, `PHASE_D: tuple[ButtonPress, ...]` — this one **does** include
the final confirmation movement, since Task 7 needs the assembled script to
end in a state already proven controllable, not one step short of it.

- [ ] **Step 5: No commit**

---

## Task 7: Assemble, re-verify continuously, regenerate `init.state`

**Files:**
- Modify: `src/pokemon_env/init_state.py` (`INTRO_SCRIPT`)
- Create locally (gitignored, not committed): `artifacts/init.state`

The spec is explicit that phase-by-phase verification does not guarantee the
concatenated whole works — PyBoy's tick-level timing is sensitive to exactly
how many frames preceded a given input, so the full script must be replayed
in one continuous run from boot before being accepted.

- [ ] **Step 1: Concatenate the four phases into `INTRO_SCRIPT`**

In `src/pokemon_env/init_state.py`, replace the current `INTRO_SCRIPT`
tuple (lines 33-51) with the concatenation of `PHASE_A + PHASE_B + PHASE_C
+ PHASE_D` from `scratch/phase_{a,b,c,d}.py` — copy the literal
`ButtonPress(...)` entries in, do not import from `scratch/` (that directory
is gitignored and must not become a runtime dependency of `src/`).

- [ ] **Step 2: Re-verify in one continuous run from boot**

```bash
uv run python -c "
from pokemon_env.emulator import PyBoyEmulator
from pokemon_env.init_state import INTRO_SCRIPT, generate_init_state
from pokemon_env import ram

emulator = PyBoyEmulator('Pokemon Red.gb')
state = generate_init_state(emulator, INTRO_SCRIPT)
print('party_size', ram.party_size(emulator))
print('badges', ram.badge_count(emulator))
print('oak_parcel', ram.oak_parcel_set(emulator))
print('oak_pokedex', ram.oak_pokedex_set(emulator))
from PIL import Image
Image.fromarray(emulator.screen_frame(), mode='L').save('scratch/final_check.png')
emulator.close()
open('artifacts/init.state', 'wb').close()  # placeholder; real write below
"
```

If any value in this one continuous run disagrees with what the
corresponding phase task recorded, the concatenation drifted — do not patch
around it with extra frames blindly; re-open `scratch/final_check.png`, find
which phase boundary the drift happened at, and redo that phase's harness
iteration (Tasks 3-6) with the corrected frame counts until a single
continuous run matches every phase's recorded acceptance values.

- [ ] **Step 3: Regenerate `artifacts/init.state` for real**

Once Step 2's continuous run matches every phase's recorded values:

```bash
uv run python -c "
from pathlib import Path
from pokemon_env.emulator import PyBoyEmulator
from pokemon_env.init_state import INTRO_SCRIPT, generate_init_state

emulator = PyBoyEmulator('Pokemon Red.gb')
state = generate_init_state(emulator, INTRO_SCRIPT)
emulator.close()
Path('artifacts').mkdir(exist_ok=True)
Path('artifacts/init.state').write_bytes(state)
print(len(state), 'bytes written')
"
```

- [ ] **Step 4: No commit for `artifacts/init.state`**

Gitignored per Global Constraints — confirm with
`git status artifacts/init.state` (expect no output / not tracked).

- [ ] **Step 5: Commit the `INTRO_SCRIPT` change**

```bash
git add src/pokemon_env/init_state.py
git commit -m "feat(pokemon_env): extend INTRO_SCRIPT past the Pokedex delivery"
```

---

## Task 8: Fix `init_state.py`'s docstrings

**Files:**
- Modify: `src/pokemon_env/init_state.py`

Two independent staleness issues, both flagged in the spec's "What changes"
section.

- [ ] **Step 1: Correct the module docstring**

Current (lines 10-13):

```python
The script below advances past the title, intro cutscene and naming screens to
the first controllable overworld step. Frame counts are generous -- the intro
animations are long and unskippable, and overshooting a menu is far cheaper
than landing in one.
```

Replace with:

```python
The script below advances past the title, intro cutscene, naming screens,
starter selection (Squirtle, specifically -- see the design spec for why),
the mandatory rival battle, and the Viridian Poke Mart parcel fetch and
delivery, landing in Oak's lab with the Pokedex received and the character
fully controllable. This mirrors PWhiddy/PokemonRedExperiments'
has_pokedex.state starting point; see
docs/superpowers/specs/2026-08-29-init-state-pokedex-skip-design.md. Frame
counts are generous -- the intro animations and scripted NPC movements are
long and unskippable, and overshooting a menu is far cheaper than landing in
one.
```

- [ ] **Step 2: Correct `state_hash`'s docstring**

Current (lines 70-75):

```python
def state_hash(state: bytes) -> str:
    """Will be recorded in checkpoints (wired in by Task 12, not yet a
    caller of this function) so a resume can detect that init.state changed
    underneath it. A different starting state would invalidate every reward
    baseline the checkpoint holds."""
    return hashlib.sha256(state).hexdigest()
```

Replace the docstring (this has been live since the PPO trainer landed —
`src/ppo/cli.py:35,222`, `src/ppo/trainer.py:150,173,187,295`, and
`tests/integration/test_ppo_smoke.py:34,131,145` already call it):

```python
def state_hash(state: bytes) -> str:
    """Recorded in every env checkpoint (src/ppo/cli.py, src/ppo/trainer.py)
    so a resume can detect that init.state changed underneath it -- see
    checkpoint.restore_env_checkpoint, which hard-rejects a mismatch. A
    different starting state invalidates every reward baseline the
    checkpoint holds."""
    return hashlib.sha256(state).hexdigest()
```

- [ ] **Step 3: Run the fast suite**

```bash
uv run pytest
```
Expected: unaffected — docstring-only change, no behavior touched.

- [ ] **Step 4: Commit**

```bash
git add src/pokemon_env/init_state.py
git commit -m "docs(pokemon_env): correct init_state.py's stale docstrings"
```

---

## Task 9: Rewrite and strengthen the smoke test

**Files:**
- Modify: `tests/integration/test_pokemon_env_smoke.py`

Research finding, this session: `pret/pokered`'s `ram/wram.asm` (fetched via
`gh api`) was searched for a "text box currently open" state flag —
`wTextBoxID` is an *input* parameter selecting which box to draw, not a
state flag, and `wAutoTextBoxDrawingControl` is a *configuration* bit
(disables automatic box drawing), not a state flag either. No documented
"is a dialogue box currently active" RAM flag exists in the disassembly.
Per the spec's own fallback ("a documented flag *if one is findable*"), this
task uses the screenshot-hash approach instead — PyBoy is deterministic
given a fixed ROM and a fixed input sequence, so the final frame's exact
bytes are reproducible run to run, the same property the existing
`test_generated_init_state_starts_in_the_bedroom_before_the_starter` already
relies on for RAM values (see its own docstring: "deliberately ROM-revision
sensitive").

**Interfaces:**
- Consumes: `pokemon_env.ram.{oak_parcel_set, oak_pokedex_set}` (Task 1),
  `pokemon_env.init_state.INTRO_SCRIPT` (Task 7), `PyBoyEmulator.screen_frame`
  (existing, `src/pokemon_env/emulator.py`).

- [ ] **Step 1: Get the real screenshot hash first**

The exact expected hash cannot be written from reasoning — it must come from
running the finished, accepted `INTRO_SCRIPT` once and reading back the
frame it actually produces:

```bash
uv run python -c "
import hashlib
from pokemon_env.emulator import PyBoyEmulator
from pokemon_env.init_state import INTRO_SCRIPT, generate_init_state

emulator = PyBoyEmulator('Pokemon Red.gb')
generate_init_state(emulator, INTRO_SCRIPT)
frame = emulator.screen_frame()
emulator.close()
print(hashlib.sha256(frame.tobytes()).hexdigest())
"
```

Record the printed hex digest — it is used as the exact expected value in
Step 2, replacing `\"<PASTE-THE-PRINTED-HASH-HERE>\"` below.

- [ ] **Step 2: Write the replacement test**

Replace `test_generated_init_state_starts_in_the_bedroom_before_the_starter`
(current lines 76-110) with:

```python
@_needs_rom
def test_generated_init_state_reaches_oaks_lab_with_the_pokedex() -> None:
    """The script's frame counts are guesses until this runs. RAM values
    alone already proved insufficient once: the previous version of this
    test asserted only position/party/money/battle-state and passed even
    though the actual screen was still mid-dialogue ("My name is OAK!
    People call me") -- none of those four values can detect "is a text box
    currently open". Oak's parcel-delivery text ends in another multi-box
    dialogue sequence with the same shape, so this asserts an exact
    screenshot hash in addition to RAM state: PyBoy is deterministic given a
    fixed ROM and a fixed input sequence, so the final frame's exact bytes
    are reproducible run to run on the same ROM revision, the same property
    this test's RAM assertions already rely on.

    This test is deliberately ROM-revision sensitive, same as its
    predecessor: a different ROM (a different release, a hacked ROM, Blue
    instead of Red) would produce different RAM values and a different
    frame, and that SHOULD fail loudly here rather than silently changing
    what all 64 environments load every reset."""
    import hashlib

    from pokemon_env import ram
    from pokemon_env.init_state import INTRO_SCRIPT, generate_init_state

    emulator = PyBoyEmulator(str(_ROM))
    generate_init_state(emulator, INTRO_SCRIPT)
    party_size = ram.party_size(emulator)
    badges = ram.badge_count(emulator)
    oak_parcel = ram.oak_parcel_set(emulator)
    oak_pokedex = ram.oak_pokedex_set(emulator)
    frame_hash = hashlib.sha256(emulator.screen_frame().tobytes()).hexdigest()
    emulator.close()

    assert (party_size, badges, oak_parcel, oak_pokedex) == (1, 0, True, True)
    assert frame_hash == "<PASTE-THE-PRINTED-HASH-HERE>"
```

(Replace `<PASTE-THE-PRINTED-HASH-HERE>` with Step 1's actual output before
running this test — it is not a placeholder left for later, it is the one
value in this plan that can only be produced by actually running the real
ROM, exactly like the frame counts throughout `INTRO_SCRIPT` itself.)

- [ ] **Step 3: Run it and confirm it passes**

```bash
uv run pytest -m slow tests/integration/test_pokemon_env_smoke.py::test_generated_init_state_reaches_oaks_lab_with_the_pokedex -v
```
Expected: 1 passed.

- [ ] **Step 4: Prove the test can fail — truncate the script**

Temporarily edit `INTRO_SCRIPT` (or a local copy of the test) to slice it
short before Phase C (`INTRO_SCRIPT[: len(PHASE_A) + len(PHASE_B)]`), rerun
the same test.

Expected: FAIL on a value mismatch (`party_size`, `oak_parcel`, or the frame
hash — whichever the truncation point leaves wrong), not a crash. Revert the
truncation.

- [ ] **Step 5: Update the `_needs_init_state` skip reason if needed**

`tests/integration/test_pokemon_env_smoke.py:113-116` already reads
`init.state`'s presence generically; no change needed there — confirm by
inspection, no edit required.

- [ ] **Step 6: Run the full slow suite once**

```bash
uv run pytest -m slow tests/integration/test_pokemon_env_smoke.py -v
```
Expected: all tests in the file pass, including
`test_a_random_agent_drives_four_real_envs_end_to_end` (unaffected in
structure per the spec — it reads `init.state` from disk generically).

- [ ] **Step 7: Commit**

```bash
git add tests/integration/test_pokemon_env_smoke.py
git commit -m "test(pokemon_env): strengthen the init.state smoke test past the Pokedex delivery"
```

---

## Task 10: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Full fast suite with coverage**

```bash
uv run pytest
```
Expected: green, coverage ≥93% (the floor `pyproject.toml:51` enforces).

- [ ] **Step 2: Full slow suite (real ROM)**

```bash
uv run pytest -m slow -v
```
Expected: green, including `test_ppo_smoke.py`'s tests (unaffected in
structure — they read `init.state` from disk and compute `state_hash` over
whatever is there).

- [ ] **Step 3: Negative-space audit on the touched source**

```bash
python /Users/theelusivegerbilfish/.claude/skills/negative-space-programming/scripts/audit_negative_space.py src/pokemon_env/ram.py src/pokemon_env/init_state.py --select NSP002,NSP003,NSP005,NSP006,NSP007
```
Expected: no findings — `generate_init_state`'s existing bound check
(`frames < 1` raises) and the pure, unbranching new readers don't introduce
an unbounded loop or a swallowed exception.

- [ ] **Step 4: Confirm the operational consequence is real, not hypothetical**

```bash
git log --oneline -- 'artifacts/' 2>&1 | head -1
ls artifacts/*.pt 2>&1
```
If any PPO env checkpoint (`env_update*.pt`) exists locally from a prior
run, note to the user that it is now unresumable (per
`checkpoint.py:51-57`'s hard rejection on a hash mismatch) — this plan does
not delete it; that is the user's call.

- [ ] **Step 5: Report status**

Summarize for the user: `INTRO_SCRIPT` now reaches Oak's lab with the
Pokédex (Whidden-equivalent starting point), `artifacts/init.state`
regenerated locally, the smoke test rewritten and strengthened, full test
suite green. No commit for this step — it is a status report, not a code
change.

---

## Self-review notes

- **Spec coverage**: RAM readers + Attribution citation (Task 1) → spec
  §"New RAM readers"; `INTRO_SCRIPT` extension via live-verified phases
  (Tasks 2-7) → spec §"The extended INTRO_SCRIPT" and §"How to build the
  script reliably"; target validation table (Task 6 Step 3, re-run in Task
  7 Step 2) → spec §"Target validation"; docstring corrections (Task 8) →
  spec §"What changes"; test rewrite + rename + strengthened assertion +
  prove-it-can-fail (Task 9) → spec §"Testing". The spec's two "No change
  needed" items (`rewards.py`, `AUX_STATE_VERSION`) are deliberately *not*
  tasks — verified against the current code in Global Constraints and the
  spec text, nothing to do.
- **Non-goals respected**: no task advances `init.state` past the Pokédex
  point, changes reward weights, or touches `VecPokemonEnv.step()`
  concurrency (that is the sibling spec's job).
- **No placeholders except one, and it's inherent, not lazy**: Task 9 Step
  1's screenshot hash and Tasks 3-6's exact button/frame sequences cannot be
  authored from reasoning — they are measurements of a real, deterministic
  ROM run, exactly like every frame count already in the pre-existing
  `INTRO_SCRIPT`. Each such step names exactly what command produces the
  real value and what to do with it, which is the bar "No Placeholders"
  actually sets.
