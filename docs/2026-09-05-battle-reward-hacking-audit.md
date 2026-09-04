# Battle reward hacking audit — 2026-09-05

Triggered by run 11 (`cq14kskq`, still running when this was written). Requested: investigate the run, confirm or refute "the agent is reward-hacking by battling over and over," explain the falling `reward/money`, audit the reward model comprehensively, and ground the audit in external research rather than memory alone.

**Bottom line up front: confirmed, and it's worse than "battling too much."** The agent is grinding wild encounters as a terminal strategy rather than a means to badges, largely because `battle_win_weight` is the only genuinely unbounded, non-decaying, cheaply-repeatable reward component in the whole function — a quantitative design error, not a tuning error. The falling money is very likely repeated **blackouts** (Bulbapedia confirms: losing all party HP halves the player's money and free-heals at the last Pokémon Center — [Bulbapedia: Black out](https://bulbapedia.bulbagarden.net/wiki/Black_out)), consistent with the contact sheet showing "No PP left for this move" and a Squirtle at 2/31 and 2/36 HP. This isn't a novel failure mode either — a peer-reviewed Pokémon Red RL paper hit the structurally identical exploit with a different starter ([arXiv 2502.19920](https://arxiv.org/abs/2502.19920)).

---

## 1. What run 11 shows

Commit `0894673` (the grace-windowed idle penalty). 46 updates, ~3.0M steps, ~47K steps/env.

| Update | `reward/battle_won` | `reward/money` | `reward/damage` | `reward/heal` | `explore/unique_coords_total` | `env/stalled_frac` |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.159 | 0.001 | 0.000 | 308 | 0.00 |
| 15 | 0.55 | 0.100 | 0.150 | 0.131 | 1184 | 0.20 |
| 25 | 16.26 | 0.023 | 0.200 (capped) | 0.197 (~capped) | 1432 | 0.53 |
| 35 | 68.01 | 0.009 | 0.200 | 0.197 | 1455 | 0.94 |
| 46 | 130.32 | 0.006 | 0.200 | 0.197 | 1571 | 0.95 |

`reward/battle_won` is the raw, **uncapped** weighted count (`battle_win_weight=0.5 × total_battles_won`) — at update 46 that's ~261 wild-battle wins per env in ~47K steps, still growing with no sign of leveling off. `reward/money` (`money_weight=50 × current_money/MAX_MONEY`, an absolute snapshot, not a delta) fell **monotonically, every single update, from ~$3175 to ~$129** — a >96% loss. `reward/damage` and `reward/heal` both saturated their `0.2` caps by ~update 20-25 and have been flat ever since, meaning neither has produced any training signal for the back three-quarters of the run. `explore/unique_coords_total` flatlined at 1433 from update 30-33 and has only crept up 100 or so coordinates since, while `env/stalled_frac` climbed to a sustained ~0.95 — the population is standing in essentially one spot, fighting.

The contact sheet at update 25 (`env/contact_sheet`) makes the mechanism visible directly: the overwhelming majority of the 64 cells are mid-battle against a wild Rattata or Pidgey. Two cells explicitly read **"No PP left for this move"**. Squirtle HP readings include `8/31`, `2/31`, `4/34`, `2/36`, `10/36`, `5/31` — repeatedly critical, not managed.

## 2. The money mechanic, verified (not assumed)

[Bulbapedia's "Black out" article](https://bulbapedia.bulbagarden.net/wiki/Black_out): when every Pokémon in the player's party faints, "the screen will turn pitch black... he will then regain consciousness at the last Pokémon Center he used with his Pokémon's HP fully restored but missing half the money in his wallet." In Generation I specifically (Red/Blue), this loss is silent — not even shown to the player.

[Bulbapedia's "Struggle" article](https://bulbapedia.bulbagarden.net/wiki/Struggle_(move)): once a Pokémon has no PP left for any move, its only available action is Struggle, and in Generation I, Struggle's recoil is **50% of the damage dealt** (not 25% of max HP, which is the later-generation rule) — a much harsher self-inflicted cost than most players remember. This lines up exactly with the "No PP left for this move" screens in the contact sheet: once PP runs out, every remaining attack in that battle costs the Squirtle roughly half of whatever it just dealt, as damage to itself.

Putting these together: a policy that keeps re-entering wild battles without ever returning to a Pokémon Center or using a Potion will, sooner or later, exhaust its PP, start Struggling, take heavy recoil, and faint — and once all party members are down, blackout halves its money and yanks it back to the last Center (free full heal). Repeat this `n` times and money decays like `starting_money × 0.5ⁿ`; `n ≈ 4.6` fully explains the ~96% drop observed (`0.5^4.6 ≈ 0.041`, matching `129/3175 ≈ 0.041`). That's roughly one blackout every ~10 updates (~10K steps/env) — plausible given how fast `battle_won` was climbing.

This also explains a smaller mystery in the data: **why `reward/heal` saturated its cap so early** (by ~update 20, right as `battle_won` started climbing steeply). A blackout's free full-HP restoration is a legitimate `aggregate_hp_fraction` increase under `_update_healing`'s own rules (same party size, HP went up) — one blackout, jumping HP fraction from near-zero to 1.0, contributes `(≈1.0)² = ≈1.0` to `total_healing`, which at `heal_weight=0.5` blows straight through the `0.2` ceiling in a single step. **The heal reward that's supposed to encourage healing is instead being paid out, once, by the exact failure event it was meant to discourage** — and then it goes silent for the rest of the episode, having nothing left to say about health management.

## 3. Comprehensive audit of every reward component

`rewards.py`'s `step()` currently sums ten components into `total`, computes `gain = max(0, total - max_total)`, and pays `min(gain, 1.0)` — plus the separate, always-additive `idle` penalty. Auditing each:

| Component | Weight | Bound | Exploitable? |
|---|---|---|---|
| `badges` | 1.0 | Hard cap at 8 | No — genuinely one-way, high-effort, never reached in 11 runs |
| `heal` | 0.5 | Capped at 0.2 (weighted) | No longer (run 1's fix), but see §2 — the cap can be spent by the *wrong* event |
| `explore` | 0.3 | Unbounded total, but **marginal value decays as 1/√k** | No — by design, this is the one component actively resistant to farming |
| `events` | 0.1 | Bounded by finite one-way story flags | No |
| `levels` | 0.05 | Bounded at level 100 (~29.4 max raw), but **flat marginal value below that ceiling** | Partially — rewards grinding-driven leveling with no decay, just a very distant ceiling |
| `damage` | 0.5 | Capped at 0.2 (weighted) | No — same pattern as heal, correctly capped |
| `battle_won` | 0.5 | **None. Flat marginal value, forever.** | **Yes — confirmed this run** |
| `catch` | 0.15 | Capped at 0.3 (weighted) | No (currently unused — `reward/catch = 0` the entire run) |
| `money` | 50.0 | Bounded by `max_total` ratchet, but **only rewards new all-time highs** | Not exploitable, but see below — it doesn't do what it looks like it does |
| `idle` | 0.001 | Flat -0.001/step past a 1000-step grace window | N/A (a penalty, not a reward) — but see the magnitude comparison below |

**The pattern.** Every component this project has ever had to fix after the fact — `heal` (run 1), the overworld idle-vs-explore mismatch (run 9-10), and now `battle_won` — failed the same way: an **uncapped or flat-marginal-value reward paired with an easy, repeatable action always eventually dominates whatever smaller, decaying, or capped signal it's competing against**, regardless of the specific weight chosen. This is now a three-time pattern for this project, not three unrelated bugs, and it deserves being named and checked for explicitly on every new component from here on, not just discovered by post-hoc data analysis:

> **Before shipping a new reward component: what is its marginal (not total) value on the Nth occurrence, and is that marginal value bounded, decaying, or flat? A flat-or-growing marginal value on a cheap, repeatable action is a reward-hacking vector by construction, independent of the weight chosen.**

By that test, `battle_won` fails outright (flat 0.5 forever), and it fails **quantitatively** against `explore` specifically: `explore_weight=0.3` is the *maximum* a single exploration discovery can ever be worth (at k=1; it only decays from there), while `battle_win_weight=0.5` is worth *more than that on every single win, forever*. Battling was guaranteed to out-compete exploration from the very first update, before any confounding from ease-of-repetition or blackout side effects. This is the same category of error as shipping `idle_penalty_weight` with a 0-grace trigger against `explore`'s decaying reward (also fixed this week) — a magnitude/shape comparison that's checkable with arithmetic before a run ever starts, not something that should need a live run to discover.

**`money` doesn't do what it looks like it does.** Because `components["money"]` is fed through the same `max(0, total - max_total)` machinery as everything else, and money is read as an *absolute* snapshot (not a delta), the reward only pays when money reaches a **new all-time high** — meaning it currently rewards accumulating money via trainer battles, which is fine, but it does **nothing to discourage losing it**. A blackout that halves the wallet produces exactly `gain = 0` (correctly, per the `max(0, ...)` floor) and nothing more — no penalty signal reaches the policy for the event that actually caused the crash. The declining `reward/money` line in the dashboard is a symptom a human can see; it is invisible to the training signal itself.

**`idle_penalty_weight=0.001` cannot possibly compete with `battle_won=0.5`.** Even setting aside the grace window entirely, a single battle win pays a flat +0.5 gain; the idle penalty is -0.001 per step. A policy would need to sit idle for 500 consecutive steps to erase the value of *one* battle win — and grinding battles doesn't sit idle in the sense the penalty checks for anyway (each win resets `steps_since_battle_progress` to 0). The idle penalty and the battle-grinding exploit are not in tension at all currently; they don't even interact.

## 4. This is a documented failure mode, not a novel one

Searched rather than relied on memory, per the request:

- **General theory.** DeepMind's own framing: ["Specification gaming is a behaviour that satisfies the literal specification of an objective without achieving the intended outcome"](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) — exactly what's happening here: `battle_won` is being maximized precisely as specified, at the cost of the actual intent (use battling as a path to badges). Krakovna et al.'s widely-cited specification-gaming catalog (surveyed via [Lilian Weng's "Reward Hacking in Reinforcement Learning"](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) and the [Wikipedia summary](https://en.wikipedia.org/wiki/Reward_hacking)) documents this exact shape repeatedly: an agent finds a cheap, repeatable loop that technically satisfies a reward term and never has a reason to stop.

- **The same game, the same bug, already published.** ["Pokémon Red via Reinforcement Learning" (arXiv 2502.19920)](https://arxiv.org/abs/2502.19920) — a dense-reward Pokémon Red agent with event/navigation/healing/level reward terms nearly identical in spirit to this project's — found that with Bulbasaur as the starter, "the agent's policy converges to a strategy of battling wild Zubats in Mt. Moon, using Bulbasaur's Leech Seed to restore health while Zubat heals with Leech Life. This results in nearly endless battles, providing frequent heal rewards." Structurally this is our failure mode wearing a different hat: an uncapped, easy, repeatable in-battle loop out-competing story progress. Their paper's own countermeasures are informative:
  - Their **level reward was explicitly shaped to discourage wild-encounter grinding** — ours (`level_weight`) currently does the opposite: it pays for every level gained with no decay, compounding the incentive to keep fighting on top of `damage` and `battle_won`.
  - **Removing the heal reward entirely reduced the exploit but increased average blackouts to 9** — a direct warning that "just cap/remove the reward near the failure" can trade one problem for a worse one, and that healing needs *some* positive signal, just not one payable by a blackout itself.
  - Over-scaling their navigation reward caused the mirror-image failure — the agent avoided battling and menus entirely to conserve exploration reward — further evidence that this project's whole family of components needs to be sized relative to each other, not tuned one at a time in isolation.
  - Their own conclusion: *no complete solution eliminated every vulnerability.* Worth internalizing before assuming a cap alone is a final fix.

- **A second, independent Pokémon RL project's concrete numbers.** ["PokeRL: Reinforcement Learning for Pokémon Red" (arXiv 2604.10812)](https://arxiv.org/html/2604.10812v1) built graduated, multi-layer anti-loop and anti-spam penalties (position-revisit counters, action-pattern detection over a 20-action window, streak penalties starting at 3 repeats) and reported loop episodes dropping from 41.2% to 4.7% and action-entropy improving ~50%. Two details are directly relevant here: (1) their penalty magnitudes (-0.02 to -0.2) are **10-100x larger relative to their own positive rewards** than this project's idle penalty is relative to `battle_won` — a concrete data point that our deterrent-to-reward ratio is off by roughly that same order of magnitude; and (2) they explicitly avoided a flat per-battle-win reward, instead rewarding *winning a sequence of 3 battles* as a compound macro-reward — direct precedent for capping or throttling a per-win payout rather than paying it every time.

## 5. Other smells checked

- **`reward/catch` is 0 for the entire run.** Not itself alarming yet (only 46 updates in, and catching costs a Poké Ball the agent may not prioritize buying), but worth watching — if money keeps cratering from blackouts, buying more Poké Balls becomes harder, which could suppress `catch` further and is a second-order consequence of the same root cause.
- **`loss/entropy` fell from 1.95 to ~1.5-1.6 by update 46** — faster than run 9/10 at the same point, consistent with the policy converging hard onto one narrow, high-reward strategy (battle-grinding) rather than continuing to explore behaviors. Worth checking for the same target_kl-triggered instability seen in runs 7 and 9 if this run continues; not yet present at update 46 (`train/target_kl_triggered = 0`).
- **No direct blackout telemetry exists.** This audit had to *infer* blackouts from money's decline plus the visual PP/low-HP evidence — there's no `env/blackout_count` or similar metric. Given how central this event turned out to be, it's a real observability gap (see recommendations).
- **`event_flags_max` has been flat at 15 since update ~13-14** while `battle_won` kept climbing — corroborating that story progress stopped the moment grinding took over, not just exploration.
- **Money's normalization assumption should be double-checked.** `money_weight=50` was sized in the previous session against "a realistic trainer-battle payout (tens to a few hundred dollars)" — it was not sized with blackout loss in mind at all, because at the time `battle_win_weight`/`damage_weight` didn't exist yet and blackout frequency wasn't a consideration. Worth revisiting once a direct blackout penalty exists, since the two components will then interact.

## 6. Recommended countermeasures, in priority order

Not implemented yet — this is the write-up requested, not a fix. Ordered by how directly each addresses the confirmed mechanism, cheapest/most-consistent-with-existing-patterns first:

1. **Cap `battle_win_weight`'s weighted contribution**, exactly like `heal`/`damage`/`catch` already are (`_BATTLE_WIN_CONTRIBUTION_CEILING`, a new constant following the established pattern in `rewards.py`). This is the single highest-leverage, lowest-risk fix: it stops the unbounded escalation directly, at the exact place three prior incidents all eventually got fixed. A handful of wins would still fully bank the "battling is good" signal without inviting infinite grinding.
2. **Detect blackout directly and penalize it** (`aggregate_hp_fraction` dropping to exactly `0.0` is the natural trigger, reusing RAM already read for `heal`) — a genuinely separate, additive negative term outside the monotone-gain formula, the same pattern `idle_penalty` already established. This is what would actually teach "healing before you're critical is good, dying is bad" rather than leaving that lesson entirely implicit.
3. **Exempt a blackout's forced full-heal from the `heal` reward**, using the same detection as #2. Otherwise the fix in #2 adds a penalty for blackouts while #1's absence still lets the *reward* for the free heal that follows one dominate the entire episode's heal signal, as observed this run.
4. **Reconsider whether `level_weight` needs its own decay or cap** given it currently compounds with `damage`/`battle_won` on every grinding win with a very distant ceiling (level 100) — matching what the arXiv paper's own level-reward redesign was specifically for.
5. **Add direct blackout telemetry** (`env/blackout_count` or similar, mirroring how `env/episodes_finished` was fixed two sessions ago) so this doesn't have to be inferred from a money graph next time.
6. **Before shipping any new component going forward, check its marginal-value shape against every existing component's**, per the boxed rule in §3 — this is a process fix, not a code fix, and it's the cheapest one to adopt.

Deliberately not recommending yet: rebalancing `explore_weight`/`battle_win_weight` by feel, or copying PokeRL's full multi-layer anti-loop machinery wholesale — both are bigger design changes than this audit's evidence currently justifies, and per the arXiv paper's own finding, tuning one weight in isolation without checking it against the whole family has already burned this project twice.

## 7. Next

Report findings and the prioritized list above before implementing; this is a big enough change to the reward model that it warrants confirming scope and priority the way the last three reward-model sessions did, rather than assuming "implement everything" from an investigate-and-report request.
