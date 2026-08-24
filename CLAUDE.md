# Pokemon RL Project

Vision-based RL agent for Pokemon Red/Blue: a frozen contrastive-pretrained CNN
(SimCLR/BYOL) feeds latent frame vectors into a RoPE/GQA transformer sequence
model, trained via PPO. Full architecture rationale: `Pokemon_RL_Architecture_Plan.pdf`.

Sub-projects are designed and planned independently (see `docs/superpowers/specs/`)
in the order: data collection -> CNN pretraining -> sequence model -> PPO agent.
Each gets its own spec and implementation plan before code is written.

## Conventions

- **Package management:** `uv` only. No bare `pip`/`venv`. `uv add <pkg>`, `uv run <cmd>`.
- **SOLID**: single-responsibility modules with injected dependencies (network/IO
  clients, filesystem, external APIs) so core logic is unit-testable without
  hitting the network, ffmpeg, or a GPU.
- **TDD**: write the failing test first for any new logic. Pure/algorithmic code
  gets fast unit tests with synthetic fixtures; anything touching real
  video/network/Hub APIs gets a separate, explicitly slow/opt-in integration test.
- **Karpathy guidelines**: minimal surgical diffs, no speculative abstraction,
  verify one stage works end-to-end before building the next stage on top of it.
- **Observability first**: every pipeline/training component logs structured
  (JSON-lines) progress and emits a live Trackio run. Anything that filters or
  drops data (dedup, anomaly detection, reward shaping) logs *why*, and produces
  a periodic visual artifact (contact sheets, sample grids, reward curves) a
  human can sanity-check without reading raw logs.
- **Interface-fit over spec-compliance**: a component's job is to fit how it
  will actually be consumed, not just satisfy the literal spec text.
  Brainstorming must produce concrete, checkable interface contracts before
  implementation starts — no vague "reusable later." Verify integration
  requirements against context7 (`ctx7` CLI), official library docs, or the
  `inspect` module on the actual installed package; never guess a calling
  convention or interface shape from memory.
- **Production readiness for long/costly runs**: any component driving a
  long-running or paid unattended job must have logging, error handling,
  retries, and resume/checkpointing decided at design time (during
  brainstorming), not discovered after.

## Infra

- **Compute**: RunPod (CPU pods for data pipeline work, GPU pods for training).
- **Storage/datasets**: Hugging Face Hub (private dataset repos), Parquet shards
  using the `datasets` library's `Image` feature.
- **Data source**: curated YouTube longplay URLs, human-approved per video
  (see the data-collection spec) — never fully automated scraping/search.
