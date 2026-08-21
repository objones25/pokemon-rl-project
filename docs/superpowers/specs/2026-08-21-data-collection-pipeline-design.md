# Data Collection Pipeline — Design Spec

Date: 2026-08-21
Status: Approved for planning

## Purpose

Produce a dataset of Game Boy-resolution (160x144) Pokemon Red/Blue gameplay
frames, suitable for contrastive pretraining (SimCLR/BYOL) of the frozen CNN
feature extractor described in `Pokemon_RL_Architecture_Plan.pdf`. Target: 20-50
hours of source video, sampled at 2 FPS, filtered for redundancy — an expected
output significantly smaller than the raw 150k-360k frame ceiling once
perceptual-hash dedup removes static/redundant stretches.

Out of scope: the contrastive training loop itself, the transformer, and the PPO
agent — each is a separate sub-project with its own spec.

## Why this design, not alternatives considered

- **Source data**: Self-play via an emulator (PyBoy) was considered and rejected.
  A random/scripted policy cannot progress far into the game, so it would yield
  poor state-space coverage — the opposite of what contrastive pretraining
  needs — in addition to consuming paid RunPod GPU/CPU hours for worse data.
  Curated YouTube longplays give broad, human-quality state coverage for free
  (bandwidth/storage cost only).
- **Storage**: Hugging Face Hub (private dataset repo) over AWS S3. No AWS
  account exists; HF Hub is free-tier friendly, versioned, and the target
  training code will load it directly via `datasets.load_dataset(streaming=True)`
  with no separate client setup.
- **Shard format**: Parquet with the `datasets` library's `Image` feature, over
  WebDataset tar shards. Native Hub dataset-viewer support and zero extra
  tooling on the training side outweigh WebDataset's raw throughput advantage
  at this dataset size.
- **Video sourcing**: A human-curated, fixed URL list, not automated
  YouTube search. Automated search risks silently ingesting filtered/cropped/
  webcam-overlay videos that would corrupt crop assumptions across an entire
  unattended run. Candidates for that list are discovered via a one-off
  `yt-dlp` search (ranked by view count, as a cheap proxy for a clean/popular
  upload) — this is a manual research aid, not a pipeline component; every
  candidate still goes through explicit human approval in Phase A before it
  enters the registry.
- **Crop-box detection**: Moved to curation time (human-approved, per video),
  not left as a runtime auto-detect step. A silent low-confidence match during
  an unattended multi-hour RunPod run could corrupt a large chunk of the
  dataset before anyone notices; a one-time human sign-off per video costs
  little and removes that risk entirely.
- **Dedup**: Perceptual hashing (pHash) over raw pixel-diff thresholding.
  pHash tolerates minor compression noise while still catching near-duplicate
  frames (static menus, dialogue boxes, standing still) that pixel-diff can
  miss (e.g. scrolling text over an otherwise static background).
- **Execution environment**: RunPod CPU pod, matching the zero-local-disk
  streaming approach — `yt-dlp` extracts a stream URL, `ffmpeg` crops/downsamples
  and pipes raw frames to stdout, nothing is ever written to disk as video.
- **Color**: Every frame is converted to grayscale (`format=gray` in the same
  `ffmpeg` filter chain as the crop) regardless of source. Original Game Boy
  Red/Blue is 4-shade monochrome — color in captured footage (a GBC recolor
  palette, a channel's own DMG tint/shader choice) is a capture-time artifact,
  not game state. Normalizing it away removes nuisance variance up front
  instead of asking the contrastive encoder to learn invariance to it, and
  incidentally shrinks stored frame size ~3x versus RGB.

## Architecture

Two phases, deliberately separated by trust level:

### Phase A — Curation (interactive, local, human-gated)

A CLI tool, given a candidate YouTube URL:

1. Downloads only the first 5-60 seconds via `yt-dlp` (smoke-test clip).
2. Runs `cv2.matchTemplate`, in grayscale, against a small bank of reference
   Game Boy UI crops (battle-menu box, text-box border, start-menu) to find
   candidate crop coordinates and a match-confidence score per template.
   Grayscale matching is robust to a source video's color/tint choice and
   matches what Phase B will actually see.
3. Renders an annotated preview: the proposed crop box drawn on a sample
   frame, plus a thumbnail grid of the cropped-only result.
4. The human reviews the preview and either:
   - **Approves** — the crop box `(x, y, w, h)`, the winning template, and its
     match-confidence score (the runtime baseline) are appended to the
     registry.
   - **Rejects** — the video is discarded; nothing is written.

The curation tool never runs unattended and never appends to the registry
without an explicit human approval step.

### Phase B — Extraction (unattended, RunPod CPU pod, per approved video)

Runs only against videos already present with approved crop boxes in the
registry.

1. `yt-dlp` extracts the direct stream URL (no video download).
2. `ffmpeg`, given the stream URL and the approved crop box, applies
   `crop=w:h:x:y,format=gray`, downsamples to 2 FPS, and writes raw
   single-channel frames to `stdout` via `image2pipe`.
3. Each frame is read into memory as a numpy array and passed through:
   - **Frame validator** — re-runs the _same_ template match inside the
     _same_ crop box. If the score falls well below the curation-time
     baseline (ad break, sponsor cutaway, a re-encode that no longer matches
     the sampled crop), the frame is dropped and logged. If the rolling
     anomaly-drop rate for a video exceeds a threshold (e.g. >20% over a
     window), extraction for that video **halts** and it's flagged for
     human review rather than continuing to burn compute against a crop
     assumption that may no longer hold.
   - **Dedup filter** — perceptual hash compared against the last _kept_
     frame; frames within a small Hamming distance are dropped.
4. Accepted frames accumulate into batches (e.g. 500 frames) with metadata
   (video id, source timestamp, game version) and are written as a Parquet
   shard using the `datasets.Image` feature (single-channel grayscale), then
   pushed to the private HF dataset repo.
5. On successful completion of a video, its id is recorded in a
   `manifest.json` (stored in the dataset repo) so a crashed/restarted pod
   resumes at video granularity instead of reprocessing completed videos.
6. Transient `yt-dlp`/`ffmpeg`/network errors retry with bounded backoff;
   a video that exhausts retries is marked failed in the manifest and
   extraction moves on to the next video rather than aborting the run.

## Components

| Module            | Responsibility                                                              | Depends on                  |
| ----------------- | --------------------------------------------------------------------------- | --------------------------- |
| `registry`        | Load/validate `configs/video_sources.yaml` into typed `VideoSource` records | none (pure)                 |
| `curation`        | Interactive smoke-test + preview tool for Phase A                           | `yt-dlp`, `cv2`, `registry` |
| `extract`         | `yt-dlp` + `ffmpeg` subprocess wrapper yielding raw frames                  | `yt-dlp`, `ffmpeg`          |
| `frame_validator` | Re-check extracted frames against curation-time crop/template baseline      | `cv2`                       |
| `dedup`           | Perceptual-hash near-duplicate filter                                       | `imagehash` (or equivalent) |
| `batcher`         | Accumulate accepted frames + metadata into Parquet shards                   | `datasets`                  |
| `hf_uploader`     | Push shards to the HF dataset repo; read/write `manifest.json`              | `huggingface_hub`           |
| `pipeline`        | Orchestrate Phase B end to end across all approved, incomplete videos       | all of the above            |
| `cli`             | Entry points for both phases                                                | `curation`, `pipeline`      |

Each module takes its external dependencies (HTTP/Hub client, subprocess
runner, filesystem) as constructor/function arguments so core logic — hash
comparisons, threshold decisions, batch chunking, manifest resume logic — is
unit-testable without a network connection, ffmpeg, or GPU.

## Data flow

```
curated URL
  -> (human, Phase A) crop approval -> registry entry
  -> (Phase B, per video) yt-dlp stream URL -> ffmpeg crop+2fps -> raw frames
  -> frame_validator (drop anomalies, halt video if drop-rate too high)
  -> dedup (drop near-duplicates)
  -> batcher (accumulate -> Parquet shard)
  -> hf_uploader (push shard, update manifest)
  -> Trackio / structured logs (per-frame and per-video metrics)
```

## Observability

- Structured (JSON-lines) logs for every stage transition and per-video
  summary counts: frames sampled / kept / dropped-dedup / dropped-anomaly.
- A Trackio run per pipeline invocation logging live: frames/sec,
  dedup-rejection rate, anomaly-drop rate, cumulative frames uploaded.
- A contact-sheet PNG (grid of sampled kept frames) written per batch, so
  dedup/anomaly-detection behavior can be spot-checked visually without
  reviewing raw video.

## Error handling

- Curation tool: if no reference template clears a minimum confidence bar,
  the tool reports "no confident match" rather than silently proposing the
  best-of-a-bad-set crop — forces an explicit human decision.
- Runtime frame validator: per-frame anomalies are dropped and logged
  individually; a sustained anomaly rate halts the video and flags it,
  rather than either silently keeping bad frames or crashing the whole run.
- Extraction errors (network, `ffmpeg`, `yt-dlp`): bounded retry with
  backoff, then the video is marked failed in the manifest and the pipeline
  continues with the next video.

## Testing strategy

- **Unit tests** (no network/ffmpeg/HF calls, synthetic numpy frame
  fixtures): registry parsing/validation, perceptual-hash distance logic,
  anomaly-threshold decision logic, batch-chunking, manifest resume logic.
- **Integration test** (marked slow, opt-in, not run by default): the real
  Phase B pipeline against one short self-provided clip, writing to a local
  scratch Parquet file instead of pushing to HF — catches wiring bugs
  before spending RunPod time.
- The curation tool's smoke-test-and-preview behavior is exercised
  interactively per video; it is Phase A's product, not a separate test
  suite.

## Tooling

- Package management: `uv` (`uv add`, `uv run`) — no bare `pip`/`venv`.

## Appendix: initial candidate videos

Found via `yt-dlp ytsearch`, ranked by view count as a cheap proxy for a
clean/popular upload. None of these are in the registry yet — each still
needs Phase A curation (crop-box + template match approval) before use.
Combined runtime already well exceeds the 20-50h target, so not all of these
need to be used.

| Views | Duration | Channel | Title | Video ID | Game |
| --- | --- | --- | --- | --- | --- |
| 946,142 | 8:56:40 | Mariofan98 Longplays | Pokémon Red Color (GBC) Longplay Part 1/2 | `OHHSQxn_IOY` | Red |
| 262,056 | 8:45:31 | RabidRetrospectGames | POKEMON BLUE Full Game Walkthrough | `uyHNAu2WoTE` | Blue |
| 105,631 | 8:13:43 | Crystal Longplays | Longplay - Pokemon Red | `WjHDuVA5k5A` | Red |
| 42,595 | 18:34:36 | Long 'n Play | Pokémon Blue Version (Game Boy) Longplay | `oRRtP7C8W8s` | Blue |
| 41,243 | 12:37:56 | FCPlaythroughs | Pokémon Red playthrough ~Longplay~ | `D1SrSFZrV7A` | Red |
| 29,406 | 11:11:00 | Kryschnack Longplay | Pokémon Blue Version (1996) Longplay | `YW29l3jJXr4` | Blue |
| 21,727 | 9:07:04 | LongplayArchive | Longplay of Pokémon Red/Blue | `V6yBLNGIeBQ` | Red/Blue |

Dropped from consideration: Nintendo Utopia's "Pokemon Red for Game Boy ᴴᴰ
Full Playthrough" (466,200 views, `hYcotDHI0dg`) — a 2:25:50 runtime is
implausibly short for a full no-glitch playthrough, suggesting skips or
speedup; excluded rather than risk polluting the dataset with non-representative
pacing.
