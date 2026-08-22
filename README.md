# Pokemon RL — Data Collection Pipeline

Turns human-curated YouTube Pokemon Red/Blue longplay videos into a grayscale,
deduplicated frame dataset on Hugging Face Hub, for contrastive (SimCLR/BYOL)
CNN pretraining. Design rationale lives in
`docs/superpowers/specs/2026-08-21-data-collection-pipeline-design.md`; this
file is the practical "how do I actually run it" guide.

The pipeline is two phases:

- **Phase A — `curate`**: interactive, runs on your machine, never touches
  the Hub. You review a candidate video and approve/reject a crop box.
- **Phase B — `run`**: unattended, meant for a RunPod CPU pod. Extracts,
  validates, deduplicates, and uploads every approved video's frames.

## Prerequisites

- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/) — this project uses `uv`
  exclusively (`uv sync`, `uv add`, `uv run`), never bare `pip`/`venv`.
- **`ffmpeg`** on your `PATH`. Required by both `curate` and `run` — neither
  works without it. macOS: `brew install ffmpeg`. Not required just to run
  the unit test suite (only the opt-in integration test and the real CLI
  commands need it).
- A Hugging Face account with a **private** dataset repo already created
  (e.g. `you/pokemon-frames`) — this pipeline does not create the repo for
  you.

## Setup

```bash
uv sync
```

Create a `.env` file at the repo root (already `.gitignore`d) with your HF
token:

```
HF_TOKEN=hf_...
```

`data-collection` loads `.env` automatically on every invocation. If no
token is found (env var or a cached `hf auth login` session), `run` fails
immediately with a clear error rather than partway through a long extraction.

## Workflow

### 1. Curate candidate videos

```bash
uv run data-collection curate "https://www.youtube.com/watch?v=<id>" --game red
```

What happens:

1. Resolves the video's real stream URL and resolution via `yt-dlp`
   (longplay uploads range from ~360p to 4K — there's no fixed assumption
   here).
2. Grabs one full, uncropped frame and proposes a crop box via ffmpeg's
   `cropdetect` filter (real pixel-based letterbox/pillarbox detection).
   Falls back to the full frame if nothing is detected.
3. Writes an annotated preview to
   `configs/templates/approved/_preview.png` and prompts:
   `[a]pprove / [m]anual entry / [r]eject`. Open the preview image to see
   the proposed box; `m` lets you type new `x`/`y`/`w`/`h` values and loop
   back to a fresh preview until it looks right.
4. On approval, appends the video — crop box, source URL, game — to
   `configs/video_sources.yaml`.

Nothing is ever written to the registry without an explicit `a`. Repeat for
every candidate video you want in the dataset. The spec's appendix has a
starting list of view-count-ranked candidates already checked for
availability.

**Smoke-testing a batch of candidate links first:** before spending time
curating, it's worth confirming every URL still resolves (uploads get taken
down, go region-locked, etc.). A quick one-off check, no `ffmpeg` needed:

```bash
uv run python -c "
from data_collection import extract
for url in ['https://www.youtube.com/watch?v=<id1>', 'https://www.youtube.com/watch?v=<id2>']:
    try:
        stream_url, w, h, _headers = extract.get_stream_info(url)
        print(f'OK   {w}x{h}  {url}')
    except Exception as exc:
        print(f'FAIL {exc!r}  {url}')
"
```

### 2. Run extraction (Phase B)

```bash
uv run data-collection run --repo-id you/pokemon-frames
```

Iterates every video in `configs/video_sources.yaml` not already marked
complete in the repo's `manifest.json`, streaming frames via `ffmpeg`
(never writing video to local disk), deduplicating near-identical frames,
and uploading Parquet shards plus a contact-sheet preview PNG per batch. A
crashed or interrupted run resumes mid-video from its last checkpoint
(`--max-concurrent-videos` controls how many videos run in parallel).

Exits nonzero if any video failed (check the JSON logs on stdout for
`"message": "video_failed"` entries, and `manifest.json` in the dataset
repo for the reasons).

This is designed to run unattended on a RunPod CPU pod, but works
identically run locally against a small registry for testing.

## Project layout

```
configs/
  video_sources.yaml       # the registry -- populated by `curate`, read by `run`
  templates/
    approved/               # preview scratch file from the last curate run
src/
  data_collection/         # this pipeline: registry, dedup, extract,
                            # batcher, hf_uploader, curation, pipeline, cli
  observability/           # shared logging/tracking/visualization -- used by
                            # every sub-project in this repo, not just this one
tests/
  unit/                    # fast, no network/ffmpeg/GPU
  integration/             # opt-in, `-m slow`, needs a real clip + ffmpeg
```

## Testing

```bash
uv run pytest            # unit suite only (default -- the slow marker is deselected)
uv run pytest -m slow    # + the opt-in integration test (needs ffmpeg + POKEMON_RL_TEST_CLIP)
```
