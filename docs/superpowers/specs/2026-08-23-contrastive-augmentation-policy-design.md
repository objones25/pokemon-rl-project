# Contrastive Pretraining — Augmentation Policy Design Spec

Date: 2026-08-23
Status: Approved for planning

## Purpose

Define the data-augmentation and positive-pair-construction policy for
contrastive pretraining (SimCLR/BYOL) of the frozen CNN feature extractor
described in `Pokemon_RL_Architecture_Plan.pdf`. The encoder's job is to
compress Game Boy-resolution (160x144, single-channel grayscale) Pokemon
Red/Blue frames into semantic latent vectors, frozen and later consumed by
the sequence transformer. Augmentation choice is the hard part of this
sub-project: the architecture plan explicitly flags that "overly aggressive
crops can destroy necessary UI state information (like HP bars)."

Out of scope (deferred to their own spec, once real dataset volume exists
to validate against): encoder architecture, the SimCLR-vs-BYOL training
objective choice, the training loop and RunPod GPU job design, and
checkpoint/latent evaluation methodology. This spec covers only the
augmentation and pair-construction policy that any of those training loops
will apply to frames pulled from the data-collection pipeline's output.

## Why this design, not alternatives considered

- **Augmentation magnitude, generally**: Stock SimCLR/BYOL recipes were
  tuned for natural photos with heavy visual redundancy (lots of texture,
  large images). This domain is a 160x144, ~4-shade, tile-based UI with
  almost no redundancy — small text and thin meter fills are load-bearing
  pixels, not noise. Every augmentation family below uses a narrower
  parameter range than the literature default for exactly this reason.
- **Geometric — drop flip/rotation/aggressive crop**: Horizontal/vertical
  flip is semantically wrong here (text reads one direction, sprite facing
  and HP-bar fill orientation are meaningful, not symmetric under
  mirroring). Rotation never occurs physically in this domain. Standard
  `RandomResizedCrop` (SimCLR default `scale=(0.08, 1.0)`) can and will
  crop the HP bar, text box, or menu cursor clean out of frame — the exact
  failure mode the architecture plan warns about. Kept instead: small
  random translation (±2-4px, sub-tile) for capture-jitter invariance, and
  a mild crop-and-resize-back restricted to ~90-100% of frame area so
  composition stays intact.
- **Color — collapses to brightness/contrast only**: Frames are already
  single-channel grayscale per the data-collection pipeline design (color
  in source footage is a capture artifact, not game state, and was
  normalized away at extraction time). There is no hue/saturation left to
  jitter. A narrow brightness/contrast range (~±10-15%) simulates
  per-channel capture-gain differences without pushing contrast far enough
  to merge adjacent shades in the 4-level palette — which is exactly where
  an HP bar's fill boundary lives.
- **Noise/blur — doing double duty for the train/deploy domain gap**: At
  PPO inference time, frames come live from PyBoy (pixel-perfect,
  uncompressed). Training frames are compressed YouTube longplay footage
  (re-encode softness, compression blockiness, per-channel capture
  quirks). Low-sigma Gaussian noise, mild small-kernel Gaussian blur, and a
  light JPEG-artifact simulation (compress/decompress roundtrip or
  blockiness approximation) are included specifically so the encoder has
  seen both compressed and relatively clean views during training, closing
  the gap to the pixel-perfect frames it will only see at inference.
- **Pair construction — single-frame two-view, not temporal pairs**:
  Considered using nearby video frames (frame_t, frame_t+k) as "natural"
  positive pairs instead of relying on synthetic augmentation to manufacture
  invariance — this would sidestep the UI-destruction risk almost entirely.
  Rejected for v1: the data-collection pipeline's dedup filter already
  removes near-duplicate consecutive frames, which means any two frames
  that both survive dedup and are close in time are, by construction,
  different enough to have passed that filter — real state changes (a step
  taken, a text line advanced, a battle triggered) are more likely between
  them than not. That makes temporal pairing a noisier positive-pair signal
  than it first appears, not a safer one. The data-collection pipeline
  already stores per-frame `video_id` and source timestamp, so this remains
  available as a future supplementary signal once the single-frame policy
  is validated (see Open Questions) — not building it now (YAGNI).
- **Pairs, not triplets**: Triplet loss (anchor/positive/negative) was
  considered and rejected. It requires an explicit negative-mining step —
  deciding which other frame is "different enough" to serve as the
  negative — that neither candidate training objective needs. SimCLR's
  InfoNCE/NT-Xent treats every other sample in the minibatch as an
  implicit negative for a given pair (batch_size-1 negatives "for free,"
  versus one negative per triplet), which is a large part of why
  InfoNCE-style losses superseded triplet loss for representation learning.
  BYOL uses no negatives at all, avoiding the mining problem entirely via a
  momentum target network and stop-gradient — precisely why it was
  attractive here. Triplets would add negative-mining design surface (risk
  of picking a near-duplicate frame as a "negative" from unlabeled
  gameplay video) in service of a loss neither deferred candidate uses.
  The pair-construction interface below produces two augmented views,
  matching both.

## Policy

### Geometric transformations

| Transform | Decision | Parameters |
| --- | --- | --- |
| Horizontal/vertical flip | Excluded | — |
| Rotation | Excluded | — |
| Aggressive area crop (`RandomResizedCrop`-style) | Excluded | — |
| Small random translation | Included | ±2-4px shift, simulates capture jitter |
| Mild crop-and-resize-back | Included | crop retains 90-100% of frame area |

### Color / intensity augmentations

| Transform | Decision | Parameters |
| --- | --- | --- |
| Hue/saturation jitter | N/A | frames are single-channel grayscale |
| Brightness/contrast jitter | Included | narrow range, ~±10-15% |

### Noise / blur augmentations

| Transform | Decision | Parameters | Purpose |
| --- | --- | --- | --- |
| Gaussian noise | Included | low sigma | general pixel-noise invariance |
| Gaussian blur | Included | small kernel, mild | re-encode softness; must not blur small glyphs into illegibility |
| JPEG-artifact simulation | Included | mild quality roundtrip or blockiness approximation | closes YouTube-compressed-training vs pixel-perfect-inference domain gap |

### Pair construction

Two independently-augmented views of the same stored frame (standard
SimCLR/BYOL two-view setup), drawn from the transform families above.
Not temporal (frame_t, frame_t+k) pairs — see rationale above.

## Validation before spending GPU time

Per this repo's observability-first convention (`CLAUDE.md`): before any
real contrastive training run, generate a contact-sheet-style preview —
grid rows of (original, view A, view B) — sampled across overworld,
battle, menu, and dialogue frame types, for a human to eyeball and confirm
no augmentation setting is destroying text legibility or meter fill. This
is a cheap, no-GPU check that catches a bad parameter choice before it
burns a training run, mirroring the contact-sheet check already used in
the data-collection pipeline.

## Testing strategy

Per this repo's TDD convention, augmentation logic gets unit tests against
synthetic numpy frame fixtures (no GPU/network required):

- Geometric transforms never shift/crop beyond their configured bounds.
- Brightness/contrast jitter output stays within the configured range.
- Noise/blur parameters (sigma, kernel size) stay within configured bounds.
- Pair construction always produces exactly two augmented views per input
  frame, each independently sampled.

The contact-sheet visual check above is the human-in-the-loop
correctness check (analogous to data collection's role for dedup/anomaly
behavior) — it is not a substitute for the unit tests, and the unit tests
are not a substitute for it; augmentation code can be "correct" per its
own bounds and still visually destroy UI state, which only the visual
check catches.

## Out of scope

- Encoder architecture (CNN backbone choice, projection head design).
- SimCLR vs BYOL training objective choice.
- Training loop implementation and RunPod GPU job design.
- Checkpoint / learned-latent evaluation methodology.

Each of the above gets its own design spec once this augmentation policy
is implemented and, ideally, once enough real dataset volume exists to
validate augmentation choices against actual frames rather than the
handful of smoke-test samples used to inform this design.

## Open questions / future extensions

- **Temporal positive pairs as a supplementary signal**: revisit once the
  single-frame augmentation policy is validated against real training
  results — could be added alongside (not instead of) synthetic
  augmentation, using the already-stored `video_id`/timestamp metadata.
- **UI-region-aware masking/cutout**: augmentations that avoid known
  crop-box sub-regions (e.g. the text-box area) were considered but not
  designed here — would require plumbing curation-time region metadata
  into the augmentation policy, adding coupling this spec avoids for v1.
