---
aliases: [recognition, estimator, pipeline]
tags: [board-reader/recognition]
---

# Recognition pipeline

How a photo becomes an 8×8 grid of 13-class distributions. This note describes the
**snapshot path** (`POST /estimate`), which is the interesting one; the streaming path is
the same estimator fed a rectified crop instead of an original photo.

## The path

```mermaid
flowchart TD
  photo[original photo] --> corners{corners known?}
  corners -->|operator tapped| geom[corner quad a8,h8,h1,a1]
  corners -->|no| detect["POST /detect — detect_board_quad"]
  detect --> geom
  geom --> boxes["square_boxes(geometry)<br/>bbox: base quads + headroom<br/>pose: projected piece prisms"]
  boxes --> crops["crop_squares_photo<br/>64 x 224² crops"]
  crops --> mask["occlusion masking (BR_CROP_MASK)<br/>gray-fill closer neighbors' prisms"]
  mask --> est[estimator]
  est --> dist["square_dist (8,8,13)"]
  dist --> demote["demote_attractor<br/>black-bishop calibration"]
  demote --> rot[autorotate?]
  rot --> event["PositionEvent.from_square_dist"]
  event --> out[hub + HTTP response]
```

## Cropping is where the accuracy is

Each square's crop is **not** a tile of a rectified board — it is cut from the *original
photo*, then clipped, padded and resized to 224². How the box is chosen is the
**`CropGeometry`** decision, and there are two answers. `square_boxes` returns the 64 boxes
standalone; `crop_squares_photo` is the clip-pad-resize on top of them.

| Geometry | The box is | Needs |
|---|---|---|
| **`BboxGeometry(headroom, side_pad)`** | the projected base quad's bounding box, extended **upward in image space** by `headroom` × the square's projected *width* | corners only |
| **`PoseGeometry(piece_height, piece_radius, focal_ratio)`** | the image-space bounding box of a piece-sized **prism standing on the square**, projected through the recovered camera pose | corners + image size |

`PoseGeometry` exists because the bbox's height is tied to the square's projected *width* —
which shrinks as the camera rises — while a real piece's projected height does not. One
scalar therefore cannot cover both a steep view and a shallow one; `headroom=1.6` patched
the symptom rather than removing the cause. `board_pose` recovers the pose from the board
homography, **solving** the focal length from the homography itself (Zhang's two plane
constraints) rather than assuming one, and falling back to `focal_ratio` only for a
near-top-down view, which genuinely carries no information about focal length. `r1`, `r2`
and `t` come straight from the homography, so the `Z = 0` face reproduces the bbox base
quads exactly — only the height is new. Measured on a real angled photo, the near-rank /
far-rank box-height ratio falls from **1.57** (bbox, tracking foreshortened width) to
**1.15** (pose, tracking real piece height).

> [!danger] Headroom is load-bearing, and it once silently broke everything
> The geometry drifted to `headroom=0.15` — near-base-only crops. At a steep camera
> elevation (~25–45°) a tall piece's head sits a full square *above* its base, so the
> classifier only ever saw lower bodies. Measured consequence on one steep dim board:
> **57/64, with every major piece wrong** — q→b, K→Q, R→P, B→P — plus phantom pawns
> hallucinated from a cut-off head one square up.
>
> Deployed geometry is **`BboxGeometry(headroom=1.6, side_pad=0.15)`, size 224**: a whole
> standing piece. Retraining with tall crops moved the unseen-style holdout from 0.665 to
> 0.835 and real photos from 415 to 430 of 448. More is not better — `headroom=1.8,
> side_pad=0.25` ties on the holdout and *loses* on real photos (409). That sweep is also
> the ceiling of what a scalar can do, which is what `PoseGeometry` is for.

Because a mismatch between training crops and serving crops is invisible until accuracy
quietly collapses, **the geometry travels with the model** in a `<model>.onnx.meta.json`
sidecar that `TrainedEstimator` reads. Never hard-code it on either side.

The `?debug=1` overlay draws from `square_boxes` under the active estimator's own geometry —
the same function recognition crops with — so the visualization cannot drift into showing a
re-derivation of the geometry.

### Occlusion masking — the deployed default since 2026-08-28

A crop's box is a *bounding* volume, so a neighboring piece leaning toward the camera can
occupy a large fraction of another square's crop and be read as if it stood there (the
"bleed" family — measured at 44% of the deployed residual in the
[[2026-08-28-retrain-program|retrain program]]'s E4). **`BR_CROP_MASK=gray`** (set by both
`compose.yaml` and `make dev`) gray-fills, in each square's crop, the pixels that fall
inside a **closer** neighbor's prism before the crop reaches the classifier **and the bank
embedder**. Nearness is camera depth from the recovered pose, not image row — an earlier
image-row version demoted the wrong neighbor at oblique angles and was fixed in review
(`a6b4463`).

Two effects, one of them surprising
([[2026-08-28-improvement-report|improvement report]]):

- The intended one: bleed pixels stop reaching the classifier.
- The dominant one: masked crops shift the DINOv2 embedding enough that the session bank's
  nearest-exemplar lookup **unlatches from a wrong exemplar** it had locked onto. That is
  what nearly doubled perfect boards on the 4-take gate (≈20 → 35), almost entirely on the
  q↔k-locked 08-13 take — a bank-side effect, not a classifier one. It is also why
  retraining the members on masked crops did *not* compose with the deployed blend: the
  bank half decides those takes.

`blur` exists as an alternative fill; unset means unmasked (the pre-08-28 behavior).

> [!note] Under `BboxGeometry` the debug boxes must be *reindexed*, not re-derived
> `headroom` and `side_pad` extend a box along fixed **image** axes, so re-deriving boxes
> from a rotated corner labelling silently rotates that extension too and samples the wrong
> pixels. When `autorotate` fires, the already-correct boxes are permuted by the same
> `rot90` as `square_dist`. `PoseGeometry` does not have this problem — a prism stands along
> the board *normal*, which relabelling cannot move — and a test pins that difference down.

## The estimators

All of them satisfy the same `Estimator` protocol (`estimate(frame) -> PositionEvent`),
plus `estimate_photo(frame, corners)` for the photo-box path.

| Estimator | What it is | Where it shines |
|---|---|---|
| `StubEstimator` | reports a fixed placement, ignores pixels | M0 walking skeleton; fresh clone with no `models/` |
| `CentroidEstimator` | an `Embedder` (DINOv2-S ONNX) + a classifier — either class centroids or a kNN exemplar bank | conditions it has been calibrated on |
| `TrainedEstimator` | multi-style supervised ResNet18, ONNX, crop → 13 logits | chess sets it has never seen |
| `CascadeEstimator` | two trained models on *different* crop geometries; the second is consulted only on squares the first was unsure about, and averaged in there | the same places a single trained model serves, more accurately |
| `AveragedEstimator` | a trained model fine-tuned on one board's calibrate frames, averaged 50/50 with the base it came from on every square — they share a geometry, so one crop pass feeds both | the one physical board and set the fine-tune was calibrated for |
| `EnsembleEstimator` | distribution-level blend of a trained member and a bank member | everywhere — it tracks the better member square by square |

### Selection at startup

Resolved from the environment at import time, in this order:

1. `BR_CENTROIDS` — an explicit curated calibration file wins outright.
2. `BR_MODEL` — the trained classifier, alone or paired with exactly one companion:
   `BR_MODEL_B` makes a `CascadeEstimator`, `BR_MODEL_TUNED` an `AveragedEstimator`.
   If `BR_SESSION_BANK` also names an existing bank, the result is wrapped in an
   `EnsembleEstimator` rather than one replacing the other.
3. A persisted session bank alone, when auto-calibration is enabled.
4. `StubEstimator`.

Step 2 is one *trained component* however it is composed — the composites replace the
model inside it, never the bank arithmetic on top of it.

A `BR_MODEL` that fails to load (missing file, ONNX Runtime mismatch, truncated download)
prints a warning and falls through to the next tier — a fresh clone with no `models/`
directory still runs `docker compose up` successfully. Neither companion gets that grace:
they are never set by default, so a broken one (or one without a geometry sidecar) fails
startup rather than quietly serving a single model. `BR_MODEL_TUNED` is stricter still —
`BR_MODEL` missing entirely is fatal rather than a fall-through, because the fine-tuned
model alone was measured to hallucinate pieces onto empty squares, and setting both
`BR_MODEL_TUNED` and `BR_MODEL_B` is refused as an unmeasured stack.

### Why blending beats either member

The bank is near-perfect under conditions it has seen and brittle outside them; the trained
model is the reverse. On the labeled eval:

| Configuration | Score |
|---|---|
| `v1` alone (short crops) | 579 / 640 |
| `d16` alone (tall crops) | 605 / 640 |
| deployed model alone (`run_e`, 11 styles) | 599 / 640 |
| trained ⊕ bank, maturity-weighted **(deployed)** | **638 / 640** |

(All four rows are historical 2026-08 measurements on the ten-board curated set,
which was pruned to five boards on 2026-08-28 after five photos were deleted —
see [[measuring-accuracy#The curated set dataevalground_truthjson]]. The
reproducible successor baseline lives there.)

Model-alone rank on this bright set is not the deployment order: `run_e` was chosen on the
dim boards (653 vs 600 / 768 squares) and on the ensemble, which is what serves. See
[[training-and-data#How run_e was decided 2026-08-06]].

The blend weight follows bank maturity: **≥1024 exemplars → 30/70 trained/bank**, otherwise
50/50. A mature bank has earned the extra trust; a one-photo bootstrap has not. (At a flat
50/50 a mature bank's correct read on backlit pawns was occasionally outvoted by the tall-crop
model's colour flip.)

### The black-bishop demotion

One post-hoc calibration ships on top of the blend. The black bishop is an **attractor** for
this stack: where it wins a square by less than **0.20** absolute probability, the runner-up
(`p`/`q`/`k`) is the right answer. `estimator.demote_attractor` swaps the top two on those
squares — a permutation, so the row still sums to one and the square's confidence is
unchanged. Measured over four live takes (133 boards, 8512 squares): 55 firings, **38 squares
fixed, 0 broken**, and +12–19 points of perfect-board rate on three of the four takes
([[2026-08-27-evidence-rerank]]). White's `B` is *not* an attractor and is never touched.

Where it applies is the whole trick:

- **Once, inside the estimator** — in `estimator.read_event`, downstream of the cascade and
  the blend, upstream of `placement`, `evidence`, `confidence` and `prior.decode`. Composites
  therefore blend members' raw `photo_dist`, never their events, so no member can demote early
  and nothing can demote twice.
- **Only on the stack it was measured on**: the trained model, its cascade/average composites,
  and the ensemble on top of one. A *bare* exemplar bank is left alone — 0.20 is an absolute
  margin fitted to the trained stack's distribution, and a kNN bank's is flatter (on the test
  fixture its own **correct** bishops read at margins 0.105 and 0.116, i.e. inside the band).

What makes 0.20 safe is a **gap**, not a mean: the lowest margin at which a top-1 `b` was
actually right is 0.210. If a future set closes that gap the rule starts eating correct
squares silently, so `eval-session` measures it on every session and warns within 0.05 of the
threshold — see [[measuring-accuracy]] and `trial.AttractorGuard`.

## Why piece type is the hard part

> [!warning] The single most important measured fact in this project
> **Occupancy transfers across chess sets; piece type does not.** Cross-style occupancy runs
> 94–99% for free. Cross-style piece type collapses — rooks, knights, bishops, queens and
> kings fall to 0–40% recall while empties and pawns survive.

Seven training-free routes were measured against this and all failed at roughly the same
~80% ceiling: kNN transfer from a public dataset, a linear probe on the same embeddings,
SAM2-segmented crops, ~50 classical embedding-alignment configs, published pretrained
models, bigger and different frozen backbones, and a shape-biased Stylized-ImageNet ResNet.

The diagnostic that settles it: a linear probe trained on ChessReD embeddings reaches 90.7%
*within* ChessReD but only 254/320 across the style gap — tying plain kNN. The gap is in the
**features**, not the classifier. No frozen backbone fixes it; scale does not help
(DINOv2 S→B is flat), and shape bias actively over-specializes (best same-style at 296/320,
worst cross-style at 219).

What did work was learning style-invariance from **many** piece styles at once — see
[[training-and-data]]. Evidence and full numbers: [[2026-07-18-zero-shot-campaign]].

One unexploited win from that sweep: DINOv2-S **central-column patch pooling** (mean of the
middle 8 of 16 token columns instead of CLS) beats CLS by +9 squares with near-perfect
occupancy. Cheap, measured, never shipped.

## Orientation

Corner geometry cannot say which corner is `a8`, and the CV layer may not infer orientation
from the pieces as a *stateful* fact (DESIGN decision #15). What it may do is resolve the
grid rotation of a single frame by chess plausibility, which is a per-frame property:

- fewest pawns on ranks 1 and 8, tie-broken by
- White's piece mass nearer rank 1.

This is `autocalib.resolve_rotation`, applied when the request carries `autorotate=1`. The
snapshot page sends that flag **only while the operator has not hand-placed or relabelled
the corners** — an operator's labelling is authoritative and is never silently overridden.

> [!failure] Two cues that look obvious and are wrong
> **Square-colour parity** and **printed border coordinates** are deliberately not used.
> Players set boards up ignoring the printed coordinates, and boards without FIDE-standard
> printing exist. Both were observed in a single field session: a position played 90° to the
> printed grid, and a shot where parity would have "corrected" an already-correct
> orientation.

Auto-correction has a sting in the tail: it hid a corner-orientation error from `/calibrate`,
which still trusted it. See [[calibration-and-the-bank#The four ways this got poisoned]].

## Advisory warnings

`/estimate` attaches `warnings` without ever refusing to answer, in keeping with "stream
what you see, flag what you doubt":

- `low_light` — mean pixel value below 55 (evening-room shots measure ~37).
- `very_low_angle` — projected board aspect below 0.62.

Both are explained in [[operating-envelope]].

## Related

- [[architecture]] — where this sits
- [[calibration-and-the-bank]] — the per-set half of the ensemble
- [[operating-envelope]] — when to distrust the output
- [[training-and-data]] — where the trained half comes from
