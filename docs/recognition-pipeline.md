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
  crops --> est[estimator]
  est --> dist["square_dist (8,8,13)"]
  dist --> rot[autorotate?]
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
| `EnsembleEstimator` | distribution-level blend of a trained member and a bank member | everywhere — it tracks the better member square by square |

### Selection at startup

Resolved from the environment at import time, in this order:

1. `BR_CENTROIDS` — an explicit curated calibration file wins outright.
2. `BR_MODEL` — the trained classifier. If `BR_SESSION_BANK` also names an existing bank,
   the two are wrapped in an `EnsembleEstimator` rather than one replacing the other.
3. A persisted session bank alone, when auto-calibration is enabled.
4. `StubEstimator`.

A `BR_MODEL` that fails to load (missing file, ONNX Runtime mismatch, truncated download)
prints a warning and falls through to the next tier — a fresh clone with no `models/`
directory still runs `docker compose up` successfully.

### Why blending beats either member

The bank is near-perfect under conditions it has seen and brittle outside them; the trained
model is the reverse. On the labeled eval:

| Configuration | Score |
|---|---|
| `v1` alone (short crops) | 579 / 640 |
| `d16` alone (tall crops) | 605 / 640 |
| deployed model alone (`run_e`, 11 styles) | 599 / 640 |
| trained ⊕ bank, maturity-weighted **(deployed)** | **638 / 640** |

Model-alone rank on this bright set is not the deployment order: `run_e` was chosen on the
dim boards (653 vs 600 / 768 squares) and on the ensemble, which is what serves. See
[[training-and-data#How run_e was decided 2026-08-06]].

The blend weight follows bank maturity: **≥1024 exemplars → 30/70 trained/bank**, otherwise
50/50. A mature bank has earned the extra trust; a one-photo bootstrap has not. (At a flat
50/50 a mature bank's correct read on backlit pawns was occasionally outvoted by the tall-crop
model's colour flip.)

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
