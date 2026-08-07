---
aliases: [glossary, terms]
tags: [board-reader/reference]
---

# Glossary

Terms that carry a specific meaning in this project. Where a word is ordinary English
elsewhere, the entry says what makes it load-bearing here.

## Domain

**Placement** — FEN field 1 only: which piece sits on which square. Not a FEN. Side-to-move,
castling rights, en passant and move counters are history-derived and cannot be read from one
image, so we never fabricate them.

**Position event** — the unit of output: capture timestamp, `seq`, orientation, the full
`square_dist`, the derived `placement` and `confidence`, `stability`, clock, and `status`.
Spec in [[DESIGN#5. Position stream protocol (L3 output)]].

**`square_dist`** — the canonical output. An 8×8 grid of 13-vectors (rank 8 → rank 1, file a →
h), each a probability distribution over `empty` plus twelve piece classes. Everything else in
the event is derived from it.

**Occupancy** — the binary question *is this square empty?*, as opposed to *what piece is on
it*. The distinction runs through the whole project: occupancy transfers across chess sets
almost for free, piece type does not.

**Style** — one physical chess set, or one rendered geometry-and-material combination. The
axis piece-type recognition fails to generalize along. "Cross-style" means evaluated on a set
never seen in training.

**Stability** — an advisory per-frame score in [0,1] (1 = at rest), computed on the edge from
inter-frame motion. A hint, never a gate: the consumer segments moves, not us.

**Transition** — a square whose occupant differs between two consecutive placements. A
difference between two strings, *not* a claim that a move happened: a hand crossing the board,
a misread and a real capture all produce transitions identically. What the headless tail
prints; telling them apart is the consumer's job.

**Status** — `ok | recalibrating | needs_setup | stale`. Notably there is no `illegal`;
legality is not ours to judge, and an illegal-looking placement is reported as `ok`.

## Geometry

**Corners** — the four playing-surface corners in `a8, h8, h1, a1` order. Everything
geometric is expressed against them.

**Corner quad** — the four corners as projected into the photo; a quadrilateral, not a
rectangle.

**Projected board aspect** — the short side of the corner quad divided by the long side. A
calibration-free proxy for camera elevation, and the basis of the `very_low_angle` warning.
See [[operating-envelope#Camera angle]].

**Headroom** — how far each square's crop box extends *upward* from the square's base, in
square-widths. `1.6` means "a whole standing piece". Getting this wrong decapitates pieces —
the single most expensive geometry bug in the project.

**Side pad** — sideways padding of the crop box, in square-widths. Deployed at `0.15`.

**Rectification** — warping the board to a canonical top-down view. The streaming path uses
it; the snapshot path deliberately does not, because the warp smears tall pieces and the tile
truncates them.

**Photo boxes / full-piece boxes** — the alternative to rectified tiles: each square's
projected base quad extended by headroom, cropped from the *original* photo.

**Autorotate** — resolving the grid rotation of a single frame by chess plausibility (fewest
pawns on ranks 1/8, tie-broken by White's mass nearer rank 1). Applied only while the operator
has not hand-placed or relabelled corners.

## Recognition

**Estimator** — anything satisfying `estimate(frame) -> PositionEvent`. The seam every
recognition approach plugs into.

**Embedder** — a frozen backbone mapping a crop to a vector (DINOv2 ViT-S/14 → 384-d). Used by
the bank path; the trained path skips it and goes crop → logits directly.

**Exemplar bank** — the calibration classifier: every calibration crop kept as its own
exemplar, a query scored by nearest exemplar per class. Beats class centroids by 33 squares on
leave-one-out. See [[calibration-and-the-bank]].

**Centroids** — the earlier per-class-mean classifier. Superseded by the bank, still loadable.

**Transductive centering** — re-centering the bank and the query batch each on their own mean
before the cosine step, which absorbs per-photo lighting and backbone shift.

**Session bank** — the persisted, growing exemplar bank for the current set. Written **only**
when `BR_SESSION_BANK` is set. The one piece of durable state in an otherwise stateless system,
and therefore the one thing that can be silently corrupted.

**Bootstrap / self-refresh** — building the bank from a start-position sighting with no user
action, and topping it up from later low-confidence start sightings to track lighting drift.

**Ensemble** — the deployed recognizer: a distribution-level blend of the trained model and
the session bank, weighted by bank maturity (≥1024 exemplars → 30/70 trained/bank).

**Zero-shot** — reading a chess set with no calibration for it. The property the multi-style
training campaign was fought for.

**Shape bias** — a model relying on silhouette over colour and texture. Pursued through
augmentation (which worked) and through Stylized-ImageNet initialization (which did not — it
over-specialized to same-set cues).

## Infrastructure

**Edge / L1** — the node at the board: capture, detect, rectify, forward. "The Pi".

**Inference / L2** — the node that recognizes: stateless per-frame estimator plus gateway.
DESIGN calls it "central / Jetson".

**Wire** — the L1→L2 frame format: a JSON header plus a JPEG. The *only* thing edge and
inference share.

**Hub** — the pub/sub that fans position events out to viewers and retains the latest per
board for late subscribers. Stateless beyond "last value".

**Position stream** — the L3 egress: the sequence of position events a board produces, served
on `WS /viewer`. Captured verbatim as NDJSON by `board-reader-inference positions --out`, where
a captured run *is* the stream rather than a rendering of it.

**Consumer / application layer** — anything downstream of the position stream. DESIGN's
out-of-scope layer: move reconstruction, legality, game state, PGN — plus the temporal
smoothing that makes them possible. Today's consumers are the viewer UI and the headless
`positions` tail, and neither smooths anything; see
[[architecture#Temporal consistency — the layer nobody has built]].

**Seasoning** — the 7-day supply-chain hold: `[tool.uv] exclude-newer` refuses to resolve any
release published in the last week, so a freshly-compromised package is never pulled.

## Related

- [[index]] · [[architecture]] · [[recognition-pipeline]] · [[calibration-and-the-bank]]
