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

**Status** — `ok | recalibrating | needs_setup | stale`. Only `ok` is emitted; the others are
reserved (plan decision D8, DESIGN §5). Notably there is no `illegal`; legality is not ours to judge, and
an illegal-looking placement is reported as `ok`.

## Geometry

**Corners** — the four playing-surface corners in `a8, h8, h1, a1` order. Legacy since
Phase C: no edge sends them; `wire.Frame` still parses them so old recordings replay, and
recognition ignores them.

**Crop rectangle** — the part of the camera frame the edge sends, in fractions of the full
frame (`board_reader_core.control.CropRect`). Drawn on `/control` for the phone; `--roi`
on the Python edge.

**Camera config / report** — the control channel (DESIGN D11): the config is what
`/control` chose for a board's camera (crop, lens, resolution, streaming); the report is
what the phone says its camera is (lenses, frame size, stream state). The phone posts the
report every 500 ms and applies the config in the reply.

**Chess frame** — the convention every placement is in: rank 8 first, file a first, as FEN
writes it, whatever way the board faces the camera. The recognizer resolves it per frame;
events, labels, truth files, eval and replay all use it. See
[[recognition-pipeline#The chess frame]].

**Corner frame** — the old convention, in which a placement was read in the frame the
operator's corners defined. Only older labels and upload sidecars carry it; `eval` skips
corner-frame labels.

**Framing** — how much of the frame the board fills. ChessQ needs the board to span at least
half of the frame width. See [[recognition-pipeline#Framing: the board must fill the frame]].

## Recognition

**Recognizer** — anything satisfying the `Recognizer` protocol (`recognizer.py`): a whole BGR
frame in, its board's `(8, 8, 13)` distribution out, in the chess frame. The seam every
recognition model plugs into.

**ChessQ Lite V6** — the served recognizer (DESIGN D10): the small model of Joël Seytre's
ChessQueries, the whole-frame board model Lichess runs in its "From image" editor.
`BR_CHESSQ`, fetched by `make models`, PolyForm Noncommercial.

**Stub recognizer** — the walking skeleton: one fixed placement, whatever the pixels. Serves,
loudly, when `BR_CHESSQ` is unset or names no file.

**Frame read** — `recognizer.frame_reader(recognizer)`: the one function from a wire frame to
its position event, run by `/estimate`, the ingest workers and replay.

**Prior / repair** — the hard-constraint decoder (`prior.py`): the nearest possible board to a
read, riding beside it as `prior`; the read itself is never rewritten.

**Settled** — the debounced board (`stabilize.py`): the last placement that held for the
settle window, riding on stream events as `settled`.

**Read scheduler** — the node's choice of which arriving frames to read (`schedule.py`,
`BR_SCHEDULE`): by default three reads 0.6 s apart at the end of a motion burst, plus a
keepalive read when 2 s of capture time passed without one.

**Replay gate** — the four recorded takes with truth that every recognition, decode or
settle change is measured on (`make gate`); the reference row is `cq-v6-tuned`.

**Zero-shot** — reading a chess set with no calibration for it. ChessQ is zero-shot on every
set; there is no calibration round any more.

The old crop stack's terms — exemplar bank, session bank, embedder, centroids, ensemble,
headroom, photo boxes, autorotate — are described in [[calibration-and-the-bank]] and
[[training-and-data]].

## Infrastructure

**Edge / L1** — the node at the board: capture, crop to the configured rectangle, measure
stability, forward. No detection or rectification. "The Pi"; in the sim, the phone's
`/camera` page, a dumb camera configured from `/control`.

**Inference / L2** — the node that recognizes: stateless per-frame recognizer plus gateway.
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

**Seasoning** — the 7-day supply-chain hold: `[tool.uv] exclude-newer` (Python) and
`minimumReleaseAge` in `pnpm-workspace.yaml` (the UI) refuse any release published in the
last week, so a freshly-compromised package is never pulled.

## Related

- [[index]] · [[architecture]] · [[recognition-pipeline]] · [[calibration-and-the-bank]]
