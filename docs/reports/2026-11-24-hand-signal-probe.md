# Hand signal probe: which signal says "this read is obscured"? (2026-11-24)

**Question.** A hand that rests still on the board makes no motion, so the burst-end
schedule reads it, ChessQ reads the covered squares as empty with p ≥ 0.96, and the
wrong board can settle. The final-class confidence of a square is no use as a hand
signal. Which cheap signal tells the node that a square or a frame is obscured, and
would it have blocked the hand settles without blocking real moves?

**Answer.** ChessQ's own logits already carry the signal, if you look at the frame
instead of the square. The **frame energy** works best: per square, minus the
logsumexp of the 13 logits; per frame, the mean of the 4 highest square values. It
separates hand frames from clear frames with ROC-AUC 0.970 (worst take 0.92). At a
threshold set from other takes, it flags 80 % of hand frames and 1 % of clear frames.

Used as a **no-commit gate**, it removes every wrong settled board from the resting-hand
test (221950: 13/22 → 22/22 settled perfect, 9 → 0 wrong boards). Under that gate, an
obscured read still feeds the stabilizer as evidence but may not commit a square. On the
seven gate takes it costs 3 of 314 settled-perfect boards and +0.06 s mean settle lag.
It needs no model change and runs on both the fp32 and the int8 file. Dropping obscured
reads outright costs 15 boards instead, because mid-move reads carry true evidence.

Plain-CV signals (background difference, skin colour, texture) and signals that compare
against a reference frame are worse. They fire on every legitimate move, on auto-exposure
and on small camera shifts.

## Data and labels

- **Reads.** Every frame of the seven gate takes and the two live takes of 2026-11-24
  (5,133 frames) went through a scratch copy of ChessQ V6 fp32. The copy adds four kinds
  of output: the 64 query embeddings, the encoder's patch tokens and the decoder's
  cross-attention, beside `logits` (`tools/probes/hand_model.py`, `hand_extract.py`;
  `models/` untouched).
- **Square label "read wrong".** A square counts as wrong when the argmax differs from
  truth on a scored frame: truth present, outside the guard window. That gives 590 wrong
  squares out of 224k, in 284 of 3,505 scored gate-take frames. 18 of the 20 frames where
  the shipped schedule's raw read was wrong outside the guard show a hand. The other 2
  are late truth boundaries (193308 rows 239 and 642).
- **Frame label "hand".** 395 frames labelled by eye from contact sheets
  (`hand_sheets.py`, `hand_labels.py`):
  - **H, 233 frames:** a hand or arm covers part of the playing area.
  - **N, 149 frames:** nothing foreign on the board.
  - **E, 11 frames:** the hand is only beyond the board's edge.
  - **S, 2 frames:** only an arm's shadow falls on the board.

  The sample is 30 frames per take: 12 near truth boundaries or bursts, 18 drawn
  uniformly. On top come rows 452–596 of 20261124-221950, labelled densely. The AUCs
  use H against N only.
- **Resting-hand test.** In 20261124-221950 the user laid a hand flat on the board from
  row ~510 to ~572 (about 7 s). The board holds one position from row 452 to the end
  (checked on every 4th frame), so that stretch gets a one-segment truth
  (`hand_signals.MINI_TRUTH`). The live stack settled wrong boards there at seq 520,
  526 and 556; the true board returned at seq 586.
- **No truth for 20261124-204025.** It enters only the H/N frame AUC.

## Candidates

All model signals are oriented so that larger means "more likely obscured". Frame
versions (`_f`) take the mean of the 4 highest square values.

| name | what it is |
|---|---|
| `nll_top` | −log of the top softmax probability (the shipped confidence) |
| `margin` | minus the gap between the top two logits |
| `energy` | −logsumexp of the 13 logits (free energy) |
| `maha` | Mahalanobis distance of the square's query embedding to the mean embedding of its argmax class. Means and shared covariance are fitted on correctly read squares of the *other* gate takes (leave-one-take-out). |
| `q_ring`, `q_oracle` | cosine distance of the query embedding to the same square's embedding in a reference. `ring` is the median of 15 earlier frames ≥ 1 s apart. `oracle` is the last earlier frame read perfectly against truth (an upper bound). |
| `tok_ring`, `tok_oracle` | distance of the encoder's patch tokens to the same reference, pooled under the square's attention footprint; `_board` pools over the whole board |
| `att_shift`, `att_spread`, `att_ent` | the localising attention head's centroid shift from its 9-frame median, its RMS radius and its entropy |
| `fr_ring`, `fr_gated`, `sq_*` | fraction of the board (or of a square's disc) whose colour moved > 30 against a background median. `ring` takes every frame ≥ 1 s apart; `gated` refuses frames with a large changed blob at the board's border. |
| `edge_blob_*` | the largest changed blob that touches the board's border, as a fraction of the board |
| `fr_skin`, `sq_skin` | the fraction of new skin-coloured pixels (YCrCb box) |
| `sq_edge` | change in Canny edge density against the background |

**Square geometry comes for free.** The node has no board corners: ChessQ reads the
whole frame. One head of decoder layer 2 (and one of layer 3) attends to exactly its
square. Fitting their attention centroids to the 8×8 grid gives R² ≥ 0.99, while the
head mean is diffuse (entropy about 6.8 nats, centroids collapse to the board centre).
The CV square signals take their square positions from layer 2, head 0. The paper
describes the same behaviour: "It localises to a single physical square and its piece"
(§5 "Query attention").

## Results: separation

| signal | square AUC: read wrong (pooled / worst take) | frame AUC: hand (pooled / worst take) | flagged at p99 of clear frames: H / N / frames with a wrong square | cost per frame | where |
|---|---|---|---|---|---|
| `nll_top` | 0.738 / 0.586 | 0.961 / 0.861 | 0.70 / 0.01 / 0.57 | < 0.1 ms, already in `square_dist` | node |
| `margin` | 0.770 / 0.615 | 0.960 / 0.922 | 0.74 / 0.01 / 0.57 | < 0.1 ms, logits | node |
| **`energy`** | 0.806 / 0.644 | **0.970 / 0.922** | **0.80 / 0.01** / 0.60 | **0.03 ms, logits** | node |
| `maha` | 0.913 / 0.835 | 0.967 / 0.922 | 0.90 / 0.08 / 0.66 | 0.3 ms + `query` output + fitted means and covariance | node |
| `q_ring` | 0.980 / 0.936 | 0.730 / 0.357 | 0.01 / 0.02 / 0.04 | 0.1 ms + `query` output + ring | node |
| `q_oracle` | 0.996 / 0.982 | 0.863 / 0.426 | 0.85 / 0.18 / 0.99 | needs "last clean read" | node |
| `tok_ring` | 0.969 / 0.866 | 0.905 / 0.800 | 0.24 / 0.00 / 0.17 | ~2 ms + `memory` (3 MB) + attention outputs | node |
| `tok_oracle` | 0.986 / 0.965 | 0.876 / 0.591 | 0.67 / 0.17 / 0.89 | as above | node |
| `att_shift` | 0.831 / 0.713 | 0.934 / 0.885 | 0.44 / 0.01 / 0.30 | attention output (layer 2) | node |
| `att_spread`, `att_ent` | 0.51, 0.58 | 0.64, 0.73 | — | — | — |
| `sq_gated` (CV) | 0.890 / 0.814 | 0.924 / 0.771 | 0.00 / 0.00 / 0.00 | ~110–200 ms in this Python probe | node (needs geometry) |
| `edge_blob_gated` (CV) | — | 0.907 / 0.674 | 0.03 / 0.03 / 0.13 | ~5 ms at 480 px | phone or node |
| `fr_skin` (CV) | 0.668 (`sq_skin`) | 0.687 / 0.324 | 0.04 / 0.03 / 0.11 | ~2 ms | phone or node |

The "flagged" column uses a threshold set leave-one-take-out: the 99th percentile of the
signal over the other gate takes' clear rows (read equal to truth, ≥ 3 s from any truth
boundary).

What the table says:

- **The covered square itself looks normal.** The per-square signals that need no
  reference rank wrong squares poorly: confidence 0.74, energy 0.81. Mahalanobis is best
  of them at 0.91.
- **The frame does not look normal.** The hand pulls down a few *other* squares: the ones
  at its edge, half covered, and the pieces it overlaps. So the top-4 frame aggregate of
  the very same numbers separates hand frames at 0.96–0.97. That is why the shipped
  per-square confidence looked useless.
- **Reference signals measure change, not occlusion.** `q_*` and `tok_*` rank wrong
  squares best (0.97–0.996) but fire on every real move. As frame gates they are weak:
  `q_ring_f` 0.73, and the ring reference absorbs a resting hand within ~7 s. The oracle
  variants need a "last clean read", which is the thing we are trying to find.
- **Mahalanobis needs calibration per setup.** Its threshold, fitted on the gate takes,
  sits below the clear frames of the dark live take 221950. The gate then blocks the
  recovery reads after the hand leaves, and the wrong board persists (see the gating
  table).
- **Plain CV fails on a wooden board.**
  - The clear-frame distributions have heavy tails: moves, auto-exposure, a camera that
    moved twice in 193308 and 204025. At a 1 % false-flag rate CV catches 0–4 % of hands.
  - Skin colour is useless: the board's wood is skin-coloured (frame AUC 0.69).
  - No CV signal is usable on the phone at a safe threshold. The phone also has no
    square geometry.

## Results: gating the shipped schedule

`hand_eval.py` runs the shipped burst-end schedule
(`burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.3`) over each take's restamped
frame index. Each chosen frame comes from the per-frame cache, and a fresh
`BoardStabilizer` settles it. Ungated, the simulation reproduces the base replay
`guard-only` on the gate takes: 446 of 446 read frames and 446 of 446 settled boards.
On 221950 it hits 107 of the live node's 109 read frames.

The threshold is the leave-one-take-out p99 from above. For `energy_f` it ranges from
−12.13 to −12.93 by held-out take, and is −12.41 for 221950.

Column key:

- **gate perfect:** settled-perfect boards out of scored gate-take events.
- **not-lag wrong:** settled boards that are neither the truth nor the previous truth
  (partial moves, hand leaks).
- **lag:** mean seconds from a truth boundary to the first settled event that equals
  truth.

| variant | gate perfect | not-lag wrong | lag | 221950 settled perfect | 221950 wrong boards |
|---|---|---|---|---|---|
| ungated (= `guard-only`) | 314/361 | 14 | 2.72 s | 13/22 | 9 |
| drop obscured reads, `energy_f` | 299/361 | 10 | 3.01 s | 22/22 | 0 |
| **no-commit, `energy_f`** | **311/361** | **13** | **2.78 s** | **22/22** | **0** |
| no-commit, `nll_top_f` | 312/361 | 13 | 2.76 s | 22/22 | 0 |
| no-commit, `maha_f` | 309/361 | 16 | 2.79 s | 22/22 | 0 |
| drop obscured reads, `maha_f` | 297/361 | 12 | 3.08 s | 7/22 | 15 |
| no-commit, `att_shift_f` | 311/361 | 16 | 2.75 s | 14/22 | 8 |
| no-commit, `edge_blob_gated` (CV) | 312/361 | 16 | 2.92 s | 13/22 | 9 |
| hold squares with high `energy` | 305/361 | 20 | 2.94 s | 22/22 | 0 |

- **Why dropping reads costs lag.** A mid-move read with the hand on the piece also says
  something true: the lifted piece's source square is empty. That read starts the
  square's challenger early, and dropping it delays every move by one read. A re-read
  hook did not help here (+1 read in total): the dropped reads fall *inside* bursts,
  not after them.
- **What no-commit keeps and blocks.** It keeps that evidence: an obscured read can start
  or extend a challenger, but no square commits on it. The first clear read after the
  hand leaves either confirms the challenger or resets it. In 221950 every read under the
  hand is flagged, so nothing commits until the hand is gone.
- **Where no-commit costs.** Per take it equals ungated on 5 of 7 gate takes. It loses one
  board on 111421 and two on 172736 (the take with the strictest threshold, −12.93).
- **Holding single squares fails.** It produces half-adopted moves (20 not-lag wrong
  boards): the frame decision is the right granularity.
- **The two gate-take "leaks" are castling.** `hand_leaks.py` lists two settled-wrong
  boards that are not lag: 111421 seq 141/147 (f1 rook) and 181802 seq 218/219 (d1 rook).
  Both are castling moves, O-O and O-O-O, where the hand still holds the rook. The
  "empty" read is physically right and the truth boundary leads it, so they are partly
  adopted moves, not hand leaks. The energy gate rightly lets them through.

## What the paper says

ChessQueries (arXiv 2608.30762v2) reports no uncertainty, calibration, out-of-distribution
or abstention analysis. Its errors "cluster on areas under heavy occlusion" (§5). The SLCC
training frames with "full occlusion of pieces or squares (e.g., by a player's hand)" were
dropped (§4), so the model never learned what a hand is. The paper's square-localising
query attention is what made the CV square geometry possible here. Its model is a
ViT-L/14; our Lite V6 is a ViT-S/14 with the same 4-layer, 64-query decoder.

## Recommendation

**Gate commits on the frame energy: an obscured read is evidence but never a commit.**
It is the only candidate that clears the resting-hand test, costs ≤ 3 gate boards,
needs no new model output, runs in microseconds and behaves the same on both release
files.

Implementation sketch:

1. **`recognition/chessq.py`.** Compute `energy = -logsumexp(logits)` per square before
   the softmax. Carry a frame score `obscured = mean of the 4 largest` on `Recognition`,
   as a new float field. The int8 file has the same `logits` output, so `make dev-cpu`
   gets it too.
2. **`stream/stabilize.py`.** Give `BoardStabilizer.observe` an `obscured: bool` argument.
   When it is true, update challengers as now but skip `_commit`. It is one condition at
   the existing `ripe` list, and it composes with the guard and the rules switch.
3. **`stream/ingest._read_with_prior`.** Pass
   `obscured = recognition.obscured > BR_OBSCURED_ENERGY`, defaulting to −12.4: the
   median of the per-take thresholds above. Put `obscured: true` on the published event,
   so the viewer can show "hand over the board" instead of a stale "pending".
4. **No scheduler change.** The re-read hook bought nothing (see above).
5. **Verify.** Run `make gate GATE_NAME=obscured-energy` against `guard-only`. Replay
   221950 live to repeat the resting-hand test, then do a fresh resting-hand take with
   a lit board.

Re-fit the threshold for every new model release. Energy is a property of the logit
scale, like the temperature the measuring rules warn about.

**Not done here.** I did not replay `make gate` through a changed node. The table above
is the offline simulation, which reproduces `guard-only` read for read. Per the
measurement rules, the shipped number must come from the gate. The no-commit gate also
does not win on *every* take (2 takes lose 1–2 boards to lag). It is a safety change
traded for lag, not a pure accuracy win.

## Shipped: the gate through the changed node (2026-11-25)

The no-commit `energy_f` gate shipped at the fixed default −12.4 (DESIGN D16). The
gate was replayed through the changed node (`make gate GATE_NAME=energy`); take 2
was replayed twice on the same code, with the gate on (`energy`) and off
(`energy-off`, `--obscured-energy off`).

| measure | base | gated (real) | simulated above |
|---|---|---|---|
| gate settled perfect | 314/361 (`guard-only`) | **312/361** | 311/361 |
| gate raw perfect | 341/361 | 341/361 | — |
| lost boards, by take | — | 111421 −1, 181802 −1 | 111421 −1, 172736 −2 |
| gate reads flagged obscured | — | 53 of 446 | — |
| mean settle lag (boundary → first settled-perfect event) | 2.82 s | 2.90 s | 2.72 → 2.78 s |
| 221950 resting-hand stretch (seq 452–596, one-segment truth) | 13/22, 9 wrong boards | **22/22, 0 wrong** | 22/22, 0 |
| 221950 whole take, `truth.json` | 80/94 settled perfect | **88/94** | — |

- **Why the lost boards differ by take.** The simulation gated each take at its own
  leave-one-take-out threshold (−12.93 on 172736); the node gates every take at −12.4.
- **No new wrong boards.** The not-lag settled errors (`hand_leaks.py energy`) are the
  base's 14, plus one: 181802 seq 110, where the settled board is two moves behind instead
  of one. That is lag the classifier counts as "other", not a hand leak.
- The lag here uses a different denominator from the simulation's (every truth boundary
  of the replay's frame index), so compare the deltas (+0.08 s against +0.06 s), not the
  absolute values.

## Reproduce

```bash
S=<scratch>
uv run --project tools/chessq_export --frozen python tools/probes/hand_model.py $S/cq-v6-probe.onnx
flock /tmp/claude-1000/br-gpu.lock uv run python tools/probes/hand_extract.py $S/cq-v6-probe.onnx $S/feat <takes>
CUDA_VISIBLE_DEVICES= uv run python tools/probes/hand_cv.py $S/feat $S/cv <takes>
uv run python tools/probes/hand_eval.py $S/feat $S/cv --sim energy_f nll_top_f maha_f att_shift_f edge_blob_gated energy
uv run python tools/probes/hand_leaks.py            # the base's not-lag settled errors
```

Extraction ran at ~3.3 frames/s on the GTX 1650 with every extra output (~25 min for
all nine takes); the cache is ~1.3 GB.
