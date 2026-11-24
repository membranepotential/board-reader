# ChessQ Lite V6 replay gate — settled perfect boards 35 → 71 of 109

_2026-11-24 · the replay-gate step of [[2026-10-09-chessqueries-handoff]]. The cold
benchmark there read gate-take frames at 99.2 % per square; this report measures the
settled stream, which the benchmark did not. Adoption is a DESIGN decision and is not
made here._

## Method

- **Candidate:** the opt-in `ChessQEstimator` (`BR_CHESSQ`, module
  `board_reader_inference.chessq`) serving
  `models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx`. It reads the whole frame
  (no crop, no warp, no bank), serves its softmax unchanged (no black-bishop
  demotion), and the prior repair and `BoardStabilizer` run downstream exactly as
  for the shipped stack.
- **Environment:** `BR_CHESSQ` only. No `BR_MODEL`, `BR_MODEL_B`, `BR_SESSION_BANK`,
  `BR_CROP_MASK`. The manifests record `estimator: ChessQEstimator` and an `env`
  holding `BR_CHESSQ` alone.
- **Replays:** `board-reader-inference replay <take> --policy burst-end --name cq-v6
  --score`, with `--restamp-stability` on 08-13 only, as in the `tkr-shipped`
  manifests. Same policy, same settle (2.5 s, 2 reads), same session per take
  (175539 replays its longer session [1]). Every take estimated exactly as many
  frames as its baseline (37 / 18 / 30 / 48), so this is replay vs replay on the
  same frames.
- **Hardware:** the CUDA provider loaded (ORT moves 222 int8 nodes to CPU and warns
  about Memcpy nodes). Measured 0.43 s per read on CUDA against 0.60 s on CPU,
  synthetic input. The four replays took 21 / 7 / 14 / 23 s, against 74 / 34 / 57 /
  86 s for the baselines.

Numbers below are read from `recordings/<take>/replays/{tkr-shipped,cq-v6}/report.json`.

## The gate

| take | raw per-sq | prior perfect | **settled perfect** |
|---|---|---|---|
| 08-13 | .9597→.9924 | 15/31 → 25/31 | 22/31 → **31/31** |
| 175539 | .9219→.9948 | 4/12 → 11/12 | 5/12 → **8/12** |
| 111421 | .9164→.9946 | 3/23 → 20/23 | 5/23 → **18/23** |
| 181802 | .8848→.9281 | 3/43 → 19/43 | 3/43 → **14/43** |
| **total** | .9169→**.9677** | 25 → **75** | **35 → 71** |

Raw perfect boards go 5 → 75 of 109. The baseline column reproduces the
`tkr-shipped` rows of [[2026-08-31-calibrate-before-after]] exactly.

**181802 is scored against truth that stops early.** Its `truth.json` says so: truth
covers the game up to seq 186, but the take runs to seq 300 and the game went on
(Qd2, O-O-O, Nf6, Bb4). The scorer applies the last segment to everything after
seq 157, so both runs are charged for reading the later moves. ChessQ reads them
(the long run of `empty->Q`, `K->empty`, `Q->R` errors is c1–e1 after queenside
castling). Scored only where truth exists (events up to seq 186, 15 boards):

| 181802, seq ≤ 186 | raw per-sq | prior perfect | settled perfect |
|---|---|---|---|
| shipped → ChessQ | .9510→1.000 | 3/15 → 15/15 | 3/15 → 9/15 |

With that row, settled perfect is **35 → 66 of 81** truth-covered boards.

## What the errors are

**Raw errors are mid-move frames, and hands read as empty squares.** On the
truth-covered events ChessQ has 10 imperfect raw boards (6 / 1 / 3 / 0 per take).
All 10 read (part of) the *next* truth segment's position before its boundary: the
from-square already reads empty and the to-square already holds the piece (e2→e4,
e7→e5, g1→f3, b8→c6, f1→b5, g8→f6, b1→c3). In three of them squares under the arm
also read empty at p ≥ 0.96. Example: 08-13 seq 268 (Nf3 being played) reads
`b8: n→empty @1.00` under the arm and the knight in hand as `e4: P→n @0.83`. This
is the benchmark's hand finding again: occluded squares are not uncertain, they
are confidently empty. On 08-13 the stabilizer absorbed all of them (settled
31/31). One is a truth error rather than a misread: 111421 seq 187 is a still frame
with no hand in it, and ChessQ reads exactly segment 11's position (knight on c3,
queen on d8) 7 frames before that segment's boundary at seq 194. That read settled
and is charged as an error.

**Settled errors are almost all debounce lag.** On the three later takes the settled
stream is *worse* than the raw and prior streams (prior → settled: 175539 11 → 8,
111421 20 → 18, 181802 19 → 14; 08-13 gains 25 → 31). Of the 15 settled-wrong
boards on truth-covered events, 8 show the previous position verbatim (in 7 of
those the raw read was already right), 5 show an older or partly updated position,
and 2 hold the early 111421 read above. The stabilizer needs two agreeing reads 2.5 s apart, and burst-end
schedules about 0.4 reads/s, so at normal playing pace it often has not switched
before the next move. The stabilizer was sized for a reader whose raw stream was
wrong on most boards; with this reader it removes hand frames and costs lag.

**The prior repair does nothing for ChessQ.** Raw and prior are identical on all
four takes: ChessQ never produced an impossible board in this gate.

## Verdict

**GO — V6 passes the replay gate by a wide margin on every take.** Settled perfect
boards double (35 → 71 of 109; 35 → 66 of 81 on truth-covered boards), every take
improves on every stream, and reads are about 3× faster than the shipped cascade +
bank on this machine.

## Caveats

- **License.** The V6 weights are PolyForm Noncommercial 1.0.0. Fine for personal
  and research use; adopting it as the default inherits that restriction
  (THIRD-PARTY.md).
- **No calibration path.** `/calibrate` answers 409 while ChessQ serves. If ChessQ
  is adopted, the bank, auto-calibration and the attractor guard stop applying to
  what serves.
- **Settle tuning was not touched.** The lag above suggests a stronger reader wants
  a shorter or adaptive settle window. That is a separate change with its own gate.
- **Four takes, one board set per take, 109 boards.** It is the same gate as every
  earlier decision, and the margin is far outside its noise, but it is not a
  cross-style result. The cold benchmark's curated and held-out stills (9/10 and
  11/11 perfect) are the cross-scene evidence.
- **Hand frames are still read confidently.** The settled stream absorbed them here.
  A reader that publishes raw reads, or a faster settle, needs a hand gate.
