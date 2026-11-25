# Latency round 3: a shorter calm hold and a shorter settle window (2026-11-25)

**Question.** After a move, the node waits `EXIT_HOLD_S = 0.7` s of calm, then reads
three frames 0.3 s apart, and the settle window (`SETTLE_SECONDS = 0.5`, 2 reads)
commits the new board. Can a shorter hold, a shorter window or both cut the time from
move to settled board, now that the energy gate (DESIGN D16) keeps obscured reads from
committing? A variant counts as a win only if the latency drops and settled-perfect
boards, hand leaks and wrong boards in the resting-hand stretch are no worse than
the shipped stack with the energy gate.

**Answer: yes, on the node alone.** Set the node's burst-end hold to **0.3 s** and the
settle window to **0.3 s** (2 reads). Keep the burst-end run at 3 reads 0.3 s apart.
Leave the phone at its 0.7 s hold.

- On live take 2 (`20261124-221950`), the only take recorded with today's phone, the
  median move→settle drops from **1.66 s to 0.94 s**. Paired per boundary, the median
  change is −0.42 s: 6 boundaries faster, 1 slower.
- On the seven gate takes, the median change is −0.36 s (34 of 50 boundaries faster,
  6 slower). The pooled median moves only from 3.12 s to 3.0 s, because most gate
  moves are first read by the keepalive, not by a burst end (see "Why the gate
  barely moves").
- Hand leaks stay at **0**, and the resting-hand stretch stays at **0** wrong boards.
  This holds in all six robustness cells: two edge streams × three energy thresholds.
- Settled perfect is better in 5 of the 6 cells and 0.002 lower in the sixth
  (`phone-nofill` at −12.9). The time-weighted share of correct frames rises from
  0.784 to 0.812. The node reads 6 % more frames.

The gain depends on the energy gate. With the gate off, the same settings leak hands
(6 on the gate takes, 15 on take 2), so the change must ship coupled to the gate.

## Method

- **Cache.** ChessQ V6 fp32 read every frame of the seven gate takes and take 2 once
  (`tools/probes/latency3_cache.py`, 4,859 frames, ~4.5 frames/s on the shared GTX 1650
  under `flock`). It stores per frame the raw board, `chessq.frame_energy`, the board
  direction, the capture time, the header stability, and the stability re-metered by
  today's `MotionMeter` (`stab_re`).
- **Simulation** (`tools/probes/latency3_sim.py`). The harness drives the real code:
  `schedule.BurstTriggered`, `stabilize.BoardStabilizer` with its excess guard and the
  shipped `observe(..., obscured=)`, `ingest.OrientationGuard`, and the edge's
  `AdaptiveForwarder` (the Python mirror of the phone's `motion.ts`). Only constructor
  parameters change. No product code was changed.
- **Two edge streams.**
  - `recorded` feeds the node the frames exactly as recorded. It reproduces the gate
    (below) and projects what `make gate` will show.
  - `phone` feeds the node what today's phone would have *sent*. The recorded frames
    go through `AdaptiveForwarder(exit_hold_s=hold, tail_s=1)`. By default the
    node's `BurstTriggered` gets the same hold.
- **The old takes' missing frames.** Six gate takes were recorded before the
  forwarder's 1 s burst tail existed. After a burst they dropped to 1 fps, so a frame
  today's phone sends in its tail may not exist. In the `phone` stream, gaps over
  0.25 s are filled with synthetic frames every 0.1 s, at stability 1.0. Each one
  borrows the read, energy and direction of the nearest recorded frame. A synthetic
  frame goes to the forwarder **only while it sends at burst rate** (in a burst or in
  its tail), so it replaces a missing tail frame and never a recorded keepalive.
  - The `phone-nofill` stream drops the synthetic frames. It is the pessimistic bound:
    a read that falls into a gap waits for the next recorded frame.
  - Shorter holds need *fewer* synthetic frames, because their burst-end reads fall
    inside the old phone's dense hold period. Synthetic reads on the gate takes: 79 at
    hold 0.7, 42 at 0.3. Take 2 needs none.
  - Numbers in the `phone` stream that rest on synthetic reads are projected. The
    recommendation holds in `phone-nofill` too.
- **Metrics.**
  - **move→settle**: a truth boundary's frame to the read that first settles the new
    board, matched as in [2026-11-24-changed-utc](2026-11-24-changed-utc.md). The
    match window runs from the previous boundary to the next one. The change must lie
    after the previous boundary.
  - **settled perfect**: per read outside the 12-frame guard window, as the gate
    scores it.
  - **frames ok**: the settled board in force at every recorded frame, against
    truth. It does not depend on where the reads fall.
  - **hand leaks**: a settled square holding a class that no earlier, current or next
    truth board has there (the round-2 definition).
  - **not-lag**: settled wrong and not the previous truth board (the probe's column).
  - **wrong changes**: settled changes to a board that is not the current truth.
    This is what a viewer sees flicker to.
  - **rest wrong**: settled boards in take 2's rows 452–596 that match neither truth
    board there. Those are the probe's resting-hand stretch, plus the e4 knight laid
    on its side.
- **Grid.**
  - hold 0.7 / 0.5 / 0.3 / 0.1 / 0 s
  - settle window 0.5 / 0.3 / 0.15 / 0 s, with 2 reads, plus a pass-through (0 s,
    1 read)
  - burst-end reads 3 / 2, spacing 0.3 / 0.15 s
  - energy gate on / off
  - both `phone` streams: 400 variants in total.

## Baseline reproduction

On the `recorded` stream with the shipped settings, the harness reproduces both gate
replays read for read. Every take's `(frame_seq, settled board, obscured)` sequence is
identical to the replay's `events.ndjson`.

| replay | reads equal | settled perfect | hand leaks | not-lag | obscured reads |
|---|---|---|---|---|---|
| `guard-only` (gate off) | 446/446 | **314/361** | 0 | 14 | 0 |
| `energy` (gate on, shipped) | 446/446 | **312/361** | 0 | 15 | 53 |

- **Take 2, gate off.** 80/94 settled perfect and 9 of 22 resting-hand boards wrong,
  which is the probe's 9.
- **Take 2, gate on.** 88/94 and 0 of 22 wrong, which is the lead's `energy` number.
- The live node itself read 117 frames of take 2. The simulation reads 117, and 115
  of them are the same frames (the live inbox drains on its own timing).

## Results

Variants on the `phone` stream with the energy gate on (−12.4) and 3 burst-end reads
0.3 s apart. Gate figures pool the seven takes. "med" and "p90" are move→settle in
seconds over the matched boundaries.

| hold | window | gate perfect | frames ok | leaks | not-lag | wrong changes | gate med / p90 | take 2 perfect | take 2 leaks | rest wrong | take 2 med / p90 | reads/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 0.5 (shipped) | 329/391 .841 | .784 | 0 | 24 | 27 | 3.12 / 4.44 | 88/94 | 0 | 0 | 1.66 / 4.19 | 42.2 |
| 0.7 | 0.3 | 332/391 .849 | .790 | 0 | 22 | 29 | 3.00 / 4.44 | 90/94 | 0 | 0 | 1.35 / 4.19 | 42.2 |
| 0.5 | 0.3 | 340/404 .842 | .788 | 0 | 28 | 28 | 2.96 / 4.43 | 86/90 | 0 | 0 | 1.14 / 4.21 | 43.2 |
| 0.3 | 0.5 | 353/419 .843 | .810 | 0 | 23 | 17 | 3.00 / 4.32 | 91/96 | 0 | 0 | 1.25 / 4.00 | 44.7 |
| **0.3** | **0.3** | **355/419 .847** | **.812** | **0** | **24** | **21** | **3.00 / 4.32** | **91/96** | **0** | **0** | **0.94 / 4.00** | **44.7** |
| 0.3 | 0 (2 reads) | 359/419 .857 | .820 | 0 | 23 | 21 | 2.82 / 4.30 | 91/96 | 0 | 0 | 0.94 / 4.00 | 44.7 |
| 0.1 | 0.3 | 393/460 .854 | .837 | 0 | 28 | 20 | 2.52 / 4.13 | 87/100 | 0 | **9** | 2.08 / 3.91 | 48.2 |
| 0.7 | pass-through | 358/391 .916 | .876 | 0 | 21 | 13 | 1.56 / 3.42 | 91/94 | 0 | 0 | 1.04 / 3.78 | 42.2 |
| 0.3 | pass-through | 377/419 .900 | .867 | 0 | 28 | 12 | 2.04 / 3.47 | 91/96 | 0 | 0 | 0.62 / 4.00 | 44.7 |
| 0.3 | 0.3, **gate off** | 354/419 .845 | .803 | **6** | 33 | 42 | 2.82 / 4.32 | 76/96 | **15** | **12** | 0.94 / 4.32 | 44.7 |

Hold 0 behaves exactly like 0.1, because the stream's frames are ~0.1 s apart.
Settle 0.15 s gives the same numbers as 0.3 s at 0.3 s spacing.

**Robustness of the shortlist.** Two edge streams (`phone`, `phone-nofill`) times
the energy thresholds −12.1, −12.4 and −12.9. That range is the hand probe's
leave-one-take-out range (−12.13 to −12.93), because the shipped threshold was fitted
on these same takes.

| variant | gate leaks (6 cells) | rest wrong (6 cells) | take 2 med (6 cells) | settled perfect vs shipped |
|---|---|---|---|---|
| shipped: hold 0.7, window 0.5 | 0 everywhere | 0 everywhere | 1.66 | — |
| **hold 0.3, window 0.3** | **0 everywhere** | **0 everywhere** | **0.94 everywhere** | **higher in 5 cells, −0.002 in one** |
| hold 0.7, pass-through | 0, 0, 0, **3**, **4**, 0 | 0 everywhere | 1.04–1.35 | higher |
| hold 0.3, pass-through | **3**, 0, 0, **4**, **1**, 0 | 0 everywhere | 0.62–0.94 | higher |

**The rollout.** The phone and the node need not change their hold together. The
phone already sends at burst rate through its own hold and then the 1 s tail, so a
node with a shorter hold still finds dense frames (window 0.3 s, gate on, `phone`
stream):

| phone hold | node hold | gate perfect | gate leaks | take 2 perfect | rest wrong | take 2 med |
|---|---|---|---|---|---|---|
| 0.7 | 0.7 | 332/391 | 0 | 90/94 | 0 | 1.35 |
| **0.7** | **0.3** | **350/414** | **0** | **91/96** | **0** | **0.94** |
| 0.3 | 0.7 | 308/365 | 0 | 81/84 | 1 | 1.30 |
| 0.3 | 0.3 | 355/419 | 0 | 91/96 | 0 | 0.94 |

**Per take**, shipped → recommended (`phone` stream; `latency3_sim.py compare`):

| take | settled perfect | hand leaks | wrong changes | median move→settle (s) |
|---|---|---|---|---|
| 20260813-193308 | 52/53 → 57/60 | 0 → 0 | 2 → 3 | 0.72 → 0.31 |
| 20260827-175539 | 11/15 → 11/15 | 0 → 0 | 4 → 4 | 4.32 → 4.32 |
| 20260828-111421 | 43/49 → 42/54 | 0 → 0 | 4 → 5 | 2.08 → 2.08 |
| 20260827-181802 | 55/73 → 59/76 | 0 → 0 | 7 → 5 | 2.21 → 3.24 |
| 20260827-163352 | 14/22 → 19/26 | 0 → 0 | 3 → 0 | 2.81 → 2.34 |
| 20260827-170333 | 53/63 → 60/67 | 0 → 0 | 3 → 0 | 3.60 → 3.00 |
| 20260827-172736 | 101/116 → 107/121 | 0 → 0 | 4 → 4 | 3.48 → 3.24 |
| 20261124-221950 | 88/94 → 91/96 | 0 → 0 | 2 → 1 | 1.66 → 0.94 |

**What `make gate` should show** (the `recorded` stream, the node-only change):
- settled perfect 312/361 → 350/405, the same rate (.864)
- frames ok .787 → .815; hand leaks 0 → 0
- not-lag 15 → 22; wrong changes 27 → 23
- gate median move→settle 3.12 → 2.70 s; reads 446 → 496 (+11 %)
- take 2: 88/94 → 91/96, resting-hand 0 wrong, median 1.66 → 0.94 s

## Findings

- **Why the gate barely moves.** On the gate takes, 45 of 60 matched boundaries are
  first read by a 2 s keepalive, not by a burst end (`latency3_sim.py explain`). On
  the held-out takes the motion meter stays in a burst for 2–3 s after the truth
  boundary: the hand moves on (to the clock, away from the board) after the board is
  already complete. A hold can only shorten the last calm stretch of that time. On
  175539 the meter never opens a burst at all (4.32 s either way). Take 2's hand
  leaves promptly, so its moves are burst-end reads, and there the hold matters.
- **The hold has a floor.** At 0.1 s (and 0) the burst ends during the pause between
  two strides of one move. On take 2 a board with the king lifted (no king on the
  board) settles during the king's return to e8 (boundary 436) and stays wrong into
  the resting-hand stretch: 9 wrong boards. That pause is what `EXIT_HOLD_S` exists
  for. 0.3 s passes every check here; 0.5 s gains less than 0.3 s.
- **The window rule from round 2 no longer holds with the energy gate.** Round 2 found
  that a window at or below the spacing (two reads span it) let hands settle, and
  `.claude/rules/stream.md` now asks for `spacing < window`. With the gate on, the
  0.3 s window at 0.3 s spacing settles no hand in any cell. With the gate off, it
  leaks (last row of the results table). The gate, not the window, now keeps hands
  out.
- **Pass-through is tempting and not safe.** It halves the flicker (wrong changes
  27 → 13) and lifts settled perfect to .90–.92, because with the gate on, most of
  what the window still blocks is lag. But it leaks hands at the looser threshold and
  on the pessimistic stream (1–4 per cell), so it fails the bar.
- **Other knobs.**
  - 2 burst-end reads cannot give the 0.5 s window its quorum, so the board waits for
    a keepalive (13 resting-hand wrong boards at hold 0.7).
  - Spacing 0.15 s buys no latency and costs settled boards.
- **The cost of the recommendation.**
  - Not-lag boards: 15 → 22 on the recorded gate, 24 → 24 on the `phone` stream.
  - Of the extra ones, 3 are the *next* truth board settled 1–2 frames before its
    truth boundary (111421 seq 73, 193308 seq 238 and 283). The others are partial
    moves of the kind the shipped stack also settles (castling rooks, 181802's
    multi-square swaps).
  - 111421 and 193308 each lose a few correct frames; the other takes gain.

## What the build changes

All of it is on the node. The phone keeps `EXIT_HOLD_S = 0.7` in
`packages/viewer/src/lib/edge/motion.ts`, and the Python edge keeps
`ForwarderConfig.exit_hold_s = 0.7`.

1. `packages/inference/src/board_reader_inference/stream/schedule.py`:
   `DEFAULT_POLICY_SPEC` becomes
   `burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.3,exit_hold_s=0.3`.
   - Keep `EXIT_HOLD_S = 0.7` as the phone's mirrored value; its comment and the
     module docstring must say that the node's hold is now deliberately shorter.
   - The phone's hold only sets how long it keeps sending at burst rate, and that
     dense stretch (hold plus 1 s tail) is what the node's run reads.
2. `packages/inference/src/board_reader_inference/stream/stabilize.py`:
   `SETTLE_SECONDS = 0.5` → `0.3`, with its comment.
3. **Couple both to the energy gate.** With `BR_OBSCURED_ENERGY=off` these settings
   leak hands. Either the build documents this next to `BR_OBSCURED_ENERGY`, or the
   node falls back to hold 0.7 and window 0.5 when the gate is off. This is a decision
   for the build.
4. `.claude/rules/stream.md`: rewrite the `spacing < window` rule. The window may
   equal the spacing only while the energy gate is on (this report).
5. Tests that pin the two constants or the spec string (`tests/stream/test_schedule.py`,
   `test_stabilize.py`, the gate-table spec), and the DESIGN §5 settle numbers if they
   are quoted.
6. **Verify.**
   - Run `make gate GATE_NAME=latency3` against `energy`. Expect the "What `make gate`
     should show" numbers above.
   - Then take a live take with moves and a resting hand. Take 2's 9 matched
     boundaries are a small sample.

## Reproduce

```bash
S=<scratch dir>
for t in <the seven gate takes> 20261124-221950; do
  flock /tmp/claude-1000/br-gpu.lock uv run python tools/probes/latency3_cache.py $S/cache $t
done                                                       # ~18 min GPU, 6.7 MB
uv run python tools/probes/latency3_sim.py $S/cache validate   # guard-only + energy, read for read
uv run python tools/probes/latency3_sim.py $S/cache sweep > grid.tsv   # 400 variants, ~30 s
uv run python tools/probes/latency3_sim.py $S/cache robust     # shortlist x streams x thresholds
uv run python tools/probes/latency3_sim.py $S/cache compare 0.7,3,0.3,0.5,2 0.3,3,0.3,0.3,2
uv run python tools/probes/latency3_sim.py $S/cache explain 0.7 3 0.3 0.5 2 1   # where the time goes
```

The cache was deleted after the measurement. It is cheap to rebuild and specific to
ChessQ V6.

## Follow-up: the build and the real gate (2026-11-25)

The build shipped the recommendation on the node alone: `DEFAULT_POLICY_SPEC` gained
`exit_hold_s=0.3` and `SETTLE_SECONDS` went to 0.3 s. The phone and the Python edge keep
their 0.7 s hold. The coupling to the energy gate is documented, not enforced: the
runbook entry for `BR_OBSCURED_ENERGY` says that `off` needs round 2's 0.5 s window and
0.7 s node hold set by hand. `latency3` is the new gate base.

`make gate GATE_NAME=latency3` against `energy` matches the projection on every number.
Leaks, not-lag, wrong changes, frames ok and latency were scored on the replays'
`events.ndjson` with this probe's `score()`:

| | `energy` | projected | `latency3` (real) |
|---|---|---|---|
| raw perfect | 341/361 | — | 374/405 |
| settled perfect | 312/361 | 350/405 | **350/405** (.864) |
| frames ok | .787 | .815 | **.815** |
| hand leaks | 0 | 0 | **0** |
| not-lag | 15 | 22 | **22** |
| wrong changes | 27 | 23 | **23** |
| median move→settle (s) | 3.12 | 2.70 | **2.70** |
| reads | 446 | 496 | **496** |

Take 2 (`20261124-221950`, replayed with the new spec): 88/94 → **91/96** settled
perfect, 0 hand leaks, **0 of 23** wrong boards in the resting-hand stretch (rows
452–596), median move→settle 1.66 → **0.94 s**.

Per take, settled perfect: 193308 46/47 → 53/55, 175539 11/15 → 11/15, 111421 30/34 →
39/48, 181802 50/60 → 54/67, 163352 14/22 → 20/25, 170333 56/63 → 61/69, 172736
105/120 → 112/126. A live take with moves and a resting hand is still open (step 6).
