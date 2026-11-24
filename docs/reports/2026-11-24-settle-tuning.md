# Settle tuning for ChessQ: 84 → 104 settled-perfect boards, 4.3 → 3.0 s to settle (2026-11-24)

> Plan steps 16–18 of [docs/plans/2026-11-24-simplify-around-chessq.md](../plans/2026-11-24-simplify-around-chessq.md).
> Code and raw tables: `scratch/experiments/settle-sweep/` (gitignored): `cache_reads.py`,
> `sweep.py`, `analyze.py`, `table.py`, `best.py`, `inspect_wrong.py`, `out/*.tsv`.

## Outcome

Shipped: **settle 1.0 s / 2 reads / 2 stale windows**, with the schedule
**`burst-end:burst_end_reads=3,spacing_s=0.6`** (keepalive 3 s, unchanged). On the
four gate takes (real `make gate` replay, `cq-v6-tuned`):

| take | raw per-sq | raw perfect | prior perfect | settled perfect |
|---|---|---|---|---|
| 20260813-193308 | .9924→.9964 | 25/31→31/35 | 25/31→31/35 | 31/31→35/35 |
| 20260827-175539 | .9948 | 11/12 | 11/12 | 8/12 |
| 20260828-111421 | .9985→.9989 | 20/21→27/29 | 20/21→27/29 | 18/21→24/29 |
| 20260827-181802 | .9996→.9985 | 36/37→41/43 | 36/37→41/43 | 27/37→37/43 |
| **total** | .9966→.9976 | 92/101→110/119 | 92/101→110/119 | 84/101→**104/119** |

(`cq-v6 → cq-v6-tuned`. The event counts differ because the new schedule reads three
times per move instead of once.) Settled perfect goes from 83 % to 87 % of scored
boards. The gap between raw and settled shrinks from 8 to 6 boards. The median
move-to-settle time drops from 4.3 s to 3.0 s (p75 5.0 → 3.8 s). No hand or misread
settles, and 08-13 settles all 35 of its scored boards.

## What changed in the code

- **Step 16** (`38a482d`): `stabilize.SettleConfig` (`seconds`, `min_reads`,
  `stale_windows`). `serve` reads it from `BR_SETTLE_SECONDS`,
  `BR_SETTLE_MIN_READS` and `BR_SETTLE_STALE_WINDOWS`, and `/healthz` reports it.
  `replay` takes `--settle-seconds`, `--settle-min-reads` and `--stale-windows`,
  and records the values in the manifest, in the default name
  (`-settle1s-2r-2w`) and in the config digest. A default config leaves names and
  digests as before.
- **Step 18**: `SETTLE_SECONDS` 2.5 → 1.0, and `DEFAULT_POLICY_SPEC`
  `burst-end:burst_end_reads=3` → `burst-end:burst_end_reads=3,spacing_s=0.6`.
  **The gate now replays `serve`'s own schedule.** It used to replay plain
  `burst-end`: one read per move, a schedule `serve` never ran. `GATE_BASE` is now
  `cq-v6-tuned`.

## Method

The stack's only per-frame state is in the schedule and the debounce. The
recognizer is stateless. So the sweep has three levels, and only the first one
uses the GPU:

0. **Every frame read once.** `cache_reads.py` reads every frame of each take's
   replayed session with the fp32 model: 4246 frames in ~15 min, cached as
   `(seq, square_dist)`.
1. **One replay per schedule.** `replay.replay()` (the real loop, prior repair,
   stabilizer and event writer) runs with the recognizer memoized from the cache:
   ~1 s per take.
2. **One re-settle per settle config.** `BoardStabilizer(config)` runs again over
   that replay's prior placements, and the result is scored with `trial.py`'s
   truth: milliseconds per config.

**Check that the cached path is exact.** At the old defaults, the cached `burst-end`
replay produces an `events.ndjson` byte-identical to the real `cq-v6-fp32` replay on
all four takes. The re-settled stream reproduces the recorded settled count
(31/8/18/27). The final config's real `make gate` replay matches the sweep's
number: 104/119.

**Grid.** Schedules: `burst_end_reads` {1, 2, 3} × `spacing_s` {0.6, 1.0, 1.3} ×
`keepalive_s` {1.5, 3.0} (18). Settle: `seconds` 0–2.5 in steps of 0.25 ×
`min_reads` {1, 2, 3} × `stale_windows` {1, 2, 3} (99). That gives 7128 scored
configurations on the four gate takes, in ~4 min.

**Metrics.**

- **Settled perfect:** scored boards (outside truth guards) whose settled board
  equals the truth.
- **Move-to-settle latency:** for each truth boundary, the capture time from the
  boundary frame to the first event in that segment whose settled board equals
  the segment's truth. A boundary is *reached* when such an event exists.
- **Hard constraint: no leak.** A *leak* is a settled square whose class appears in
  neither the current truth segment, the next one, nor any earlier one at that
  square. That is a hand or a misread that settled. A move adopted only in part,
  or a lag of several moves, does not count as a leak. In addition, 08-13 must
  settle every scored board.

## Results (four gate takes, pooled)

| schedule | settle s / min reads | settled perfect | boundaries reached | median (p75) latency | leaks | 08-13 |
|---|---|---|---|---|---|---|
| `burst-end` (old gate) | 2.5 / 2 (old) | 84/101 (.832) | 19/51 | 4.32 (5.04) s | 0 | 31/31 |
| `burst-end` | 1.75–2.0 / 1–2 (best) | 86/101 (.851) | 20/51 | 4.02 (4.92) s | 0 | 31/31 |
| `burst-end` | 0 / 1 (pass-through) | 92/101 | 34/51 | 1.32 s | **5** | **25/31** |
| `…reads=3` (old serve) | 2.5 / 2 (old) | 88/111 (.793) | 19/51 | 4.16 (5.51) s | 0 | 36/36 |
| `…reads=3` | 0.25–1.5 / 1–2 | 92–94/111 | 21–22/51 | 3.2–3.4 s | 0 | **35/36** |
| `…reads=3` | 1.75–2.0 / 1–2 | 93/111 (.838) | 21/51 | 3.42 (4.44) s | 0 | 36/36 |
| `…reads=2,spacing_s=1.0` | 0.75–1.0 / 1–2 | 95/109 (.872) | 22/51 | 2.94 (3.43) s | 0 | 32/32 |
| `…reads=3,spacing_s=0.6` | 2.5 / 2 | 94/119 (.790) | 18/51 | 4.56 (5.20) s | 0 | 35/35 |
| **`…reads=3,spacing_s=0.6`** | **0.5–1.0 / 1–2** | **104/119 (.874)** | 21/51 | **3.00 (3.78) s** | 0 | 35/35 |

What the grid shows:

- **The settle window matters only relative to the read spacing.** A move settles
  on its own burst-end reads when the window is no longer than those reads span:
  (`burst_end_reads` − 1) × `spacing_s`. Otherwise it waits for the 3 s keepalive.
  With one read per move (the old gate schedule), no window helps: the best is
  86/101 at a median of 4.0 s.
- **Pass-through breaks the hard constraint.** With 0 s and 1 read, the settled
  stream equals the raw one (92/101), but hands leak (5) and 08-13 drops to 25/31.
  So the debounce still has a job: it filters out hand frames.
- **`stale_windows` changed nothing** on any row of the grid; it stays 2.
  **`min_reads` 1 and 2 tie** wherever the window is longer than 0. It stays 2,
  because one read alone is never evidence.
- **The old served schedule fails the constraint with short windows:** at
  ≤ 1.5 s, 08-13 settles a wrong board once (35/36).

**Choice: 1.0 s, not 0.5 s.** On the gate, 0.5 s and 1.0 s give identical results.
At 0.5 s, two agreeing burst-end reads 0.6 s apart settle a square. At 1.0 s it
takes all three (1.2 s). The held-out takes favour 0.5 s (below). We chose 1.0 s
anyway, because it keeps the window about one hand-crossing long (~1 s, the
original sizing argument). That matters on a denser stream (`BR_SCHEDULE=every-frame`,
or a faster edge), where 0.5 s would be only a few frames.

## Hand inspection (chosen config, gate takes)

All 15 settled-wrong boards were classified (`inspect_wrong.py`). None is a hand:

- **9 are lag.** The settled board is the previous position verbatim on the
  squares that differ (castling twice, Bc5, Nf3, Bxe3, d3, …).
- **6 are a move adopted in part.** The from-square has already settled empty
  while the to-square has not yet adopted its arrival (`c3:N->empty`,
  `f1:R->empty`, `f4:P->empty`, `d1:R->empty`). Every square still holds a value it
  had in the previous or current position. The old default shows the same pattern
  (6 such boards at 2.5 s), because squares settle independently.
- **The 4 boards whose own frame is a hand frame** (raw read wrong: 175539 seq 76,
  111421 seq 141, 181802 seq 218 and 274) were checked by eye. The arm is over the
  board in each. In all four the settled board shows the position *before* the
  move, not the hand's read.

One trace in detail. On 175539 at seq 327, a frame with the knight in hand reads b1
and c3 both empty. That read is one of the two that settle b1 as empty, which is
correct after Nc3. c3 has not yet ripened at seq 352. The leak check therefore
holds; what remains is that, for one read, the display can show a board with the
moving piece missing.

## Held-out check (three non-gate takes)

20260827-163352, 170333 and 172736 have chess-frame truth. They were recorded with
the broken motion meter, so they are replayed with `--restamp-stability` (what a
phone sends today). Without restamping, burst-end never sees calm and every
schedule reduces to the 3 s keepalive. Real replays:

| take | `cq-v6` (old gate, unrestamped) | old served (`reads=3`, 2.5 s) | **tuned** |
|---|---|---|---|
| 20260827-163352 | 5/11 | 4/15 | **8/18** |
| 20260827-170333 | 13/28 | 17/43 | **19/50** |
| 20260827-172736 | 24/50 | 32/82 | **46/92** |
| **total settled perfect** | 42/89 (.472) | 53/140 (.379) | **73/160 (.456)** |

Like for like (all restamped, from the cache): plain `burst-end` at 2.5 s gives
48/112 (.429), median 6.0 s. The old served config gives 53/140 (.379), median 5.6 s.
**Tuned gives 73/160 (.456), median 4.2 s.** The tuned schedule at 0.5 s would give
84/160 (.525), median 3.6 s. The ranking matches the gate's, so the choice is not
fitted to the four gate takes. The unrestamped `cq-v6` column reads only every 3 s,
mostly at rest, so it is a different population and not comparable.

**The held-out truths look incomplete.** Their "leaks" (32–50 events at *every*
config, nearly independent of the window) fall on four square pairs: d2/d3, d8/e7,
f8/c5 and b1/c3. The truths never put a pawn on d3 or a knight on c3 in 170333 and
172736, yet the reads show d3 and c3 occupied for stretches. These are the same
opening moves (d3, …Bc5, Nc3, …Qe7) that were missing from the 111421 truth before
its fix. Raw is only 104/160 perfect here for the same reason. This was not checked
by eye; fix these truths before using these takes as a gate.

## Not done

- **Count-drop guard** (the optional variant: a read whose piece count falls by more
  than one against the settled board is not evidence for emptying squares). Not
  built. The gate shows no hand leak at the chosen config, so it had nothing to fix
  here. It would also touch the D1 boundary, because the debounce would start
  judging a read's plausibility.
- **`keepalive_s`.** 1.5 s reached two more boundaries but costs twice the
  resting reads. Its perfect rate (.855) is no better than 3 s at the same settle.
  It stays 3 s.

## Caveats

- **One physical set, seven takes, 119 + 160 scored boards.** The gate's margin
  (84 → 104) is large, but the 0.5 s vs 1.0 s question rests on the held-out takes,
  whose truth is suspect.
- **Replay cannot test the edge.** Restamped stability is today's motion meter on
  recorded JPEGs. A live take is the real check that `spacing_s=0.6` burst-end reads
  land after the hand has left.
- **Latency is measured on reached boundaries only.** About 21 of 51 gate
  boundaries are never settled before the next move (fast play: the next move
  arrives first). Their latency is not in the median.
- **GPU cost.** Three reads per move instead of one: 119 vs 101 events on the gate,
  each ~0.2 s with fp32 on the GPU.
