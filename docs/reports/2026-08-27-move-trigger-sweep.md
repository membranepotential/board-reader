# Estimation scheduling: burst-end + keepalive vs every frame

_2026-08-27 night · code `5f9212f` · four 08-27 takes replayed offline, 25 replays,
~3 800 GPU-reads · `scratch/experiments/move_trigger/`_

## TL;DR

- **Ship `burst-end:keepalive_s=3,burst_end_reads=3,spacing_s=1.3`.** It reads a
  take **12× less** (2 500 → 209 reads over the four takes) for a **−0.9 pp**
  pooled per-square cost that is inside the noise on three of four takes.
- The real prize is not accuracy, it is **the node keeping up.** At ~1.3 GPU-s per
  read this hardware needs ~11 GPU-seconds per second to read an 8.3 fps stream
  and delivers one; burst-end asks for **0.4–0.6 reads/s**, a ~55–80 % duty cycle.
  Every-frame is not a thing the live node can do — it is what the latest-wins
  inbox has been silently *pretending* to do by dropping frames.
- **The "mid-burst frames are chaos" premise did not pay.** On three takes the
  boards a policy selects are statistically indistinguishable from a random
  subsample of every frame (bootstrap below). Skipping the burst buys GPU and a
  calmer display, not a better read.
- **Read sparsity does not fix the settled lock-in** — it perturbs it. The control
  take's every-frame settled stream is wrong on 253/253 boards (`q->k`, d8 locked);
  sparse policies get 2–3 of 7 segments settled instead of 0, and the hard take
  goes the other way (10/10 → 6–7/10). `stabilize.py`'s adopt-first rule is the
  problem and needs its own fix, not a read-rate one.
- **The 12–30× number is inflated by history.** Three of the four takes were
  recorded at a flat ~8.3 fps because the motion meter was broken. On take 5 — the
  only one recorded through the *fixed* adaptive forwarder — the same policy buys
  only **5×**, because the edge already stopped sending most of the rest frames.
  Expect ~5×, not 30×, live.

## 1. What was built

`packages/inference/src/board_reader_inference/schedule.py`

| policy | semantics |
|---|---|
| `EveryFrame` | reads everything — the control, and the only reading of a take that leaves no frame unread |
| `BurstTriggered(keepalive_s, burst_end_reads, spacing_s, enter, exit, exit_hold_s)` | tracks bursts with the browser forwarder's own hysteresis (`stability < 0.7` enters, `> 0.85` sustained for `0.7 s` exits); a burst **end** owes `burst_end_reads` reads `spacing_s` apart, starting immediately; nothing is read mid-burst except the floor |
| `WhenStill(keepalive_s, theta, spacing_s)` | no edge detection: read any frame at `stability >= theta`, at most every `spacing_s` |

Every policy carries the same **hard keepalive floor**: if `keepalive_s` of
*capture* time has passed without a read, the next frame is read whatever the
meter claims. This is DESIGN D3 and it is not switchable off (`keepalive_s <= 0`
raises). The dead-meter case is a test: fed 60 recorded seconds of a meter stuck
at 0.2, `BurstTriggered` never believes the move ended and still reads every 3 s,
with no gap above the floor. That is the incident this policy family was written
after — a burst-end scheduler trusting a stuck meter reads a session zero times.

Policies see the header only (`seq`, `capture_ts`, `stability`, `sharpness`,
`burst_id`) and never a wall clock, so a declined frame costs not even a JPEG read
and two replays schedule identically. `name` is the canonical spec string and
round-trips through `parse_policy`, so a replay manifest's `policy` field is
literally the `--policy` argument that reproduces it.

### Restamping

`--restamp-stability` re-runs today's motion meter over the recording's JPEGs in
seq order and overwrites each header's `stability` before the policy sees it,
marking the replayed index `stability_restamped: true`. It is mandatory for the
three older takes, whose recorded field came from the broken absolute-scale meter
(<0.35 on every frame of a mostly-still board). One meter per streaming session;
every frame is metered including the declined ones, because a floor learned from
a subsample is a different meter — that is the whole cost of the flag.

**Layer separation.** `tests/test_layer_separation.py` forbids
`board_reader_inference` from importing `board_reader_edge` (AST scan over
`packages/*/src`), so the meter is **ported**, not imported — a third mirror
beside the Python edge's and the browser's. The copy is pinned by
`test_schedule.py::test_the_ported_meter_matches_the_edges_value_for_value`,
which imports both in a *test* (tests are outside the scan) and asserts equal
output over 50 frames.

### Restamping changes the schedule and nothing else — verified

`stability` is copied onto the published event and read by nothing else. Two
every-frame replays of the control take on the identical stack, one restamped and
one not: **337/337 events byte-identical outside the `stability` field**, 337/337
differing in it. Restamped every-frame is therefore a legitimate control, and this
doubles as a determinism check on the real GPU stack.

Stronger: across all 4 takes × 5 sparse policies, **every** read of a given frame
produced the byte-identical raw and prior board as the every-frame run
(919/919 events). The raw/prior stream of any policy is a pure *subset* of the
every-frame run's — which is both a determinism proof and a large future speedup
(a new policy's raw/prior score can be computed from one every-frame replay; only
the stateful `settled` stream needs a real replay).

## 2. Caveat: the read-only comparators are a different stack

`baseline-3452334` and `bdemotion-fe8a7bb` were produced by a **`CascadeEstimator`**
(their manifests say so — the session bank did not load in that session). Every row
of this sweep is the **`EnsembleEstimator`** the given environment builds
(cascade ⊕ bank, 50/50). Their numbers are *not* comparable to this table's. The
restamped `mt-every-frame` row is the control, and it is the only thing the sparse
rows are diffed against.

## 3. What the restamped meter sees

| take | frames | span | arrival fps | `<0.7` | `>0.85` | bursts found | truth moves | frames in-burst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control 16:33 | 337 | 35 s | 9.6 | 27 % | 66 % | **7** | **7** | 55 % |
| hard/dim 17:27 | 1429 | 174 s | 8.2 | 35 % | 59 % | 26 | 10 | 55 % |
| fast 17:55 | 433 | 52 s | 8.4 | 33 % | 58 % | 7 | 14 | 58 % |
| take 5 18:18 | 301 | 142 s | 2.1 | 49 % | 40 % | 11 | 12 | 82 % |

The detector is honest on the control take (7 bursts, 7 moves) and on take 5
(11 vs 12). It **over-segments the dim take** 26:10 — hands hovering in poor light
break one move into several bursts, which is why that take's sparse rows still
spend 47–121 reads. It **under-segments the fast take** 7:14 — two moves inside one
busy window are one burst, exactly as `truth.json`'s own comment describes.

Take 5's 82 % in-burst is the adaptive forwarder working: it *already* sends 12 fps
during a move and 1 fps at rest, so the stream that arrives is mostly move frames
and there is little left for a node-side scheduler to skip.

## 4. The sweep

Reads / frames · prior per-square · prior perfect · settled per-square · settled
perfect · settle latency (median/max over truth segments, and how many of them ever
settled correctly).

| take | policy | reads/frames | scored | prior sq | prior pf | settled sq | settled pf | settle latency | GPU s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| control | **every-frame** | 337/337 | 253 | 97.2 % | 30.0 % | 96.1 % | **0.0 %** | never 0/7 | 452 |
| control | burst-end k=3 | 15/337 | 15 | 97.7 % | 46.7 % | 95.5 % | 13.3 % | 5.8 s / 6.9 s · 3/7 | 23 |
| control | burst-end k=5 | 11/337 | 11 | 97.0 % | 27.3 % | 94.0 % | 0.0 % | never 0/7 | 18 |
| control | burst-end k=3 n=2 | 19/337 | 15 | 97.4 % | 33.3 % | 96.6 % | 13.3 % | 6.6 s / 6.9 s · 2/7 | 28 |
| control | burst-end k=3 n=3 | 21/337 | 16 | 97.0 % | 31.2 % | 95.3 % | 0.0 % | 5.7 s / 5.9 s · 2/7 | 28 |
| control | when-still k=3 s=1 | 28/337 | 23 | 97.2 % | 26.1 % | 95.4 % | 0.0 % | 5.1 s / 5.2 s · 2/7 | 36 |
| hard | **every-frame** | 1429/1429 | 1309 | 94.9 % | 30.9 % | 97.2 % | 49.9 % | 3.4 s / 4.8 s · 10/10 | 1795 |
| hard | burst-end k=3 | 68/1429 | 61 | 94.8 % | 27.9 % | 96.2 % | 24.6 % | 6.7 s / 11.6 s · 7/10 | 92 |
| hard | burst-end k=5 | 47/1429 | 44 | 94.6 % | 31.8 % | 96.5 % | 27.3 % | 8.3 s / 8.8 s · 6/10 | 68 |
| hard | burst-end k=3 n=2 | 86/1429 | 81 | 94.8 % | 27.2 % | 96.4 % | 28.4 % | 6.3 s / 10.0 s · 6/10 | 114 |
| hard | burst-end k=3 n=3 | 102/1429 | 94 | 94.8 % | 29.8 % | 95.8 % | 26.6 % | 5.9 s / 14.8 s · 7/10 | 134 |
| hard | when-still k=3 s=1 | 121/1429 | 114 | 95.0 % | 31.6 % | 96.2 % | 29.8 % | 5.5 s / 11.3 s · 10/10 | 161 |
| fast | **every-frame** | 433/433 | 270 | 98.0 % | 69.6 % | 98.4 % | 62.6 % | 3.6 s / 4.6 s · 7/14 | 540 |
| fast | burst-end k=3 | 20/433 | 11 | 96.3 % | 45.5 % | 96.7 % | 36.4 % | 4.5 s / 5.2 s · 2/14 | 27 |
| fast | burst-end k=5 | 14/433 | 8 | 97.7 % | 62.5 % | 97.5 % | 50.0 % | 5.2 s / 7.9 s · 6/14 | 21 |
| fast | burst-end k=3 n=2 | 25/433 | 16 | 98.0 % | 62.5 % | 97.0 % | 56.2 % | 4.2 s / 7.3 s · 5/14 | 34 |
| fast | burst-end k=3 n=3 | 28/433 | 18 | 98.3 % | 66.7 % | 97.7 % | 66.7 % | 4.0 s / 5.5 s · 5/14 | 36 |
| fast | when-still k=3 s=1 | 39/433 | 26 | 98.1 % | 76.9 % | 97.9 % | 46.2 % | 4.6 s / 7.1 s · 5/14 | 50 |
| take 5 | **every-frame** | 301/301 | 266 | 92.3 % | 31.2 % | 93.4 % | 27.1 % | 5.5 s / 13.9 s · 4/12 | 358 |
| take 5 | burst-end k=3 | 52/301 | 49 | 90.7 % | 20.4 % | 91.4 % | 12.2 % | 4.1 s / 5.5 s · 5/12 | 65 |
| take 5 | burst-end k=5 | 31/301 | 27 | 90.1 % | 18.5 % | 90.9 % | 7.4 % | 6.1 s / 6.2 s · 3/12 | 40 |
| take 5 | burst-end k=3 n=2 | 53/301 | 47 | 90.3 % | 17.0 % | 91.2 % | 19.1 % | 7.2 s / 14.9 s · 3/12 | 65 |
| take 5 | burst-end k=3 n=3 | 58/301 | 55 | 91.5 % | 20.0 % | 91.9 % | 21.8 % | 7.1 s / 14.9 s · 3/12 | 71 |
| take 5 | when-still k=3 s=1 | 81/301 | 77 | 90.5 % | 13.0 % | 91.3 % | 22.1 % | 4.6 s / 14.9 s · 6/12 | 98 |

`k` = `keepalive_s`, `n` = `burst_end_reads` (spacing 1.3 s). GPU seconds are on a
contended GTX 1650; the constant is **~1.3 s per read** and it does not vary with
the policy, so the reads column *is* the cost column.

Totals over the four takes:

| policy | reads/2500 | factor | segments ever settled | median settle latency |
|---|---:|---:|---:|---:|
| every-frame | 2500 | 1.0× | 21/43 | 3.5 s |
| burst-end k=3 | 155 | 16.1× | 17/43 | 5.5 s |
| burst-end k=5 | 103 | 24.3× | 15/43 | 6.2 s |
| burst-end k=3 n=2 | 183 | 13.7× | 16/43 | 5.7 s |
| burst-end k=3 n=3 | 209 | 12.0× | 17/43 | 5.5 s |
| when-still k=3 s=1 | 269 | 9.3× | 23/43 | 5.1 s |

Pooled (note the take mix differs between rows because sparse policies shed the
1429-frame take hardest, so read these as directional only): prior per-square
95.2 % every-frame vs 93.7–94.3 % sparse; prior perfect 35.8 % vs 28.3–30.6 %.

## 5. Is the difference the policy, or just fewer boards?

The raw/prior board of a frame is identical whichever policy read it (§1), so a
policy's prior score differs from the control's *only* through which frames it
picked. Bootstrap: 2 000 random subsamples of the every-frame scored population at
the policy's own n, 95 % band vs the policy's observed prior-perfect rate.

| take | policies inside the random-subsample band |
|---|---|
| control | 5/5 |
| hard | 5/5 |
| fast | 5/5 |
| take 5 | 3/5 — `k=3 n=2` (17.0 % vs band [19.1, 44.7]) and `when-still` (13.0 % vs [22.1, 40.3]) fall **below** it |

So on the three flat-cadence takes, **a scheduling policy selects boards no better
and no worse than chance**. Two reasons, and both matter:

1. `trial.py` already excludes `guard_seqs=12` frames after every truth boundary —
   i.e. the mid-move window — from scoring. The chaos frames a burst-end policy
   exists to skip were **never in the scored population**, so skipping them cannot
   register as a score. The gain is GPU and display stability, and the score is
   correctly flat.
2. The policies *do* select much calmer frames (median stability of read frames:
   0.94–0.98 every-frame → 0.99 burst-end; fraction above 0.85 up from ~60 % to
   72–93 %). Calmness simply is not what these takes' errors are made of — they
   are persistent piece-type confusions (`q->k`, `p->b`), not motion blur.

On take 5 the selection is mildly *negative*, which is the honest warning in this
report: its arriving stream is 82 % in-burst, so there are few calm frames to
select and a sparse policy lands preferentially near motion.

## 6. The settled lock-in

The control take's every-frame settled stream is wrong on **253 of 253 scored
boards**, all of it `q->k` (d8) — the debounce adopted a board with the king/queen
swap on its first read and the alternating prior repair never assembled 2.5 s of
agreeing evidence to displace it. Its settled per-square (96.1 %) is *worse* than
its own prior (97.2 %): on this take the debounce actively subtracts.

Read sparsity **perturbs the lock-in without fixing it**:

- control: 0/7 segments settled → 2–3/7, and settled-perfect 0 % → 13.3 % (k=3, n=2).
  Fewer reads mean the adopted-first board has fewer chances to be re-confirmed.
- hard: 10/10 → 6–7/10, settled perfect 49.9 % → 24.6–29.8 %. Here density was
  *helping* (settled 97.2 % vs prior 94.9 %) and sparsity gives that back.
- fast: 7/14 → 2–6/14. take 5: 4/12 → 3–6/12.
- `when-still` settles the most segments overall (23/43, better than every-frame's
  21/43) but not the most *perfectly*.

`stabilize.py` was not touched. The data says its adopt-first-wholesale rule is the
defect, at every read rate, and that a scheduler cannot substitute for a tiebreak.

## 7. Settle latency

Median recorded seconds from a truth boundary (`from_seq`, taken just before the
piece is lifted, so the move's own duration is inside the number) to the first
settled event carrying that segment's placement:

- every-frame 3.5 s → burst-end 5.5–6.2 s. The +2 s is the honest price of sparsity
  and is dominated by the exit hold (0.7 s) plus the `spacing_s` run.
- `burst_end_reads` is what buys it back. `n=1` cannot settle from a burst end at
  all — one read is never a quorum — so it waits for the keepalive; `n=3` at 1.3 s
  spacing spans 2.6 s ≥ `SETTLE_SECONDS`, so a burst end can settle a board *on its
  own evidence*. Measured medians are nonetheless within ~1 s of each other, so
  this is a design argument the data does not contradict rather than one it proves.

## 8. Recommendation

**Ship `BurstTriggered(keepalive_s=3.0, burst_end_reads=3, spacing_s=1.3)`** in the
live ingest worker.

Why these constants:

- `keepalive_s=3.0` — two settle windows. `k=5` saves a further 30 % of reads and
  costs 0.7 s of settle latency plus two more never-settled segments; not worth it.
  This is the D3 floor and the only thing standing between a future meter
  regression and a node that reads nothing.
- `burst_end_reads=3, spacing_s=1.3` — the only setting where a burst end alone
  spans `SETTLE_SECONDS=2.5` and can therefore move the settled board without
  waiting on the keepalive. It costs 209 reads against `n=1`'s 155 (still 12×
  cheaper than every-frame) and has the best sparse prior per-square on 3 of 4 takes.
- burst hysteresis left at the forwarder's `0.7 / 0.85 / 0.7 s`. They describe the
  same bursts the edge sent; changing one without the other is how a policy starts
  reasoning about a stream nobody transmitted.

What it costs and buys, honestly:

- **buys**: 12× fewer estimates; ~1.3 GPU-s each, so 12× less GPU. The node stops
  needing to drop frames to survive, which removes a whole class of "why did the
  board not update" from the live path. Reads land on calm frames (median stability
  0.99 vs 0.94–0.98), so the *displayed* board stops flickering through hands.
- **costs**: pooled prior per-square −0.9 pp, prior perfect −5 pp, both inside the
  random-subsample band on 3 of 4 takes; settle latency +2 s; and on take 5 —
  the one recorded through the fixed forwarder — prior perfect 31.2 % → 20.0 %,
  which is at the low edge of chance and should be re-measured on the next take.
- **`when-still` is not the pick** despite the most settled segments: it reads 30 %
  more, has no notion of a transition (so it cannot spend reads where they are worth
  most), and is the worst row on take 5.

Two things the recommendation wants next, neither of them this change:

1. **Apply the policy to the ingest *inbox*, not the raw arrival stream.**
   `ingest.prefer_frame` already keeps the sharper of two contenders within a
   one-second window. A burst-end read that consults it picks the best frame of the
   moment instead of whichever frame happened to arrive — which is exactly the
   weakness take 5 exposes.
2. **`stabilize.py` needs an adopt-first tiebreak** (§6). Sparse reads do not fix
   the lock-in and were never going to; at 12× fewer reads the debounce's evidence
   is thinner, so the fix matters more, not less. Do not raise `SETTLE_MIN_READS`
   without re-running this sweep — at 0.5 reads/s a quorum of 3 is ~6 s of latency.

## 9. Reproducing

```bash
BR_MODEL=models/run_e/model.onnx BR_MODEL_B=models/run_f/model.onnx \
BR_SESSION_BANK=models/exemplars_session.npz BR_GPU_MEM_LIMIT_MB=1024 \
uv run python scratch/experiments/move_trigger/sweep.py     # 25 replays, ~75 min
uv run python scratch/experiments/move_trigger/analyse.py   # the table + table.json
uv run python scratch/experiments/move_trigger/check_restamp.py
```

Replays land in `recordings/<take>/replays/mt-*`; each is a recording-shaped
directory `eval-session` scores unchanged.
