# Framing drift in the 08-27 takes — measurement and mechanism

_Analysis-only. Scripts in this directory; recordings untouched._
Frames: `recordings/2026 08-27 * /frames.ndjson` + `jpeg/`. Detector:
`board_reader_core.boarddetect.detect_board_quad` (CPU, unmodified).

## 0. Two corrections to the premise, up front

**(a) The client does not stamp a fixed quad — it tracks it every frame.**
`packages/viewer/ui/src/lib/geometry/tracker.ts` is a pyramidal-LK plane tracker;
the corners on the wire are its output, refreshed per frame. So the failure is not
"a stale quad while the phone moved". It is **tracker drift**: the tracked quad
slides off a board that is standing still. That is why re-detection is the fix and
why it must land in the client.

**(b) The recorded corners are in CROP space, not camera space.**
`+page.svelte:sendFrame` cuts `cropBox(quad)` (margin 0.1, headroom 0.3) out of the
video and rebases the corners into it (`cornersInCrop`). The JPEG height therefore
varies frame to frame (708–796 px at a fixed 960 width). Consequence: displacement
of the stamped quad *from the session-initial stamped quad* is a poor drift proxy —
the crop follows the quad, so a real camera move largely cancels. Every number below
is instead **stamped quad vs. what the detector says the board is, in the same
frame**, normalised by that frame's square width.

## 1. Drift characterisation (Q1)

Per-corner displacement of the stamped quad from the session-initial quad
(`drift.py`, CSVs in `drift/`). Wire corner order is semantic a8,h8,h1,a1.

| take | recording | frames | a8 | h8 | h1 | **a1** | worst px | worst /diag | biggest 1-frame step |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 20260827-163352 | 337 | 27.1 | 13.0 | 11.7 | 43.7 | 43.7 | 0.035 | 7.3 px @109 |
| 2 | 20260827-170333 | 755 | 50.3 | 38.6 | 20.9 | 96.2 | 96.2 | 0.078 | 13.0 px @670 |
| 3 | 20260827-172736 | 1429 | 73.2 | 24.8 | 41.7 | 20.5 | 73.2 | 0.059 | 12.0 px @191 |
| 4 | 20260827-175539 | 433 | 22.1 | 11.9 | 17.3 | 23.7 | **23.7** | 0.019 | 6.0 px @37 |
| 5 | 20260827-181802 | 301 | 43.3 | 32.1 | 27.2 | **82.5** | 82.5 | 0.067 | **77.9 px @177** |

Take 5's a1 corner reaches 82.5 px ≈ one square — the reported ~79 px reproduces.
Take 5's 77.9 px single-frame step at seq 177 is **a re-anchoring, not a bump**: §2
shows the quad *agrees with the detector* immediately afterwards. It is the operator
manually re-detecting mid-session — the natural experiment for this whole report.

> **Data note.** take 4's `frames.ndjson` holds a **48-frame false start** — the page
> streamed, dropped and reconnected, restarting `seq` at 0 (`restarts_at_idx=[48]`;
> the only take affected). Measuring drift from that first stream's frame 0 invents
> an 84 px "step" at the reconnect. All take-4 numbers here are from the second
> stream only (`_last_stream()` in `drift.py`). Q2–Q4 were never affected: they
> compare each frame's stamped quad against the detector on **that same frame**.

**Take 4 is the sharpest control.** With the false start removed it has the *least*
framing drift of all five takes (23.7 px = 0.19 sq, never once crossing
`QUAD_VERIFY_SHIFT`) — and yet the *worst* accuracy (95.6 %, 6.06 wrong/read). Its
errors are not framing (dim light / steep angle), which is exactly the session on
which a drift mechanism must stay silent. It does (§4).

**Take 5 shape: a creep, not a step.** a1 climbs monotonically from ~seq 40 to a
peak at seq ~160-176, is reset by the operator at seq 177, then **starts creeping
again** from seq ~275 (55 px by seq 300). Two episodes in 141 s.

### Onset vs. the errors

Scored reads joined to the drift at their frame (`errors_vs_drift.py`):

| stamped drift | reads | perfect | wrong squares | wrong/read |
|---|---:|---:|---:|---:|
| 0–20 px | 13 | 9 | 4 | **0.31** |
| 20–40 px | 6 | 1 | 13 | 2.17 |
| 40–60 px | 5 | 2 | 3 | 0.60 |
| >60 px | 11 | **0** | 57 | **5.18** |

Errors track the creep and vanish the instant the operator re-detects: read at seq
176 = 5 wrong, read at seq 179 = **0 wrong**. 58 of take 5's 77 wrong squares fall
in seqs 137–176.

## 2. Detectability (Q2)

`detect_run.py` ran `detect_board_quad` on 239 resting frames (inside a truth
segment, past the guard, ≥3 frames before the next boundary), 6-frame stride.

**The detector is cheap and reliable on these crops.** 230/239 hits (96 %); serial
cost median 75–182 ms, worst 1.7 s on a miss (`cost_cadence.py`). Repeatability on
identical frames over 5 runs: median spread **0.00 px, max 0.72 px**
(`repeat.py`) — effectively deterministic here, despite the nondeterminism the
`boarddetect` docstring warns about on broadcast stills.

Detector vs. stamped quad, max same-corner distance after best relabelling:

| take | hits/n | median | p90 | max | median ms |
|---|---|---:|---:|---:|---:|
| 1 | 38/38 | 24.8 | 30.6 | 32.0 | 268 |
| 2 | 60/60 | 25.6 | 28.7 | 30.1 | 222 |
| 3 | 58/60 | 32.7 | 38.5 | 42.4 | 226 |
| 4 | 38/41 | 16.7 | 22.3 | 22.4 | 726 |
| 5 | 36/40 | **13.0** | 39.8 | **59.0** | 695 |

Two things this exposes:

* **On take 5's drifted stretch the detector tracks the TRUE board.** Its quad sits
  4–6 px from the stamped quad on the clean stretches and **39–59 px away exactly
  where the drift is** (seqs 124–172), i.e. it disagrees by ≈ the drift. See
  `t5_160_quads.png`: red (detected) is on the playing surface, green (stamped) has
  slid off it.
* **There is a per-session CONSTANT offset that a naive threshold would trip on.**
  Takes 1–4 were framed by finger-drag, so the operator's corner line differs from
  the detector's playing-surface line by a fixed 17–33 px. Take 5's framing came
  from `/detect`, hence its 4–6 px agreement. **An absolute threshold on
  detector-vs-stamped distance false-alarms on every hand-framed session.** The
  signal must be baselined per session.

### False-alarm floor

The right unit is **square widths**, not fraction of frame diagonal — a crop lands
on the wrong square when the grid slips by a square, not by a diagonal.
`misaddress.py` projects every square centre of the stamped grid onto the detector's
grid:

| take | worst square-centre offset | frames with a mis-addressed square |
|---|---:|---:|
| 1 | 0.44 sq | 0 / 38 |
| 2 | 0.48 sq | 0 / 60 |
| 3 | 0.56 sq | 1 / 58 (1 square, last sample) |
| 4 | 0.31 sq | 0 / 38 |
| 5 | **0.92 sq** | **7 / 36 (up to 6 squares)** |

**All four controls stay under 0.56 square and mis-address essentially nothing.**
Take 5 crosses 0.6 at seq 124 and reaches 0.92 sq / 6 mis-addressed squares at
seq 160–172 — which matches the 5–8 wrong squares its reads produced there.

Note the controls are **not** drift-free either: every take shows a monotone
staircase of tracker drift (take 1 → 0.33 sq, take 2 → 0.40, take 3 → 0.46,
take 4 → 0.21 on the baselined signal). Take 5 is the same failure, further along.
Take 3 was still climbing at its last frame. **This will recur on every session.**

`FRAC_OF_DIAGONAL` reference: `QUAD_VERIFY_SHIFT = 0.03` ≈ 37 px ≈ **0.47 square**
here — right in the middle of the control distribution. Too tight to reuse as an
adoption threshold.

## 3. Correctability (Q3)

Re-cut take 5's drifted frames with the detector's quad
(`correct.py`, `board_reader_inference.squares.crop_squares_photo`, the production
geometry):

* `t5_160_crops_stamped.png` — pieces straddle cell boundaries; several squares
  labelled empty contain a piece and vice versa. Rank 1 and rank 2 are visibly
  shifted by most of a square.
* `t5_160_crops_detected.png` — every piece is centred in its own cell and matches
  its truth label.

Quantitatively: the detected quad drops mis-addressed squares from **6 → 0** at
seq 160/166/172, from 1 → 0 at seq 124/136/142, and 2 → 0 at seq 298.

**The strongest evidence is not synthetic.** At seq 177 the operator did exactly
this by hand: the stamped quad jumped 77.9 px in one frame and landed 4.9 px from
the detector's. The next scored read (seq 179) went from 5 wrong squares to 0.
Re-detection demonstrably corrects the framing on this recording.

Corner relabelling is safe: best-vs-second-best correspondence margin is **9.7×
minimum** (best ≤ 59 px, second-best ≥ 533 px) across all 230 detections. Adopting
"detected quad, relabelled onto the tracked quad by nearest corner" never risks the
a8 ambiguity that DESIGN decision #15 warns about, as long as drift stays well under
half a board.

## 4. Recommendation (Q4)

> **Client-side periodic re-detect on resting frames, with baselined adoption:
> re-anchor the tracker to the detector's quad when the implied square-grid shift
> exceeds 0.45 square on two consecutive successful detections.**

### Why client-side, decisively

The crop the node receives is cut from `cropBox(tracked quad)`. A correction applied
only in the node is overwritten by the next frame's crop — and if the tracker keeps
drifting the crop eventually walks the board out of the sent image entirely. **Only
the client can fix framing at its root.** The wire is unchanged, `/detect` already
exists, and the auto-detect state machine (`shouldAutoDetect`, `AUTO_BACKOFF_MS`,
the activity lock) is already there — this is a gate relaxation plus an adopt rule,
not new machinery. Continuous corner tracking is rejected because that is precisely
what is already running and drifting.

### Constants

| constant | value | measured from |
|---|---|---|
| `REDETECT_INTERVAL_MS` | `10000` | catching take 5 at seq 136 needs a detection every ≤12 frames ≈ 6 s at 2.1 fps; every 24 frames only catches it at seq 160. 10 s gives margin. |
| gate: `stability >= AUTO_STABILITY` (0.9) | reuse | take 5's fixed meter reads ≥0.9 on 69 % of frames; ≥0.7 on 79 %. Resting frames are identifiable. |
| `REDETECT_BASELINE_N` | `3` | R₀ = mean(detected − stamped) over the first 3 successful detections after framing is set — cancels the 17–33 px operator-vs-detector convention offset. |
| `ADOPT_SHIFT_SQUARES` | **`0.45`** | max control drift 0.46 (take 3, one sample); take 5 reaches 0.51 at seq 124 and 0.75 at 160. |
| `ADOPT_CONSECUTIVE` (K) | **`2`** | K=1 at θ=0.45 fires once on take 3; K=2 gives zero control adoptions. A detector MISS neither arms nor disarms the counter. |
| `ADOPT_MAX_SHIFT_SQUARES` | `2.0` | sanity cap: beyond this it is a reposition (DESIGN §4.4.1(b)), not drift — hand it to the operator, never self-adopt. |
| `ADOPT_RELABEL_MARGIN` | `4.0` | require second-best corner correspondence ≥ 4× the best. Measured minimum is 9.7×, so this never blocks a real correction but refuses an ambiguous one. |

Adopt by setting the tracked quad to **`detected − R₀`** (preserves the operator's
declared corner convention *and* the a8 labelling) and re-seeding the tracker there.

### What it would have done

Sweep over all five takes (`mechanism.py`), θ = 0.45 sq, K = 2:

| take | adoptions | first at |
|---|---:|---|
| 1 | **0** | — |
| 2 | **0** | — |
| 3 | **0** | — |
| 4 | **0** | — |
| 5 | 3 | **seq 136** |

Zero adoptions on all four controls, at every cadence from 6 to 24 frames. Take 5
is caught at seq 136 — before the worst stretch and before the operator noticed.

Projected take 5 score (`projection.py`; the counterfactual uses take 5's **own**
error rate on its drift-free reads, 0.67 wrong/read and 60 % perfect, so no
cross-take assumption):

| | actual | projected |
|---|---:|---:|
| scored reads | 35 | 35 |
| wrong squares | 77 | **32** |
| accuracy | 0.9656 | **0.9857** |
| perfect reads | 12 | **18** |
| perfect rate | 0.343 | **0.520** |

45 of take 5's 77 wrong squares — 58 % — are framing drift, recoverable with a
detection that costs ~80 ms of CPU every 10 s.

### Interaction with auto-calibrate

Adoption changes the crop geometry, so any calibration bank keyed to the old quad is
now keyed to a *better* one — this only helps. But the two must not run at once:
gate adoption behind the same `activity.kind === 'idle'` lock `shouldAutoDetect`
already uses, and treat an adoption as re-origining the framing to `'detected'`
(exactly what the restored-quad verify already does).

## 5. Caveats

* The projection in §4 is arithmetic on take 5's own drift-free rate, **not a
  replay**. Re-running `board-reader-inference replay` with corrected corners would
  confirm it; that needs the GPU, which another agent owns.
* Resting frames were sampled at a 6-frame stride (239 of ~3300 frames). Behaviour
  between samples is interpolated, and the exact adoption seq shifts with cadence
  (136 → 160 as the interval grows).
* θ = 0.45 sq separates these five takes with a thin margin against take 3 (0.46
  max, one sample, and still climbing at session end). Take 3 was genuinely starting
  to mis-address a square there, so firing on a longer take 3 would be a true
  positive — but the constant should be revisited once more sessions exist.
* `stability` on takes 1–4 comes from the **old** meter (never exceeds 0.7); only
  take 5 has the adaptive one. The "resting frame" gate is validated on take 5 only.
* Detector misses (4 % overall, 10 % on take 5) cluster on the drifted frames
  themselves — hence K counts successful detections and a miss is neutral, not a
  reset.

## Files

`drift.py` · `errors_vs_drift.py` · `detect_run.py` · `analyse_detect.py` ·
`residual.py` · `repeat.py` · `squares.py` · `misaddress.py` · `correct.py` ·
`mechanism.py` · `cost_cadence.py` · `projection.py` · `draw.py`
Data: `drift/*.csv`, `detect_results.json`, `residual.json`, `misaddress.json`,
`mechanism.json`. Images: `t5_{124,160,172}_{quads,crops_stamped,crops_detected}.png`.
