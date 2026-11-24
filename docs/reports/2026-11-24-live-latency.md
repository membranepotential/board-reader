# Move-to-settle latency on the live phone: a blind motion meter and a 2 s keepalive (2026-11-24)

> Live take `recordings/20261124-200631` (iPhone `/camera` as a dumb camera, ChessQ fp32 on
> the GPU, ~0.2 s per read). Scratch scripts: `scratch/experiments/settle-sweep/` (cached
> reads, gitignored), plus the live take's cache `cache/20261124-200631.npz`.

## Outcome

- **The phone's motion meter was blind.** After the 4 s startup, every frame of the take
  carried `stability` 0.9999999999999999, so the forwarder opened no burst for any of the
  four moves. The node read only on its 3 s keepalive, and each move needed two agreeing
  keepalive reads to settle: 3.1–4.2 s from the first hand-free frame (median 3.1 s).
- **Cause: the browser thumbnail was point-sampled, not area-averaged.** Fixed in the
  `/camera` page (`MOTION_OVERSAMPLE`); the edge's Python meter was already correct.
- **Safety net: the keepalive is now 2 s** (`DEFAULT_POLICY_SPEC`
  `burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.6`). Gate median move-to-settle
  3.0 → 2.0 s, live take 3.1 → 2.1 s, no hand settles. The settle window stays 1.0 s.

## Root cause

The page drew the ~950×550 crop straight onto a 32×32 canvas with `drawImage`. A canvas
downscale samples a few source pixels per output pixel, so the thumbnail kept the camera's
per-pixel noise. The meter learns its floor from the resting difference and drops only
when a frame exceeds ~2.4× that floor.

Evidence:

- **The startup ramp measures the phone's resting difference.** The meter restarts at
  streaming-on with its floor at the 0.0015 noise bound and climbs it slowly; fitting that
  recursion to the 40 recorded burst frames (stability 0.0001 → 1.0 over 4 s) gives a
  constant resting difference of **~0.020** per 50 ms meter tick.
- **Point sampling reproduces that number; area averaging does not.** On the recorded
  frames, consecutive-frame mean difference of a 32×32 gray thumbnail:

  | thumbnail | rest, full frame (100 ms) | rest, crop (1 s) | during moves, crop (1 s, median) | move / rest |
  |---|---|---|---|---|
  | `INTER_NEAREST` (one sample per pixel) | 0.0146 | 0.0201 | 0.0282 | 1.4 |
  | `INTER_LINEAR` | 0.0108 | 0.0159 | 0.0252 | 1.6 |
  | `INTER_AREA` (the edge's meter) | 0.0020 | 0.0025 | 0.0142 | 5.7 |
  | 8× oversample, then 8×8 block mean (the fix) | — | 0.0029 | 0.0143 | 5.0 |

  A hand moved the point-sampled thumbnail by about 1.4× its rest even a second apart, and
  less between 50 ms ticks, so it never crossed the tolerance band. The recorded JPEGs are
  already smoothed by compression; the raw camera frames the phone meters are noisier still.
- **The crop switches were the only bursts.** `b1` (6.4 s) and `b2` (72 s) coincide with
  the control UI setting and clearing the crop: one frame differenced across two different
  pictures (stability exactly 0.3 = 1 − attack), then rest again.
- **Exposure hunting is not the cause.** Sharpness and JPEG size are flat across the
  startup ramp; the ramp is the floor's slow, gated climb from 0.0015 to ~0.02.

## The fix (commit "fix(camera): area-average the motion thumbnail…")

- The page draws the crop at 256×256 (`MOTION_SIDE × MOTION_OVERSAMPLE`, 8×) and
  `sampleThumbnail` averages 8×8 blocks into the 32×32 thumbnail, which mirrors the edge's
  `cv2.INTER_AREA`. On the take's frames that rests within 20 % of `INTER_AREA`.
- `MotionMeter.rebase()`: a new crop or a reopened camera starts a new reference but keeps
  the learned floor, so a crop change no longer opens a burst.
- Tests: a seeded noisy-camera scene in `motion.test.ts` (exposure ramp, rest, a hand
  crossing, σ chosen so a point-sampled thumbnail rests at ~0.019 and an area one at
  ~0.0024): point-sampled thumbnails never drop below `ENTER_BURST`, area ones do and
  still read rest as rest. `camera.spec.ts` checks in Chromium that a crop change alone
  sends no burst frame and a drawn hand does.
- **Not yet verified on the phone.** The gate replays recorded `stability`, so the burst
  path on a real iPhone is validated only by the next live take.

## The keepalive and the settle window

Cached reads (every frame read once with fp32, then the real `replay.replay` loop and a
re-settle per window; byte-identical to the real `cq-v6-tuned` replay at today's
defaults). Schedule `burst-end:burst_end_reads=3,spacing_s=0.6` with the keepalive varied.

Gate (four takes; `make gate GATE_NAME=ka2-tuned` reproduced the 2 s row exactly):

| keepalive | settle | settled perfect | boundaries reached | median (p75) latency | leaks | reads |
|---|---|---|---|---|---|---|
| 3 s (was) | 1.0 / 0.75 / 0.5 s | 104/119 (.874) | 21/51 | 3.00 (3.78) s | 0 | 166 |
| **2 s** | 1.0 / 0.75 / 0.5 s | **138/154 (.896)** | 26/51 | **2.04 (2.76) s** | **0** | 214 |
| 1.5 s | 1.0 / 0.75 / 0.5 s | 161/183 (.880) | 26/51 | 2.04 (2.34) s | 0 | 246 |
| 1 s | 1.0 / 0.75 / 0.5 s | 242/264 (.917) | 34/51 | 1.07 (1.98) s | **4** | 362 |

Held-out (20260827-163352/170333/172736, restamped; their truth is known to be incomplete,
so their "leaks" are not hands — see the settle-tuning report):

| keepalive | settle 1.0 s | settle 0.5 s | median latency (1.0 / 0.5 s) |
|---|---|---|---|
| 3 s | 73/160 (.456) | 84/160 (.525) | 4.20 / 3.60 s |
| 2 s | 107/202 (.530) | 112/202 (.554) | 3.40 / 3.18 s |
| 1.5 s | 139/230 (.604) | 142/230 (.617) | 2.88 / 2.82 s |
| 1 s | 186/298 (.624) | 187/298 (.628) | 2.40 / 2.40 s |

Live take (no truth; latency from the first hand-free frame, picked by eye, to the first
event whose settled board shows the move; e4, e5, Nf3, Nc6):

| keepalive | latencies | median |
|---|---|---|
| 3 s (reproduces the live run) | 3.12, 4.17, 3.12, 3.12 s | 3.12 s |
| **2 s** | 2.08, 2.08, 2.08, 3.12 s | **2.08 s** |
| 1.5 s | 2.08, 2.08, 2.08, 3.12 s | 2.08 s |
| 1 s | 0.00, 0.00, 1.04, 1.04 s | 0.52 s |

The settle window made no difference on the gate or the live take at any keepalive.

**Settled-wrong boards, inspected:**

- **1 s lets hands settle** (4 gate leaks): on 175539 a b8 knight read empty under the
  hand on two reads 1 s apart; on 181802 a c8 bishop under the hand. Rejected.
- **2 s:** no leak. 08-13 settles 44/45; the one wrong board (seq 239, e5 already on the
  board, e7 empty) is a late truth boundary — frames 234–239 show e5 played, the truth
  starts that segment at 240. The live take settles one board that is no position at seq
  99: b8 empty, c6 not yet adopted, during Nc6 — the "move adopted in part" pattern the
  settle-tuning report accepts (both squares hold values of the previous or next position).
- **1.5 s** shows the same 08-13 pattern at seq 186/236/239/281/642 (186 and 236–239
  checked by eye: the move is already on the board).

**Choice: 2 s keepalive, settle window unchanged at 1.0 s.** 2 s gives the same median
as 1.5 s on the gate and the live take at 29 % instead of 48 % more reads, and keeps more
distance from 1 s, where hands settle. The keepalive matters most when the meter is blind,
which is exactly when reads also land on hands. A 0.5 s window gained only on the
held-out takes, whose truth is suspect, and the settle-tuning report kept 1.0 s on purpose
(one hand-crossing long, for denser streams), so it stays.

## Caveats

- Four moves on the live take; latency there is quantised by its 1 fps keepalive frames.
- With the meter fixed, burst ends should drive latency (three reads 0.6 s apart after
  0.7 s of calm, so ~2 s from the hand leaving to settled); the keepalive is the floor
  under that. Only a new live take measures it.
