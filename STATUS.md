# STATUS — board-reader

_reconciled: 2026-08-07 @ `5a6a19a`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md); this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- **Broadcast capture tooling absorbed + the streams audited (main session, 2026-08-07 04:00–).**
  `../otb_video_recognition` is pulled in and deleted: scripts → `tools/video/`, videos/frames →
  `data/streams/` (git-ignored), skill → `.claude/skills/get-frames-from-yt/`. Legacy Hough/scipy
  detector + notebook dropped (superseded by `boarddetect`). **`quad_aspect` + `LOW_ANGLE_ASPECT`
  moved into `board_reader_core.boarddetect`** (was private in `inference.server`, which now
  imports them) so the capture-side gate and the `very_low_angle` warning cannot drift.
  **The angle gate rejected 4 of 5 broadcasts** — medians 0.445 / 0.450 / 0.508 / 0.573 vs the
  0.62 floor; only `JTmovXvZnPM` survives at **0.65, i.e. "degraded", never the ≥0.73 good band**.
  Their videos are deleted; ids + crop coords + frames are kept, so `yt-dlp` reproduces any of
  them. One lost crop (`62h7O9Ns_eA`) was recovered by template-matching a frame back into the
  source (corr 1.0). Also measured: `detect_board_quad` is *not* usable to find the crop on full
  broadcast frames (3/6 failures, one latched onto the 2D overlay board) — crop stays manual,
  detection is the QC step. 3 parallel implementers in flight, disjoint files: **video-tools**
  `tools/video/**` (registry rekeyed on video id, `check_angle.py`, tests) · **skill-rewrite**
  `.claude/skills/get-frames-from-yt/SKILL.md` · **stream-docs** `docs/youtube-streams.md`,
  `tools/README.md`, `aspects.yaml`.
- **New broadcast candidate `2sRJ4JuL_m8` (main session, 2026-08-07 04:2x–).** Searched YouTube for
  a *higher-angle* stream than the 5 already audited: ~40 candidates screened by `yt-dlp ytsearch`
  + a sampled frame each + `detect_board_quad`/`quad_aspect`. Winner: **Chess League | CCBF |
  Round 5 Board 1** (Chess Club Bobby Fischer, 3:04:36) — aspect **0.633–0.639 across 5 frames
  spanning the full 3 h**, fixed camera that never moves, one board, big in frame. Still only the
  *degraded* band, but the steadiest source found and better than every rejected stream.
  **The honest finding: nothing on YouTube is both a long real OTB game and ≥0.73** — broadcast
  cameras frame players, not boards; the ≥0.73 footage that exists is short CV-demo/overhead-rig
  clips (`kg_whG0Ijh4` 0.790, `B_3v5sqqU0Y` 0.743). **Landed:** 1080p60 on disk (3.9 GB; board
  ~1150 px in-crop, ~144 px/square — 720p would have given ~430 px, under the skill's own marginal
  line), crop row `ccbf-r5` (403,228,1280×764 @1920×1080), gate **passed at median 0.635**
  (11/16), 7 operator-picked frames spread over the 3 h in `data/streams/frames/ccbf-r5_2sRJ4JuL_m8/`
  auditing at **0.637 (6/7)**. Crop keeps headroom over the far rank, so tall pieces aren't
  decapitated by the crop itself.
  **New measurement — the detector's misses here are false negatives, not bad frames.** All 6
  no-detection frames were pulled and inspected: every one is a clean, fully-framed board (incl.
  a textbook opening position), zero camera cuts. So `detect_board_quad` has a ~30% false-negative
  rate on this crop while the *source* is ~100% usable — the earlier "69% detection" number
  understates stream quality and overstates detector reliability. Consequence for M2: the
  `Rectifier`'s cadenced re-detection will miss roughly 1 in 3 attempts on real footage, so
  `feed --corners` is the safe path. Suspected cause (unverified): the board's visible barrel
  distortion breaks the straight-edge quad model.
  Next: M2 over this video (`feed --video --adaptive --corners`), still not started.
- **Live-viewer upgrade — DONE, uncommitted, browser-verified.** Gateway retains the latest
  JPEG per board → `GET /boards/{id}/frame` (X-Seq/X-Capture-Ts, no-store, 404 pre-ingest);
  the live page renders via **chessground** (as /snapshot), low-confidence squares ringed
  yellow / just-changed ringed green, an **expandable preview** docked left of the board
  showing the live ingested frame, and a **calibrate** button (disabled while stale/moving/
  not-ok) that posts the live frame through the guarded `/calibrate`. Playwright-verified
  end-to-end against a live game feed: preview showed the hand mid-burst, calibrate returned
  "13/13 classes, 256 exemplars" and the board then tracked the game with real estimates.
  **565 tests green** (`make test`; GPU pair quarantined to `make test-gpu` by the other
  session). Awaiting operator review + commit word.
- **M2 — live position stream over a recorded game: DONE, committed** (`c601dea`…`b565d68`, 6
  logical commits). Built 2026-08-07 by
  4 parallel opus implementers + main session finishing the two operator-stopped workstreams
  inline. Shipped: edge `MotionMeter`+`AdaptiveForwarder` (WP-C2/C3, keepalive ~1 fps / burst
  12 fps, hysteresis, `burst_id`) · cadenced `Rectifier` re-detection (~1 s capture-time; dropped
  frames never pay the ~100 ms detection) · capture-time throttle · `GameSource`/`LoopSource` +
  synthetic recorded game with occluding hand (`core.sim`) · gateway `/ingest` per-board
  latest-wins async workers (ack ≠ estimate) · viewer liveness (age/stale, ev/s, moving chip,
  change flash, dimmed occluded squares) · `feed --game --adaptive --loop --throttle` = the
  compose demo. **477/477 tests green (serial + parallel)** incl. new e2e `tests/test_m2_stream.py`;
  CLI smoke: 540-frame game → 208 adaptively forwarded through a real gateway. Docs updated
  (TESTING.md Option A/C, architecture.md, index.md).
- **run_f — the pose-aware-crop bet** — **NOT RUNNING; stopped by the operator at e0 (03:24),
  ready to relaunch.** The harness is fixed and verified (below); nothing blocks it. Relaunch
  with `tools/train/run_f.sh` — no `PY=`, the script now resolves and preflights its own
  interpreter. 20 epochs, `patience: 8`, ~40 min, crop cache warm. The partial dir
  (`scratch/runs/run_f/`, one epoch, val 0.666) is disposable; the driver overwrites it.
  **The early evidence is good — the first attempt (02:42) proved the bet by accident:**
  `patience: 5` truncated it at e8
  with a mere **0.8466 @ e3** checkpoint, and that badly undertrained model still scored
  **0.855 on `chesscog_transfer` vs run_e's 0.819** — with piece-type errors cut 173 → **115**,
  exactly the steep-angle decapitation pose crops were meant to fix. It *loses* on the user's
  own sets so far (user_photos 0.894 vs 0.938, uploads_calibrate 0.804 vs 0.864), which may
  just be the undertraining. Both numbers come from a same-code, same-data re-eval of run_e
  (`scratch/runs/run_e/eval_rerun.txt`); run_e's on-disk `eval.txt` is from 19 Jul and predates
  the `chesscog_transfer` repair, so it is NOT comparable. Evidence kept in
  `scratch/runs/run_f.e3-truncated/`. Next: relaunch, then compare fully-trained run_f
  against **0.898** holdout / **0.819** transfer on `===F_DONE===`.
- **Training harness: env preflight + a screening tier** (main session, 03:05–03:20) — **done.**
  Root cause of the above: the run was launched with `PY=/usr/bin/python` (CUDA
  torch, *no* ONNX stack) instead of the documented `scratch/torchvenv`, so training completed
  and then `export.py` (needs `onnxscript`: torch ≥2.9 exports via dynamo) and `evaluate.py`
  (needs `onnxruntime`) both died — while `run_f.sh`, lacking `set -e`, printed a blank
  `CANDIDATE_RESULT` and `F_DONE` over the wreckage. Third instance of "the pipeline reports
  success over nothing". Fixed: `onnxscript`+`matplotlib`+`pytest` installed into the venv via
  `uv` (seasoned to the repo's pinned date); new `_env.sh` defaults `PY` to the venv and
  **preflights** it before any GPU time; `set -e` in `run_f.sh`; `patience: 8`.
  New `screen.sh` + `val_max` give a ~13-min **ranking** tier (8 epochs, val capped 4000,
  stratified) — measured: training data volume is *not* a time lever (`steps_per_epoch` already
  caps each epoch at 2.2% of the corpus), val and epochs are. 406 workspace tests green + 7 new
  harness tests (`tools/train/tests/`, run under the torch venv — the workspace stays torch-free).

## Done (recent)

- 08-07: **Training harness observable + fail-fast** (`c3fecbb`, `34c88ad`); run dirs self-document,
  and the GTX 1650 measures compute-bound at ~165 img/s fp32. **`make test` 35s → ~10s** (`5a6a19a`).
- 08-07: **Hard-constraint prior decoder ships and is wired** (`bbddded`). Flips the lowest-margin
  offending square on *impossible* boards: **5/23 → 10/23** perfect trained-alone, 13/23 → 14/23
  ensemble, **3/9 → 6/9** on uncalibrated broadcast frames — where the bank does nothing. A decoder
  over `square_dist`, never an estimator: the raw distribution stays on the wire (D1).
- 08-07: **`chesscog_transfer` repaired, not replaced** (`f362f78`). Never geometrically broken: 14 of
  22 labels were rotated 180° by eye, now resolved from pixels. `run_e` **0.500 → 0.819**, and it is
  the only real-photo set no bank and no run has ever seen.
- 08-07: **Pose-aware crop geometry** (`f362f78`). A `CropGeometry` ADT rides the `.onnx` sidecar and
  keys the crop cache, so a train/serve mismatch is unrepresentable; the focal length is *solved* from
  the homography. Near/far box height **1.57 → 1.15**, and the boxes are `rot90`-invariant.
- 08-07: **Temperature scaling measured** (`f362f78`): LOBO ECE **0.255 → 0.004**, AUC 0.862 → 0.877, **0 answers changed**. Not served — see Next 3.
- 08-07: **The honest eval, 23 confirmed labels.** Ensemble **13/23 perfect · 1450/1472** vs 5/23
  alone — *the bank is the entire margin, and only where calibrated*: 2/13 → 9/13 on own photos,
  **zero boards** on nine broadcast frames ([docs](docs/measuring-accuracy.md)).
- 07→08: **M0 skeleton · M1 single-frame · snapshot app · `run_e` deployed** — photo in, position out, zero taps; the labelling loop; kNN bank ⊕ trained model.

## Next

1. **Score run_f against run_e** on both holdouts; deploy only if it wins both. If pose crops lose, the geometry bet is answered and item 6 is what survives it.
2. **Attack the type confusions.** Trained-alone: 53 type / 5 hallucination / 4 colour, led by
   `p->b` (11). The ensemble's residual differs in kind — `empty->p`/`empty->P` is a third of it.
3. **Decide whether calibrated confidences go on the wire** — a decision, not an experiment: argmax is invariant, and DESIGN §9 says the downstream smoother wants them.
4. **Run M2 over real footage.** The synthetic demo works; the milestone's spirit wants a real
   recorded game (operator is curating clips → `docs/youtube-streams.md`): `feed --video <clip>
   --adaptive --corners ...`, judge the stream, then decide if `stability` earns its keep (D3).
5. **Central-column pooling A/B.** `dinov2_small_central.onnx` is exported and verified; build
   `exemplars_chessred_central.npz` and score it under the uncontaminated cross-style protocol the
   +9/320 came from — no deployed bank moves before it wins.
6. **Crop attribution / occlusion** — pose fixed coverage and resolution, not attribution: the box is
   still the prism's *bounding* box, so a piece behind can fall inside it unflagged.

## Open questions

- **We have no real footage inside the good angle band.** All five absorbed broadcasts measure
  ≤0.65 projected aspect and four are below the 0.62 floor; table-level broadcast cameras are
  simply outside the envelope. M2's recorded-game demo therefore rests on one *degraded* video.
  A steeper/overhead source (or own footage) is now a sourcing requirement, not a nice-to-have.
- **Bank writes are not monotone.** A board that read 64/64 lost a square after two `/calibrate`
  writes, and trained-alone still gets it right. Nothing measures that before a write lands.
- **Confidence is not usable today.** Trained-alone has no board-level signal (0.978 vs 0.939) and the ensemble is under-confident, though it ranks boards roughly right. Top risk in DESIGN §9.
- **The 638/640 baseline is unreproducible** — 5 of 10 curated boards point at deleted photos, and
  the same erosion makes `run_e`'s bright-board regression unjudgeable. Recover them or prune.
- Does `stability` earn its keep (D3)? Undecided until M2 produces a stream to judge.
- Licensing: ChessReD (CC BY-NC-SA) + murky geometry → personal use only.
- No lint gate: `ruff` has no project config, ~109 findings unaddressed. Adopt it or drop it.
