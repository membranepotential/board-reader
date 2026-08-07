# STATUS — board-reader

_reconciled: 2026-08-07 @ `b7932da`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md); this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- **Phone camera page — the browser edge (main session, 2026-08-07).** New viewer-served
  `/camera` page (DESIGN §6 "virtual Pi"): getUserMedia preview → operator crop (4 draggable
  corners, `/detect` assist, rotatable labels) → adaptive JPEG streaming into WS `/ingest`
  (JS mirror of `MotionMeter`+`AdaptiveForwarder`) + **calibrate** button through the guarded
  `/calibrate`. Wire grows optional per-frame `corners`; ingest gains the photo-path policy;
  viewer becomes single-origin TLS terminator (`--ssl`, self-signed) + WS/HTTP proxy so phone
  browsers get a secure context. **2 opus implementers in flight, disjoint files:**
  wire-ingest `packages/core/**`+`packages/inference/**` · camera-page `packages/viewer/**`
  (+uv.lock). Docs + STATUS reconcile + commit after both land.

Also staged and waiting on a decision to start:

- **M2 over real footage — staged, not started.** The recorded-game demo works synthetically;
  the milestone's spirit wants a real game. The source is now picked: `ccbf-r5`
  (`2sRJ4JuL_m8`, 3 h, fixed camera, 1080p60 on disk). Run
  `feed --video --adaptive --corners`, judge the stream, then decide whether `stability`
  earns its keep (D3). **Use `--corners`, not auto-detect** — see the false-negative
  finding below.
- **`run_f` (the pose-aware-crop bet) — ready to relaunch,** stopped by the operator at e0.
  Harness fixed and verified; `tools/train/run_f.sh` with no `PY=` (it resolves and
  preflights its own interpreter). 20 epochs, `patience: 8`, ~40 min, crop cache warm.
  `scratch/runs/run_f/` holds a disposable one-epoch partial the driver overwrites.
  **Early evidence favours the bet:** a truncated e3 checkpoint (0.8466) still beat run_e on
  `chesscog_transfer` — **0.855 vs 0.819**, piece-type errors 173 → **115**, exactly the
  steep-angle decapitation pose crops target. It loses on the user's own sets so far
  (user_photos 0.894 vs 0.938), possibly just undertraining. Compare against **0.898**
  holdout / **0.819** transfer on `===F_DONE===`; deploy only if it wins both.

> `tools/video/crops.csv` has an uncommitted row (`ccbf-r5`). The docs describe it; commit
> the registry with them or the reproducibility claim is false.

## Done (recent)

- 08-07: **Live viewer is genuinely live** (`cb7877d`, `b7932da`). Gateway retains the latest
  JPEG per board → `GET /boards/{id}/frame`; the page renders via **chessground** (low-confidence
  squares ringed yellow, just-changed green), an expandable preview shows the live ingested
  frame, and **calibrate-from-stream** posts it through the guarded `/calibrate`.
  Playwright-verified end to end. **565 tests green.**
- 08-07: **Broadcast capture tooling absorbed + streams audited** (`3458578`…`4dba64a`).
  `../otb_video_recognition` folded into `tools/video/` + `data/streams/` + the
  `get-frames-from-yt` skill; legacy Hough detector dropped. `quad_aspect`/`LOW_ANGLE_ASPECT`
  moved into `board_reader_core.boarddetect` so the capture gate and the serving warning
  cannot drift. **The gate rejected 4 of 6 broadcasts;** both survivors are only *degraded*.
  Culled videos are deleted but reproducible from `crops.csv` alone.
- 08-07: **`ccbf-r5` sourced deliberately** — ~40 candidates screened for a steeper camera.
  Gate passed at median **0.635** (11/16); 7 spread frames audit at **0.637** (6/7).
  **New measurement: the detector's misses here are false negatives.** All 6 undetected
  frames are clean, fully-framed boards — `detect_board_quad` runs ~30% FN on footage that
  is ~100% usable, so cadenced re-detection will miss ~1 attempt in 3 on real video.
  Suspected cause (unverified): barrel distortion breaking the straight-edge quad model.
- 08-07: **M2 — live position stream, committed** (`c601dea`…`b565d68`). Edge `MotionMeter` +
  `AdaptiveForwarder` (keepalive ~1 fps / burst 12 fps, hysteresis), cadenced `Rectifier`
  re-detection, `GameSource` + synthetic game with an occluding hand, gateway latest-wins
  `/ingest` workers (ack ≠ estimate), viewer liveness. 477/477 green incl. `tests/test_m2_stream.py`.
- 08-07: **Training harness fail-fast + observable** (`b896db9`, `34c88ad`, `c3fecbb`). Root
  cause of the run_f wreck: launched on the wrong interpreter, and `run_f.sh` lacked `set -e`
  so it printed success over nothing (third instance). Now `_env.sh` preflights before any GPU
  time. New `screen.sh` gives a ~13-min ranking tier. **`make test` 35s → ~10s** (`5a6a19a`).
- 08-07: **Hard-constraint prior decoder ships** (`bbddded`). Flips the lowest-margin offending
  square on *impossible* boards: **5/23 → 10/23** perfect trained-alone, **3/9 → 6/9** on
  uncalibrated broadcast frames. A decoder over `square_dist`, never an estimator (D1).
- 08-07: **`chesscog_transfer` repaired** (`f362f78`) — 14 of 22 labels were rotated 180°, now
  resolved from pixels. `run_e` **0.500 → 0.819**. Plus **pose-aware crop geometry** (a
  `CropGeometry` ADT rides the `.onnx` sidecar, so train/serve mismatch is unrepresentable)
  and **temperature scaling measured** (LOBO ECE 0.255 → 0.004, **0 answers changed**).
- 08-07: **The honest eval, 23 confirmed labels.** Ensemble **13/23 perfect · 1450/1472** vs
  5/23 alone — *the bank is the entire margin, and only where calibrated*: **zero boards** on
  nine broadcast frames ([docs](docs/measuring-accuracy.md)).
- 07→08: **M0 skeleton · M1 single-frame · snapshot app · `run_e` deployed.**

## Next

1. **Score run_f against run_e** on both holdouts; deploy only if it wins both. If pose crops
   lose, the geometry bet is answered and item 5 is what survives it.
2. **Run M2 over `ccbf-r5`** with `--corners`, then settle D3 (`stability`).
3. **Attack the type confusions.** Trained-alone: 53 type / 5 hallucination / 4 colour, led by
   `p->b` (11). The ensemble's residual differs in kind — `empty->p`/`empty->P` is a third of it.
4. **Decide whether calibrated confidences go on the wire** — a decision, not an experiment:
   argmax is invariant, and DESIGN §9 says the downstream smoother wants them.
5. **Central-column pooling A/B.** `dinov2_small_central.onnx` is exported; build
   `exemplars_chessred_central.npz` and score it under the uncontaminated cross-style protocol.
6. **Crop attribution / occlusion** — pose fixed coverage and resolution, not attribution: the
   box is still the prism's *bounding* box, so a piece behind can fall inside it unflagged.

## Open questions

- **We have no real footage inside the good angle band,** and a deliberate ~40-candidate search
  found none: no long real OTB game on YouTube reaches 0.73, because broadcast cameras frame
  players, not boards. Own footage is now a **sourcing requirement**, not a nice-to-have.
- **Is `detect_board_quad` good enough for auto re-detection?** ~30% false negatives on
  ccbf-r5, cause unverified. Until it is understood, real-footage runs need operator corners.
- **Bank writes are not monotone.** A board that read 64/64 lost a square after two `/calibrate`
  writes, and trained-alone still gets it right. Nothing measures that before a write lands.
- **Confidence is not usable today.** Trained-alone has no board-level signal (0.978 vs 0.939)
  and the ensemble is under-confident. Top risk in DESIGN §9.
- **The 638/640 baseline is unreproducible** — 5 of 10 curated boards point at deleted photos,
  making `run_e`'s bright-board regression unjudgeable. Recover them or prune.
- Licensing: ChessReD (CC BY-NC-SA) + murky geometry → personal use only.
- No lint gate: `ruff` has no project config, ~109 findings unaddressed. Adopt it or drop it.
