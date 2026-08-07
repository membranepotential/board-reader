# STATUS — board-reader

_reconciled: 2026-08-07 @ `5f2a0a4`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md); this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- **Single-origin viewer + tunnel access (this session, 2026-08-07) — done, uncommitted.**
  All three pages are now same-origin: the viewer relays the node's whole HTTP API under
  a catch-all `/api/<path>` (was: two hardcoded routes) and injects nothing into the HTML.
  `/` and `/snapshot` previously carried an absolute `http://<lan-ip>:8000`, which made
  them LAN-only — blocked mixed content over https, unreachable behind a tunnel. Deletes
  snapshot.html's localhost-hostname-rewrite guess and the index.html origin derivation.
  New project skill `.claude/skills/tunnel` (cloudflared quick tunnel, one port).
  **Verified live through a real tunnel:** all three pages 200, `/api/uploads` returned
  node data, `wss://…/ws/viewer` connected, no LAN address in any page. 783 tests green,
  ruff clean. Touches viewer + Makefile/compose/docs only — disjoint from the camera-page
  session. **`hcloud` is not involved:** `boardreader.peppert.de` cannot point at a quick
  tunnel (measured: TLS handshake failure, and 403 on Host-header routing), and the paid
  Cloudflare tiers that would fix it were declined — see the skill.
- **ccbf-r5 live calibration (calibration session, 2026-08-07) — WORKS; landed in `63923c7`.**
  Root cause of "won't calibrate": the guard's probe reads the *angled* t=0 start frame at
  28/32 but the edge-rectified *warp* at 6/32 (aspect 0.635), and an angled-photo bank reads
  warps at 11/32 — warp streams are unusable at this angle, full stop. Fix rides the
  wire-corners work above: new `feed --forward-original` (edge implementer, landed in
  `packages/edge/**`, 589 tests green, in review) forwards ORIGINAL frames + semantic
  `--corners`; ingest photo-paths them. **Live e2e verified on the mp4:** frames ingest,
  `X-Corners` served, `/calibrate` from the retained frame accepts (13/13, `label_rotation 0`
  — a8-first corners `536,918,631,336,1458,320,1638,896` yield white's-view placements),
  start reads 63-64/64 and tracks 1. e4; persistent d8 q→b far-side error (decapitation
  weakness, degraded band). The camera-page landing delivered the live calibrate button
  preferring `X-Corners` (index.html `cornersOf`); remaining: optional per-board display
  rotation (user-approved). Docs (TESTING/index: the new flag + ccbf runbook) in the
  joint reconcile.

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

- 08-07: **The position stream has a consumer, and the gateway has one routing decision**
  (`chore/mvp-cleanup`, merged `5f2a0a4`; 776 green, lint clean). The project's output was
  observable only in a browser, so a run could never be captured, diffed or asserted —
  **`board-reader-inference positions`** now tails `WS /viewer` headlessly, or captures it
  verbatim as NDJSON (`--out`), and `tests/test_position_stream_e2e.py` drives rendered
  frames through the real edge → ingest → gateway → hub into a file and asserts the captured
  positions follow the game. **`EstimatorSession`** lifts the calibration session out of
  `create_app`'s 1060-line closure and its four `nonlocal est` mutations (server.py 1307 →
  979); **`routing.py`** gives `POST /estimate` and the stream one four-way decision, with
  `Uncalibrated` unable to produce a `PositionEvent` by construction — and unifying them
  settled a live divergence (a photo-mode estimator read a corner-less frame through the
  photo path over HTTP but the plain path on the stream). Edge **`--corners` is now
  validated before any frame moves** (bowtie/sliver/degenerate refused, low angle warns),
  plus `--jpeg-quality`. **Ruff adopted** (`make lint`), closing that open question.
  **Temporal consistency written down** as the missing **L4 consumer**, not a CV-layer gap:
  D1/D3 put smoothing in the consumer *by design*, and what is absent is that consumer —
  `stability` is computed, shipped on every event, and read by nobody.
  **Suite 19.5s → 15.0s while gaining ~90 tests**, and an intermittent whole-suite *hang*
  fixed: uvicorn teardown burning `join(timeout=5)` after the assertions passed
  (`timeout_graceful_shutdown=1`), a dozen xdist workers against one 4GB card (deselecting
  `gpu` tests ≠ denying the process a GPU), and a real ack≠estimated race. `tools/wtest.sh`
  runs a worktree against the main venv with no second build.

- 08-07: **`/camera` becomes a viewfinder, and finds the board by itself.** Landscape-first
  full-screen preview (iOS Safari has no `screen.orientation.lock()`, so landscape is the base
  stylesheet and portrait the `@media` override); six rail controls — `more · position ·
  detect board · rotate labels · calibrate · start` — with everything else in a slide-up sheet,
  and a draggable/resizable live-position overlay. **Detection is automatic**: first sight of an
  unframed board, and after a *sustained* loss of track. Four gates, each a measurement:
  `MotionMeter` returns 1.0 on its first frame so a bare stability gate passes before AE/AF
  settle (settle delay); `PlaneTracker.lose()` self-heals so raw `lost` fires on essentially
  every move of a real game (sustained-loss threshold); and `detect_board_quad` is **91-198 ms
  on a hit but 2066-3865 ms on a miss** (exponential back-off + give-up). Browser-measured:
  4 attempts, gaps 8.7/18.5/32.7 s, then stop. `/detect` also moved off the event loop
  (`asyncio.to_thread`) — a prerequisite, not a nicety. **The calibration walk was removed**
  (`/calibrate` is additive, so tapping calibrate from several angles already is a walk); its
  novelty and blur gates are the honest loss. Geometry uses a JS-sized `.picture` wrapper, not
  `object-fit: contain`, so `pts` stay in video pixels and screen space cannot leak onto the
  wire. Verified in Chromium at both orientations, which caught a 31 px recovery button.
  **701 tests green.**
- 08-07: **Board-detection FN hunt closed** (`3f93e57`, calibration session, worktree).
  The ~30% misses on clean ccbf-r5 frames root-caused and fixed — NOT barrel distortion
  (refuted: k1 unidentifiable from the interior grid). Causes: `findChessboardCornersSB`
  is scale-brittle **and nondeterministic** (same frame flips 7x7↔nothing, single-threaded
  too) → wider ladder (900/800) + per-rung retry; phase scoring killed correct hypotheses
  whose extrapolated centers left the frame → partial-coverage scoring; crowded start
  positions sit below the validation floor even for the TRUE quad → crowded floor
  (min≥3 ∧ median≥30) guarded by a new **outside-ring check** (a shifted quad always
  leaves a true board line just outside: 30–51 vs ≤19 for correct) + best-of selection.
  **ccbf 6/7 → 21/21 (3 passes, 0 shifted-quad outliers; t=0 start frame 0/6 → 6/6 at
  ≤25 px); every other stream ≥ old code.** Opus-reviewed, fix-first items in,
  mutation-verified pin. Suite 697 green. Follow-ups → Next.
- 08-07: **Camera page becomes an instrument: tracking · calibration walk · lens picker.**
  New `static/tracker.js` — pyramidal Lucas-Kanade over the board's **own 9×9 lattice of
  square corners** + RANSAC homography, ~3 ms/frame at 400 px. The quad now follows the
  board with its `a8` labelling intact (DESIGN §4.4.1(a) realised as *tracking, not
  re-detection* — a fresh detect returns an unlabelled quad, decision #15), freezes on
  loss and re-locks by itself. That unlocks the **calibration walk**: carry the phone
  round a start position, the page collects one full-res frame per genuinely new vantage
  (novelty = quad *shape*, translation/scale divided out) and posts them one at a time to
  the additive `/calibrate`; forwarding pauses throughout; a refused vantage is reported,
  not fatal. Plus a **lens picker** for the iPhone ultra wide (iOS ≥16.3 exposes every
  rear lens but nothing identifies them — remembered by label; no `zoom` capability
  exists on iOS Safari to fake it). Tracker maths is *executed* under node
  (90 checks, `tracker_checks.mjs`), not string-asserted. **697 tests green.**
  Browser-verified end to end, which caught the one real bug: the pyramid aliased the
  page's reusable capture buffer, so each frame was compared against *itself* — a
  confident lock with a quad that never moved.
- 08-07: **Phone camera page — the browser edge ships** (DESIGN §6 "virtual Pi"). Viewer-served
  `/camera`: getUserMedia → draggable 4-corner crop (`/detect` assist, rotatable labels) →
  adaptive JPEG streaming into WS `/ingest` with per-frame `corners` (JS mirror of
  `MotionMeter`+`AdaptiveForwarder`) + guarded **calibrate** from a full-res grab + live
  mini-board. Viewer = single-origin WS/HTTP relay + `--ssl` cached self-signed TLS
  (serverAuth EKU for iOS) — one cert accept on the phone, or `adb reverse` for zero.
  Opus-reviewed: 6 findings (2 critical) all fixed + regression-tested; e2e-verified
  through real servers. The wire/ingest half landed with the calibration session's
  `63923c7`; the page itself with `0442177`. Docs: architecture browser-edge section,
  DESIGN §4.5, TESTING Option J.
- 08-07: **Ingress recording + replay — runs are reproducible now (WP-H2, `f35638f`).**
  New `board_reader_core.recording` (manifest + append-only `frames.ndjson` + the JPEGs
  as received). The gateway records at `WS /ingest` — the one socket *every* edge crosses,
  so the CLI feed, a Pi and the phone `/camera` page (via the viewer proxy) are all
  captured, and **before** the latest-wins pre-filter, so a recording is the run's *input*.
  `board-reader-edge feed --recording <dir> --throttle` re-sends the stored bytes
  **verbatim** (no JPEG round trip) and refuses `--adaptive/--rectify/--corners/
  --forward-original/--loop`. On by default for `serve` (`--no-record` opts out) and for
  compose (`./recordings` bind mount); off for `create_app()` in tests. 43 new tests;
  full suite re-run green at 684 before the commit.
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
2. **Run M2 over `ccbf-r5`** with `--forward-original --corners` (verified path; the warp
   path is unusable at this angle — see Now), then settle D3 (`stability`).
   Review follow-ups on the new flag (non-blocking): geometry sanity-check (convex/in-frame)
   on `--corners` at startup; `--jpeg-quality` knob (full frames are ~6x a 384px crop —
   matters for a real Pi); throttle+loop interaction tests; `--corners` help text is
   mode-dependent now; `docs/TESTING.md` needs the manual path for the flag.
3. **Attack the type confusions.** Trained-alone: 53 type / 5 hallucination / 4 colour, led by
   `p->b` (11). The ensemble's residual differs in kind — `empty->p`/`empty->P` is a third of it.
4. **Decide whether calibrated confidences go on the wire** — a decision, not an experiment:
   argmax is invariant, and DESIGN §9 says the downstream smoother wants them.
5. **Central-column pooling A/B.** `dinov2_small_central.onnx` is exported; build
   `exemplars_chessred_central.npz` and score it under the uncontaminated cross-style protocol.
6. **Crop attribution / occlusion** — pose fixed coverage and resolution, not attribution: the
   box is still the prism's *bounding* box, so a piece behind can fall inside it unflagged.
7. **Event loop: `/estimate` and `/calibrate` still block it.** Same bug class as `/detect`,
   which is now **fixed** (`asyncio.to_thread`, regression-tested by reverting) — measured
   91-198 ms on a hit but **2066-3865 ms on a miss**, since a miss exhausts every coarse
   candidate. Still synchronous inside their `async def`: `/estimate` (`server.py:743`,
   `photo_fn`/`est.estimate`/`rectify_photo` — ONNX) and `/calibrate` (`server.py:875`, the
   orientation-guard `probe_fn` and `build_exemplars`, likely the most expensive of the three
   since it embeds a crop per square). Both stall `/ingest` acks and the `/viewer` hub.
8. **Detection review follow-ups** (`3f93e57`, non-blocking):
   `_alternation`'s partial scoring can read a pure brightness gradient as alternation
   (mean-subtract per row/col, or require full 2x2 neighbourhoods); hexamind carries one
   pre-existing 142 px outlier detection worth a look.

## Open questions

- **We have no real footage inside the good angle band,** and a deliberate ~40-candidate search
  found none: no long real OTB game on YouTube reaches 0.73, because broadcast cameras frame
  players, not boards. Own footage is now a **sourcing requirement**, not a nice-to-have.
- **Is `detect_board_quad` good enough for auto re-detection?** Largely answered (`3f93e57`):
  ccbf-r5 now detects 21/21 (was ~70%), causes measured, barrel distortion refuted. Residual:
  the corner finder stays nondeterministic (a cadence retries for free), one piece-dense
  frame class stays undetectable, and a boardless frame costs ~2 s. Operator corners remain
  the recommendation for *reproducible* runs; the `/detect` assist and re-detection are now
  trustworthy enough to lean on.
- **Bank writes are not monotone.** A board that read 64/64 lost a square after two `/calibrate`
  writes, and trained-alone still gets it right. Nothing measures that before a write lands.
- **Confidence is not usable today.** Trained-alone has no board-level signal (0.978 vs 0.939)
  and the ensemble is under-confident. Top risk in DESIGN §9.
- **The 638/640 baseline is unreproducible** — 5 of 10 curated boards point at deleted photos,
  making `run_e`'s bright-board regression unjudgeable. Recover them or prune.
- Licensing: ChessReD (CC BY-NC-SA) + murky geometry → personal use only.
- **Nothing consumes the position stream.** Not a CV-layer gap — D1/D3 put temporal
  smoothing in the consumer deliberately — but that consumer does not exist, here or
  anywhere, so `stability` is measured and read by nobody and D3's "does it earn its keep?"
  cannot be answered until something uses it. See
  [architecture.md](docs/architecture.md) "Temporal consistency".
