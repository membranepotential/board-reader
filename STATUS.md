# STATUS — board-reader

_reconciled: 2026-08-10 @ `90ed183`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md); this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- repair-uploads-labels (agent) — photometric rotation repair of the 20 mislabeled `uploads_calibrate` sidecars + corrected eval numbers — running — valid comparison set.
- ensemble-serving — DONE 08-11, reviewed + committed: `BR_MODEL_B` turns the trained
  component into a τ-gated `CascadeEstimator` (default 0.95, `BR_CASCADE_TAU`); each
  member crops with its OWN sidecar geometry, sidecar-less members refused at startup;
  bank ensemble arithmetic untouched; `/healthz` exposes deferred-fraction. 735 tests
  green; live-verified (real photo → sane placement, 5/64 deferred). **Deploy step
  still open: promote `run_f` into `models/` + compose env.**
- confirm-finetune (agent) — confirming run DONE: **NOT CONFIRMED as drop-in** — the
  gain replicates on fresh seeds (0.968 [0.963–0.975], q→b confusion → 0, no
  forgetting) but each arm fails one occupancy condition by a hair (halluc 3.2 vs ≤2 /
  wrongtype 10.4 vs ≤10). Same coupled type-vs-occupancy trade as run_f. Now running
  the registered follow-up: tuned⊕run_e 50/50 composition under the same rule.
  `scratch/experiments/calib_finetune_confirm/report.md`.

- exp-ensemble-ef — DONE 08-11, **prob_50_50 ensemble is DEPLOY-WORTHY under the
  pre-registered rule**: beats BOTH models on every set — user_photos 0.938→**0.956**
  (halluc 1, wrongtype 19→13), chesscog_transfer 0.856→**0.861** (wrongtype 118),
  OVERALL 0.696→**0.718**. The decoupled routing variant (occ from run_e, type from
  run_f) answered-negative: 0.928 user_photos, −70 squares vs averaging — combine
  evidence across geometries, don't route. Sanity gate reproduced both baselines
  exactly. `scratch/experiments/ensemble_ef/{PREREGISTRATION,report}.md`. Next
  decision: ship ensemble serving in `board_reader_inference` (2 sessions, 2 crop
  passes, ~2× inference cost).
- exp-central-pooling — DONE 08-11, **swap answered-negative (Next #5 closed)**: the
  bakeoff's +9 doesn't survive the shipped 400-board bank — CLS 255 vs central 254
  cross-style (tie), same-style LOBO 294 = 294. Bank size was the real lever (CLS
  80→400 boards = 248→255). Surviving nuance: central is better at cross-style
  *occupancy* (p→e misses 10→1) at the cost of type — right embedding if a
  zero-setup occupancy-only head ever ships. `scratch/experiments/central_pooling/report.md`.
- exp-calib-finetune — DONE 08-11: **prereg (2 frames/1 session) NOT validated**
  (0.935 vs 0.938), but the sweep found the real lever — **session diversity**: 3 frames
  from 3 lighting sessions → user's board 0.938→**0.969** [0.966–0.972, 3 seeds],
  wrongtype 19→**7**, occupancy held, NO forgetting (chesscog 0.819→0.856, it gains).
  Single-session frames are worthless for tuning no matter the count. Needs one
  confirming pre-registered run before productionizing (trigger: board's stored
  calibrate frames span ≥3 sessions). `scratch/experiments/calib_finetune/report.md`.
  **Side discovery:** 20/25 non-user `uploads_calibrate` boards have ROTATED corner
  labels (run_e 17–26/64 on them) — the ~0.59 uploads numbers were floored by a label
  bug all along; same repair as `chesscog_transfer` (f362f78) applies. Until then that
  set is not a valid comparison.
- research-crossstyle — DONE 08-11: ranked report at `scratch/experiments/research_crossstyle.md`. Top-2: (1) decouple occupancy/type onto own geometries (zero-cost, = exp-ensemble-ef variant C), (2) calibration-time fine-tune of the type head (chesscog measured 9.33→0.11 errors/board from 2 start photos). Rejected with evidence: YOLO detection (inherits style problem, CS231N 2025), end-to-end 64-head whole-image (collapses cross-set, CVChess), TTA/TENT (measured harmful). Prerequisite P: real-photo val set for checkpoint selection — synthetic holdout proxy is refuted externally too (chesscog 0.15→42.87).

- **ccbf-r5 live calibration (calibration session, 2026-08-07) — WORKS; landed in `63923c7`.**
  Root cause of "won't calibrate": the guard's probe reads the *angled* t=0 start frame at
  28/32 but the edge-rectified *warp* at 6/32 (aspect 0.635), and an angled-photo bank reads
  warps at 11/32 — warp streams are unusable at this angle, full stop. Fix rides the
  wire-corners work: new `feed --forward-original` (edge implementer, landed in
  `packages/edge/**`, 589 tests green, in review) forwards ORIGINAL frames + semantic
  `--corners`; ingest photo-paths them. **Live e2e verified on the mp4:** frames ingest,
  `X-Corners` served, `/calibrate` from the retained frame accepts (13/13, `label_rotation 0`
  — a8-first corners `536,918,631,336,1458,320,1638,896` yield white's-view placements),
  start reads 63-64/64 and tracks 1. e4; persistent d8 q→b far-side error (decapitation
  weakness, degraded band). The camera-page landing delivered the live calibrate button
  preferring `X-Corners` (now `lib/components/live/calibration.ts`); remaining: optional
  per-board display rotation (user-approved). Docs (TESTING/index: the new flag + ccbf
  runbook) in the joint reconcile.

Also staged and waiting on a decision to start:

- **M2 over real footage — staged, not started.** The recorded-game demo works synthetically;
  the milestone's spirit wants a real game. The source is now picked: `ccbf-r5`
  (`2sRJ4JuL_m8`, 3 h, fixed camera, 1080p60 on disk). Run
  `feed --video --adaptive --corners`, judge the stream, then decide whether `stability`
  earns its keep (D3). **Use `--corners`, not auto-detect** — see the false-negative
  finding below.

## Done (recent)

- 08-11: **Per-board display rotation on the live page** (the ccbf-r5 remainder) —
  UNCOMMITTED in the working tree. A rotate button on each live board card cycles
  0/90/180/270° via CSS `transform` on the square `.board-wrap` (chessground's
  `orientation` can only flip, so the whole picture turns, labels included — deliberate,
  like turning a printed diagram). Pure `rotation.ts` + Vitest; persisted per board id in
  localStorage (`board-reader.live.rotation.<id>`), Playwright covers cycle/isolation/
  reload. 339 Vitest · 17 Playwright · 720 pytest green, `svelte-check` clean.

- 08-10: **`run_f` (the pose-aware-crop bet) ran to completion 21:57→22:38, and it does
  NOT deploy.** 20/20 epochs on the GPU, best val **0.8942 @ e17** (`scratch/runs/run_f/`,
  `eval.txt`, exported `model.onnx` + pose sidecar, round-trip `2.98e-07` / argmax 8/8).
  Clean run: preflight passed on `scratch/torchvenv`, cache warm (no rebuild), `set -e` held,
  `===F_DONE===` printed over a *real* eval. Scored against run_e re-run on the same 60 boards
  (`run_e/eval_rerun_20260810.txt`) — each model on its own geometry sidecar:

  | set | run_e (bbox hr1.6) | run_f (pose h1.9/r0.5) | verdict |
  |---|---|---|---|
  | synthetic holdout `chessr_vlab_wooden` | **0.898** | 0.8942 | loses by 0.004 (tie) |
  | `chesscog_transfer` (unseen real photos) | 0.819 · wrongtype 173 | **0.856** · **121** | **wins +3.7pp, −30% wrongtype** |
  | `user_photos` (the deployed target) | **0.938** · halluc 1 | 0.894 · halluc **12** | **loses −4.4pp on occupancy** |
  | `uploads_calibrate` | 0.592 | 0.593 | tie (both weak) |
  | OVERALL | 0.696 · perfect 1/60 | **0.706** · perfect 0/60 | wins +1.0pp |

  **The pre-registered rule was "deploy only if it wins both" — it wins neither the holdout
  nor `user_photos`, so run_e stays deployed and the geometry bet is ANSWERED, not pending.**
  What pose crops actually trade: they fix cross-style piece **type** (the decapitation
  weakness they targeted — wrongtype 173 → 121 on unseen photos) and pay for it in
  **occupancy** on the user's own board (hallucinations 1 → 12, occ 0.997 → 0.963), which is
  consistent with a taller prism swallowing part of the neighbouring piece.
  **The more valuable finding: 14 extra epochs bought nothing real.** e3-partial (val 0.8466)
  vs final (val 0.8942) → transfer 0.855→0.856, user_photos 0.900→0.894, OVERALL 0.707→0.706.
  **+4.8pp of synthetic holdout transferred as 0.0pp on real photos**, so the synthetic
  holdout is a near-useless proxy for deployment quality on this task — training longer is not
  where the remaining accuracy lives. Item 5 (central-column pooling) is what survives this.
  **Trap found:** with no sidecar `evaluate.py` silently assumes `BboxGeometry(headroom=0.15)`
  and scores a pose model on short crops — it warns, but does not refuse.
  Corpus note: `uploads_calibrate` grew 7 → 35 boards since 08-07, so the 08-07
  `eval_rerun.txt` is no longer corpus-comparable — quote neither its `uploads_calibrate` nor
  its `OVERALL`.
- 08-10: **The three pages become one typed SvelteKit app** (`fc2cae2`, docs `90ed183`).
  `index.html` 465 · `camera.html` 1682 · `snapshot.html` 2407 + `tracker.js` 536 —
  ~4,550 hand-written lines, carrying three near-identical copies of `fenToGrid`, corner
  labelling and `squareKey` — are now `packages/viewer/ui` (SvelteKit 2, Svelte 5 runes,
  TS strict + `noUncheckedIndexedAccess`), prerendered by `adapter-static` into
  `src/board_reader_viewer/web/`: generated, gitignored, built by the new **`make ui`**
  (which `make dev` depends on) and by a Docker `ui-build` stage only the viewer image
  pulls from. **`proxy.py`/`tls.py` untouched** — the viewer is still a Python
  byte-forwarder importing no `board_reader_*` package, so the single-origin/tunnel
  property survives, and is now *asserted* by a test that greps the built bundle for any
  absolute origin, IP literal or explicit port. **The prize is `src/lib/**`:** one tested
  version of the protocol parser, the always-relative URL builder, FEN⇄grid, the
  board-plane tracker + homography, `MotionMeter`/`AdaptiveForwarder` and the ingest
  framing. **`create_app()` deliberately does not require the bundle** (missing pages 503;
  `main()` is what refuses to start), so `proxy`/`tls` stay testable on a fresh clone with
  no Node toolchain. Two more fixes on top of the port: the ingest client regained its
  *pre-encode* backpressure check, and a chessground-creating `$effect` now depends on the
  element alone — reading a prop there rebuilt the board per placement and leaked a
  ResizeObserver each time. **720 Python · 333 Vitest · 16 Playwright green**, ruff +
  `svelte-check` clean; verified live in Chromium against the real stack (both
  orientations) and as a rebuilt Docker image. **Dependency note:** vendored chessground 9
  → `@lichess-org/chessground` 10.1.1 — the unscoped `chessground` package is
  **deprecated**, so don't "fix" it back; npm deps are 7-day seasoned
  (`minimumReleaseAge: 10080`). **Known gap:** Playwright is Chromium-only and drives
  touch as mouse events, so iOS Safari — the actual target device — stays manual-only.
- 08-07: **Single-origin viewer + tunnel access** (`9468cc2`). All three pages are
  same-origin: the viewer relays the node's whole HTTP API under a catch-all `/api/<path>`
  (was: two hardcoded routes) and injects nothing into the HTML. `/` and `/snapshot`
  previously carried an absolute `http://<lan-ip>:8000`, which made them LAN-only —
  blocked mixed content over https, unreachable behind a tunnel. Deletes snapshot's
  localhost-hostname-rewrite guess and the index origin derivation. New project skill
  `.claude/skills/tunnel` (cloudflared quick tunnel, one port). **Verified live through a
  real tunnel:** all three pages 200, `/api/uploads` returned node data,
  `wss://…/ws/viewer` connected, no LAN address in any page. **`hcloud` is not involved:**
  `boardreader.peppert.de` cannot point at a quick tunnel (measured: TLS handshake
  failure, 403 on Host-header routing), and the paid Cloudflare tiers that would fix it
  were declined — see the skill.
- 08-07: **The flaky/hanging suite, root-caused — and a testing guide agents can act on.**
  The `ack≠estimated` race closed earlier was a symptom, not the cause. `WS /ingest` acks on
  *receipt* and estimates on a per-board worker task; a `TestClient` **not entered as a
  context manager gets a fresh event loop per call**, so that worker died with the websocket
  and its position was never published (`WebSocketTestSession.__exit__` also cancels the app's
  scope before the handler's own `drain()` runs). Three faces of the one bug: a flaky
  `GET /boards/A1` 404; `test_viewer_replays_retained_latest_on_connect` **blocking forever**
  on a `/viewer` read — the silent 14-minute whole-suite hang, caught in the act by a
  faulthandler dump; and two sockets on one bare client landing on *two* loops, so
  `hub.publish()` wakes a Future owned by the other loop. **Production is unaffected** —
  uvicorn has one long-lived loop. Fix: `packages/inference/tests/conftest.py` with `gateway`
  (one portal per client) + `drain_ingest` (waits on the workers' own idle events); 6 tests
  moved onto it, 3 sleep-poll loops with 5 s deadlines deleted. Measured under identical heavy
  load: **old 6/600 frames lost, fixed 0/600**; 10/10 loaded full-suite runs green.
  `faulthandler_timeout = 120` now makes any future hang name its test instead of burning a
  session. **[TESTING.md §0](docs/TESTING.md)** is the new 60-second agent briefing — measured
  cost per command, the habits that waste time, hang/port triage, and the ingest rule; linked
  from CLAUDE.md. (Note the trap it documents: `pytest packages/inference` is 20 s, *slower*
  than `make test`'s 12 s for all 783.)
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

1. **~~Score run_f~~ — done 08-10; it lost `user_photos` and tied the holdout, so run_e stays
   deployed.** What replaces it: the geometry knob is exhausted (bbox swept 07-18, pose
   answered 08-10), and the synthetic holdout is now known not to predict real-photo accuracy,
   so **stop tuning crops and epochs** — go to item 5 (central-column pooling), the one
   measured +9 win still unspent. A pose/bbox *ensemble* is the only geometry idea left worth
   a run: the two models fail differently (pose −52 wrongtype on unseen photos, bbox −11
   hallucinations on the user's board), which is the precondition for one helping the other.
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
5. ~~**Central-column pooling A/B.**~~ Answered-negative 08-11 (see Now): the +9 was an
   80-board-bank artifact; at the shipped 400-board bank CLS ties central and same-style
   ties at 294. Occupancy nuance kept in `scratch/experiments/central_pooling/report.md`.
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
