# STATUS — board-reader

_reconciled: 2026-08-27 @ `3406d2d`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md), detail in `git log` + `scratch/experiments/*/report.md`; this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- public-release — repo is **push-ready** (Apache-2.0 + THIRD-PARTY.md, details
  scrubbed, history re-signed onto the current author email). Before pushing: add that
  address as a UID on the signing key and re-upload it, or every commit lands
  Unverified. Optional: a CI workflow, CONTRIBUTING.md. Pre-rewrite SHAs recoverable
  from `docs/commit-rewrite-{map.tsv,pre.bundle}` (git-ignored); `tools/check_commit_refs.py`
  translates stale citations.
- phone-live-test — control trial **run and scored** (08-27, see Done); the fixes are
  measurably in. Still to do: the *hard* take (rank-8 pieces, different lighting) —
  and the old phone must be held **landscape**, or crops are cut along the wrong axis
  (see the portrait finding in Done). Then decide the tall-crop retrain's shape from
  the evidence margins.

## Done (recent)

- 08-27: **Control trial run, scored, and the 08-13 fixes are confirmed landing.**
  `recordings/20260827-163352` (Ruy Lopez through 4.Ba4, 337 frames, 31 reads, truth
  authored + `report.json` kept). **prior 99.0 % per-square, 58.3 % perfect boards**
  against 08-13's 97.0 % / 28.8 % — the kings-evidence repair (`faaa4ed`) doing exactly
  what it was built for. Latency **median 1.38 s, max 1.56 s** vs 6.59 s on 08-13 (the
  event-loop fix). Throughput unchanged: 9.65 fps in, 0.89 reads/s out.
  *Evidence margins, the new measurement:* of 38 wrong raw squares, truth was the
  runner-up in only 21. Split by confusion it answers the retrain question — `q->k`
  (22×) has truth in the top 2 just **11/22** at median margin 0.152, so the crop often
  does not contain the answer and re-ranking cannot fix it; `p->b` (7×, the b7 pawn
  behind the leaning knight) has truth ranked **7/7** at margin 0.144, so that one is
  re-rankable. *Eval-design finding:* `guard_seqs` must cover the **settle lag** or the
  settled stream is scored while it is correctly still catching up — settled reads 29.2 %
  perfect at guard 12 and 77.8 % (99.7 % per-square) at guard 45, on the same 7 boards.

- 08-27: **Calibrate failed all afternoon because the phone was portrait.** Not a
  regression (08-13's photos score 28-31/32 through today's code) and not the angle
  (aspect 0.86-0.90, floor 0.62). `squares.py:264` extends every crop **upward in image
  space** — "piece tops point up in a normal photo" — so a board rotated 90° in the frame
  has each crop reaching into the neighbouring square. Proof: the same photo scores
  18/32 REJECT as sent and **30/32 PASS rotated 90° CCW** (piece confidence 0.47 → 0.94).
  Occupancy stayed 64/64 throughout, which is why it looked like a piece-type problem.
  `autorotate` cannot help — it rotates *labels* from `square_dist`, one layer above the
  crop axis. A quad-only heuristic for the upright direction was tried and **rejected**:
  it predicts 270° where the truth is 90° (3/7 correct), so the usable signal is the
  device orientation angle, not perspective foreshortening.

- 08-27: **A trial now records why, not just what** — three gaps closed before the
  re-trial. *Evidence*: every recorded event carries a top-2-per-square digest
  (~1.8 kB vs ~25 kB for `square_dist`), and `eval-session` reports per confusion pair
  how often truth was the runner-up and the median `p(read) − p(truth)` margin — a
  near-miss the crop nearly answered vs a conviction the evidence never held.
  *Timing*: fps in, reads/s out, frames per read and capture→publish latency, off the
  frame index alone; it confirms the hand-derived 08-13 figures (**7.47 fps in, 0.85
  reads/s out, latency median 1.29 s / max 6.59 s**). *Heartbeat*: `/camera` beats every
  10 s into `client-logs/`, so a quiet stretch says which kind of quiet. Scoring now
  always writes `<recording>/report.json`.
- 08-27: **The 08-13 headline numbers were wrong, and are corrected.** Re-scoring with
  unchanged code gives prior **28.8 %** perfect, settled **25.8 %** (97.7 % per-square)
  — not 45.4 % / 66.6 % (98.96 %). The truth.json was revised after those numbers were
  published (66 scored events / guard 25 today vs 81 / guard 12 in the learning) and no
  report artifact was kept. README corrected. Same failure as the 638/640 entry — hence
  the always-write-the-report rule.

- 08-20: **Pre-publish pass — the repo can go public.** Full object-graph audit: no
  photos, datasets, weights or secrets were ever committed. Apache-2.0, with
  `THIRD-PARTY.md` for the two exclusions — the built viewer bundle is
  **GPL-3.0-or-later** (chessground + cburnett; our source stays Apache and
  core/edge/inference carry no GPL dep), and the trained model is **not
  redistributable** (ChessReD CC BY-NC-SA + murky geometry).

- 08-13: **Trial 19:33 analyzed end-to-end; four fixes landed** (`faaa4ed`, `ef9a675`,
  `78bc4d2`, `37a1972`). Kings repair demotes the lowest-`p(k)` square, not the smallest
  k→runner-up margin (that margin is q-tail noise: wrong 55% of 628 two-king replay
  frames vs 27% for the evidence rule; EMA fixes nothing, the misread is systematic).
  `settled` ships — a per-square *time* debounce (2.5 s + ≥2 reads; count-based K does
  NOT transfer across a 10× read-rate range) on the wire beside `prior`, drawn with
  dashed-amber pending rings. `eval-session` scores the three streams apart. And
  /estimate + /calibrate moved off the event loop (the 6.6 s latency spike). Trial
  facts: the "phantom b6 rook" was a real c6 knight leaning across the boundary
  (attribution bleed, also corrupts b7); rank-8 crops at phone height are decapitated,
  so k/q is indistinguishable in-crop.

- 08-13: **The node stops starving the display GPU** — `cuda_providers()` ships CUDA EP
  options (`BR_GPU_MEM_LIMIT_MB`, default 1024 MiB/session) and `make dev-cpu` hides the
  card. A `/calibrate` batch had twice taken the VRAM X needed for the 4K monitor's
  reverse-PRIME surface. Peak measured ~1.6 GiB of 4 GiB across 3 sessions.

- 08-11: **Recognition reached its current shape** (`20b063a`, `364bea3`, `2b64ada`,
  `155704c`, `3fc8831`) — τ-gated two-model cascade behind `BR_MODEL_B`, calibration
  fine-tune behind `BR_MODEL_TUNED`, ensemble-with-the-bank beating either member
  (user_photos 0.956), per-board display rotation. Answered-negative along the way:
  decoupled routing, central-column pooling, and run_f deployed alone. Session
  *diversity* is the fine-tune lever, not frame count. Detail: `git log`,
  `scratch/experiments/*/report.md`.
- 08-10: **The synthetic holdout does not predict deployment** — run_f's +4.8pp there
  transferred as 0.0pp on real photos. Crops and epochs are exhausted as knobs.

## Next

0. **The portrait crop axis** — hold the phone landscape for now. The fix is to
   canonicalise the frame client-side from `screen.orientation.angle` (B) and to have
   the calibrate gate name this cause instead of refusing generically (A); a
   pose-derived crop axis (C) is the principled version and belongs with the retrain.
0b. **The hard take** — rank-8 pieces, different lighting; the control is done.
1. **Tall-crop retrain, gated on `eval-session`** — the d8/e8 k/q residual is a crop
   problem (decapitated rank-8); score candidates on recordings/20260813-193308 truth,
   and read the new evidence margins first: truth ranked second by a hair says the crop
   carries the answer, truth outside the top 2 says a better head will not find it.
2. **Estimator throughput** — measured 0.85 reads/s against 7.47 fps in; profile the
   cascade+ensemble path (bank embed batching?); faster reads shrink settle lag too.
3. **Run M2 over `ccbf-r5`** with `--forward-original --corners` (the warp path is unusable at
   this angle), judge the stream, then settle D3 — does `stability` earn its keep?
4. **Give a user's `tuned.onnx` a deploy story** — per-operator artifact, no compose path today.
5. **Attack the residual type confusions** — `p->b` leads trained-alone (confirmed live:
   b7 pawn read `b` behind the leaning knight); a third of the ensemble's residual is
   `empty->p`/`empty->P`, a different animal.
6. **Decide whether calibrated confidences go on the wire** — a decision, not an experiment:
   argmax is invariant, DESIGN §9 says the downstream smoother wants them.
7. **Crop attribution / occlusion** — the prism is a *bounding* volume: the c6 knight's lean
   hallucinated a b6 rook and corrupted b7 all trial. Base-weighted attribution over
   bounding-prism membership. Plus detection follow-ups (`73efd60`, non-blocking).

## Open questions

- **No real footage sits inside the good angle band**, and a ~40-candidate search found none:
  broadcast cameras frame players, not boards. Own footage is a sourcing requirement.
- **Nothing consumes the position stream** — deliberate (D1/D3 put smoothing in the consumer),
  but that consumer does not exist, so `stability` is measured and read by nobody.
- **Confidence is not usable today** — no board-level signal; the ensemble is under-confident.
- **Bank writes are not monotone** — a 64/64 board lost a square after two `/calibrate` writes.
- **The 638/640 baseline is unreproducible** — 5 of 10 curated boards point at deleted photos.
