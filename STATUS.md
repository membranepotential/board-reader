# STATUS — board-reader

_reconciled: 2026-08-20 @ `f196f24`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md), detail in `git log` + `scratch/experiments/*/report.md`; this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- public-release — repo is **push-ready**: Apache-2.0 + THIRD-PARTY.md, personal
  details scrubbed, history rewritten onto `felix@peppert.de` (97 commits, all
  re-signed). Remaining before the push: add `felix@peppert.de` as a UID on GPG key
  `BB114CEF` and re-upload it to GitHub, or every commit shows Unverified. Optional:
  a CI workflow (`make test` is 25 s) and CONTRIBUTING.md.
- phone-live-test — 19:33 re-trial **worked** (748/748 frames, watchdog recovered two
  stalls) and its defects are diagnosed + fixed in-tree (uncommitted, see Done). Next:
  commit, restart `make dev`, re-trial; then attack the one residual — d8/e8 k/q model
  confusion (decapitated rank-8 crops) via the tall-crop retrain, gated on
  `eval-session` numbers instead of the synthetic holdout.

## Done (recent)

- 08-20: **Pre-publish pass — the repo can go public.** Audited the full object graph:
  no photos, datasets, weights or secrets were ever committed (one UI screenshot is the
  only binary in history). Licensed **Apache-2.0** (patent grant; MIT gives none) with
  `THIRD-PARTY.md` recording the two things it does not cover — the built viewer bundle
  is **GPL-3.0-or-later** because `packages/viewer/ui` links chessground + cburnett
  sprites (our source stays Apache; `core`/`edge`/`inference` carry no GPL dep), and the
  trained model is **not redistributable** (ChessReD CC BY-NC-SA + murky geometry).
  Scrubbed the tunnel skill's DNS/mail disclosure, de-named eval provenance, repointed a
  dead agent-scratchpad path, refreshed the README off the unreproducible 633/640.

- 08-13 (eve, uncommitted): **Trial 19:33 analyzed end-to-end; four fixes landed.**
  (1) *Kings repair chooses by evidence* — surplus kings demote the lowest-`p(k)` square,
  not the smallest k→runner-up margin (that margin is q-tail noise: wrong 55% on the 628
  two-king replay frames, evidence rule wrong 27%; EMA smoothing fixes nothing — the
  misread is systematic). (2) *`settled` ships* — advisory per-square time debounce
  (2.5 s + ≥2 reads; count-based K does NOT transfer across the 10× read-rate range) on
  the wire beside `prior`, drawn by viewer + /camera with dashed-amber pending rings.
  (3) *`eval-session`* — truth.json per recording, three-stream scorer, labeled crop
  export; trial scored: raw 0% perfect boards, prior 45.4%, settled **66.6%** (98.96%
  per-square), residual is *only* d8/e8 k/q. (4) `_parse_corners` deduped into core;
  /estimate + /calibrate off the event loop (the trial's 6.6 s latency spike). Trial
  facts: estimator sustains ~0.85 reads/s vs 7.5 fps input; the "phantom b6 rook" was a
  real c6 knight leaning across the b6 boundary (attribution bleed, also corrupts b7);
  rank-8 crops at phone height are decapitated — k/q indistinguishable in-crop.

- 08-13: **The node stops starving the display GPU** — `cuda_providers()` now ships CUDA EP
  options (`BR_GPU_MEM_LIMIT_MB`, default 1024 MiB/session, `kSameAsRequested`, `HEURISTIC`)
  and `make dev-cpu` runs the same stack with the card hidden. Diagnosed from Xorg logs: a
  `/calibrate` batch took the VRAM X needed for the 4K monitor's reverse-PRIME surface, twice.
  Measured: 3-session peak ~1.6 GiB of the 4 GiB card; DINOv2 @batch 64 needs ≥896 MiB.

- 08-11: **Calibration fine-tune ships** (`9fe975f`) — `tools/train/calibrate_finetune.py`
  (≥3-session gate) + `AveragedEstimator` behind `BR_MODEL_TUNED`, 50/50 with base; live
  `/estimate` 63/64 vs base 58/64. Untested on unseen footage (same 5 `user_photos` throughout).
- 08-11: **Two-model cascade deployed** (`34a4ae7`, `3a55d41`) — τ-gated `CascadeEstimator`
  behind `BR_MODEL_B` (0.95), each member on its own sidecar geometry; `run_f` promoted into
  `models/` + compose, so the canonical demo runs it.
- 08-11: **`uploads_calibrate`'s 0.59 was a label bug** (`c4dd8b2`) — 19 rotated sidecars
  repaired, 0.59 → **0.75**, halluc 254 → 30. Quote the 17-real-photo subset (**0.919**); the
  13 pre-rectified 384px warps (0.63) are out-of-domain and drag the mean.
- 08-11: **Session diversity is the fine-tune lever** (pre-registered confirm passed): 3 frames
  from 3 lighting sessions → 0.938 → **0.969**, wrongtype 19 → 9, no forgetting. Single-session
  frames are worthless at any count.
- 08-11: **Ensemble beats both members** — user_photos 0.938 → **0.956**, OVERALL 0.696 →
  **0.718**. Decoupled routing answered-negative: combine evidence, don't route.
- 08-11: **Central-column pooling answered-negative** — the bakeoff's +9 was an 80-board-bank
  artifact; bank size was the real lever. Also: per-board display rotation (`bb7c2da`).
- 08-10: **run_f (pose-aware crops) does NOT deploy alone** — fixes cross-style type, pays
  occupancy. **+4.8pp of synthetic holdout transferred as 0.0pp on real photos**: the holdout
  does not predict deployment, and crops/epochs are exhausted as knobs.

## Next

1. **Tall-crop retrain, gated on `eval-session`** — the d8/e8 k/q residual is a crop
   problem (decapitated rank-8); score candidates on recordings/20260813-193308 truth.
2. **Estimator throughput** — 0.85 reads/s vs 7.5 fps input; profile the
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
   bounding-prism membership. Plus detection follow-ups (`3f93e57`, non-blocking).

## Open questions

- **No real footage sits inside the good angle band**, and a ~40-candidate search found none:
  broadcast cameras frame players, not boards. Own footage is a sourcing requirement.
- **Nothing consumes the position stream** — deliberate (D1/D3 put smoothing in the consumer),
  but that consumer does not exist, so `stability` is measured and read by nobody.
- **Confidence is not usable today** — no board-level signal; the ensemble is under-confident.
- **Bank writes are not monotone** — a 64/64 board lost a square after two `/calibrate` writes.
- **The 638/640 baseline is unreproducible** — 5 of 10 curated boards point at deleted photos.
