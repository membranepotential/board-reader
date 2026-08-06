# STATUS — board-reader

*Canonical dashboard. Last reconciled 2026-08-07 against `git log`, `data/uploads/` (23 labels), a re-run eval.*
Design decisions live in [docs/DESIGN.md](docs/DESIGN.md); this file is state, not design.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- **Snapshot quality push** — make the snapshot app read a board reliably before M2.
  Landed: ground-truth capture + the eval harness (below). Next: act on what it measured.
- **snapshot.html rewrite** — the page is a stage-by-stage pipeline inspector; `test_viewer.py`
  matches. Committed in `bf2ae0b` (this entry previously said uncommitted; it is clean).
- **Snapshot orientation + labelling UX** (uncommitted, 2026-08-06) — auto-rotation now
  reaches the whole page, not just the board; hover magnifier; server-backed Recent list.
  Detail in Done. Still uncommitted, along with the new `Makefile` (`make dev` = the
  deployed env) and the 23 labels the two labelling sessions produced.
- **Act on the first honest eval** (opened 2026-08-06). The number exists now (Done below),
  and the labels behind it are confirmed correct (2026-08-07, operator), so it is a real
  measurement rather than the ensemble grading itself. Now re-run on 23 labels: **13/23
  perfect, 1450/1472**, and the bank accounts for all of the margin on the set it was
  calibrated on while doing nothing on the nine broadcast frames. Next: decide the prior
  (Next 1), then temperature scaling.
- **Hard-constraint decoder** (`implementer` agent, spawned 2026-08-07) — objective: a pure
  `prior.py` in `inference` that repairs *impossible* boards (≤1 king/colour, no pawn on rank
  1/8, ≤16 pieces/colour, promotion budget) by flipping the lowest-margin offending square.
  Landed: `prior.py`, the eval scoring it, and the circular-frame fix below
  (`quarter_turns=None` → the back-rank constraint sits out). **406 tests green.**
  `kings` does nearly all the work — fired 19× trained-alone / 11× ensemble, vs `piece_count`
  6/3 and `promotion_budget` 5/1.
  **Re-measured on all 23 labelled boards** — trained-alone **5/23 → 10/23 perfect**, 1410 →
  1428 squares (fixed 21 · broken 3); ensemble 13/23 → 14/23, 1450 → 1458 (fixed 9 · broken 1).
  The domain split is the point: on the nine uncalibrated broadcast frames it goes 3/9 → 6/9
  alone, which is the regime the bank cannot touch. Framed as a **decoder over
  `square_dist`**, never an estimator — the raw distribution stays on the wire so a downstream
  legality smoother cannot double-count the prior (D1). Soft priors (two queens, bishop
  parity, castled motifs) deliberately excluded until confidences are calibrated (Next 3).
  **Decided 2026-08-07 (operator): it serves, banked and unbanked — and it is wired.**
  `/estimate` keeps `square_dist` + `placement` as the estimator produced them (§5 invariant
  intact, hub still publishes the raw event) and adds `result["prior"] = {placement, flips,
  residual}`; the snapshot page shows it as pipeline stage 7, *Impossible-board repair*, with
  every relabelled square listed, and takes the repaired board as its corrections baseline.
  The eval is prior-**on** by default (`--no-prior` isolates) because the harness scores what
  serves. Both full-position server paths carry it; the occupancy-only branch does not (no
  piece identity, no colour, nothing the constraints can read). `quarter_turns` is never passed
  in serving either — same circular guesser — so the back-rank rule sits out there too, with a
  comment at the call site saying why. Verified end-to-end: a two-white-king read comes back
  raw *and* repaired, and all four eval numbers reproduce (5/23 → 10/23 alone, 13/23 → 14/23
  ensemble). Label sidecars record `prior_flips`, un-rotated into the corner frame and locked
  by a property test against `rotate_placement`, so decoder-inherited truth stays *detectable*.
- **The back-rank constraint consumed a circular frame guess (found 2026-08-07).** The eval fed
  `decode()` the rotation from `autocalib.resolve_rotation`, which picks the frame that
  *minimizes pawns on ranks 1/8* — the very quantity `BACK_RANK_PAWNS` measures. So when the
  guess is right the constraint can never fire, and when it *can* fire that residual is
  evidence the guess is wrong. 16/21 boards get a nonzero turn; on `20260806-235526-0005`
  (counts `[2,1,2,1]` → `k=1`) it destroyed a correct 3.70-nat pawn at `a3` — provably a bad
  frame, since ground truth has a pawn there and no pawn can stand on a back rank. **Fixed**
  (`evaluate.py:526` now passes `None`; the eval prints "frame unknown here, so the back-rank
  pawn constraint sits out"): `quarter_turns: int | None`, `None` = frame unknown = skip the one
  rank-dependent constraint; the three count constraints are rotation-invariant and carry the
  whole win. Pass a real `int` only where orientation is *established* rather than inferred —
  the snapshot operator confirms it. **Caveat: the constraint therefore never fires in the
  eval, so it is the one rule with no evidence behind it on real boards.**

- ~~**Another session is mid-refactor of the crop geometry, and the suite is red**~~ —
  **finished and green, committed `f362f78` (2026-08-07).** The refactor was the
  `CropGeometry` ADT below; the red suite was its call sites mid-migration. **406 tests
  pass.** The warning stands though: two sessions had `server.py` open at once, which the
  working agreement forbids. Nothing was lost this time.

- **The `run_f` campaign** (2026-08-07) — set up the next training run and land the four
  things it needs. Three are done and committed; the run itself is in flight.
  1. **`chesscog_transfer` repaired** (was Next 7) — see Done.
  2. **Pose-aware crop geometry** (was Next 6) — see Done.
  3. **Temperature scaling** (was Next 3) — see Done.
  4. **Central-column patch pooling** (was Next 5) — exported to
     `models/dinov2_small_central.onnx` and verified (384-d, dynamic batch, cosine
     0.58–0.80 vs CLS, i.e. genuinely different features). **Not wired into any deployed
     bank**: the A/B is running now — a ChessReD bank at central pooling
     (`models/exemplars_chessred_central.npz`) vs the CLS one, the same uncontaminated
     cross-style protocol the +9/320 was measured on. Deployed bank untouched until it wins.
  - **State:** pose crop cache building (`data/external/_cropcache/pose_h1.9_r0.5_f1`,
    ~9 GB, 11 styles). Then `train.py --config configs/run_f.yaml` on the GPU — fp32, no
    AMP (the Turing NaN/slowdown learning). Targets to beat, both `run_e`: holdout
    `chessr_vlab_wooden` **0.898**, and the repaired real-photo set **0.819**.

## Done

- **M0 — walking skeleton.** `edge → inference → viewer` over WebSocket, `docker compose up`.
- **M1 — single-frame estimation.** Board detect + rectify on the edge; per-square 13-class
  distributions on the inference node. Exceeded: it reads real phone photos, not just clips.
- **Snapshot application** (built on M1, not a milestone). `GET /snapshot` on the viewer →
  `POST /estimate`: photo in, position out, zero taps. Corner auto-detect (`/detect`),
  drag-editable corners, orientation auto-correction, distribution correction panel,
  Lichess links. Mobile-verified.
- **Zero-shot recognition.** Seven training-free routes measured dead (~80% ceiling, all
  gated by piece *type*, never occupancy). A multi-style ResNet18 (`d16`, tall crops) reads
  an unseen set at 92.5% per-square with no calibration. Full write-up:
  [docs/reports/2026-07-18-zero-shot-campaign.md](docs/reports/2026-07-18-zero-shot-campaign.md).
- **Calibration as a booster.** kNN exemplar bank ⊕ trained model, blended by bank maturity.
  Superseded as a *number* by the honest eval below; the mechanism is what stands.
- **`run_e` deployed (2026-08-06).** The wave-2 11-style retrain replaced `d16` as `BR_MODEL`
  on the strength of the dim regime (**653 vs 600 / 768 squares**) and the deployed-ensemble
  comparison; it loses 6 squares model-alone on the bright set. Verified live.
  Detail: [docs/training-and-data.md](docs/training-and-data.md#how-run_e-was-decided-2026-08-06).
- **Guardrails from four bank-corruption incidents.** Opt-in persistence, orientation probe
  on `/calibrate`, piece-square agreement gate (422), `very_low_angle` / `low_light` warnings.
- **The labelling loop.** `Confirm as ground truth` → `POST /labels/{upload}`, stored beside
  the retained photo in the corner frame (autorotation undone, so truth never inherits a model
  guess); labels are scoring data only, never fed back into the bank. Supported by a
  server-backed Recent list (`GET /uploads`), a hover magnifier, and auto-rotation that
  reaches the whole page rather than just the board (commits `bf2ae0b`, `5c3fe59`).
- **Eval harness.** `uv run board-reader-inference eval` scores the estimator
  `_default_estimator()` builds — i.e. what actually serves — and leads with **perfect-board
  rate**. Reports error kinds, confusions, a reliability diagram, and warns when a bank is
  active. Write-up: [docs/measuring-accuracy.md](docs/measuring-accuracy.md).
- **The honest eval, on 23 confirmed labels (2026-08-07).** Ten more labels took the set from
  13 to **23**, all correct, and **all ten new ones post-date the bank's last write** (23:55),
  so they cannot be in it. The domain split does not soften with the bigger set — it hardens:

  | | boards | trained alone (`run_e`) | deployed ensemble |
  |---|---|---|---|
  | broadcast stream frames (1116×646, **uncalibrated**) | 9 | 3/9 perfect · 557/576 | **3/9 · 560/576** |
  | own phone photos (the set the bank **was** calibrated on) | 13 | 2/13 perfect · 798/832 | **9/13 · 826/832** |
  | one 748×396 stream frame (see caveat) | 1 | 0/1 · 55/64 | **1/1 · 64/64** |
  | all | 23 | 5/23 · 1410/1472 (95.8%) | **13/23 (56.5%) · 1450/1472 (98.5%)** |

  Read it as: **the bank is the entire margin, and only where it was calibrated.** On the
  calibrated set it turns 2/13 into 9/13 (+28 squares); across the nine same-format broadcast
  frames it moves **zero boards** and 3 squares. At 13 labels that row read 2/6 → 3/6 and
  looked like a modest win; at 9 boards it is flat. The headline 56.5% perfect is a
  calibrated-condition number. Caveat on the 748×396 row: that photo is from 22:58, *before*
  the three `/calibrate` writes, so unlike the rest it may actually be in the bank — its
  55→64 jump is the one gain here that cannot be trusted. Reproduce with
  `BR_MODEL=models/run_e/model.onnx [BR_SESSION_BANK=models/exemplars_session.npz] uv run
  board-reader-inference eval [--prior] data/uploads`.
- **Next 1 done — the label set nearly doubled (2026-08-07).** 13 → 23 confirmed labels; all
  ten new ones land after the last bank write and five of them needed corrections. This is
  what made the flat broadcast row visible.

## Next

1. **Decide whether the hard-constraint prior serves.** It is now the only thing measured to
   move the *uncalibrated* regime: trained-alone on the nine broadcast frames it goes
   **3/9 → 6/9 perfect** (557 → 573 squares), where the bank does nothing. On the ensemble it
   adds 7 squares on that same set and one board overall (13/23 → 14/23). Cost across all 23:
   fixed 21 · broken 3 alone, fixed 9 · broken 1 with the bank — positive in every cut
   measured. Remaining call: ship it as a decode step behind the raw `square_dist` (D1), and
   decide whether the viewer shows the repaired board, the raw one, or both.
2. **Attack the type confusions.** On the 23-board set, trained-alone errors are 53 type / 5
   hallucination / 4 colour, led by `p->b` (11) then `b->k` · `p->r` · `q->k` · `Q->K` (5
   each) — the same silhouette family, with pawn→bishop now clearly the single worst. The
   ensemble cuts it to 14 type / 7 hallucination / 1 colour, and what it leaves is different
   in kind: `empty->p` (3) and `empty->P` (2) are now a third of its remaining errors, i.e.
   hallucinated pieces on empty squares — the failure mode Next 6's crop attribution predicts.
3. **Calibrate confidence.** Trained-alone ECE is fine (0.016) and still useless at board
   level (perfect 0.975 vs imperfect 0.948, on 5 perfect boards). The ensemble is badly
   under-confident (**ECE 0.255** — every bin from 0.4 to 0.9 is *100%* correct while claiming
   45–85%) yet its board confidence separates: perfect 0.789 vs imperfect 0.654. Both patterns
   held from 13 boards to 23, so this is structure, not noise. Temperature-scale the ensemble
   and re-check the separation.
4. **M2 — live position stream over a full recorded game** (the headline demo). The whole
   recognition stack so far is single-frame; M2 is adaptive forwarding + stability + streaming
   through hands and transitions. This is the actual next milestone.
5. **Central-column patch pooling** for the DINOv2/kNN path (+9 squares cross-style, measured,
   never shipped).
6. **Pose-aware crop boxes — never tried, don't confuse it with the headroom sweep** (idea
   logged 2026-08-07). What *was* tried is a **scalar** sweep of `photo_square_boxes`'
   `headroom`/`side_pad` (hr 0.15 → **1.6** → 1.8, sp 0.15/0.25; `d16` won, `d18` lost on real
   photos). Every variant is still an *axis-aligned bbox extended along fixed image axes* by a
   multiple of the square's projected width. Searched code, docs, learnings and all 23 commits
   on 2026-08-07: no 3D/camera-pose crop work exists. The untried idea: decompose the board
   homography (`cv2.decomposeHomographyMat`, approximate `K` from image size) into R,t and
   project a **piece bounding cylinder** standing on each square centre. Three defects it
   addresses, worth separating: (a) *coverage* — piece pixel-height depends on camera
   elevation, so a fixed multiple of a foreshortened square width decapitates at steep angles
   (the known steep+dim weakness `hr1.6` only patched); (b) *attribution* — a tall box also
   contains the piece on the square behind it, the likely `empty->p` source; a cylinder makes
   occlusion explicit so contaminated crops can be flagged rather than guessed; (c) *effective
   resolution* — a ~2.75:1 box padded square and resized to 224 leaves the piece in a thin
   strip. Cost: crop geometry is baked into the `.onnx.meta.json` sidecar and the exemplar
   bank, so this is a retrain + rebank + re-eval campaign, not a quick fix. Cheaper fallback
   for M2 only: a video stream gives an empty-board background model, which classical
   segmentation cannot get from a single frame. A piece *detector* (base-keypoint → square via
   inverse homography) is the bigger alternative; new model family, same ChessReD licensing.
7. **Repair or replace the `chesscog_transfer` holdout** — it is geometrically broken
   (0.597 self-kNN) and made a good run look like a failure once already.
8. **Optional: rebuild the session bank against `run_e`.** Not required — the crop geometry is
   identical (hr1.6/sp0.15), so the existing bank stays valid, and it is validated at 638/640
   with `run_e`. `tools/rebuild_session_bank.py` now points at `run_e` for whenever it is next
   run for other reasons.

## Open questions

- ~~**The eval's truth is partly the model's own answer**~~ — **closed 2026-08-07.** The
  worry was that 11 of the first 13 labels saved with `corrections: 0` meant the operator had
  rubber-stamped the displayed read. They did not: the labels are confirmed correct, so the
  numbers are real. What actually explains the high read rate is **the session bank**, not
  soft labelling — see the domain split in Done. Both contamination questions are closed for
  the 22 post-bank-write boards; only the 748×396 frame from 22:58 remains unresolved.
- **Calibration is not monotone (found 2026-08-06).** `20260806-231719-0000` read 64/64 live
  at 23:17 and was confirmed as read; re-scored after the 23:28 and 23:33 `/calibrate` writes
  it is 63/64 (`e6 empty->p`) — and trained-alone still gets it right, so the *bank* caused
  the regression. Adding exemplars can cost squares elsewhere, and nothing measures that
  before a write lands. Fifth entry in the bank-incident family.
- **Confidence is measured and it is not usable — but the ensemble may be salvageable.**
  Trained-alone has no board-level signal at all (perfect 0.978 vs imperfect 0.939). The
  ensemble is systematically under-confident (ECE 0.226) yet ranks boards roughly right
  (0.801 vs 0.678). DESIGN §9 flags this as a top risk because downstream smoothing consumes
  confidences; a consumer trusting them today would still be misled. See Next 3.
- **The 638/640 baseline can no longer be reproduced.** Five of the ten curated boards
  (`up_20260718_*`) point at `data/uploads/` photos that have since been deleted, so
  `ground_truth.json` now loads only `img_0565`–`img_0569`. Either recover those photos or
  prune the dead entries so the curated set stops silently shrinking under us.
- Is `run_e`'s bright-board regression real or eval noise? Still open, and now on thinner
  ground: the 10-board bright set it was measured on is down to 5 loadable boards.
- Does `stability` earn its keep (DESIGN D3)? Undecided until M2 produces a stream to judge.
- Licensing: the deployed model is trained on ChessReD (CC BY-NC-SA) + murky-license
  geometry — personal use only. A clean-data retrain is the path if that ever changes.
- No lint gate. `ruff` is a system binary with no project config; ~109 findings and 41
  unformatted files sit unaddressed. Adopt it deliberately or drop it — currently it is
  neither.
