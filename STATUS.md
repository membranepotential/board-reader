# STATUS — board-reader

*Canonical dashboard. Last reconciled 2026-08-06 against `git log`, `scratch/runs/`, `data/eval/`.*
Design decisions live in [docs/DESIGN.md](docs/DESIGN.md); this file is state, not design.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- **Snapshot quality push** — make the snapshot app read a board reliably before M2.
  Landed: ground-truth capture + the eval harness (below). Next: act on what it measured.
- **snapshot.html rewrite** (yours, uncommitted) — the page is now a stage-by-stage pipeline
  inspector; `test_viewer.py` was updated to match. Not committed yet.
- **Snapshot orientation + labelling UX** (uncommitted, 2026-08-06) — auto-rotation now
  reaches the whole page, not just the board; hover magnifier; server-backed Recent list.
  Detail in Done. Still uncommitted, along with the new `Makefile` (`make dev` = the
  deployed env) and the 13 labels the labelling session produced.
- **Act on the first honest eval** (opened 2026-08-06). The number exists now (Done below).
  Next: a *blind* second labelling session (see Next 1) so the truth stops being the
  ensemble's own output, then temperature scaling.

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
- **First generalisation number, on boards the bank never saw (2026-08-06).** A labelling
  session produced **13 confirmed labels** in `data/uploads/*.label.json`, none of them in
  the bank. They split into two domains, and the split is the whole story:

  | | boards | trained alone (`run_e`) | deployed ensemble |
  |---|---|---|---|
  | broadcast stream frames (1116×646, unseen set/camera/lighting) | 6 | 2/6 perfect · 364/384 | **3/6 · 376/384** |
  | own phone photos (the set the bank was calibrated on that evening) | 7 | 2/7 perfect · 434/448 | **6/7 · 447/448** |
  | all | 13 | 4/13 · 798/832 (95.9%) | **9/13 · 823/832 (98.9%)** |

  Read it as: **calibration is what carries the product** (2/7 → 6/7 on the calibrated set),
  and **uncalibrated it is roughly a coin flip per board** (3/6). The contaminated 638/640 =
  9/10 was the right *shape* — the ensemble does win — but the honest perfect-board rate is
  69%, not 90%, and 50% on a set nobody calibrated on. Reproduce with
  `BR_MODEL=models/run_e/model.onnx [BR_SESSION_BANK=models/exemplars_session.npz] uv run
  board-reader-inference eval data/uploads`.

## Next

1. **A second, *blind* labelling session.** 13 labels exist, but 11 of them were confirmed
   with zero corrections against the ensemble's own displayed answer — where operator and
   model agree on a wrong square, the eval scores it correct, so today's 823/832 is an upper
   bound on the ensemble specifically. Next session: read the position off the *board* (or
   the source video) before looking at the overlay, and cover a second lighting. Target ~20
   more, and keep the two domains separable so whole sessions can be held out.
2. **Attack the type confusions.** On the honest set, trained-alone errors are 27 type / 4
   hallucination / 3 colour, led by `p->b` (4) and `q->k` (4) then `Q->K` (3) — the same
   silhouette family as before, and the queen/king pair is back alongside pawn→bishop. The
   ensemble erases most of them but leaves `Q->K` (2) and `empty->p` (2).
3. **Calibrate confidence — now unblocked.** Trained-alone ECE is fine (0.021) but useless at
   board level (perfect 0.978 vs imperfect 0.939). The ensemble is badly under-confident
   (ECE 0.226 — every bin from 0.3 to 0.8 is *100%* correct while claiming 35–75%) yet its
   board confidence does separate: perfect 0.801 vs imperfect 0.678, and the three worst
   boards are at 0.56/0.64/0.69. Temperature-scale the ensemble on these 13 and see whether
   the separation survives; the bins say there is real signal being thrown away.
4. **M2 — live position stream over a full recorded game** (the headline demo). The whole
   recognition stack so far is single-frame; M2 is adaptive forwarding + stability + streaming
   through hands and transitions. This is the actual next milestone.
5. **Central-column patch pooling** for the DINOv2/kNN path (+9 squares cross-style, measured,
   never shipped).
6. **Repair or replace the `chesscog_transfer` holdout** — it is geometrically broken
   (0.597 self-kNN) and made a good run look like a failure once already.
7. **Optional: rebuild the session bank against `run_e`.** Not required — the crop geometry is
   identical (hr1.6/sp0.15), so the existing bank stays valid, and it is validated at 638/640
   with `run_e`. `tools/rebuild_session_bank.py` now points at `run_e` for whenever it is next
   run for other reasons.

## Open questions

- **The eval's truth is partly the model's own answer (found 2026-08-06).** 11 of the 13
  confirmed labels were saved with `corrections: 0` — the operator agreed with the displayed
  read. Squares where both were wrong are scored correct, so the honest 823/832 is still
  biased toward the ensemble. This is the *label protocol*, not the harness; Next 1 is the
  fix. The old contamination question is closed: the bank has never seen these 13 boards.
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
