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
  Detail in Done. Next: commit, then act on item 1 below (collect clean labels) — the
  save-labelling button and the all-photos list exist to make that session practical.

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
- **Calibration as a booster.** kNN exemplar bank ⊕ trained model, blended by bank maturity:
  638/640 on the labeled eval vs 599 (trained alone) / 579 (`v1`). **That 638 is
  not a generalization number** — see Open questions.
- **`run_e` deployed (2026-08-06).** The wave-2 11-style retrain replaces `d16` as `BR_MODEL`.
  It wins the regime it was built for — **653 vs 600 / 768 squares** on the dim calibrate
  boards, and **+39** there against a bank that has never seen them — and wins the deployed
  ensemble on `data/eval` (**9/10 perfect, 638/640** vs 7/10, 633). It *loses* 6 squares
  model-alone on the bright set, which is why the ensemble comparison was the deciding one.
  Verified live: dim boards read 64/64 at conf 0.86 through a real server.
  Detail: [docs/training-and-data.md](docs/training-and-data.md#how-run_e-was-decided-2026-08-06).
- **Guardrails from four bank-corruption incidents.** Opt-in persistence, orientation probe
  on `/calibrate`, piece-square agreement gate (422), `very_low_angle` / `low_light` warnings.
- **Ground-truth capture.** The snapshot correction panel used to discard its labels on
  reload; **Confirm as ground truth** now POSTs them to `/labels/{upload}`, stored beside the
  retained photo in the corner frame (autorotation undone, so truth never inherits a model
  guess). Labels are scoring data only — nothing feeds them back into the bank.
- **Auto-rotation reaches the whole page (2026-08-06).** A read posted with `autorotate=1`
  came back rotated, but only the board knew: the corner handles, corner table, rectified
  preview and the inspector's crop thumbnail all still described the pre-rotation frame.
  Two halves, both fixed: the page now absorbs the server's `k` into its own label rotation
  (`rot -= k`, fresh reads only), and `/estimate`'s `?debug=1` boxes/mosaic are permuted by
  the same `np.rot90` as `square_dist` — sampled from the original corners, since `headroom`
  and `side_pad` extend along fixed *image* axes and re-deriving from shifted corners would
  sample the wrong pixels. Verified live on a real 90°-rotated board: handles land right,
  d8's crop shows the black queen the board shows, and the saved label still stores truth in
  the posted-corner frame.
- **Snapshot labelling UX (2026-08-06).** "Save labelling" is a real button under the board
  (disabled *with a reason* when the node retains no photos, instead of vanishing); Recent
  positions lists every photo the node still holds via new `GET /uploads` +
  `GET /uploads/{id}/photo`, not just this tab's reads — clicking one loads it and reads it
  for real. Opt-in hover magnifier that draws the corner handles inside the loupe, and
  handles shrunk 44→34px so they stop hiding the edges they align to.
- **Eval harness.** `uv run board-reader-inference eval` scores the estimator
  `_default_estimator()` builds — i.e. what actually serves — and leads with **perfect-board
  rate**. Reports error kinds, confusions, a reliability diagram, and warns when a bank is
  active. Write-up: [docs/measuring-accuracy.md](docs/measuring-accuracy.md).

## Next

1. **Collect clean labels.** The harness reports `10 curated, 0 confirmed` — every labeled
   board we own is in the bank, so there is still no honest generalisation number. Serve with
   `BR_SAVE_UPLOADS=data/uploads`, photograph ~20 boards across two sessions/lightings, and
   confirm each. Then re-run `eval` and hold out whole sessions.
2. **Attack pawn→bishop.** The failure signature moved with the model: `d16`'s was queen→king
   (10 of 35 errors), `run_e`'s is **`p->b`, 14 of 41**, with `b->k` and `p->r` next. Type
   errors are 37 of its 41 — occupancy is fine, the crop is telling us the wrong silhouette.
   Worth checking whether wave-2's dim renders taught it to over-read pawn heads as mitres.
3. **Calibrate confidence.** Still no board-level signal: `run_e` alone scores its ten boards
   at 0.950 with *zero* of them perfect, and the ensemble is under-confident everywhere
   (ECE 0.142). Temperature scaling on held-out labels is the cheap first move — blocked on
   item 1.
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

- **We still have no clean eval data (found 2026-08-06, now measurable).** The bank's base is
  built *from* `data/eval/ground_truth.json`, so it has seen 9 of the 10 eval boards. Measured
  with the new harness: trained alone **0/10 perfect** (599/640), deployed ensemble **9/10**
  (638/640) — and that gap is the bank recognising its own exemplars. `up_20260718_0001`, the
  one board it never saw, is the ensemble's worst at 62/64. Note the trained-alone perfect
  rate went 2/10 (`d16`) → 0/10 (`run_e`) even as per-square accuracy improved elsewhere: on
  ten boards that is noise, but it is the number that matters and nobody has moved it. The
  harness prints this warning itself; the fix is labels, not tooling. See
  [docs/measuring-accuracy.md](docs/measuring-accuracy.md).
- **Confidence is measured and it is not usable.** Trained model: all ten boards average 0.950
  and none of them is right — no board-level signal whatsoever, and mid-range squares are badly
  overconfident. Ensemble: under-confident everywhere, ECE 0.142. DESIGN §9 flags this as a top
  risk because downstream smoothing consumes confidences; a consumer trusting them today would
  be misled.
- Is `run_e`'s bright-board regression real or eval noise? 10 boards is thin — it cost 6 squares
  and 2 perfect boards model-alone, and the ensemble more than covered it, but that is a bet on
  the bank always being there.
- Does `stability` earn its keep (DESIGN D3)? Undecided until M2 produces a stream to judge.
- Licensing: the deployed model is trained on ChessReD (CC BY-NC-SA) + murky-license
  geometry — personal use only. A clean-data retrain is the path if that ever changes.
- No lint gate. `ruff` is a system binary with no project config; ~109 findings and 41
  unformatted files sit unaddressed. Adopt it deliberately or drop it — currently it is
  neither.
