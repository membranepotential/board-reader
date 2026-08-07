# STATUS — board-reader

_reconciled: 2026-08-07 @ `5a6a19a`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md); this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

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
4. **M2 — live position stream over a recorded game.** The headline demo and the actual next
   milestone: adaptive forwarding, stability, streaming through hands and transitions.
5. **Central-column pooling A/B.** `dinov2_small_central.onnx` is exported and verified; build
   `exemplars_chessred_central.npz` and score it under the uncontaminated cross-style protocol the
   +9/320 came from — no deployed bank moves before it wins.
6. **Crop attribution / occlusion** — pose fixed coverage and resolution, not attribution: the box is
   still the prism's *bounding* box, so a piece behind can fall inside it unflagged.

## Open questions

- **Bank writes are not monotone.** A board that read 64/64 lost a square after two `/calibrate`
  writes, and trained-alone still gets it right. Nothing measures that before a write lands.
- **Confidence is not usable today.** Trained-alone has no board-level signal (0.978 vs 0.939) and the ensemble is under-confident, though it ranks boards roughly right. Top risk in DESIGN §9.
- **The 638/640 baseline is unreproducible** — 5 of 10 curated boards point at deleted photos, and
  the same erosion makes `run_e`'s bright-board regression unjudgeable. Recover them or prune.
- Does `stability` earn its keep (D3)? Undecided until M2 produces a stream to judge.
- Licensing: ChessReD (CC BY-NC-SA) + murky geometry → personal use only.
- No lint gate: `ruff` has no project config, ~109 findings unaddressed. Adopt it or drop it.
