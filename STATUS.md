# STATUS — board-reader

_reconciled: 2026-08-13 @ `9fe975f`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md), detail in `git log` + `scratch/experiments/*/report.md`; this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- phone-live-test — trial 08-13 evening **failed twice over**: the phone camera froze
  (nothing recovered it) and its stream delivered **zero frames** to the node — which was
  additionally invisible because CLI `serve` built the app at import time and silently ran
  **without ingress recording** all along. Fixed in-session (uncommitted): camera stall
  watchdog + track recovery on /camera; prior repair now rides the live stream (two-kings
  reproduced from the 18:22 calibration frame and repaired); recording default actually
  works, plus events.ndjson, per-read INFO logs, refused-frame counter, and a phone→viewer
  client-log sink. Next: **restart `make dev`**, re-trial on the phone, then read
  recordings/, client-logs/ and the serve log.

## Done (recent)

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

1. **Run M2 over `ccbf-r5`** with `--forward-original --corners` (the warp path is unusable at
   this angle), judge the stream, then settle D3 — does `stability` earn its keep?
2. **Give a user's `tuned.onnx` a deploy story** — per-operator artifact, no compose path today.
3. **Attack the residual type confusions** — `p->b` leads trained-alone; a third of the
   ensemble's residual is `empty->p`/`empty->P`, a different animal.
4. **Decide whether calibrated confidences go on the wire** — a decision, not an experiment:
   argmax is invariant, DESIGN §9 says the downstream smoother wants them.
5. **`/estimate` and `/calibrate` still block the event loop** (`server.py:743`, `:875`) — same
   class as the `/detect` bug already fixed with `asyncio.to_thread`. Both stall ingest acks.
6. **Crop attribution / occlusion** — the prism is a *bounding* volume, so a piece behind can
   fall inside the crop unflagged. Plus detection follow-ups (`3f93e57`, non-blocking).

## Open questions

- **No real footage sits inside the good angle band**, and a ~40-candidate search found none:
  broadcast cameras frame players, not boards. Own footage is a sourcing requirement.
- **Nothing consumes the position stream** — deliberate (D1/D3 put smoothing in the consumer),
  but that consumer does not exist, so `stability` is measured and read by nobody.
- **Confidence is not usable today** — no board-level signal; the ensemble is under-confident.
- **Bank writes are not monotone** — a 64/64 board lost a square after two `/calibrate` writes.
- **The 638/640 baseline is unreproducible** — 5 of 10 curated boards point at deleted photos.
- Licensing: ChessReD (CC BY-NC-SA) + murky geometry → personal use only.
