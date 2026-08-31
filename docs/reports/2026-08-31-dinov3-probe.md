# DINOv3 frozen-ceiling probe — NO-GO: the ceiling moves 3 points, not enough to build on

_2026-08-31 · closes [[ideas]] §2 ("the cheap probe, do first regardless") —
done, and it does not change what we build. Official Meta weights
(`facebook/dinov3-vits16-pretrain-lvd1689m`, license-gated, converted to ONNX
ourselves via the generalized `tools/export_dinov2_onnx.py`, which now reads
patch size / embed dim / register-token count from the model config — DINOv3's
4 register tokens would otherwise have been averaged into central pooling).
Detail: `scratch/experiments/dinov3_probe/report.md`._

**Preflight held:** DINOv3's `preprocessor_config.json` specifies the identical
ImageNet normalization to DINOv2 — the A/B is not sandbagged. **Sanity gate
held:** re-embedding with DINOv2 from scratch reproduced the archived
central-pooling numbers exactly (248/255/294), so every delta is the backbone.

## Result (pooled 10 boards, 640 squares, 400-board ChessReD bank)

| variant | acc | occupancy | piece-id on occupied |
|---|---|---|---|
| DINOv2 CLS | .803 | .975 | .622 |
| DINOv2 central | .808 | .992 | .595 |
| **DINOv3 CLS** | **.827** | .975 | **.674** |
| DINOv3 central | .808 | **.994** | .591 |

1. **~80 % → ~83 %.** +2.4 pp at the shipped bank size; McNemar p = 0.137 on
   the paired squares (37 vs 52 of 89 disagreements — differently wrong, not
   uniformly better). Same order as levers already shrugged at. 95 of 291
   occupied squares still get the wrong piece.
2. **Central pooling does not compose with DINOv3.** The +9 was
   DINOv2-specific; under DINOv3 it lands 12 squares behind CLS and gives back
   the piece-id gain. Its durable property — best occupancy embedding in the
   grid — survives as an occupancy-head asset only.
3. **The deployed same-style path gains nothing and can regress** (LOBO:
   v3 CLS 283 vs v2's 294). A bank-embedder swap costs an ONNX change plus
   every bank rebuild for an unmeasurable point estimate. Not worth it.
4. **Worth banking:** DINOv3 CLS is the first frozen encoder measured here
   that cuts cross-style piece-id errors (110 → 95, colour errors to 0)
   instead of trading them against occupancy — a mildly better *fine-tuning
   initialisation* for a future retrain, nothing more. The exports
   (`models/dinov3_vits16*.onnx`) are kept for that day.

## Consequence for the roadmap

The recommended sequence in [[ideas]] loses its first step and gains nothing to
wait for: **the data engine (idea A, [[2026-08-31-autolabel-engine|plan]]) is
now unambiguously the front of the queue.** No frozen encoder reaches the
board-level reliability target; only real labeled data at scale plausibly does.
