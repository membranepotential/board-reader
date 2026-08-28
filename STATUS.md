# STATUS — board-reader

_reconciled: 2026-08-28 @ `c0138f0`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md), detail in `git log` + `scratch/experiments/*/report.md`; this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- **The 08-27 night wave is COMPLETE** (see Done). Estimator, scheduler,
  stabilizer, framing, recorder, harness all landed, reviewed, and
  replay-verified; suite 1048 Python + 417 vitest + 50 e2e, all green.
- tall-crop-retrain — small experiments first, then scale; improvement report
  at the end — starting (main session).
- docs-wrap (fork) — trial report + baseline docs — in progress.
- ui-mockups — awaiting user review in `docs/mockups/` (user: redo later as
  clickable simulations, not storyboards).

- retrain-E7 — masked-crop retrain (gray fill, both members; screen gate →
  2 scoring runs → 4-take replay + curated gates) — implementer training,
  owns the GPU.

## Done (recent)

- 08-28 retrain-E6-redo (fixed rule): d8 q<->k win REAL (25→9, byte-identical
  post-fix); b1/b5 hallucination tax REAL OOD (not the rule bug); pooled
  inference-only net +9 settled / +11 perfect, fill choice take-dependent
  (blur pooled-wins, gray kept for training). Masking-alone not shippable;
  retrain carries the load. summary.json updated, provisional kept.
- 08-28 review: BR_CROP_MASK commit reviewed — nearness rule was image-row,
  inverted under the rot90 probe and masked ~12 % of rectified crops
  spuriously; fixed to camera depth + tie epsilon (`a6b4463`), env typo now
  warn-once not per-frame raise, fill hygiene. E6's provisional numbers
  (incl. some of its OOD tax) superseded by the e6f rerun.
- 08-28 retrain-E6: occlusion masking (BR_CROP_MASK, committed, 1085 green) —
  mechanism confirmed: 08-13 settled +30, perfect 6→22/31 (d8 q<->k 25→9
  wrong); small OOD tax on non-pathological takes (model never saw masks) →
  E7 masked-crop retrain. Report: scratch/experiments/e6_mask_crops/report.md.
- 08-28 retrain-E5: bleed-demotion decoding rule — **NO-GO**, rigorously:
  4,480-config sweep, no config clears 0-broken, no safety gap (correct reads
  inside every firing band), LOTO negative 4/7, breaks are pawn chains, fixes
  mostly redundant with the prior. The signal is in the pixels → E6 masks
  crops upstream. Report: scratch/experiments/e5_bleed_rule/report.md.
- 08-28 retrain-E4: bleed attribution measured — 44 % of pooled raw residual
  is neighbor-prism bleed @0.30 (08-28 take: 66 %, +6.6 pp oracle ceiling);
  overlap alone is ANTI-discriminative (correct squares overlap more); the
  signal is identity coincidence. c5→b5 and c6→b6 confirmed; d8's residual and
  R->P are largely bleed. Report: scratch/experiments/e4_bleed_attribution/.
- 08-28 retrain-E3: leave-session-out crop fine-tune (`finetune_crops.py`,
  tested) — clean cross-session gains are partial: 08-13 settled 97.5→98.2 %
  (perfect 6→13/31, q<->k −28 %); 08-28 degrades 93.3→90.8 % (missed up). E2's
  sweep was mostly leakage. Fine-tune helps type residual, can't touch
  drift/bleed. Report: scratch/experiments/e3_crops_lso/report.md.
- 08-28 retrain-E2: start-position fine-tune (7 sessions) vs deployed cascade,
  replay-gated — settled 08-13: 97.5%→100% (q<->k erased, 31/31 perfect);
  settled 08-28: 93.3%→92.6% (hallucinations 14→23). Caveat: harvest included a
  start frame of each scored take (same-session leakage) → E3 goes
  leave-session-out. Report: scratch/experiments/e2_calib_finetune/report.md.
- 08-28: **proof trial done and scored** (docs/reports/2026-08-28-proof-trial.md,
  recording 20260828-111421). Zero-tap checklist passed: auto-calibrate fired
  (bank 768→1024 mid-take), re-detector adopted real drift at 78 s with zero
  false adoptions, 36 reads/243 frames, backstopped 0, latency p50 1.55 s, no
  OOM. Settled 91.9 % per-square, 4/22 perfect; per-segment decline tracks the
  drift; post-adoption residual is recognition, not framing. **Demotion A/B**
  (replay-vs-replay): "0 broken" is dead — the rule trades eaten `b`s against
  `empty->b` hallucinations, net +4 settled; keep it, revisit with the retrain.
- 08-28: **curated baseline re-pinned** — the mid-trial calibrate invalidated
  the morning's 768-bank number; baseline now pins its own bank snapshot
  (`data/eval/bank-baseline-20260828.npz`): **304/320, 0/5 perfect**, ECE 0.34.
- 08-28: **curated set pruned to 5 boards** (dead photos removed, survivors
  moved into `data/eval/images/`; the only bank-held-out board was among the
  dead). The 638/640 headline is historical.
- 08-28: **repo pushed public** (pre-rewrite SHAs: `docs/commit-rewrite-*`,
  `tools/check_commit_refs.py`); **own footage confirmed** (trial recordings are
  the source; broadcast sourcing closed); **b-demotion display stays as-is**
  (user: recognition is the ambition, confidence marking clutters the UI).
- **08-27 night wave** (detail: `git log 866653f..9167dfd`,
  docs/reports/2026-08-27-*): replay harness; GPU arena ratchet fixed;
  b-demotion shipped (+2–10 pp perfect); framing re-detect (drift −58 %);
  burst-end read scheduling (~5× fewer GPU reads); settled-stream coherence
  (impossible boards 233→0); per-session recorder; zero-tap calibrate
  root-caused + e2e-pinned; test cascade filled (1048 py / 417 vitest / 50
  e2e). All takes scored; every non-trivial commit adversarially reviewed.
- 08-27 (day): adaptive meter floor, label-rotation auto-absorb, portrait
  block, sharpness-preferring inbox, latency batch fix. Earlier: `git log`.

## Next

1. **Tall-crop retrain — GO (user 2026-08-28)**: small experiments first, then
   scale; consider smart/pose-derived crops (prior work exists); iterate on the
   test data; deliver an improvement report at the end. ~78 % of residual errors
   sit outside the top-2 ceiling; take 4 and 08-13's d8 `q->k` are pure
   recognition errors. In progress (main session).
2. **Estimator throughput / latency** — profile the cascade+ensemble read
   (~1.3 GPU-s/read measured in the sweep) and attack the biggest term (bank
   embed batching?). Faster reads shrink settle lag and the guard window both.
   Burst-end scheduling made this cheaper to afford but reads are still the cost
   unit.
3. **Per-square motion masks** (deferred from the scheduler design): regional
   motion feeding the stabilizer for hovering hands; only worth it if the proof
   take shows hand-hover errors surviving the new stack.
4. **Run M2 over `ccbf-r5`** with `--forward-original --corners` (the warp path
   is unusable at this angle) and judge the stream at that angle.
5. **Give a user's `tuned.onnx` a deploy story** — per-operator artifact, no
   compose path today.
6. **Attack the residual type confusions** — `p->b` leads trained-alone; a third
   of the ensemble's residual is `empty->p`/`empty->P`, a different animal.
   Feeds item 1.
7. **Decide whether calibrated confidences go on the wire** — a decision, not an
   experiment: argmax is invariant, DESIGN §9 says the downstream smoother wants
   them.
8. **Crop attribution / occlusion** — the prism is a *bounding* volume: the c6
   knight's lean hallucinated a b6 rook and corrupted b7 all trial.
   Base-weighted attribution over bounding-prism membership. Plus detection
   follow-ups (`73efd60`, non-blocking). The 08-28 demotion A/B pins both sides
   of the `b` trade on this (c5's crop bleeding into b5).
9. **Client-side session video** (user proposal 08-28): `MediaRecorder` on the
   camera track, chunked upload during the session, stored beside the recording
   — edge/detector debugging plus trivial truth-authoring (~25–50 MB/take).

## Open questions

- **Position stream has no consumer** — user decision 2026-08-28: deferred; the
  position stream *is* the product output for now. (`stability` earns its keep
  internally: forwarder + read scheduler run on it.)
- **Confidence is not usable today** — under-confident ensemble, no board-level
  signal. User 2026-08-28: calibrated confidences would be nice, **not a priority**.
- **Bank writes are not monotone** — adding exemplars via `/calibrate` can regress
  squares on *other* boards (2026-08-06: a confirmed 64/64 board re-scored 63/64
  after two later writes; trained-alone still read it right, so the bank caused
  it). No regression gate on bank writes exists. Open issue.
