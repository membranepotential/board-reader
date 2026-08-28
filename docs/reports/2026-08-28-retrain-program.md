---
tags: [board-reader/experiments]
---

# 2026-08-28 — the retrain program (E1–E7): from "tall-crop retrain?" to masked-crop retraining

One day, seven gated experiments. The question going in (STATUS Next 1) was
"do the tall-crop retrain"; the answer coming out is sharper: **most of the
retrainable residual is neighbor-prism occlusion bleed, decode-time rules
cannot separate it, crop masking fixes it in the pixels, and a masked-crop
retrain matches the synthetic baseline while beating it on real photos.**
Detail per experiment lives in `scratch/experiments/e{2..7}_*/report.md`
(gitignored, on this machine); this report is the citable summary.

Frozen baselines for every comparison (all replay-vs-replay or pinned):
curated 5-board set **304/320, 0/5 perfect** (pinned bank
`data/eval/bank-baseline-20260828.npz`, artifact `data/eval/baseline-20260828.json`);
gate replays `e2-control` (08-28, 08-13 takes) and `e6f-control` (two 08-27
takes), deployed cascade ⊕ pinned bank, burst-end.

## The ladder

- **E1 — data**: every scored trial recording exported as labeled crop styles
  (`eval-session --export-crops`): 7 sessions, ~3,450 real crops
  (`data/external/trial_*`). Real deployment-domain training data exists for
  the first time.
- **E2 — start-frame fine-tune** (7 sessions): settled 08-13 97.5→**100 %**
  (31/31 perfect) but the harvest contained a start frame of each gated take —
  session leakage. 08-28 slightly worse. Personalization is powerful; the
  measurement was contaminated.
- **E3 — leave-session-out crop fine-tune** (new `tools/train/finetune_crops.py`,
  tested): the clean number — 08-13 settled 97.5→98.2 % (perfect 6→13/31,
  q↔k −28 %); 08-28 *degrades* 93.3→90.8 %. Most of E2's sweep was leakage;
  fine-tuning helps type residual, cannot touch drift/bleed. Bonus: no
  foreign-board forgetting (curated 304→308/320).
- **E4 — bleed measurement**: 44 % of the pooled raw residual (66 % on the
  08-28 take) sits under a closer neighbor's projected prism @0.30 overlap.
  Overlap alone is ANTI-discriminative (correct squares overlap MORE); the
  signal is identity coincidence. d8's q↔k residual and the R→P class are
  substantially bleed. c5→b5 and c6→b6 confirmed (overlap 0.6–0.97).
- **E5 — bleed-demotion decode rule: NO-GO.** 4,480-config sweep, zero configs
  reach 0-broken, no safety gap (correct reads inside every firing band), LOTO
  net-negative 4/7, breaks are pawn chains, fixes redundant with the prior.
  The discriminating information is in the pixels, not the digest.
- **E6 — occlusion masking at inference** (`BR_CROP_MASK`, shipped `59fafb2`,
  nearness fixed to camera depth after review, `a6b4463`): the 08-13 far-rank
  d8 lock breaks (settled q→k 25→9, perfect 6→22/31 gray / 19 blur) — but an
  OOD tax appears on takes without the pathology (the model never saw masks).
  Pooled inference-only: +9 settled, +11 perfect. Mechanism proven, not
  shippable alone.
- **E7 — masked-crop retrain** (`run_g`/`run_h` configs; cropcache/evaluate/
  export grew a mask dimension): screen gate passed (0.876 @e8 vs run_e band
  0.835–0.859). **run_g (bbox+mask) complete: holdout 0.8983 ≡ run_e's 0.898,
  chesscog_transfer 1089/1280 ≈ 0.851 vs run_e 0.819 (+3.2 pp)** — on the one
  real-photo set nothing in the stack ever saw, scored *unmasked*: masked
  training regularizes beyond its own serving domain. `run_h` (pose+mask) and
  the replay/curated gates are pending (see Continuation).

## The incident

Mid-E7 the machine hard-froze and was force-rebooted: concurrent cache
building + torch training (resident multi-GB dataset) + the rest of the box
exhausted 31 GiB RAM (kernel page-allocation failures, swap write errors
14:14–14:16). Orchestration error — heavy jobs were allowed to overlap.
Rules adopted for every future run: one memory-heavy job at a time,
`systemd-run --user --scope -p MemoryHigh=12G -p MemoryMax=16G`, DataLoader
`num_workers: 4`, a low-memory watchdog. System-level mitigations (earlyoom,
zram sysctls) are the operator's, tracked outside this repo. The crash cost
one screen-run export (0-byte files, retrained artifacts unharmed) and
`run_g` was re-run to completion under the caps.

## Continuation (cloud GPU)

The remaining work, in order — all commands from repo root, torch venv per
`tools/train/README.md` §Setup:

1. `tools/train/run_h.sh` — builds the masked pose cache (partial locally:
   only chessred + chessrender360 of 11 styles; `cropcache.py` is idempotent
   and re-runs safely) and trains run_h (~40 min on a GTX 1650; faster on
   anything cloud). Needs `data/external/` sources and `models/dinov2_small.onnx`.
2. Replay gates (GPU, needs `recordings/` + `models/` + the pinned bank):
   `BR_CROP_MASK=gray BR_MODEL=scratch/runs/run_g/model.onnx BR_MODEL_B=scratch/runs/run_h/model.onnx BR_SESSION_BANK=data/eval/bank-baseline-20260828.npz uv run board-reader-inference replay recordings/<take> --policy burst-end --score --name e7-masked`
   over takes 20260828-111421, 20260813-193308 (`--restamp-stability`),
   20260827-181802, 20260827-175539. Compare vs `e2-control`/`e6f-control`
   and the `e6f-gray`/`e6f-blur` inference-only rows.
3. Curated gate: same env,
   `uv run board-reader-inference eval data/eval/ground_truth.json` vs 304/320.
4. Ship decision + the improvement report (E2/E3 fine-tune deploy story is a
   separate follow-up: STATUS Next 5).

Note `data/`, `recordings/`, `scratch/` are gitignored — the cloud machine
needs them synced from this box, not from the repo.
