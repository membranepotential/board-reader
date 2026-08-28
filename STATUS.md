# STATUS — board-reader

_reconciled: 2026-08-28 @ `c0138f0`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md), detail in `git log` + `scratch/experiments/*/report.md`; this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- **The 08-27 night wave is COMPLETE** (see Done). Estimator, scheduler,
  stabilizer, framing, recorder, harness all landed, reviewed, and
  replay-verified; suite 1048 Python + 417 vitest + 50 e2e, all green.
- ui-mockups — awaiting user review in `docs/mockups/` (user: redo later as
  clickable simulations, not storyboards).

- **08-28 retrain program CLOSED** — masking shipped, cloud pipeline live
  (see Done). Nothing running; RunPod volume standing ($2.10/mo), pods reaped,
  balance $7.25.
- **cloud-phase-2+3** — main session: pod tuning grid (amp/batch/workers +
  loader ceiling) → tuned config → full-data run_i — IN PROGRESS — next: bench
  pod.
- **two-kings-coherence-fix** — implementer agent: settled coherence must demote
  the lower-confidence duplicate king, not the incumbent — PENDING SPAWN.

## Done (recent)

- **08-28 retrain program E1–E7** (full numbers + narrative:
  docs/reports/2026-08-28-retrain-program.md; details
  scratch/experiments/e{2..7}_*/report.md): trials exported as crop styles;
  fine-tunes measured clean (E2 leaked, E3 partial win); bleed measured (44 %
  of residual); decode rule NO-GO; occlusion masking shipped (`BR_CROP_MASK`,
  depth-fixed after review) and proven on the d8 lock; **run_g masked retrain
  matches holdout, +3.2 pp chesscog**.
  Mid-program RAM-overcommit freeze → serialize + systemd-run caps adopted.
- **08-28 evening: program closed on RunPod + gates**
  (docs/reports/2026-08-28-improvement-report.md): rp runner + S3 staging +
  cache-backed style resolution landed (`684507d`..`c2f3c58`); parity PASS
  (0.8995 vs 0.8983); run_h_cloud trained (0.8944). Gates: masked-retrained
  pair NO-GO deployed (E6 d8 win = bank-lookup effect; members benched as OOD
  assets, +4pp chesscog each); **shipped BR_CROP_MASK=gray in compose**
  (`38c40ac`) — perfect boards ~20→35 on the 4-take gate, curated 305/320.
  Total cloud spend $0.66. New defect: settled coherence demotes the correct
  king on a raw two-kings read (see Next 6).
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

1. **Cloud Phase 2+3** (docs/plans/2026-08-28-runpod-migration.md): tune the
   pod (bf16/batch/workers grid — amp:false and batch 48 are Turing relics;
   L4 measured only 3.7×), then full-data run_i (~$3). Rematch run_g/run_h
   after **re-embedding the bank from masked crops** (the E6 lesson: the bank
   half decides these takes). Then the fine-tune deploy story (E2/E3; feeds
   Next 5).
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
   of the ensemble's residual is `empty->p`/`empty->P`, a different animal. And
   the 08-28 gate found **settled coherence demotes the CORRECT king** when raw
   hallucinates a second one (q->k doubled into a full swap, 31/31 events on
   08-13) — prefer demoting the lower-confidence duplicate. Feeds item 1.
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
