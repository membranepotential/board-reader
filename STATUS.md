# STATUS — board-reader

_reconciled: 2026-08-28 @ `c0138f0`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md), detail in `git log` + `scratch/experiments/*/report.md`; this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- **The 08-27 night wave is COMPLETE** (see Done). Estimator, scheduler,
  stabilizer, framing, recorder, harness all landed, reviewed, and
  replay-verified; suite 1048 Python + 417 vitest + 50 e2e, all green.
- trial-run — live proof-session stack (`make dev`, recorder on, session
  20260828-111421) — **up and verified**; user plays the planned Italian line
  at `/camera`; truth pre-authored, score with `eval-session` after.
- **Next live take = the proof session.** It should demonstrate, zero-tap:
  auto-detect → auto-verify → start → auto-calibrate ("Calibrated
  automatically…", trigger bug fixed in `b9b886c`), burst-end reads (~5× fewer
  GPU calls, watch `/healthz` ingest.unscheduled, backstopped==0), drift
  re-lock ("The frame had drifted — re-locked"), no OOM at any length, coherent
  settled stream. Author truth with `guard_seqs=3` (sparse events by design).
- ui-mockups — awaiting user review in `docs/mockups/` (user: redo later as
  clickable simulations, not storyboards).

## Done (recent)

- 08-28: **curated baseline made reproducible** — 5 dead `up_*` entries pruned
  (their photos were deleted; among them the set's only bank-held-out board),
  surviving photos moved into `data/eval/images/`. New baseline, deployed
  cascade⊕bank stack: **307/320 per-square, 0/5 perfect** (prior: 308/320, 1/5),
  ECE 0.24. The 638/640 headline is historical. Docs updated.
- 08-28: **own real footage exists** (user-confirmed) — trial recordings
  (08-13, 08-27 takes 1–5, today's proof session) are the footage source;
  broadcast sourcing closed.
- 08-28: **repo pushed public**. Pre-rewrite SHAs recoverable from
  `docs/commit-rewrite-{map.tsv,pre.bundle}` (git-ignored);
  `tools/check_commit_refs.py` translates stale citations.
- 08-28: **b-demotion confidence display: user decided leave as-is** — rescued
  squares reporting ~0.34–0.47 stay unflagged; correct recognition is the
  ambition, confidence marking would only clutter the UI.
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

1. **Tall-crop retrain decision** — the case sharpened all night: the `b` rule
   ships (residual shrunk), the control's `q->k` is NOT re-rankable (truth
   outside top-2 half the time), 08-13's d8 q->k is a pure recognition error the
   prior reads wrong 90 % of the time, and take 4 (least drift, worst accuracy)
   is pure classification error. ~78 % of residual errors sit outside the top-2
   ceiling. A pose-derived crop axis belongs with whatever retrain survives.
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
   follow-ups (`73efd60`, non-blocking).

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
