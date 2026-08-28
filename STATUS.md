# STATUS — board-reader

_reconciled: 2026-08-28 (post night-wave) @ `9167dfd`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md), detail in `git log` + `scratch/experiments/*/report.md`; this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- **The 08-27 night wave is COMPLETE** (see Done). Estimator, scheduler,
  stabilizer, framing, recorder, harness all landed, reviewed, and
  replay-verified; suite 1048 Python + 417 vitest + 50 e2e, all green.
- trial-run — live proof-session stack (`make dev`, recorder on) — being
  brought up 2026-08-28; user does the live take at `/camera` when ready.
- **Next live take = the proof session.** It should demonstrate, zero-tap:
  auto-detect → auto-verify → start → auto-calibrate ("Calibrated
  automatically…", trigger bug fixed in `b9b886c`), burst-end reads (~5× fewer
  GPU calls, watch `/healthz` ingest.unscheduled, backstopped==0), drift
  re-lock ("The frame had drifted — re-locked"), no OOM at any length, coherent
  settled stream. Author truth with `guard_seqs=3` (sparse events by design).
- ui-mockups — awaiting user review in `docs/mockups/` (user: redo later as
  clickable simulations, not storyboards).

## Done (recent)

- 08-28: **repo pushed public**. Pre-rewrite SHAs recoverable from
  `docs/commit-rewrite-{map.tsv,pre.bundle}` (git-ignored);
  `tools/check_commit_refs.py` translates stale citations.
- 08-28: **b-demotion confidence display: user decided leave as-is** — rescued
  squares reporting ~0.34–0.47 stay unflagged; correct recognition is the
  ambition, confidence marking would only clutter the UI.
- **08-27 night wave** (detail: `git log 866653f..9167dfd`, docs/reports/2026-08-27-*):
  all five takes scored; **replay harness** (recorded takes → rerunnable offline
  benchmark; replay-vs-replay is the valid comparison); **GPU arena ratchet
  fixed** (fixed-shape batches; 0 failures over every replay since, was 91/take);
  **b-demotion shipped** (evidence re-rank study GO: +2–10 pp perfect, 0 broken);
  **framing re-detect** (take 5's drift residual −58 %, baselined adoption, 0
  false adoptions on controls); **burst-end read scheduling live** (~5× fewer GPU
  reads, D3 floor + wall-clock backstop, `BR_SCHEDULE` escape hatch, events now
  sparse by design → author truth with guard_seqs=3); **settled stream
  coherence-fixed** (excess-monotone commits + bounded staleness ≤3 windows;
  impossible boards 233→0 on control, perfect 0→28 %); **recorder rolls per
  session** (seq-collision wart gone); **zero-tap calibrate root-caused**
  (occupancy-vs-classname bug — it had never fired; fixed, e2e-pinned); **test
  cascade filled** (1048 py / 417 vitest / 50 e2e; every feature unit +
  integration + e2e; 4 silently-red e2e specs repaired). Every non-trivial
  commit adversarially reviewed; findings fixed same-night.
- 08-27 (day): meter was dead on the phone → adaptive floor (proven live in take
  5); label-rotation auto-absorb; portrait hard block; sharpness-preferring
  inbox; latency batch fix. Pre-wave history: `git log` before `866653f`.

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

- **No real footage sits inside the good angle band**, and a ~40-candidate search found none:
  broadcast cameras frame players, not boards. Own footage is a sourcing requirement.
- **Nothing consumes the position stream** — deliberate (D1/D3 put smoothing in
  the consumer), but that consumer app still does not exist. (`stability` now
  earns its keep internally: forwarder + read scheduler both run on it.)
- **Confidence is not usable today** — no board-level signal; the ensemble is under-confident.
- **Bank writes are not monotone** — a 64/64 board lost a square after two `/calibrate` writes.
- **The 638/640 baseline is unreproducible** — 5 of 10 curated boards point at deleted photos.
