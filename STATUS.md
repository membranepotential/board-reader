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

- **08-28/29 improvement cycle CLOSED** (see Done): masking shipped, cloud
  pipeline live, bank rematch NO-GO — **the shipped stack stands**. RunPod:
  volume destroyed by user 08-31 (no standing costs), balance $14.41, DC
  re-pinned EU-RO-1→EUR-IS-1 after availability survey (`fd5e104`; US DCs
  carry no 4090s — US-NE-1 is the recorded US fallback).

- **08-31 new-style batch labeled + measured cold — DONE, nothing running**
  (see Done). Follow-ups live in Next 11.

- **08-31 experiment wave toward board-level reliability (user-approved, 4 tracks)**:
  - two-kings-reopen — **CLOSED, shipped stands** (docs/reports/2026-08-31-two-kings-reopen.md): fresh challenger `p(K)−p(Q)` won stills then collapsed 08-13 22/31→0/31 in the replay gate — queen-mass polarity inverts between datasets, no safe λ; ceiling confirmed against 14 rules; `_kings_flip` docstring corrected (cascade, not bank, owns the d8 misread); bank measured 73%/66% OOD on stills at 0.70 blend weight
  - calibrate-before-after — **CLOSED** (docs/reports/2026-08-31-calibrate-before-after.md): decoded Q->K killed 4->0 on the fresh style, ECE 0.345->0.229 — but 08-13 settled perfects 22->12 (total 35->25) while EVERY per-square metric rose. Non-monotone confirmed; bank writes must gate on 4-take settled-perfect rows; points at style-scoped banks. Deployed bank untouched.
  - dinov3-probe — frozen-ceiling rerun + bank embedder swap (ideas.md §2) — HF access granted, ONNX exports (cls+central) running; export tool generalized (`a6a572a`)
  - autolabel-engine — **plan landed**: docs/plans/2026-08-31-autolabel-engine.md (Viterbi decoder, 6 gates, ~7.5 d) + explainer docs/viterbi-auto-labeling.md — awaiting user review; Stage 0 (null-hypothesis train on hand-truth crops) is the cheap first step
  - yt-angle-search — **DONE, positive** (`d0f39bf`, docs/youtube-streams.md): ASMR chess niche reliably clears the 0.73 good band with full multi-hour games (@ASMRChess goldmine); broadcast conclusion stands; idea B (PGN-aligned labeling) un-parks — next: user picks a video to ingest via get-frames-from-yt

## Done (recent)

- **08-31: a new board style measured cold — it transfers; the queen does not**
  (docs/reports/2026-08-31-new-style-batch.md). 17 iPhone photos of an unseen
  style (vinyl roll-up, plastic Staunton, outdoor) in
  `data/own/2026-08-31_iphone/`: 15 start positions + 2 Ruy Lopez
  (`0763`/`0764`). Split **5 calibrate** (→ `ground_truth.json`) / **12 held
  out** (`data/eval/heldout-20260831/*.label.json`) — the split matters because
  `build_exemplar_bank.py` reads `ground_truth.json`. Truth oriented without the
  model (back-rank luminance, 27–62 grey margin; `a1`-dark chirality check;
  eye-verified). **No bank write**, so all 16 labeled boards are cold.
  Held-out **680/704 = 96.6 % per-square, ABOVE the curated set's 95.3 %** —
  cuts against the standing cross-style-piece-type prior. But perfect boards are
  1/11, from **one** systematic defect: `Q->K` ×13 / `q->k` ×8 on `d1`/`d8`
  (21 of 41 errors) — both back-rank kings read as kings, and the decoder demotes
  the *true* king on `img_0756`/`0758`. Reproduces the two-kings question on 13
  fresh boards vs the 7-board ceiling that closed it (`8c42b83`). ECE 0.345.
  `0760` corner-detect FAILed (unlabeled).

- **08-29 (night): the rematch cycle closed — every challenger NO-GO, three
  mechanisms found** (docs/reports/2026-08-29-bank-rematch.md): masked bank
  loses the 4-take gate 35→22 perfect (its curated +3 inverted — the baseline
  bank's value is auto-calibrate exemplars grown from the sessions' own frames,
  which uploads cannot contain); masked members + masked bank 35→17; run_i
  full-data lifts holdout to 0.9283 but drops chesscog to 0.828. Also:
  two-kings decode claim refuted (`8c42b83`, docs/reports/2026-08-28-two-kings-
  decode.md — shipped rule Pareto-best, tie-break ceiling ~7 boards);
  `make dev` unmasked-drift fixed (`4d6ce3e`); `rebuild-bank` landed review-
  hardened (`6572b76` — guard shares live `verify_labels`, 10/54 uploads
  genuinely bad); cloud Phase 2 closed (loader-bound, ~110 img/s/vCPU,
  workers 8; README "Cloud perf"); lint backlog cleared. Suite 1147. Cloud
  spend $3.30 total.

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

1. **Bank provenance, then the session-fair rematch**: persist the source crop
   (or masked+unmasked embedding pair) for every exemplar written — calibrate
   AND auto-calibrate — so bank re-embeds become possible at all. A session-
   fair rematch is runnable today by mining the auto-calibrate frames out of
   `recordings/` (frames + settled placements are recorded); that is the
   remaining way the masked-bank question gets a real answer
   (docs/reports/2026-08-29-bank-rematch.md §residue). Cloud Phase 3.2–3.4
   (unfreeze, architectures, resolution) remain open on the measured pipeline;
   the fine-tune deploy story (E2/E3) feeds Next 5.
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
   (The 08-28 "two-kings demotes the CORRECT king" claim was investigated and
   refuted — see `8c42b83`: the decoder already prefers the lower-confidence
   duplicate, the evidence genuinely crosses on 08-13, and the residual lever
   is the bank lock on d8, not the tie-break.) Feeds item 1.
7. **Decide whether calibrated confidences go on the wire** — a decision, not an
   experiment: argmax is invariant, DESIGN §9 says the downstream smoother wants
   them.
8. **Crop attribution / occlusion** — the prism is a *bounding* volume: the c6
   knight's lean hallucinated a b6 rook and corrupted b7 all trial.
   Base-weighted attribution over bounding-prism membership. Plus detection
   follow-ups (`73efd60`, non-blocking). The 08-28 demotion A/B pins both sides
   of the `b` trade on this (c5's crop bleeding into b5).
9. **Step-back reevaluation ideas parked in [docs/ideas.md](docs/ideas.md)**
   (user 08-29, "work on it some other time"): recommended sequence is DINOv3
   ceiling probe → legality-constrained auto-labeling of game video →
   detection-first (RF-DETR) prototype; SAM 3 as labeling tooling.
11. **Follow-ups from the 08-31 new-style batch**
   (docs/reports/2026-08-31-new-style-batch.md): (a) **calibrate the five and
   re-measure** — the direct test of whether user calibration kills `Q->K` on a
   fresh set; gated on the un-gated non-monotone bank write below, so run it as a
   before/after on this batch, now the cleanest regression probe we have.
   (b) **Reopen the two-kings tie-break** — 13 fresh boards beat the 7-board
   ceiling that closed it. (c) Ask the user for **varied positions, not more
   start positions**: every calibrate candidate is the start position, so each
   piece type is only seen on its home square. (d) `0760` needs hand-picked
   corners (`tools/corner_picker.html`).
12. **`replay.CONFIG_ENV` omits `BR_CROP_MASK`** — masked and unmasked replays
   of a take share a `config_digest` (found in the 08-31 two-kings replay gate:
   `e6-control` vs `e6-gray` both `cc530662`, scoring 0.9139 vs 0.9121). Add
   the mask to the digest env; small, wants its own tests.
10. **Client-side session video** (user proposal 08-28): `MediaRecorder` on the
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
  it). **Measured at full strength 2026-08-31**
  (docs/reports/2026-08-31-calibrate-before-after.md): a five-board calibrate
  raised every per-square metric and dropped settled perfects 35→25 via
  cross-style contamination of the king margin. The gate must be the 4-take
  settled-perfect rows; still unbuilt. Design lean: style-scoped banks.
- **`verify_labels` cannot reject an all-empty placement** (`agree < 0.6 × 0` never
  fires) — a piece-filled photo labeled empty would embed piece pixels as `empty`
  exemplars. Pre-existing in live `/calibrate`, found 08-28 writing the rebuild
  guard tests; fixing changes live calibration semantics, wants its own review.
  Related: 6 non-portrait guard rejects are probably repairable (~+20% corpus).
