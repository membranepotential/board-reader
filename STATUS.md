# STATUS — board-reader

_reconciled: 2026-08-27 evening @ `b1032ec`_ · decisions live in [docs/DESIGN.md](docs/DESIGN.md), detail in `git log` + `scratch/experiments/*/report.md`; this is state.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

- public-release — repo is **push-ready** (Apache-2.0 + THIRD-PARTY.md, details
  scrubbed, history re-signed onto the current author email). Before pushing: add that
  address as a UID on the signing key and re-upload it, or every commit lands
  Unverified. Optional: a CI workflow, CONTRIBUTING.md. Pre-rewrite SHAs recoverable
  from `docs/commit-rewrite-{map.tsv,pre.bundle}` (git-ignored); `tools/check_commit_refs.py`
  translates stale citations.
- **Iteration is offline now** — recorded takes replace live phone takes (operator
  decision 08-27 eve); a replay harness turns each recording into a rerunnable
  benchmark. Dev stack stopped, GPU free.
- bdemotion-verify — rule landed (`fe8a7bb`: `estimator.read_event` seam;
  composites now blend raw dists via `PhotoDistReader` so it applies exactly
  once; guardrail stat in trial scorer). NOW: after-replays of all four takes
  (background, GPU, worktree-pinned) + code review of the blend refactor —
  running — next: before/after table vs `baseline-3452334`.
- phone-live-test — FOUR takes recorded, three scored (control 99.0 %/58.3 %,
  hard 98.6 %/46.5 %, fast 98.1 %/58.1 % prior). Take 5 (18:18, unscored, no truth)
  proved the meter fix live — send rate 1 fps at rest with real bursts (was a flat
  8.2), and the restored-quad verify fired (`framing-verified, moved: false`) —
  but hit the arena OOM again and exposed **the ratchet** (next item).
- **GPU arena ratchet — the top fix before the next take.** Batch-16 bounds the
  peak but the cascade secondary's batches still VARY in size (unsure-count tail,
  1..16), every distinct shape reserves its own BFC chunks, and an arena never
  shrinks — so over a session the secondary's arena creeps to its 1 GiB cap and
  every big read fails from then on (91 failures in take 5; node idles at 2.2 GiB
  after). Fix: **fixed-shape batches** — pad every batch (trained.py crop_dist,
  embed.py) to exactly batch_size, slice the outputs; one working set, reused
  forever. Consider ORT `arena_extend_strategy=kSameAsRequested` on top.
- **Zero-tap calibrate still unproven live** — the operator calibrates manually
  before pressing start (habit), and the trigger needs the position stream, which
  only flows while streaming. Decide: teach the flow (calibrate pill only after
  start?) or let the idle page probe /estimate itself so the trigger works
  pre-stream.
- ui-mockups — full-system mockups **awaiting user review** in `docs/mockups/`
  (also http://localhost:8766/ while the ad-hoc server runs).
- tall-crop-retrain — decision input now in from BOTH trials: the dim-light residual
  (`p->b` 17×, `q->b` 8×) is 25/25 truth-ranked-second, i.e. re-rankable WITHOUT a
  retrain; the control's `q->k` (truth outside top-2 11/22) still argues for one.
  Weigh a cheap re-rank/calibration pass against the retrain before training.

## Done (recent)

- 08-27 (night): **baseline replays of all four 08-27 takes** (dirs
  `recordings/*/replays/baseline-3452334/`, log in session scratchpad): 2500
  frames, **0 estimate failures, GPU flat ~660 MiB** — the ratchet fix holds at
  session length (live take 5 had 91 failures on 301 frames). Prior stream
  (every-frame population — NOT comparable to recorded scores, see replay docs):
  control 97.2 %/30.8 %, hard 94.9 %/24.7 %, fast 97.0 %/54.1 %, take 5
  92.1 %/28.6 %. These are the diff-base for every estimator change from here.

- 08-27 (night): **replay harness shipped** (`e759631`) — `board-reader-inference
  replay <recording>` re-runs recorded frames through the live route/prior/settle
  path on the recorded capture_ts clock; output under `<src>/replays/<name>/` is
  eval-session-scoreable; deterministic (byte-identical, proven on the real
  cascade); `FramePolicy` seam ready for move-triggered estimation. First result:
  replaying 08-13 at full 7.5 fps locks the settled stream onto a d8/e8 k/q swap
  for the whole session (needs ~19 agreeing reads to displace vs ~3 at trial
  rate) — the stabilizer's adopt-first rule wants a tiebreak, and read RATE is
  now known to change settled behaviour, not just lag.

- 08-27 (night): **re-rank study: GO on one rule, NO-GO wider**
  ([report](docs/reports/2026-08-27-evidence-rerank.md), `f66f9a4`+`4d3c165`).
  Demote a `b` argmax when top-2 margin < 0.20 (swap top two, keep runner-up
  prob): 38 squares fixed / **0 broken** over all four takes; anchored-prior
  perfect boards control 58.3→70.8, hard 46.5→65.1, fast 58.1→71.0 (take 5
  drift-bound, unchanged). Captures only ~22 % of top-2-reachable errors, so the
  tall-crop retrain argument stands; control's q→k is NOT re-rankable (half
  outside top-2). Guardrail wanted: margin-gap assertion in the trial scorer
  (tightest observed gap 0.210). → implement next (Next 0b).

- 08-27 (night): **arena ratchet fixed** (`dcfcf28`) — every GPU batch padded to
  exactly batch_size (repeat-last-crop) and sliced back, in trained.py crop_dist
  (covers the cascade secondary + tail) and embed.py (incl. the symbolic-width
  probe, which now probes at full batch shape). Soak on the deployed config: flat
  1142 MiB over 300 mixed-shape calls, 0 failures; a control with padding patched
  out crept +774 MiB in 100 calls. Steady state 1142 MiB / 3 sessions.
- 08-27 (night): **all five takes now scored** (`866653f`; truth+report in each
  recording dir — recordings/ stays git-ignored). Take 5 prior 97.3 %/45.7 %
  (guard_seqs=3, NOT guard-comparable — adaptive 1 fps rest); aborted 17:03 take
  prior 98.3 %/23.5 %. Two new signals: (1) take 5's dominant residual is
  **framing drift** — the a1 corner wandered ~79 px (≈1 square) over the session,
  one coherent a-file shift, while drift-free segments score 100 % — the
  restored-quad verify only runs at page open, nothing re-detects mid-session;
  (2) the 17:03 raw stream shows the k/q-swap calibration plainly, and the
  prior's kings constraint repaired **all 12** q/k errors — the constraint layer
  earning its keep.

- 08-27 (late): **The motion meter was dead on the deployment phone — root-caused
  and rebuilt** (`5af963d`). Resting inter-frame difference measures ~6× the old
  fixed noise floor on this rig, so stability lived under 0.35 all day: auto-detect
  never fired (every detect was a tap), the forwarder sat in permanent burst (the
  flat 8.2 fps), and the zero-tap trigger could never open. The meter (TS+Python,
  mirrored) now learns each rig's own resting floor (asymmetric chase, gated so
  sustained motion is not adopted as rest) and scores the ratio to it — tuned by
  replaying all three takes (rest ≥0.9 on 56–62 % of frames, moves <0.7 on 27–35 %).
  The page also samples motion from the board's crop box, not the full frame. Plus:
  a **restored framing is auto-verified** by one detection at page open — a stale
  saved quad after a perspective change read the board a full rank off and starved
  the zero-tap gate (which correctly refused to calibrate from it).
- 08-27 (late): **Fast take scored — realistic pace costs almost nothing.**
  `recordings/20260827-175539` (report kept there; scored on a capture_ts-filtered
  copy): the full Italian incl. the 6.d3 Qe7 7.Nc3 Qd8 tail in ~45 s (~3 s/move,
  hands near the board throughout): prior **98.1 % per-square, 58.1 % perfect** —
  the same perfect rate as the slow control. Settled drops to 32 % (settle lag ≫
  segment length at this pace, by design). Latency median 1.60 s / max 2.56 s,
  0.89 reads/s, **zero OOM failures** (batch-16 held). New residual at the shifted
  perspective: `B->b` colour flips (18×, near-ties) and `empty->p` hallucinations
  (24×, runner-up 23/24 — re-rankable). Eval gotcha: ingest seq resets per
  streaming session but the recorder appends ONE dir per server run, so sessions'
  seqs collide — the recorder should roll a dir per session.

- 08-27 (evening): **Hard take run and scored** (`recordings/20260827-172736`,
  1.e4–5...Re8 with both castlings, fresh one-photo bank; lighting NOT dimmed,
  operator-confirmed): prior
  **98.6 % per-square / 46.5 % perfect** on 43 events. The rank-8 k/q residual did
  NOT reproduce — castled/rook-lift segments score 99.5–100 % prior; the hard take's
  residual is bishop-attractor (`p->b` 17×, `q->b` 8×) with truth ranked second
  25/25 at ~0.18 margin → re-rankable. The 6.d3+ tail is lost to the OOM below.
- 08-27 (evening): **Both live stalls diagnosed: CUDA arena OOM, fixed** (`c2e9bf1`).
  The cascade secondary fed a near-full-board unsure batch (a fresh one-photo
  bank leaves run_e unsure almost everywhere; lighting was unchanged) pushes
  its session arena to the cap (728 MiB@768, 1001 MiB@1024) and a BFC arena never
  shrinks — every later read fails, stream goes quiet, worker survives. Fix: GPU
  batches bounded at 16 (¼ the peak, ms of overhead). Also: never run browsers
  during a take on the shared 4 GiB card.
- 08-27 (evening): **Zero-tap session flow shipped** (`b1032ec`, `0b2857c`, `ec96b98`):
  aim → start → play. Auto-detect (existing) + the first calibration firing itself
  when the reading looks like a start position at rest (server gate stays the
  authority); calibrate's `label_rotation` folded into the page labelling
  automatically (sign pinned end-to-end); portrait hard-blocked with an overlay;
  rotated-board refusals say the one actionable sentence. Plus shake mitigation:
  per-frame sharpness on the wire, sharpest-recent-wins inbox (`prefer_frame`).
  The mis-calibrated bank was set aside (`models/exemplars_session.npz.poisoned-20260827`).

- 08-27: **Control trial run, scored, and the 08-13 fixes are confirmed landing.**
  `recordings/20260827-163352` (Ruy Lopez through 4.Ba4, 337 frames, 31 reads, truth
  authored + `report.json` kept). **prior 99.0 % per-square, 58.3 % perfect boards**
  against 08-13's 97.0 % / 28.8 % — the kings-evidence repair (`faaa4ed`) doing exactly
  what it was built for. Latency **median 1.38 s, max 1.56 s** vs 6.59 s on 08-13 (the
  event-loop fix). Throughput unchanged: 9.65 fps in, 0.89 reads/s out.
  *Evidence margins, the new measurement:* of 38 wrong raw squares, truth was the
  runner-up in only 21. Split by confusion it answers the retrain question — `q->k`
  (22×) has truth in the top 2 just **11/22** at median margin 0.152, so the crop often
  does not contain the answer and re-ranking cannot fix it; `p->b` (7×, the b7 pawn
  behind the leaning knight) has truth ranked **7/7** at margin 0.144, so that one is
  re-rankable. *Eval-design finding:* `guard_seqs` must cover the **settle lag** or the
  settled stream is scored while it is correctly still catching up — settled reads 29.2 %
  perfect at guard 12 and 77.8 % (99.7 % per-square) at guard 45, on the same 7 boards.

- 08-27: **Calibrate failed all afternoon because the phone was portrait.** Not a
  regression (08-13's photos score 28-31/32 through today's code) and not the angle
  (aspect 0.86-0.90, floor 0.62). `squares.py:264` extends every crop **upward in image
  space** — "piece tops point up in a normal photo" — so a board rotated 90° in the frame
  has each crop reaching into the neighbouring square. Proof: the same photo scores
  18/32 REJECT as sent and **30/32 PASS rotated 90° CCW** (piece confidence 0.47 → 0.94).
  Occupancy stayed 64/64 throughout, which is why it looked like a piece-type problem.
  `autorotate` cannot help — it rotates *labels* from `square_dist`, one layer above the
  crop axis. A quad-only heuristic for the upright direction was tried and **rejected**:
  it predicts 270° where the truth is 90° (3/7 correct), so the usable signal is the
  device orientation angle, not perspective foreshortening.

- 08-27: **A trial now records why, not just what** — three gaps closed before the
  re-trial. *Evidence*: every recorded event carries a top-2-per-square digest
  (~1.8 kB vs ~25 kB for `square_dist`), and `eval-session` reports per confusion pair
  how often truth was the runner-up and the median `p(read) − p(truth)` margin — a
  near-miss the crop nearly answered vs a conviction the evidence never held.
  *Timing*: fps in, reads/s out, frames per read and capture→publish latency, off the
  frame index alone; it confirms the hand-derived 08-13 figures (**7.47 fps in, 0.85
  reads/s out, latency median 1.29 s / max 6.59 s**). *Heartbeat*: `/camera` beats every
  10 s into `client-logs/`, so a quiet stretch says which kind of quiet. Scoring now
  always writes `<recording>/report.json`.
- 08-27: **The 08-13 headline numbers were wrong, and are corrected.** Re-scoring with
  unchanged code gives prior **28.8 %** perfect, settled **25.8 %** (97.7 % per-square)
  — not 45.4 % / 66.6 % (98.96 %). The truth.json was revised after those numbers were
  published (66 scored events / guard 25 today vs 81 / guard 12 in the learning) and no
  report artifact was kept. README corrected. Same failure as the 638/640 entry — hence
  the always-write-the-report rule.

- 08-20: **Pre-publish pass — the repo can go public.** Full object-graph audit: no
  photos, datasets, weights or secrets were ever committed. Apache-2.0, with
  `THIRD-PARTY.md` for the two exclusions — the built viewer bundle is
  **GPL-3.0-or-later** (chessground + cburnett; our source stays Apache and
  core/edge/inference carry no GPL dep), and the trained model is **not
  redistributable** (ChessReD CC BY-NC-SA + murky geometry).

- 08-13: **Trial 19:33 analyzed end-to-end; four fixes landed** (`faaa4ed`, `ef9a675`,
  `78bc4d2`, `37a1972`). Kings repair demotes the lowest-`p(k)` square, not the smallest
  k→runner-up margin (that margin is q-tail noise: wrong 55% of 628 two-king replay
  frames vs 27% for the evidence rule; EMA fixes nothing, the misread is systematic).
  `settled` ships — a per-square *time* debounce (2.5 s + ≥2 reads; count-based K does
  NOT transfer across a 10× read-rate range) on the wire beside `prior`, drawn with
  dashed-amber pending rings. `eval-session` scores the three streams apart. And
  /estimate + /calibrate moved off the event loop (the 6.6 s latency spike). Trial
  facts: the "phantom b6 rook" was a real c6 knight leaning across the boundary
  (attribution bleed, also corrupts b7); rank-8 crops at phone height are decapitated,
  so k/q is indistinguishable in-crop.

- 08-13: **The node stops starving the display GPU** — `cuda_providers()` ships CUDA EP
  options (`BR_GPU_MEM_LIMIT_MB`, default 1024 MiB/session) and `make dev-cpu` hides the
  card. A `/calibrate` batch had twice taken the VRAM X needed for the 4K monitor's
  reverse-PRIME surface. Peak measured ~1.6 GiB of 4 GiB across 3 sessions.

## Next

0c. **Move-triggered estimation** (operator-endorsed direction): the node reads
   continuously today, wasting reads on resting boards and on mid-burst chaos
   frames (whose huge unsure batches are the OOM path). Frames already carry
   `stability` + `burst_id`; estimate at BURST END (the move-complete moment) +
   a slow keepalive at rest (~3-5 s), skip mid-burst. Hard floor rate is
   non-negotiable (D3 hint-never-gate; the dead-meter incident is the cautionary
   tale). Later, per-square motion masks feed the stabilizer for hovering hands;
   shadows stay the classifier's problem — a false burst is cheap, keepalive
   bounds a missed move.
0b. **Ship the `b` demotion** (study said GO — see Done): swap top-2 in
   `square_dist` when argmax is `b` and margin < 0.20, upstream of
   placement/evidence/prior; plus the margin-gap guardrail assertion in the trial
   scorer. Then re-verify via replay on all four takes.
0d. **Mid-session framing drift** (take 5's dominant residual, ~79 px a1 wander):
   nothing re-detects/verifies the quad after streaming starts. Cheap candidates:
   periodic restored-quad-style verify off keepalive frames, or corner tracking.
1. **Retrain case, post-re-rank** — what the `b` rule leaves: control's `q->k`
   (truth outside top-2 half the time) is the true crop case; ~78 % of residual
   errors are outside the top-2 ceiling. A pose-derived crop axis belongs with
   whatever retrain survives.
2. **Estimator throughput / latency** — 0.89 reads/s (control) collapsing to 0.48
   under the (now fixed) OOM; profile the cascade+ensemble read at batch 16 and
   attack the biggest term (bank embed batching?). Faster reads shrink settle lag
   and the guard window both.
3. **Run M2 over `ccbf-r5`** with `--forward-original --corners` (the warp path is unusable at
   this angle), judge the stream, then settle D3 — does `stability` earn its keep?
4. **Give a user's `tuned.onnx` a deploy story** — per-operator artifact, no compose path today.
5. **Attack the residual type confusions** — `p->b` leads trained-alone (confirmed live:
   b7 pawn read `b` behind the leaning knight); a third of the ensemble's residual is
   `empty->p`/`empty->P`, a different animal.
6. **Decide whether calibrated confidences go on the wire** — a decision, not an experiment:
   argmax is invariant, DESIGN §9 says the downstream smoother wants them.
7. **Crop attribution / occlusion** — the prism is a *bounding* volume: the c6 knight's lean
   hallucinated a b6 rook and corrupted b7 all trial. Base-weighted attribution over
   bounding-prism membership. Plus detection follow-ups (`73efd60`, non-blocking).

## Open questions

- **No real footage sits inside the good angle band**, and a ~40-candidate search found none:
  broadcast cameras frame players, not boards. Own footage is a sourcing requirement.
- **Nothing consumes the position stream** — deliberate (D1/D3 put smoothing in the consumer),
  but that consumer does not exist, so `stability` is measured and read by nobody.
- **Confidence is not usable today** — no board-level signal; the ensemble is under-confident.
- **Bank writes are not monotone** — a 64/64 board lost a square after two `/calibrate` writes.
- **The 638/640 baseline is unreproducible** — 5 of 10 curated boards point at deleted photos.
