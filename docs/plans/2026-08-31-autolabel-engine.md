# Plan — legality-constrained auto-labeling (the data engine)

> [!important] Stage 0 ran 2026-08-31 and returned **NO-GO on today's corpus**
> ([[2026-08-31-autolabel-stage0]]): hand-perfect labels move no uncontaminated
> gate, the corpus holds only **30 distinct positions** (not 73), and the
> 4-take gate becomes a memorization readout for any candidate trained on its
> sessions (100 % of gate boards within 5 frames of a trained image). Stages
> 1–6 are **deferred until the corpus has real positional and stylistic
> spread** — varied games, a second set, a second camera. The export path,
> style layout, and staged volume survive; re-running Stage 0 later is cheap.

_2026-08-31 · idea: [docs/ideas.md](../ideas.md) §A · position 2 of 3 in that note's recommended sequence · adjacent: [docs/measuring-accuracy.md](../measuring-accuracy.md), [docs/training-and-data.md](../training-and-data.md) · the algorithm explainer: [docs/viterbi-auto-labeling.md](../viterbi-auto-labeling.md)_

A recorded take is not unlabeled data. The reader is ~92–97 % per-square and ~0–50 % per *board*, but the sequence of boards it emits is constrained by a graph so brutal that most single-frame ambiguity cannot survive it: from one position, only ~30 successors exist, and a persistent single-square misread almost never lands on one of them. This plan builds the offline decoder that exploits that, turning imperfect reads over a whole game into per-frame labels good enough to train on.

> [!warning] The layering irony is deliberate, and it has exactly one rule
> This builds the temporal, legality-aware consumer that [DESIGN D1/D3](../DESIGN.md) forbid in the live system. It is allowed **only** because it runs offline at labeling time. The rule: `tools/autolabel/` may import `packages/`; **nothing in `packages/` may ever import `tools/`**, and no decoder state may reach the wire. Stage 5 lands a guardrail test that enforces it the same way `tests/test_layer_separation.py` enforces edge↔inference.

---

## Ground state (verified against the repo, 2026-08-31)

**The corpus.** Seven recordings, all with hand-authored `truth.json` and a scored `report.json`:

| Take | frames | truth segments | guard | frames ≥guard from a boundary | raw / prior per-square |
|---|---|---|---|---|---|
| `20260813-193308` | 748 | 6 | 25 | 623 | 0.970 / 0.970 |
| `20260827-163352` | 337 | 8 | 12 | 253 | 0.975 / 0.990 |
| `20260827-170333` | 755 | 8 | 12 | 671 | 0.975 / 0.984 |
| `20260827-172736` | 1429 | 11 | 12 | 1309 | 0.976 / 0.986 |
| `20260827-175539` | 481 | 15 | 12 | 302 | 0.905 / 0.926 |
| `20260827-181802` | 301 | 13 | 3 | 266 | 0.966 / 0.973 |
| `20260828-111421` | 243 | 12 | 3 | 212 | 0.909 / 0.917 |
| **total** | **4294** | **73 positions** | | **3636** | |

Every frame carries `corners`. So a perfect decoder over today's corpus yields **~3 636 labeled board frames ≈ 232 000 square crops — but only ~73 distinct positions, one chess set, one camera.** That number is the honest frame for everything below: **the engine multiplies frames, not positions.** It still roughly quadruples this project's real labeled board count (5 curated + 54 calibrate uploads + 20 chesscog), and it scales linearly with games played, which is the point.

**What a recording contains.** `manifest.json`, `frames.ndjson` (wire header per frame: `seq`, `capture_ts`, `stability`, `sharpness`, `burst_id`, `corners`, jpeg relpath), the JPEGs, `events.ndjson` (published `PositionEvent`s), and — if authored — `truth.json` + `report.json`. `replays/<name>/` holds recording-shaped re-runs.

**`events.ndjson` does not carry `square_dist`.** `server.event_line` strips it and substitutes the top-2 digest (`protocol.square_evidence`, `EVIDENCE_TOP_K = 2`). And the measured ceiling on that digest is decisive: **only 65.5 % of raw errors have the truth inside the top-2 at all.** A decoder fed the digest starts a third of its work impossible. The engine must therefore generate full 8×8×13 distributions itself, from the JPEGs, through the deployed estimator — a replay-shaped pass.

**Reads already happen at move ends.** `schedule.DEFAULT_POLICY_SPEC = "burst-end:burst_end_reads=3"`; the gateway and `replay --policy` share one implementation. The move-segmentation problem is therefore *already half-solved by the recorder*: `burst_id` in the frame header marks the hand-over-the-board windows, and the reads the node spends are clustered at the ends of them.

**Two plies per segment step is real, not hypothetical.** Measured over the seven `truth.json` files, three takes contain segment steps where 4 squares change. On `20260828-111421`, `from_seq 168 → 181` is d2–d3 *and* Qd8–e7, and `181 → 194` is Nb1–c3 *and* Qe7–d8: two plies with no read between them, exactly the collapse recorded in the learnings store ("plies with no event between them collapse into one segment carrying the later placement; `guard_seqs=3` absorbs the noisy mid-move reads"). Meanwhile `131` in the same take is O-O — **also 4 squares changed, but one ply.** No diff heuristic can separate those two cases. Legality can, trivially.

**`prior.decode` already exists and is the static half of this idea.** It enforces the hard *multiset* constraints (king counts, army sizes, promotion budget, back-rank pawns) by relabelling the fewest lowest-margin squares, and it is emphatic that it never rewrites `square_dist` "so a downstream legality smoother cannot double-count this prior." This plan is that smoother. **It consumes raw `square_dist`, never `prior.placement` and never `settled.placement`** — the latter carries `stabilize.py`'s adopt-first lock-in, which is measured to settle *wrong* (253/253 boards on one take).

**No chess library in the workspace.** `python-chess` is **GPL-3.0-or-later**; the repo is Apache-2.0 with a carefully maintained [THIRD-PARTY.md](../../THIRD-PARTY.md). See §6.

**One gotcha with teeth:** `recordings/20260827-175539` holds **two streaming sessions with colliding `frame_seq`** (the scored one is `capture_ts > 1787846505`). Any pass over a recording must go through `replay.find_sessions()` / `longest_session()` or it will decode two games as one.

---

## 1. The decoder — concrete formulation

### 1.1 State space

A trellis node is a **full chess position**, not a placement: `python-chess`'s `Board` — piece placement **+ side to move + castling rights + en-passant square**. Legality of the next transition needs all four; a placement alone cannot tell you whether O-O is available or whether that pawn capture is en passant.

Nodes at step *t* are deduplicated on `board.fen()` truncated to those four fields, keeping the max-score path (standard Viterbi merge). Positions repeat across a game; merging is what keeps the beam honest.

**Root.** `chess.Board()` (the start position) with weight 1.0 — every take in the corpus starts from the initial position, and `truth_skeleton()` already assumes it. For a take adopted mid-game, the root set is seeded from `prior.decode(dist_0)` over the first few reads, with castling rights inferred conservatively (rights present iff king and rook sit on home squares) and every frame in that prefix stamped `provenance=cold_adopt`.

### 1.2 Emission model

For a state *p* and an observation *o* = `square_dist` (8×8×13, rank 8→1, file a→h, matching `protocol`):

```
log P(o | p) = α · Σ_{r,c} log max(dist[r][c][class_of(p, r, c)], MIN_PROB)
```

Two deliberate choices:

- **`MIN_PROB = 1e-12`, borrowed from `prior.py`.** An estimator can emit an exact 0.0, and `log 0` poisons every comparison with `-inf`. A flip costing ~27.6 nats is expensive without being fatal.
- **`α` is a single global emission temperature and it is the most important parameter in the system.** The reader is measurably uncalibrated — the deployed ensemble is *under*confident throughout (ECE 0.130–0.336), and the middle confidence bins are badly over-confident in the trained member. `α = 1` treats 64 squares as 64 independent pieces of evidence, which they are not (crop bleed correlates neighbours), and the resulting likelihood ratios are astronomically over-sharp. `α < 1` is what lets the transition prior actually win an argument. It is fitted once (§4) and frozen.

Observations are **the frames a scheduling policy selects**, not every frame — one observation per burst-end read plus keepalives, i.e. `schedule.parse_policy("burst-end:burst_end_reads=3")`. Mid-burst frames are never observations (they are a fist over the d-file); they receive their labels by *interpolation* from the decoded segment they fall in, stamped `provenance=in_burst`, and are excluded from export by default.

### 1.3 Transition model

From state *s* at observation *t−1* to *s′* at *t*, with elapsed capture time Δt and a flag `crossed_burst` (the `burst_id` changed between the two frames):

| Transition | Successor | Log weight |
|---|---|---|
| **null** — nothing moved | `s` | `log(1 − p_move)` |
| **one ply** | `s.push(m)`, `m ∈ s.legal_moves` | `log p_move + log P(m∣s)` |
| **k plies**, 2 ≤ k ≤ K | k-fold legal composition | `log p_move + (k−1)·log p_extra + Σ log P(m_i∣s_i)` |
| **re-adopt** (escape) | any `p ∈ C_t` | `log ε_restart` |

- `P(m∣s) = 1/|legal_moves(s)|`. **Uniform, on purpose.** A chess-strength policy prior (Stockfish, a small policy net) would make the labels a function of an engine's taste and would quietly "correct" the human blunders that are the most interesting frames in the corpus. The prior stays *structural*, which is the entire claim of idea A.
- `p_move` is where **move segmentation meets burst-end scheduling**, and the answer is that it is never segmented at all — the burst structure merely tilts a prior:
  ```
  p_move = p_burst   if crossed_burst          (≈0.5, fitted)
         = 1 − exp(−λ · Δt)                    otherwise (a hazard rate, λ fitted)
  ```
  A hard segmenter has to be right; a soft prior only has to be *informative*, and it degrades gracefully when the motion meter lies — which it demonstrably has (every take before 2026-08-27 evening was recorded with the broken absolute-difference meter, `stability < 0.35` from first frame to last). `replay --restamp-stability` recomputes the field with today's `MotionMeter` and is the right preprocessing for the older takes; restamp uniformly across a corpus or not at all.
- **K = 3.** K = 2 is measured to be necessary (§Ground state); K = 3 covers a long read gap (08-13 has a 315-seq gap between segments). Cost is `|legal|^K` per node in the worst case, so K-ply expansion is only attempted when `Δt` or the read gap exceeds a threshold — otherwise the branching is wasted work.
- **`ε_restart` is the unreadable-stretch and camera-drift valve.** `C_t` is the small candidate set `{rot_k(prior.decode(dist_t))}` for the locked orientation. It is armed only when the best path's per-observation likelihood has sat below a floor for M consecutive reads — i.e. the decoder has *evidence* that it has lost the board (someone bumped it, a hand rested on it for ten seconds, the phone slid). Every frame downstream of a restart is `provenance=cold_adopt` until R clean reads confirm. **How often this fires is a headline diagnostic, not a footnote:** a decoder that restarts twice a game is not decoding, it is guessing.

### 1.4 Beam, orientation, posteriors

- **Beam B = 128** by default (a sweep over 32/64/128/256 in Stage 4 settles it). Expansion is ~30B successors per step with a cheap emission; 4 000 observations × 128 × 30 is seconds of CPU. This is not a performance problem, which is why the beam can be generous.
- **Orientation is decided by the decoder, not inherited.** Events carry `"orientation": "a1_bottom_left"`, but that is `autocalib.resolve_rotation`'s guess, and `prior.py` documents at length why feeding a guessed frame into a legality constraint is circular. The decoder instead runs all four quarter-turns over the first ~30 observations and keeps the k with the best total log-likelihood. Sequence likelihood over legal games is enormously stronger evidence than "minimise back-rank pawns", and the margin between the winner and runner-up is itself a sanity check — if it is small, the take is flagged and not exported. If the winner disagrees with the recorded orientation, that is a finding, and the run says so loudly.
- **Per-frame confidence comes from forward–backward, not from the Viterbi max.** Re-run the same trellis with log-sum-exp to get the marginal posterior of each state at each t; a frame's confidence is the marginal of the position the backtrace assigned it. Approximate under a beam — say so in the report — but it is a genuine per-frame number rather than a global one, and it is what the export threshold reads.

### 1.5 Why this attacks the *measured* residual

Worth stating precisely, because the obvious objection is "Viterbi fixes flicker, and our errors are not flicker."

The measured top failure modes are persistent piece-type confusions: `Q→K` (10 of 35 errors on the curated set), `q→k` locked by a bank exemplar on d8 for a whole take, `p→b`. These are *stable across every frame of a position* — smoothing cannot touch them. But:

- `q→k` produces **two black kings**, which is illegal under the multiset constraint `prior.decode` already enforces. That is the static half, and it already fires.
- `p→b` on d6 violates nothing in the multiset — and is killed dead by **reachability**. A bishop appearing on d6 while a pawn leaves it requires a legal move that does exactly that, in one ply, and no such move exists. The temporal constraint is strictly stronger than the multiset constraint, and it is the half that does not exist today.

That is the real argument for this idea, and it is falsifiable: Stage 3 reports `fixed` and `broken` squares relative to the raw stream, in the same paired form the project already uses for `prior.decode`.

---

## 2. Output spec

Written recording-shaped, beside the replays, so nothing has to learn a new layout:

```
recordings/<session>/autolabel/<name>/
  manifest.json      # provenance: source, git describe, CONFIG_ENV digest, decoder params
  obs.npz            # (N,8,8,13) float16 square_dist + parallel seq/capture_ts/burst_id arrays
  decoded.json       # truth.json FORMAT 1, plus extensions
  report.json        # decode diagnostics, and truth agreement when truth.json exists
```

`<name>` follows `replay`'s convention: `<git-describe>-<env-digest>`, refused on a dirty tree, so two decoder configurations never overwrite each other.

**`decoded.json` is a superset of `truth.json` format 1.** That is the load-bearing choice: `trial.load_truth()` reads it unchanged, so `eval-session`, `export_crops`, and every existing consumer work on a decoded take on day one, and validating against a hand-authored file is a segment-by-segment diff of two files in the same schema.

```jsonc
{
  "format": 1,
  "board_id": "phone",
  "guard_seqs": 3,
  "segments": [
    {"from_seq": 0,  "placement": "rnbqkbnr/...",
     "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     "move": null, "san": null,
     "posterior": 1.0, "provenance": "decoded"},
    {"from_seq": 39, "placement": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR",
     "fen": "...", "move": "e7e5", "san": "e5", "plies": 2,
     "posterior": 0.99997, "provenance": "decoded"}
  ],
  "autolabel": {
    "orientation_k": 0, "orientation_margin": 41.7,
    "params": {"alpha": 0.35, "beam": 128, "k_max": 3, "p_burst": 0.5, "lambda": 0.08,
               "eps_restart": -60.0},
    "restarts": 0, "pgn": "1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 4. O-O O-O 5. Re1 Re8 6. d3 Qe7 7. Nc3 Qd8"
  }
}
```

`provenance ∈ {decoded, boundary, in_burst, cold_adopt, low_evidence}` is carried **per frame** in `report.json` (a seq→provenance/posterior table), not only per segment; `boundary` is the `guard_seqs` window after a move, and the guard is written into the file so downstream scoring keeps honouring it.

**The training-facing output** is a **board-mode** `labels_normalized.json`, not the crop-mode export `trial.export_crops` writes:

```
data/external/autolabel_<set>/
  labels_normalized.json   # [{"image": "jpeg/0000/000123.jpg", "placement": "...",
                           #   "corners_a8_h8_h1_a1": [[x,y]x4], "group": "20260828-111421",
                           #   "kind": "real"}, ...]
  jpeg -> <recording>/jpeg   # symlink; discover_styles() walks with followlinks=True
  sources.json               # which sessions, which decoder, which gate verdict
```

Board mode, because `styles._load_board_manifest` keeps the crop geometry a *training-time* choice — `cropcache.py` cuts with whatever `headroom`/`side_pad`/`mask` the run config asks for. A crop-mode export bakes today's geometry into the dataset forever, and this project has already paid for a geometry drift once (the `headroom=0.15` decapitation, ~15 points of piece-type accuracy).

**One manifest directory per physical chess set, `group` = session id.** A board manifest yields exactly one `Style` named after its directory, and `dataset.split_by_group` holds out whole groups. So this layout gives leave-**game**-out inside a set and leave-**style**-out across sets, which is what the LOSO protocol needs and what makes "hold out whole games" (idea A's own stated mitigation) mechanical rather than aspirational.

---

## 3. Validation gates — nothing is trusted before these

This project has twice shipped a number that flattered itself (the bank built from the eval set; the `uploads_calibrate` label bug). The gates are correspondingly paranoid, and they run **in this order**.

**G1 — the observation dump is the estimator, not a copy of it.** The top-2 digest of the dumped `square_dist` must equal `events.ndjson`'s recorded `evidence` to 4 decimals, per square, for every frame a replay read. This is checkable today and it is what stops the engine from silently decoding a different reader than the one being measured.

**G2 — agreement with hand-authored truth on the four scored takes.** `20260813-193308`, `20260828-111421`, `20260827-175539`, `20260827-181802`. Bar: **every decoded segment matches the hand-authored segment, move for move, with boundaries inside the take's own `guard_seqs`.** Not "95 % of frames" — segments. A single disagreement is adjudicated by eye against the JPEG and resolved one of three ways: the decoder is wrong (fix it), the hand truth is wrong (fix it, and say so in the report), or the frame is genuinely ambiguous (widen the guard). **No crop is exported until this is 4/4.**

**G3 — held-out games, parameters frozen.** `α`, `p_burst`, `λ`, `K`, `B`, `ε_restart` are fitted on the **fit set** (`163352`, `170333`, `172736`) and then *frozen* before the held-out set (the four of G2) is scored. Any tuning after a held-out number is looked at invalidates it and the fit restarts. Bar: zero segment errors on held out. Seven takes is a small fit set and this gate is the only thing standing between the plan and a decoder tuned to seven games — which is why §6 lists it as the top risk anyway.

**G4 — PGN cross-check.** Every decode emits a PGN. For the existing corpus these are casual games with no published score, so the check is (a) `chess.pgn` round-trips it, (b) the opening is nameable and the operator recognises the game. For future takes and for idea B's broadcast footage the check becomes real: decoded PGN vs published PGN, move for move, and that is the strongest validation this engine can ever have. **Ask the operator to keep a scoresheet for the next few takes** — it costs a person nothing and converts G4 from an eyeball into a gate.

**G5 — the training gate, unchanged from every other challenger.** A model trained with the autolabel style goes through `tools/train/evaluate.py` and then the **local replay + curated + settled gates** (`tools/train/gates.sh`). The learnings store is unambiguous here: *"Member-level model gains do not compose with a bank-dominant cascade — run_g/run_h beat their baselines by +4 pp on unseen real photos yet LOST every replay take. Gate deployed configurations on take replays, never on member evals."*

**G6 — contamination guard, enforced in code.** The export refuses to include a session that is in the gate set unless `--allow-gate-sessions` is passed explicitly, and `sources.json` names every session in the style so a trainer can hold whole sessions out. A style built from the takes it will be scored on is the 2026-08-06 incident with a new coat of paint.

> [!danger] Auto-labels never feed the exemplar bank
> [docs/measuring-accuracy.md](../measuring-accuracy.md) states the rule for hand labels — *"Nothing reads a label back into calibration or training. A bank fitted on labels would make the score meaningless again."* Auto-labels are strictly worse: they are labels **derived from the bank's own output**. Feeding them back closes a loop with no measurement outside it. If a bank experiment is ever wanted it builds a *separate* `exemplars_autolabel.npz`, scored only on sessions excluded from it, and it is out of scope here.

---

## 4. Build order

Each stage is independently runnable and observable, per principle 1. Effort is working days for one person with an agent.

### Stage 0 — measure the prize, and test the null hypothesis (0.5 day)

Before writing any decoder: `eval-session --export-crops` all seven takes using the **hand-authored** truth that already exists, build the style, and train one run against it. This is the null hypothesis — *if hand-perfect labels from these takes do not move the gates, auto-generated ones certainly will not*, and the whole plan should be re-scoped toward recording more games instead.

- **Gate:** a written number for (a) exportable frames and distinct positions (3 636 / 73 as computed above, re-verified after session splitting), and (b) whether `run_e + trial_crops` beats `run_e` on `gates.sh`. Both outcomes are informative; only one of them justifies stages 1–6.

### Stage 1 — the observation dump (1 day)

`tools/autolabel/observe.py`, run in the **workspace venv** (`uv run`): opens a `Recording`, picks the longest session via `replay.find_sessions`/`longest_session`, builds the estimator through `replay.build_session()` (same `CONFIG_ENV`, same digest), walks frames through the same `schedule` policy replay uses, and writes `obs.npz` + `manifest.json`. Optional `--restamp-stability` for pre-08-27 takes.

- **Gate: G1.** Plus a test that a second run over the same recording produces a byte-identical `obs.npz` (replay's determinism property, inherited).
- Cost: ~1.3 GPU-s per read × ~250 reads per take ≈ 5 min/take, once. After this the decoder never needs a GPU again — which is the point of the file boundary.

### Stage 2 — the decoder, pure and offline (2 days)

`tools/autolabel/viterbi.py`: `decode(obs, params) -> Decode`. No file I/O, no estimator, no recording — a function from an `(N,8,8,13)` array plus timing metadata to a segment list. Runs in a small dedicated venv (`scratch/autolabelvenv`: `python-chess`, `numpy`), created by `tools/autolabel/_env.sh` mirroring `tools/train/_env.sh`.

Tests, all synthetic (`board_reader_core.sim` supplies board rendering helpers; distributions can be fabricated directly):
- a clean game round-trips exactly;
- a game whose reader is corrupted on 8 % of squares at random round-trips exactly;
- a **persistent** single-square type error across every frame of a position is corrected (the `p→b` reachability case);
- a two-ply step with no observation between is recovered;
- a castling step (4 squares changed, one ply) is **not** decoded as two plies;
- a ten-observation unreadable stretch is bridged by nulls rather than by a restart;
- a mid-take board bump triggers exactly one restart and re-adopts.

- **Gate:** those tests green under the autolabel venv, in `tools/autolabel/tests/`.

### Stage 3 — run it on the corpus, diff against hand truth (1 day)

`tools/autolabel/run.py <recording>` glues stages 1 and 2 and writes `decoded.json` + `report.json`. `--against-truth` prints the side-by-side.

- **Gate: G2 (4/4 takes).** Plus the paired `fixed`/`broken` square counts vs the raw stream, reported apart, because a repair that trades a right square for a wrong one is not an improvement.

### Stage 4 — fit, freeze, hold out (1 day)

A small sweep over `α ∈ [0.1, 1.0]`, `B ∈ {32,64,128,256}`, `λ`, `p_burst`, `K` on the **fit** takes; freeze; score the **held-out** takes. Because decoding reads `obs.npz` and never the GPU, the whole sweep is minutes of CPU — which is exactly why Stage 1 exists as a separate artifact.

- **Gate: G3 + G4.** Land the frozen parameters as module constants with the measurement beside them, in this codebase's style (`ATTRACTOR_MARGIN` is the precedent, including its cautionary tale about a constant fitted on the takes it was measured on).

### Stage 5 — the export, with its guards (1 day)

`tools/autolabel/export.py` → `data/external/autolabel_<set>/`, board-mode manifest, `jpeg` symlink, `group` = session, `sources.json`. Filters: `provenance == decoded`, `posterior ≥ 0.999`, corners present, `--every N` decimation (consecutive frames of a static board are near-duplicates; `trial.DEFAULT_EVERY = 5` is the precedent).

Also in this stage, the two guardrails:
- extend `tests/test_layer_separation.py` with `test_packages_never_import_tools` (AST scan for any `tools`/`autolabel` root import under `packages/*/src`);
- `tools/autolabel/` is **not** a uv workspace member and appears in no package's dependencies.

- **Gate: G6, plus** `styles.discover_styles()` finds the style, `cropcache.py` builds it, class counts are sane (empties dominate ~60 %, and the inverse-frequency sampler in `dataset.py` handles that), and a schema test locks the manifest.

### Stage 6 — the training gate (1 day + one cloud run)

`configs/run_j.yaml` = the deployed recipe + the autolabel style, leave-session-out and leave-style-out folds. Runs on RunPod per [the migration plan](2026-08-28-runpod-migration.md) (~$0.55 for a full-data run).

- **Gate: G5.** Ship or NO-GO like any other challenger. A NO-GO here is a perfectly good outcome and gets a report, the way the bank rematch did.

**Total: ~7.5 days + one ~$1 cloud run.**

---

## 5. The smallest end-to-end demo

One command, one screen, and a person can see whether it worked:

```bash
uv run python tools/autolabel/run.py recordings/20260828-111421 --against-truth --pgn
```

```
20260828-111421 · 243 frames · 1 session · orientation k=0 (margin 41.7 nats) · 0 restarts

  ply  seq   decoded            truth              posterior
   1    39   e4                 e4                   1.0000  ok
   2    39   e5                 e5                   1.0000  ok
   3    75   Nf3                Nf3                  1.0000  ok
  ...
  16   194   Qd8                Qd8                  0.9997  ok

  12/12 truth segments agree · squares fixed vs raw stream: 137 · broken: 0
  labelable frames: 212 / 243  (decoded 212 · boundary 31 · in_burst 0 · cold 0)

  1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 4. O-O O-O 5. Re1 Re8 6. d3 Qe7 7. Nc3 Qd8
```

That last line is the demo: **the reader wrote down the game, from 243 JPEGs, with no human in the loop.** It is observable, it is a claim anyone can check against the recording, and on a take with no `truth.json` it is still the whole product — the operator reads the PGN and recognises the game they played.

---

## 6. Risks and how each is handled

| Risk | Handling |
|---|---|
| **Label error lock-in** — a wrong decode trains the next reader to be wrong the same way | G2 at *segment* granularity, not per-square; frozen-parameter held-out gate (G3); per-frame posterior floor on export; `provenance` carried into the manifest; no export until 4/4 |
| **Only 73 distinct positions** — the engine multiplies frames, not boards | Stage 0 measures the prize *before* building; the payoff is on future recordings, and the plan says so rather than hiding it. Pairs with STATUS Next 10 (client-side session video) and idea C |
| **Decoder tuned to seven games** | Fit/held-out split with a hard freeze; parameters land as documented constants with their measurement, per the `ATTRACTOR_MARGIN` precedent |
| **Move segmentation on real footage** | Never segmented — `burst_id` tilts a soft `p_move` prior. Degrades to a pure hazard rate when the motion meter lies (which it did on every pre-08-27 take); `--restamp-stability` repairs those, applied uniformly or not at all |
| **Illegal real events** — a knocked-over piece, a takeback, adjusting, an operator resetting the board | `ε_restart` escape hatch, armed only on sustained low likelihood; **restart count is a headline diagnostic** and a take that restarts more than once is not exported |
| **The reader is confidently wrong in a way that is also legal** | Report `fixed`/`broken` paired against the raw stream. If `broken > 0` on any gate take, the engine does not ship. Emission temperature `α` is the lever that decides how much a confident read can overrule the graph |
| **Camera drift / corner misaddressing** | Detected as a likelihood collapse and marked `low_evidence`; **not fixed here**. Geometry repair is STATUS Next 8 / the framing-drift work, and conflating them would make both unmeasurable |
| **`python-chess` is GPL-3.0+ in an Apache-2.0 repo** | Lives only in `scratch/autolabelvenv` (gitignored, mirroring the torch venv), imported only by `tools/autolabel/`, never by `packages/`, never in a wheel or image. THIRD-PARTY.md gets an entry saying exactly that. Labels are facts about board states, not a derivative work. Alternative if this ever becomes uncomfortable: a ~150-line pseudo-legal generator — but correctness first, and castling/ep/promotion/PGN for free is worth a lot |
| **D3 layering erosion** — someone imports the decoder into the gateway "just for the viewer" | `test_packages_never_import_tools` (AST scan); `tools/autolabel` is not a workspace member; the decision is restated at the top of the module docstring, as `prior.py` does |
| **Two sessions in one recording** (`175539`) | Mandatory `find_sessions`/`longest_session`; a multi-session recording without `--all-sessions` is a warning, and decoding two games as one would otherwise be silent |
| **Same-set leakage into LOSO** — the takes and `uploads_calibrate` may be the *same physical chess set* | Must be checked before Stage 6 (open question below). If they are the same set, they must share a style or at minimum never straddle a LOSO split |
| **Feeding auto-labels into the bank** | Forbidden, in the export code and in the docs. See the callout in §3 |

---

## 7. Open questions, honestly

1. **Are the trial takes the same physical set as `data/uploads` / the curated boards?** If yes, an `autolabel_<set>` style held out by LOSO while `uploads_calibrate` stays in training is leakage, and the "unseen style" number is fiction. This must be answered before Stage 6, by eye if necessary.
2. **Is `α` stable across takes, or per-take?** A per-take temperature would be a much weaker claim (it is a free parameter per unit of evaluation). Stage 4 reports its fold-to-fold spread; stability is the evidence it is real, exactly as it was for the `T = 0.251` temperature fit.
3. **Does the decoder consume the black-bishop-demoted distribution or the pre-rule one?** The attractor swap happens *inside* the estimator, before `placement`/`evidence`/`prior.decode` are derived, so the dumped `square_dist` already carries it. That is probably right (it is what serves), but it means the decoder inherits a rule fitted on four takes, and a decode that has to *undo* the demotion is a signal worth watching.
4. **Should the export include `boundary` frames with the position they are transitioning *from*?** These are the hardest, most valuable frames in the corpus (a hand partially over the board) and the reason `guard_seqs` exists. Including them with an honest provenance flag is defensible; including them silently is not. Deferred to after G5.
5. **Do 232 000 near-duplicate crops from one camera actually help, or do they just re-weight the sampler?** `--every` decimation and the class sampler both bear on this; the answer is a Stage 6 measurement, not an argument.
6. **What does this do for idea B (broadcast footage)?** The decoder is the same; the PGN becomes an *input constraint* rather than a check, which collapses the beam to one path. That is a strictly easier problem and a natural follow-on — but the [angle audit](../youtube-streams.md) says 4/6 broadcasts sit below the operating envelope, so it extends the envelope downward rather than polishing the in-envelope stack.
7. **Does `stability` finally earn its keep?** DESIGN §9 has carried that open question since the start, unanswerable because nothing consumed it. This engine consumes it (via `burst_id` and the hazard rate). An ablation — decode with `p_move` constant vs burst-tilted — answers it for the first time, offline, for free.

---

## 8. Out of scope

Live decoding of any kind; anything on the wire; move/PGN output from the gateway; geometry or crop-attribution repair (STATUS Next 8); the detection-first re-architecture (idea 1) and SAM 3 masks (idea 3), which slot in *after* this and consume its output; feeding anything into `models/exemplars_*.npz`; and the DINOv3 probe (idea 2), which is a separate, cheaper experiment that should run first and under everything else.

### Documentation and registration

A new `autolabel` aspect in `aspects.yaml` (`files: ["tools/autolabel/**"]`), affecting `training` (it produces a style the harness consumes) and `docs`; a section in [docs/training-and-data.md](../training-and-data.md) describing the new data source and its provenance rules; a THIRD-PARTY.md entry for `python-chess`; and a manual path in [docs/TESTING.md](../TESTING.md) for the Stage-3 demo, per principle 1.
