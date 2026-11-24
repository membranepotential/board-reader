---
aliases: [ideas, reevaluation ideas, architecture options]
tags: [board-reader/ideas, board-reader/recognition, board-reader/data]
---

# Ideas

_Parked, not planned. Nothing here is committed work; when one is picked up it gets a
plan in `docs/plans/` and a gate like any other challenger._

## 2026-11-24 — a fully on-phone, portable app (user idea)

Run recognition in the phone's browser, so the phone is the whole system. Feasible
because ChessQ Lite int8 targets browser WASM (Lichess runs it client-side) and ChessCam
runs in the browser at ~100 ms per frame on a Pixel 7a CPU. The control channel and the
recognizer contract carry over; the read scheduler, prior repair and settle debounce
would need a TypeScript port. Open: the **sink** (Lichess broadcast or PGN, a WebSocket
endpoint, a local file) and latency on a mid-range phone. Full note, with a first
experiment: [[on-phone-app]].

## Live-viewer proposals from the 2026-08-27 mockups

The static UI mockups (`docs/mockups/`, deleted 2026-11-24; `git show 7f56f58:docs/mockups/viewer.html`)
carried two proposals that were never built: a **numbers** toggle on the live card
printing each square's top-class probability, and a **raw · prior · settled** switch naming
which board the card draws. The user asked for future mockups as interactive simulations,
not storyboards.

## What needs attacking now (2026-11-24)

Since ChessQ became the recognizer (DESIGN D10) the crop-stack residuals below are gone
with the crop stack. What remains, from the gate and the framing run
([[2026-11-24-settle-tuning]], [[2026-11-24-chessq-framing]]):

1. **Hands read as confident empties.** The burst-end scheduler and the settle debounce
   absorb them on the gate, at a median move-to-settle time of 3.0 s.
2. **Framing.** The board must fill at least half of the sent width.
3. **Data.** The gate is four takes of one physical set and one camera. More varied
   recordings (idea C) are the binding need, for measurement and for any offline LoRA
   (DESIGN M5).

## The 2026-08-29 step-back reevaluation (crop-stack era)

After the 08-28/29 improvement cycle closed with every challenger NO-GO
([[2026-08-29-bank-rematch]]), this survey reevaluated the system against the modern CV
landscape (research pass 2026-08-29). It was written for the crop stack. Status after
ChessQ: ideas A–C (data) still apply; ideas 1–3 (architecture) are moot, because the
end-to-end model they worked around now serves; idea 4 was refuted by ChessQ.

### The frame then — what needed attacking (historical)

The measured residual was **not** "classifier too weak" in general. It was three
specific things, and an idea that attacked none of them was noise:

1. **Per-square crops break under occlusion/lean** — 44 % of residual is crop
   bleed ([[2026-08-28-retrain-program]]); decapitation at phone height; the c6
   knight hallucinating a b6 rook (STATUS Next 8). The gray prism mask helps
   (occlusion masking, [[calibration-and-the-bank|deleted with the crop stack]])
   but is a patch on the crop assumption, not a fix.
2. **Piece-type transfer has a ~80 % frozen-DINOv2 ceiling**
   ([[training-and-data]], backbone bakeoff) — no frozen encoder beat it; scale
   was flat.
3. **Almost no labeled real data** — and the synthetic holdout provably does
   not predict real-photo accuracy (geometry sweep: +4.8 pp val → 0.0 pp real;
   run_i: holdout +2.9 pp, chesscog −3.2 pp).

## Data ideas

### A. Legality-constrained auto-labeling of game video (the data engine)

_Still applies; the reader it decodes is now ChessQ (~99 % per square). ChessQueries'
own PGN-aligned producer is the off-the-shelf alternative ([[2026-10-09-chessqueries-handoff]])._

A game video is not "unlabeled" — the legal-move graph is a brutal prior. Run
the current ~92 %-per-square reader over a whole game, then Viterbi-decode the
position *sequence* over legal transitions: single-frame ambiguities collapse,
yielding near-perfect frame labels from an imperfect reader. Those frames then
train the next reader (self-training with a *structural* teacher, not a model
teacher).

- Reuses the replay/recording machinery — frames + settled placements are
  already recorded per session.
- Risks: needs move segmentation to hold up on the footage; label errors can
  lock in. Mitigate by holding out games and verifying decoded streams against
  published PGNs where available.
- Note the layering irony: this builds (offline, at labeling time) exactly the
  temporal consumer DESIGN D3 says the live system must not be.

### B. PGN-aligned broadcast footage

For tournament footage the PGN is *published* — align settled frames to moves
and labels are free. [CVChess](https://arxiv.org/abs/2511.11522) (Nov 2025) did
this with filmed recreations of famous games.

**The catch is our own angle audit** ([[youtube-streams]]): 4/6 broadcasts sit
below the envelope (aspect < 0.62), and nothing on YouTube is a long game at
≥ 0.73. YouTube is therefore inherently **low-angle OOD data** — valuable *if*
the goal is extending the operating envelope downward, useless for polishing
the in-envelope stack.

### C. Own-recordings flywheel

Half-built already: recordings hold frames + settled placements;
`truth.json` authoring plus the queued client-side session video (STATUS
Next 9) makes every real game played a labeled dataset. Highest label quality,
exactly our domain; volume grows only as fast as games get played. A + C
compose: C supplies the in-envelope data, A makes labeling it nearly free.

### Refuted: VLMs as labeler/oracle

Even Qwen2.5-VL-72B sits ~54 % on FEN-adjacent tasks and accuracy collapses on
dense boards ([SEAM](https://arxiv.org/pdf/2508.18179),
[BYO-Eval](https://arxiv.org/pdf/2506.05440)). Not worth an experiment.

## Model / architecture ideas

### 1. Detection-first re-architecture — pieces as objects, not squares as crops

_Moot since 2026-11-24: ChessQ reads the whole frame end to end, so there are no crops
to bleed._

Detect every piece with a fine-tuned detector, take each detection's **base
point**, map it through the homography to a square. This dissolves the
bleed/decapitation/attribution family **by construction** — a leaning knight is
one object with one base, not pixels contaminating three crops.

- Candidate model: [RF-DETR](https://blog.roboflow.com/rf-detr/) (Apache 2.0,
  Mar 2025, DINOv2 backbone) — SOTA real-time accuracy, strong fine-tuning on
  small custom datasets, explicitly good on overlapping objects
  ([vs YOLO11](https://roboflow.com/compare/rf-detr-vs-yolo11)). Community
  chess pipelines already run this shape
  ([example](https://github.com/siromermer/Dynamic-Chess-Board-Piece-Extraction)).
- Labels: the Blender pipeline can emit boxes/keypoints for free
  ([[training-and-data|synthetic pipeline]]); real fine-tune data comes from
  idea A.
- Cost: a genuine second pipeline, and the bank — our best asset — needs
  rethinking (likely a per-detection embedding bank instead of per-square).

### 2. DINOv3 bank/backbone swap — PROBED 2026-08-31: NO-GO as a swap

_Closed; there is no bank or backbone of ours any more._

> [!important] Measured ([[2026-08-31-dinov3-probe]]): the frozen cross-style
> ceiling moves ~80 % → ~83 % (n.s., p = 0.137), central pooling does **not**
> compose (DINOv2-specific), and the deployed same-style path gains nothing.
> DINOv3 CLS is the first frozen encoder to cut piece-id errors rather than
> trade them, so it is banked as a mildly better fine-tune *initialisation*
> (exports kept in `models/`). The sequence below therefore starts at A.

The ~80 % frozen ceiling and the shape-bias refutation were measured on
**DINOv2**. [DINOv3](https://ai.meta.com/research/dinov3/) (Meta, Aug 2025)
fixed exactly what we lean on — dense/patch feature quality (Gram anchoring,
+6 mIoU on dense tasks, [paper](https://arxiv.org/html/2508.10104v1)) — ships
distilled [ViT-S 21M](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
/ [ViT-B 86M](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
sizes that fit our deploy, first-party Meta weights (per the official-sources
rule), commercial-friendly license.

Experiment: re-run the frozen-ceiling probe + central-column patch pooling (the
+9 win should compose), and swap the bank embedder. Days, not weeks; the local
GPU suffices for the probe.

### 3. SAM 3 as offline labeling/masking tooling

_Masking is moot with ChessQ; SAM 3 as labeling tooling for idea A still could apply._

[SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
(Nov 2025) segments *every instance* of a text concept ("chess piece") at once.
Too heavy to deploy; as tooling it upgrades two things: real instance masks
instead of the gray prism (train-time occlusion-aware crops), and auto-labeling
for A/1 (masks → base points → squares). Low-risk, bounded scope.

### 4. End-to-end whole-board model — explicitly a non-idea (REFUTED 2026-11-24)

> [!important] Refuted by ChessQ ([[2026-11-24-chessq-v6-replay-gate]]): ChessQ Lite V6,
> an end-to-end whole-board model, read 92/101 raw perfect boards on our gate, cold,
> against the crop stack's 5. It is now the recognizer (DESIGN D10). The paragraph
> below is the 08-29 reasoning, kept as written.

The literature's best end-to-end result on real photos
([ChessReD ResNeXt](https://arxiv.org/abs/2310.04086)) reaches **15.26 %
full-board accuracy** — celebrated as 7× the prior SOTA. Our pipeline does 35
perfect settled boards across 4 takes at ~92–99 % per-square. The field's e2e
frontier is behind us; don't chase it.

## Recommended sequence (2026-08-29, crop-stack era)

_Superseded by ChessQ: stage 3 is moot, and the RunPod pipeline is deleted. Stage 2's
prerequisite, varied recordings, is still open._

A chain, not a menu — each stage independently gateable on the existing 4-take
harness:

1. ~~**DINOv3 probe** (idea 2)~~ — done 2026-08-31, NO-GO as a swap
   ([[2026-08-31-dinov3-probe]]); banked as a fine-tune init.
2. **Legality-constrained auto-labeling** (A, on own recordings first, C;
   YouTube second — and the 08-31 [[youtube-streams|ASMR-niche finding]] makes
   B's PGN-aligned variant in-envelope too). **Plan landed
   ([[2026-08-31-autolabel-engine]]) and Stage 0 measured NO-GO on today's
   corpus** ([[2026-08-31-autolabel-stage0]]): 30 distinct positions, no
   uncontaminated gate moves. Deferred until the corpus has spread — the
   prerequisite is now explicitly **recording varied games on more than one
   set/camera**, which is idea C plus new hardware, not more engineering.
3. **Detection-first prototype** (idea 1) — fine-tuned on synthetic + those
   labels, gated like every other challenger. SAM 3 (idea 3) slots in when A
   needs masks.

## Related

- [[on-phone-app]] — the fully on-phone app idea
- [[2026-08-29-bank-rematch]] — the cycle whose close triggered the 08-29 survey
- [[training-and-data]] — the deleted harness these ideas would have extended
- [[youtube-streams]] — the angle audit constraining idea B
- [[operating-envelope]] — what "extending the envelope" means precisely
