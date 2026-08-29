---
aliases: [ideas, reevaluation ideas, architecture options]
tags: [board-reader/ideas, board-reader/recognition, board-reader/data]
---

# Ideas — the 2026-08-29 step-back reevaluation

_Parked, not planned. After the 08-28/29 improvement cycle closed with every
challenger NO-GO ([[2026-08-29-bank-rematch]]), this survey reevaluated the
system against the modern CV landscape (research pass 2026-08-29). Nothing here
is committed work; when one is picked up it gets a plan in `docs/plans/` and a
gate like any other challenger._

## The frame — what actually needs attacking

The measured residual is **not** "classifier too weak" in general. It is three
specific things, and an idea that attacks none of them is noise:

1. **Per-square crops break under occlusion/lean** — 44 % of residual is crop
   bleed ([[2026-08-28-retrain-program]]); decapitation at phone height; the c6
   knight hallucinating a b6 rook (STATUS Next 8). The gray prism mask helps
   ([[recognition-pipeline#Occlusion masking — the deployed default since 2026-08-28]])
   but is a patch on the crop assumption, not a fix.
2. **Piece-type transfer has a ~80 % frozen-DINOv2 ceiling**
   ([[training-and-data]], backbone bakeoff) — no frozen encoder beat it; scale
   was flat.
3. **Almost no labeled real data** — and the synthetic holdout provably does
   not predict real-photo accuracy (geometry sweep: +4.8 pp val → 0.0 pp real;
   run_i: holdout +2.9 pp, chesscog −3.2 pp).

## Data ideas

### A. Legality-constrained auto-labeling of game video (the data engine)

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
  ([[synthetic pipeline|training-and-data]]); real fine-tune data comes from
  idea A.
- Cost: a genuine second pipeline, and the bank — our best asset — needs
  rethinking (likely a per-detection embedding bank instead of per-square).

### 2. DINOv3 bank/backbone swap — the cheap probe, do first regardless

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

[SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
(Nov 2025) segments *every instance* of a text concept ("chess piece") at once.
Too heavy to deploy; as tooling it upgrades two things: real instance masks
instead of the gray prism (train-time occlusion-aware crops), and auto-labeling
for A/1 (masks → base points → squares). Low-risk, bounded scope.

### 4. End-to-end whole-board model — explicitly a non-idea

The literature's best end-to-end result on real photos
([ChessReD ResNeXt](https://arxiv.org/abs/2310.04086)) reaches **15.26 %
full-board accuracy** — celebrated as 7× the prior SOTA. Our pipeline does 35
perfect settled boards across 4 takes at ~92–99 % per-square. The field's e2e
frontier is behind us; don't chase it.

## Recommended sequence (when picked up)

A chain, not a menu — each stage independently gateable on the existing 4-take
harness, cheap on the RunPod pipeline:

1. **DINOv3 probe** (idea 2) — cheapest, may raise the ceiling under
   everything else.
2. **Legality-constrained auto-labeling** (A, on own recordings first, C;
   YouTube second) — the only scalable source of real labels.
3. **Detection-first prototype** (idea 1) — fine-tuned on synthetic + those
   labels, gated like every other challenger. SAM 3 (idea 3) slots in when A
   needs masks.

## Related

- [[2026-08-29-bank-rematch]] — the cycle whose close triggered this
- [[training-and-data]] — the current harness these ideas would extend/replace
- [[youtube-streams]] — the angle audit constraining idea B
- [[operating-envelope]] — what "extending the envelope" means precisely
