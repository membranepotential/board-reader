---
aliases: [docs home, board-reader docs]
tags: [board-reader/moc]
---

# board-reader — map of content

A camera watches a chess board and streams **timestamped piece placements with
per-square confidences**. It reads *positions*, not moves — legality, move
reconstruction and PGN belong to whatever consumes the stream.

> [!abstract] Two kinds of document
> **[[DESIGN]]** owns *decisions* — why the system is shaped this way, what is
> locked, what is out of scope. The notes below own *behaviour* — how it works
> right now, what it measures, where it breaks. When they disagree, DESIGN wins
> on intent and the notes are wrong about fact.

## Start here

| Note | Answers |
|---|---|
| [[architecture]] | What the pieces are, what may import what, how a frame travels |
| [[recognition-pipeline]] | How a frame becomes an 8×8 distribution (ChessQ, the chess frame, framing) |
| [[measuring-accuracy]] | What "it works" means, where truth comes from, why the old number flattered |
| [[operating-envelope]] | Where it works, where it fails, and the measured thresholds |
| [[runbook]] | How to run it, what to check after a restart, how to recover |
| [[glossary]] | Terms that mean something specific here |
| [[calibration-and-the-bank]] · [[training-and-data]] | Historical: the deleted crop stack's calibration and training, and what they taught |

## Reference documents

- [[DESIGN]] — decisions (live and superseded), L3 protocol spec, milestones (`M0`–`M7`)
  with their status, work packages and the roadmap
- [[TESTING]] — the automated suite and the manual demo path
- [[STATUS]] — what is done, what is next, what is unresolved
- [[2026-11-24-chessq-v6-replay-gate|ChessQ V6 replay gate]] — why ChessQ became the
  recognizer (92/101 raw, 84/101 settled vs the crop stack's 5 and 35);
  [[2026-11-24-chessq-framing|framing]] and [[2026-11-24-chessq-gpu-formats|GPU formats]]
  measure its limits; [[2026-11-24-settle-tuning|settle tuning]] sets today's schedule and
  debounce (104/119 settled perfect)
- [[on-phone-app]] — idea: the whole system in the phone's browser

**History (the crop-stack era, before 2026-11-24).** These reports measured a recognizer
that no longer exists; their lessons on measurement still hold.

- [[2026-07-18-zero-shot-campaign|Zero-shot campaign report]] — the investigation that
  produced the old crop classifier
- [[2026-08-27-move-trigger-sweep|Move-trigger sweep]] · [[2026-08-27-framing-drift|Framing
  drift]] · [[2026-08-27-evidence-rerank|Evidence re-rank]] — the night-wave studies behind
  the read scheduler, the client re-detector and the b-demotion (both since removed)
- [[2026-08-28-proof-trial|Proof trial]] — the first live take over the full stack, scored
- [[2026-08-28-retrain-program|Retrain program E1–E7]] — bleed diagnosis → occlusion masking →
  masked-crop retrain; frozen baselines and the cloud-GPU continuation runbook
- [[2026-08-28-improvement-report|Improvement report]] — the program closed: serve-time
  masking shipped (perfect boards ≈20→35 across the gate takes), masked-retrained
  members benched with the why, RunPod pipeline measured ($0.66 total)
- [[2026-08-28-two-kings-decode|Two-kings decode]] — the follow-up refuted: the shipped
  demotion rule is already confidence-based and Pareto-best; ~7 settled boards is the
  ceiling of any per-frame tie-break, the residual lever is the bank lock
- [[2026-08-29-bank-rematch|Bank rematch]] — masked banks and full-data training both
  NO-GO; the bank's value is session adaptation (exemplars only recordings hold), and
  a curated gain is again not a deployment verdict
- [[2026-08-31-new-style-batch|New-style batch]] · [[2026-08-31-calibrate-before-after|Calibrate
  before/after]] · [[2026-08-31-two-kings-reopen|Two-kings reopen]] ·
  [[2026-08-31-dinov3-probe|DINOv3 probe]] · [[2026-08-31-autolabel-stage0|Auto-label stage 0]] ·
  [[2026-08-28-cloud-training-migration|Cloud training migration]] — the last crop-stack wave

**Ideas and data.**

- [[ideas]] — parked ideas: the on-phone app (2026-11-24), then the 2026-08-29 step-back
  survey (data engines still apply; its architecture options are moot since ChessQ)
- [[viterbi-auto-labeling]] — the algorithm behind ideas §A: how sequence decoding
  over the legal-move graph turns an imperfect per-frame reader into a near-perfect labeler
- [[youtube-streams]] — curated static-camera OTB footage for bring-up

## The one-paragraph version

The **edge** node (a Pi; a phone browser in the sim — `/camera`, a dumb camera configured
from `/control` on the laptop) forwards a rectangle crop in which the board fills at least half of
the width, adaptively since M2: a ~1 fps keepalive at rest, a burst
through motion, with a measured `stability` on every frame. The **inference** node reads each
frame with one model call — ChessQ Lite V6 finds the board and its orientation and returns the
8×8 distribution in the chess frame — repairs impossible boards beside the read, debounces a
settled board, and publishes a position event. It does not read every frame: a burst-end
scheduler reads three times at the end of each move plus a keepalive every 3 s
(`BR_SCHEDULE`, see [[runbook]]). The **viewer** renders the stream; the headless `board-reader-inference positions`
tails or captures it. Edge and inference never import each other — they meet only on the wire.

> [!warning] The load-bearing failure modes
> Hands read as confident empty squares, so confidence cannot keep a hand off the board;
> read scheduling and the settle debounce do. And a board that fills too little of the frame
> is misread. See [[recognition-pipeline#Known weaknesses]].

> [!important] The layer above us has not been built
> The CV layer is a stateless per-frame estimator by decision, and `stability` is advisory:
> the *consumer* is supposed to smooth across frames and segment moves
> ([[DESIGN#D3 — Stability is an advisory signal; the consumer segments moves|D3]], §5).
> No such consumer exists — here or anywhere else — so nothing yet turns this noisy
> per-frame stream into a stable position history. See
> [[architecture#Temporal consistency — the layer nobody has built]].
