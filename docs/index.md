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
| [[recognition-pipeline]] | How a photo becomes an 8×8 distribution |
| [[calibration-and-the-bank]] | How it adapts to *your* chess set, and how that goes wrong |
| [[measuring-accuracy]] | What "it works" means, where truth comes from, why the old number flattered |
| [[operating-envelope]] | Where it works, where it fails, and the measured thresholds |
| [[runbook]] | How to run it, what to check after a restart, how to recover |
| [[training-and-data]] | The offline harness, the data tiers, the licensing ledger |
| [[glossary]] | Terms that mean something specific here |

## Reference documents

- [[DESIGN]] — decisions, L3 protocol spec, milestones (`M0`–`M7`), work packages
- [[TESTING]] — the automated suite and the manual demo path
- [[STATUS]] — what is done, what is next, what is unresolved
- [[2026-07-18-zero-shot-campaign|Zero-shot campaign report]] — the investigation that
  produced the current recognizer
- [[2026-08-27-move-trigger-sweep|Move-trigger sweep]] · [[2026-08-27-framing-drift|Framing
  drift]] · [[2026-08-27-evidence-rerank|Evidence re-rank]] — the night-wave studies behind
  the read scheduler, the client re-detector and the b-demotion
- [[2026-08-28-proof-trial|Proof trial]] — the first live take over the full stack, scored
- [[2026-08-28-retrain-program|Retrain program E1–E7]] — bleed diagnosis → occlusion masking →
  masked-crop retrain; frozen baselines and the cloud-GPU continuation runbook
- [[2026-08-28-improvement-report|Improvement report]] — the program closed: serve-time
  masking shipped (perfect boards ≈20→35 across the gate takes), masked-retrained
  members benched with the why, RunPod pipeline measured ($0.66 total)
- [[2026-08-28-two-kings-decode|Two-kings decode]] — the follow-up refuted: the shipped
  demotion rule is already confidence-based and Pareto-best; ~7 settled boards is the
  ceiling of any per-frame tie-break, the residual lever is the bank lock
- [[youtube-streams]] — curated static-camera OTB footage for bring-up
- `aspects.yaml` — which parts must stay in sync with which, and the contract between them

## The one-paragraph version

The **edge** node (a Pi; a phone browser in the sim — `/camera`) finds the board and
forwards frames — rectified, or original with the corner quad on each frame header —
adaptively since M2: a ~1 fps keepalive at rest, a burst through motion, with a
measured `stability` on every frame. The browser edge also **tracks the board plane**,
so the quad follows the camera with its `a8` labelling intact (DESIGN §4.4.1(a))
through a bump, a hand sweeping over the board, or the phone moving to a new angle.
The **inference** node runs a stateless estimator over 64 square crops and publishes a
position event — since 08-27 not for every frame: a burst-end scheduler reads at the end of
each move plus a keepalive, ~12× fewer reads (`BR_SCHEDULE`, see [[runbook]]). The **viewer** renders it; the headless
`board-reader-inference positions` tails or captures it. Edge and inference never import
each other — they meet only on the wire. Recognition is a multi-style-trained crop classifier
blended with a per-set nearest-exemplar bank; the trained half is robust where it has never
been, the bank half is near-perfect where it has, and the blend tracks the better of the two
square by square.

> [!warning] The load-bearing failure mode
> Occupancy — *is this square empty?* — transfers across chess sets almost for free.
> Piece **type** does not. Nearly every accuracy problem in this project is a piece-type
> problem wearing a different hat. See [[recognition-pipeline#Why piece type is the hard part]].

> [!important] The layer above us has not been built
> The CV layer is a stateless per-frame estimator by decision, and `stability` is advisory:
> the *consumer* is supposed to smooth across frames and segment moves
> ([[DESIGN#D3 — Stability is an advisory signal; the consumer segments moves|D3]], §5).
> No such consumer exists — here or anywhere else — so nothing yet turns this noisy
> per-frame stream into a stable position history. See
> [[architecture#Temporal consistency — the layer nobody has built]].
