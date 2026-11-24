---
aliases: [training, datasets, styles, licensing]
tags: [board-reader/training, board-reader/data, board-reader/history]
---

# Training and data (historical)

> [!info] Removed on 2026-11-24 (plan decision D7)
> No in-house model is served any more: ChessQ Lite V6 is the recognizer (DESIGN D10). The
> training harness (`tools/train/`), the RunPod tooling (`tools/runpod/`, the `runpod`
> skill), the DINOv2 export and the crop export from recorded takes are deleted. The last
> commit that has all of it is `457cf96`; this note's full text is in that commit too.

## What it was

An offline harness that trained the crop classifier on many piece styles at once (photos,
Blender renders, public datasets), with leave-one-style-out splits, run observability and
an ONNX export carrying its crop geometry as a sidecar. It ran on its own torch interpreter,
locally and later on rented RunPod GPUs.

## What it taught

- **Piece type does not transfer across chess sets on frozen features; occupancy does.**
  Seven training-free routes stopped at roughly the same ~80 % ceiling; learning from many
  styles at once reached 92.5 % per square on an unseen set
  ([[2026-07-18-zero-shot-campaign]]).
- **Member-level wins did not survive the gate.** Retrained cascade members that won on
  held-out crops lost every gate take once deployed together
  ([[2026-08-28-improvement-report]], [[2026-08-28-retrain-program]]).
- **Keep the eval tiers apart.** Training, validation, test and auxiliary data were held in
  separate tiers so that no number was graded by data that shaped the model.

## Data and licences

Nothing here is served any more, but the datasets are still on disk under the git-ignored
`data/`: ChessReD is **CC BY-NC-SA 4.0** (non-commercial), the ChessR render geometry is of
unclear licence, ChessRender360, the chesscog transfer set and the Zenodo crops are CC BY.
No dataset file or derived artifact was ever committed, and none may be. The served model's
licence is in `THIRD-PARTY.md`.

## Related

- [[recognition-pipeline]] — what reads boards now
- [[calibration-and-the-bank]] — the per-set half of the old stack
- [[measuring-accuracy]] — how models are compared now (the replay gate)
