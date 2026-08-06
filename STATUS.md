# STATUS — board-reader

*Canonical dashboard. Last reconciled 2026-08-06 against `git log`, `scratch/runs/`, `data/eval/`.*
Design decisions live in [docs/DESIGN.md](docs/DESIGN.md); this file is state, not design.

## Goal

A camera reads a live OTB board and streams timestamped **piece placements + per-square
confidences**. No moves, no legality, no PGN — those are the downstream application layer's.

## Now

Nothing in flight. The 2026-07-17/18 recognition campaign landed and is committed (5 commits,
202 tests green); the working tree that held it is now empty. The GPU is free.

## Done

- **M0 — walking skeleton.** `edge → inference → viewer` over WebSocket, `docker compose up`.
- **M1 — single-frame estimation.** Board detect + rectify on the edge; per-square 13-class
  distributions on the inference node. Exceeded: it reads real phone photos, not just clips.
- **Snapshot application** (built on M1, not a milestone). `GET /snapshot` on the viewer →
  `POST /estimate`: photo in, position out, zero taps. Corner auto-detect (`/detect`),
  drag-editable corners, orientation auto-correction, distribution correction panel,
  Lichess links. Mobile-verified.
- **Zero-shot recognition.** Seven training-free routes measured dead (~80% ceiling, all
  gated by piece *type*, never occupancy). A multi-style ResNet18 (`d16`, tall crops) reads
  an unseen set at 92.5% per-square with no calibration. Full write-up:
  [docs/reports/2026-07-18-zero-shot-campaign.md](docs/reports/2026-07-18-zero-shot-campaign.md).
- **Calibration as a booster.** kNN exemplar bank ⊕ trained model, blended by bank maturity:
  633/640 on the labeled eval vs 605 (trained alone) / 579 (previous model). **That 633 is
  not a generalization number** — see Open questions.
- **Guardrails from four bank-corruption incidents.** Opt-in persistence, orientation probe
  on `/calibrate`, piece-square agreement gate (422), `very_low_angle` / `low_light` warnings.

## Next

1. **Decide `run_e`.** The wave-2 retrain (11 styles) finished 2026-07-19 and was never
   evaluated as an ensemble or deployed. Alone vs the deployed `d16`: **+53 squares** on the
   12 dim calibrate boards, **−6** on the 10 bright boards (+47 net over the common 19).
   Run `d16 ⊕ bank` vs `run_e ⊕ bank` on the full eval, then deploy or drop it.
2. **M2 — live position stream over a full recorded game** (the headline demo). The whole
   recognition stack so far is single-frame; M2 is adaptive forwarding + stability + streaming
   through hands and transitions. This is the actual next milestone.
3. **Central-column patch pooling** for the DINOv2/kNN path (+9 squares cross-style, measured,
   never shipped).
4. **Repair or replace the `chesscog_transfer` holdout** — it is geometrically broken
   (0.597 self-kNN) and made a good run look like a failure once already.

## Open questions

- **The headline accuracy is measured on contaminated data (found 2026-08-06).** The
  session bank's base is `models/exemplars_dinov2.npz`, which `tools/build_exemplar_bank.py`
  builds *from `data/eval/ground_truth.json`* — the eval set. So the bank has seen 9 of the
  10 eval boards, and scores 64/64 on exactly those 9; on `up_20260718_0001`, the one board
  it never saw, it scores **43/64**. The honest generalization figure is the trained model
  alone: 605/640 per-square but only **2/10 perfect boards**. Whole-board correctness, not
  per-square, is the number that matters for reading a game, and it has never been the
  number we report. Needs: an eval set disjoint from the bank, and perfect-board rate as the
  headline metric.
- **Confidence is not calibrated.** A novel board (`20260806-...-0002`) read with two white
  kings and no black king at **confidence 0.97**. DESIGN §9 lists confidence calibration as
  a top risk because the whole downstream-smoothing plan leans on confidences being
  meaningful; it is still unmeasured. Reliability diagrams, not accuracy.
- Is `run_e`'s bright-board regression real or eval noise? 10 boards is thin.
- Does `stability` earn its keep (DESIGN D3)? Undecided until M2 produces a stream to judge.
- Licensing: the deployed model is trained on ChessReD (CC BY-NC-SA) + murky-license
  geometry — personal use only. A clean-data retrain is the path if that ever changes.
- No lint gate. `ruff` is a system binary with no project config; ~109 findings and 41
  unformatted files sit unaddressed. Adopt it deliberately or drop it — currently it is
  neither.
