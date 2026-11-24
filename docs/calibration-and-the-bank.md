---
aliases: [calibration, exemplar bank, session bank]
tags: [board-reader/recognition, board-reader/calibration, board-reader/history]
---

# Calibration and the exemplar bank (historical)

> [!info] Removed on 2026-11-24 (DESIGN D10, plan decision D5)
> ChessQ reads every piece set as shipped, so there is no calibration round any more.
> `POST /calibrate`, the occupancy-only start, start-position auto-calibration, the
> per-set exemplar bank (`BR_SESSION_BANK`, `BR_OCCUPANCY_BANK`, `BR_CENTROIDS`) and
> `rebuild-bank` are deleted. The last commit that has all of it is `457cf96`; this note's
> full text is in that commit too.

## What it was

The old crop classifier read piece *type* badly on a chess set it had never seen. The fix
was a per-set **exemplar bank**: one photo of a known position (the start position by
default) gave 64 labeled crops, embedded once, and every later square was blended 50/50
between the trained classifier and a nearest-exemplar lookup in that bank. The bank could
be built by a tap (`/calibrate`), with zero taps (a start position the classifier already
read confidently), and offline from retained uploads.

## What it taught

- **A bank scores near-perfectly on the photos it was built from.** Any eval that reuses
  them is an upper bound, never a measurement — the rule behind keeping labels as eval data
  only (`/labels` never feeds recognition). See [[measuring-accuracy]].
- **Calibration poisons quietly.** A wrong corner orientation, a mid-game photo sent as the
  start position, or a too-flat angle wrote wrong exemplars that then read with high
  confidence. Each needed its own guard; the lesson is that an adaptation step is a second
  model to validate, not a free improvement.
- **Bank writes are not monotone.** Rebuilt banks lost 13 perfect boards on the four gate
  takes ([[2026-08-29-bank-rematch]]). A calibrate on a new set helped that set and cost
  the old takes 35 → 25 settled perfect boards, while per-square accuracy rose on every
  take — per-square metrics hid the regression ([[2026-08-31-calibrate-before-after]]).

Any future adaptation to a specific set is an offline LoRA producing a new ONNX file
(plan decision D5), measured on the replay gate like any other model.

## Related

- [[recognition-pipeline]] — what reads boards now
- [[training-and-data]] — the other deleted half of the old stack
