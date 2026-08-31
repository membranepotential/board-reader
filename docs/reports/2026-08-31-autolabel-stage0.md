# Auto-label Stage 0 — the null hypothesis wins: NO-GO on the decoder, record more games

_2026-08-31 · Stage 0 of [[2026-08-31-autolabel-engine|the auto-label engine plan]]:
before building the Viterbi decoder, test whether HAND-PERFECT labels from the
recorded corpus move the gates — if they don't, auto-labels won't. They don't.
Cloud spend $0.43 (secure 4090, EUR-IS-1, landed by a 60 s retry loop after a
cross-DC probe). Detail: `scratch/experiments/autolabel_stage0/`._

## Gate (a) — the prize, measured

The seven hand-truthed takes yield **3,604 exportable frames → 46,336 crops**
(every=5) — but only **30 globally distinct positions**, not the plan's 73:
the takes replay the same opening (the start position in all 7, three more in
6 of 7). One opening, one chess set, one camera, one operator. The plan's own
caveat — "the engine multiplies frames, not positions" — is ~2.4× more binding
than it estimated.

## Gate (b) — a landslide that is actually a memorization readout

`run_j0` (deployed run_e recipe + the 7 hand-truth styles) vs the `s0-control`
(shipped pair, reproduces the 35-perfect baseline exactly):

| settled | control | run_j0 |
|---|---|---|
| 08-13 | 22/31 · .9899 | **31/31 · 1.0000** |
| 111421 | 5/23 · .9266 | 12/23 · .9660 |
| 175539 | 5/12 · .9609 | 8/12 · .9870 |
| 181802 | 3/43 · .9113 | 4/43 · .9233 |
| **total** | **35 · .9424** | **55 · .9612** |

Naively +20 perfect boards. But 4 of the 7 training styles ARE the gate takes,
and the proximity measurement kills the number dead:

```
gate boards with the exact frame in training:  26/109  (23.9 %)
within 1 frame of a trained frame:             69/109  (63.3 %)
within 5 frames (≈ <1 s of recorded time):    109/109  (100 %)
```

**Every scored board is within a second of an image the model trained on.**
The replay gate silently stops being a gate for a candidate trained on its
sessions. The uncontaminated signals all say nothing happened:

| held-out signal | run_e | run_j0 |
|---|---|---|
| chesscog (different set, untrained) | 0.819 | 0.823 |
| curated stills through the deployed cascade+bank | 608/640 | **606/640** |
| synthetic holdout | 0.898 | 0.897 |

(The one member-level gain, user_photos +1.5 pp, is the trained-on domain and
does not survive into the served cascade+bank — the run_g/run_h lesson again.)

## Verdict

**NO-GO on the 7.5-day decoder build, on today's corpus.** Hand-perfect labels
move no uncontaminated gate; the decoder would multiply near-duplicates of 30
positions in one style. The measured bottleneck was never frame count — it is
style-entangled features (four frozen-backbone negatives; run_i's
holdout-up/transfer-down). The engine's value scales with *games played*, and
that axis is near-empty: **the priority is recording — varied openings,
mid-game positions, and above all a second physical set and a second camera.**
The export path, style layout, run_j0 config, and staged volume all survive
and make the re-run cheap once the corpus has real spread.

## Follow-ups

1. **`gates.sh` needs a scoring-side contamination guard.** Plan G6 guards the
   exporter; this run shows a candidate trained on a gate session can print a
   20-board "win" with no warning. The gate should detect training/gate
   session overlap (e.g. via the style's `sources.json`) and refuse or label
   the row.
2. The 08-28 "curated 304/320" style docs pins are stale — the curated set
   grew to 10 boards / 640 squares when the 08-31 calibrate five entered
   `ground_truth.json`; reconcile on next docs pass.

## Ops notes (the day's cloud lesson)

RunPod stock listings are near-worthless ("Low" everywhere, creates failing
everywhere); the only capacity signal is a create attempt. A cross-DC probe
(24 create+instant-delete attempts, 8 DCs, ~1¢) found exactly one creatable
spec (US-IL-1 secure 4090) — and minutes later a 60 s retry loop landed a
secure 4090 in EUR-IS-1 anyway. Method now in the runpod skill: probe by
create, retry fast, never trust the listing. Volume `1heed8wd1u` (EUR-IS-1,
staged) stands; the US-IL-1 detour volume was deleted same-hour.
