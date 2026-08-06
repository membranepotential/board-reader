---
aliases: [operating envelope, limits, when it fails]
tags: [board-reader/recognition, board-reader/ops]
---

# Operating envelope

Where the reader works, where it does not, and the numbers behind each boundary. Everything
here was measured on real photographs, not inferred.

## The short version

| Condition | Threshold | What the server does |
|---|---|---|
| Camera elevation (projected board aspect) | usable ≥ 0.73; warn below **0.62**; garbage at 0.56 | `very_low_angle` warning |
| Light level (mean pixel value) | warn below **55**; evening rooms measure ~37 | `low_light` warning |
| Chess set | any — zero-shot; a calibration photo makes it better | nothing |
| Position | any — no start-position requirement | nothing |

Neither warning blocks a read. The design streams what it sees and flags what it doubts;
suppressing output is reserved for a *geometrically* invalid state (a bumped or repositioned
camera), because a mis-localized camera produces confidently-wrong crops and a consumer
cannot tell a confident lie from a real read.

## Camera angle

The measure is the **projected board aspect**: the short side of the corner quad over its
long side. It is a proxy for elevation that needs no calibration — just the corners you
already have.

| Aspect | Elevation, roughly | Behaviour |
|---|---|---|
| 0.73 – 0.88 | ~45–55° | the band every good result was measured in |
| 0.62 – 0.73 | ~40° | degraded; the warning fires below 0.62 |
| ~0.56 | seated across the table, phone barely raised | **near-random output** |

At 0.56 the failure is geometric, not statistical: pieces occlude whole ranks, and each
square's crop mostly contains the *next* rank's piece. Corners and orientation can both be
perfectly correct and the read is still worthless. The regime is simply outside anything
trained or evaluated.

> [!tip] The fix is free
> Raise the phone. The UI says so when the warning fires.

Synthetic training data now samples aspect 0.5–0.9 (mode ~0.75, with a deliberate ~23% tail
in 0.5–0.65), so the flat-angle regime is being *learned* rather than only warned about.

## Light

Warm dim tungsten room light is the hardest realistic condition, and it interacts badly with
steep angles. The trained model alone drops to ~84% there, which is the one regime where a
per-set calibration still clearly beats it — so the calibrate path covers exactly the gap the
zero-shot path leaves.

The self-refresh mechanism ([[calibration-and-the-bank#Zero-interaction calibration]]) is the
practical answer: photographing the board before each game in that evening's light took
dim-room accuracy from ~50% to 98.8%.

## The steep-and-dim corner

The two worst conditions together are the known weakness. Its mechanism is understood — see
[[recognition-pipeline#Cropping is where the accuracy is]] — and the fix was to retrain with
tall crops, which moved the hardest board from 57/64 to 59/64 with the ensemble. It is
improved, not solved.

> [!example] What failure looks like
> Not noise. **Systematic major-piece confusion**: q→b, K→Q, R→P, B→P, plus phantom pawns one
> square above a decapitated head. If you see that signature, suspect crop geometry or camera
> elevation — not the classifier.

## What is measured on, and what that means

Every accuracy figure in this repo is relative to a small labeled eval — currently 10 boards
/ 640 squares of user photos plus 12 dim calibrate uploads. Two consequences worth holding
onto:

- Figures from different eval sizes are **not comparable**. A model evaluated on 19 boards
  and one evaluated on 22 do not have comparable "overall" numbers; compare only on the
  shared boards.
- 10 boards is thin. A 6-square difference between two models is inside the noise you should
  expect; a 50-square difference is not.

`BR_SAVE_UPLOADS` exists to fix this over time: every phone photo is retained with a sidecar
recording the predicted placement and whether it was auto-rotated, so every field shot becomes
eval data without forensics.

## Related

- [[recognition-pipeline]] — the mechanisms behind these limits
- [[calibration-and-the-bank]] — what to do about the dim regime
- [[runbook]] — checking a live server against these expectations
