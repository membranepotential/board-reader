---
aliases: [operating envelope, limits, when it fails]
tags: [board-reader/recognition, board-reader/ops]
---

# Operating envelope

Where the reader works, where it does not, and the numbers behind each boundary. Everything
here was measured on real photographs, not inferred. The recognizer is ChessQ Lite V6
([[recognition-pipeline]]); it reads the whole frame and finds the board itself.

## The short version

| Condition | Threshold | What the server does |
|---|---|---|
| Board size in the frame | board ≥ **50 %** of frame width; ~83 % is best; 35 % fails | nothing |
| Light level (mean pixel value) | warn below **55**; evening rooms measure ~37 | `low_light` warning |
| Chess set | any — zero-shot, no calibration | nothing |
| Position | any — no start-position requirement | nothing |

The warning does not block a read. The design streams what it sees and flags what it
doubts (DESIGN D1): an illegal-looking board is still reported as `ok`.

## Framing

ChessQ squashes the whole input to 644×644, so a board that fills little of the frame
loses pixels and its squares get distorted. Measured on 467 images
([[2026-11-24-chessq-framing]]):

| Board share of frame width | Per-square | Perfect boards |
|---|---|---|
| ~83 % (as photographed) | 99.4 % | 82 % |
| 50 % (16:9, centered) | 99.3 % | 79 % |
| 35 % (16:9, centered) | 91–99 % (padding-dependent) | 5–69 % |
| 80 % of a portrait 9:16 frame | 98.4 % | 57 % |

> [!tip] The fix is free
> Frame the board so it fills at least half the width of a landscape shot. Do not send a
> wide room shot; crop to the board with a margin first.

## Camera angle and light

These limits were measured on the earlier crop-classifier stack (removed; last in
`457cf96`). There, a projected board aspect below 0.62 (a camera barely raised above the
table) gave near-random output, and warm dim room light was the hardest realistic
condition. ChessQ has not been measured separately on flat angles or dim rooms; treat both
as the likely weak spots until it is. Only the `low_light` warning remains on the server.

## What is measured on, and what that means

Every accuracy figure in this repo is relative to a small labeled eval: 21 stills
(`ground_truth.json` 10, `heldout-20260831` 11), the chess-frame upload labels (47 boards
in total with the stills), and the replay gate's four recorded takes (101 truth-covered
boards). Two consequences worth holding onto:

- Figures from different eval sets are **not comparable**. Compare only on the shared
  boards.
- The sets are thin. A few squares' difference between two runs is inside the noise you
  should expect; a large difference is not.

`BR_SAVE_UPLOADS` exists to fix this over time: every phone photo is retained with a sidecar
recording the predicted placement and the frame it was read in, so every field shot becomes
eval data without forensics.

## Related

- [[recognition-pipeline]] — the mechanisms behind these limits
- [[measuring-accuracy]] — how the numbers above are produced
- [[runbook]] — checking a live server against these expectations
