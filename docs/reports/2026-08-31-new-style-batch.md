# 2026-08-31 — a new board style, measured cold

17 iPhone photos of a **board+piece style the stack had never seen** (brown/white
vinyl roll-up, cheap plastic Staunton, outdoor picnic table, overcast) —
`data/own/2026-08-31_iphone/`. Labeled and scored against the **deployed**
estimator with **no bank write**, so every new board is genuinely cold.

## The batch

| | |
|---|---|
| `0748`–`0762` | 15× the **start position**, angles from near-overhead (aspect 0.95) to steep oblique (0.53) |
| `0763`, `0764` | the **Ruy Lopez** after 1.e4 e5 2.Nf3 Nc6 3.Bb5, side-on, board rotated 90° in frame |
| `0760` | corner detection **FAILed** — hardest angle in the set; unlabeled, needs `tools/corner_picker.html` |

Split (user call): **5 calibrate** (`0748/0749/0753/0755/0761` →
`data/eval/ground_truth.json`) / **12 held out** (`data/eval/heldout-20260831/*.label.json`).
The split is load-bearing because `tools/build_exemplar_bank.py` reads
`ground_truth.json` — anything landed there is bank material, never eval. The two
Ruy Lopez boards are held out unconditionally: they are the batch's only
non-start positions, hence its only real recognition test.

**No bank was rebuilt this pass**, so the calibrate five are *also* cold here —
all 16 labeled boards are unseen.

### Authoring the truth without asking the model

Corners came from `detect_board_quad` (16/17). Which corner is `a8` is not
geometric (DESIGN #15), and the live path resolves it with
`autocalib.resolve_rotation` — which reads the *model's* predictions. Using that
to orient a truth label would make the label partly a model output. Two
model-free checks were used instead:

1. **Back-rank luminance** — pick the quarter-turn where the top two rows are
   darkest (black men on ranks 7–8). Won by a **27–62 grey-level margin** on all
   16; unique, since only one of four rotations puts black on top.
2. **Chirality** — a rotation cannot produce a mirror, but a wrongly-wound quad
   can. Checked `a3`/`b3`/`a4`/`b4` against the `a1`-is-dark invariant on the
   empty middle ranks. All 16 passed with ~70 vs ~240 separation.

Both then verified by eye on a rectified contact sheet.

## Result — deployed stack, cold

`Ensemble(Cascade(run_e, run_f) ×0.30 + Bank(1024 exemplars) ×0.70)`,
`BR_CROP_MASK=gray`.

| group | squares | per-square | perfect boards |
|---|---|---|---|
| old curated (5) | 305/320 | 95.3 % | 0/5 |
| new **calibrate** (5, still cold) | 303/320 | 94.7 % | 0/5 |
| new **held-out** (11) | 680/704 | **96.6 %** | 1/11 |

**The style transfers.** Per-square accuracy on a never-seen board and piece set
is *higher* than on the curated set the bank was built from. That cuts against
the standing prior that cross-style piece-type recognition is the weak point
(`docs/…/backbone-bakeoff`, the ChessReD finding at 79.7 %). Whatever else is
true, this style is not hard for the shipped ensemble at the square level.

## The one defect: the queen reads as a king

Perfect-board rate is 1/11 not because errors are spread thin, but because a
**single systematic confusion** lands on almost every board:

```
Q->K  13    q->k   8      <- 21 of 41 new-style errors
B->P   5    b->k   3    N->B  3    b->p  2    n->r  2    Q->B  2
```

Nearly always on the **home squares** `d1`/`d8`. The raw estimator reads *both*
`d1` and `e1` as kings; the hard-constraint decoder then has to demote one, and
the repair log shows it choosing wrong on real boards:

```
img_0756  e1 K->Q  cost 1.82  kings  truth K   BROKEN
img_0758  e1 K->Q  cost 1.70  kings  truth K   BROKEN
img_0761  d1 K->Q  cost 1.48  kings  truth Q   FIXED
```

This is the two-kings scenario from `2026-08-28-two-kings-decode.md`, which
concluded the shipped tie-break was Pareto-best against a **~7-board** ceiling on
the 08-13 trial. This batch reproduces it on **13 boards of a fresh style** — a
substantially better test bed for that question than the one that closed it.

Physically plausible: on this set the queen's coronet and the king's cross are
both small top features, and `BR_CROP_MASK=gray` is masking occluders right where
the discriminative detail sits.

`img_0759` is the only perfect board in all 21.

## Confidence

ECE **0.345**, under-confident in the same direction as ever (the 0.3–0.4 bin is
actually 92 % correct). Board confidence barely separates the one perfect board
(0.664) from the twenty imperfect ones (0.612) — consistent with STATUS's
"confidence is not usable today".

## What this opens

1. **Calibrate the five and re-measure.** The obvious test of whether user
   calibration fixes `Q->K` on a new set. Blocked on the un-gated non-monotone
   bank write (STATUS open question) — worth doing behind a before/after on this
   very batch, which is now the cleanest available regression probe.
2. **Reopen the two-kings tie-break** with 13 fresh boards instead of 7.
3. **Ask for varied positions, not more start positions.** Every calibrate
   candidate here is the start position, so each piece type is only ever seen on
   its home square under one session's light. The two Ruy Lopez boards are the
   only square-diversity in the batch, and they must stay held out.
4. `0760` still needs hand-picked corners.
