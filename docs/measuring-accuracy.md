---
aliases: [eval, evaluation, perfect-board rate, ground truth]
tags: [board-reader/recognition, board-reader/evaluation]
---

# Measuring accuracy

How well the reader works is a harder question than it looks, and for most of this
project's life we answered it with a number that flattered us. This note is what
we measure now, why, and where the truth comes from.

> [!warning] The number that was wrong
> The headline was **633/640 per-square (98.9%)**. The bank that produced it was
> built *from the eval set*. On the one board it had never seen it scored 43/64.
> See [[#Contamination]].

## Perfect boards, not per-square

Per-square accuracy is the natural metric and the wrong headline. 94.5% of 64
squares is ~3.5 wrong squares per board, and a position with three wrong squares
is not a position — you cannot play from it, analyse it, or send it to Lichess.

The deployed model alone, on the ten curated boards:

| Metric | Value |
|---|---|
| **Perfect boards** | **2 / 10 (20%)** |
| Per-square | 605 / 640 (94.5%) |

Same estimator, same photos. The first row is what a person experiences; the
second is what we used to report. Both are in the harness output, in that order.

> [!info] Why the CV layer is still allowed to be this wrong
> DESIGN [[DESIGN#D1|D1]] puts recovery in the *application* layer: dense streaming
> plus per-square confidences let a consumer smooth across frames. The snapshot app
> is that architecture with the recovery half missing — one frame, nothing
> downstream. A 94.5%-per-square estimator cannot produce reliable single-photo
> boards, and no amount of tuning changes that arithmetic. See
> [[operating-envelope]].

## The error structure

Errors are counted in four kinds, because they have different fixes:

| Kind | Meaning | Usually caused by |
|---|---|---|
| `hallucination` | empty square read as a piece | crop geometry, a neighbour leaning in |
| `missed` | piece read as empty | occlusion, flat camera angle |
| `colour` | right type, wrong colour | lighting, exposure |
| `type` | wrong piece type | the hard one — see [[recognition-pipeline#Why piece type is the hard part]] |

Measured on the ten curated boards (trained model alone, 35 errors):

```
type            26
colour           6
hallucination    3

Q->K  6    p->P  5    q->k  4    n->r  2    empty->B  2
```

**Ten of the 35 errors are a queen read as a king** (`Q->K` 6, `q->k` 4), and three
more are a queen read as a bishop. Queen/king discrimination is the single largest
failure mode in the deployed recognizer — one crowned piece for another, which is
also the confusion a short crop creates by cutting the crown off (the decapitation
signature that the `headroom` geometry exists to prevent).

## Confidence is uninformative at board level

The harness prints a reliability diagram: for each confidence bin, what the reader
*claimed* against what it actually achieved.

```
bin          n   claimed   actual    gap
0.4-0.5      7     0.473    0.286 +0.187
0.5-0.6     12     0.573    0.333 +0.239
0.9-1.0    560     0.993    0.984 +0.009
board confidence: perfect 0.957 (2) · imperfect 0.958 (8)
```

Aggregate ECE is 0.019, which looks excellent — but it is dominated by the 560 easy
squares in the top bin. Two facts matter more:

- **The middle bins are badly overconfident.** A square claiming 0.5–0.6 is right
  about a third of the time.
- **Board confidence carries no signal at all.** A wrong board averages 0.958; a
  perfect board averages 0.957. You cannot tell them apart. A novel board once read
  with two white kings and no black king at 0.97.

The deployed ensemble has the opposite flaw — every bin is *under*confident (ECE
0.130, gaps to −0.47), because averaging two distributions flattens the peak even
when both members agree. It does at least separate boards (0.885 perfect vs 0.800
imperfect). Neither member is calibrated; the blend is not calibrated either, and
the blend's error is not the average of theirs.

This matters beyond cosmetics: DESIGN §9 lists confidence calibration as a top risk
precisely because the downstream smoothing plan consumes confidences. Right now a
consumer that trusted them would be misled.

## Where truth comes from

Two sources, both read by `board_reader_inference.labels`, both keyed to the frame
the photo's **corners** define rather than to any model output.

### Confirmed labels (`*.label.json`)

The snapshot page's correction panel is the only place a human ever states what a
board really holds. **Confirm as ground truth** in stage 8 writes that statement
next to the retained photo. Confirming a flawless read counts just as much as
fixing four squares — the perfect-board rate is made of both.

If the read was `autorotated`, the turn is undone before storing, so truth never
inherits the model's orientation guess and survives a model that guesses differently.

> [!warning] A confirmed label can now inherit the *decoder's* answer
> Since the repaired board is what the page displays, confirming a read with no
> corrections can hand back truth that came from the hard-constraint decoder rather
> than from the board. The label therefore records `prior_flips` — the squares the
> decoder relabelled, un-rotated into the corner frame like the placement itself —
> so an eval can find these boards later. It does not make the label wrong; it makes
> the risk **detectable**, which is the difference between this and the bank
> incidents. When labelling, read the repaired squares off the board deliberately.

> [!danger] Labels never feed the bank
> Nothing reads a label back into calibration or training. That is the whole point:
> a label's only job is to score, and a bank fitted on labels would make the score
> meaningless again.

### The curated set (`data/eval/ground_truth.json`)

Ten hand-labeled boards from 2026-07. Still read by the harness, but **the
calibration bank was built from nine of them** — see below.

## Contamination

`tools/build_exemplar_bank.py` builds the bank's base *from
`data/eval/ground_truth.json`*. So the bank has seen nine of the ten eval boards,
and it shows:

| Board | trained | bank | ensemble |
|---|---|---|---|
| the nine it was built from | 51–64 / 64 | **64 / 64** each | 63–64 / 64 |
| `up_20260718_0001` (never seen) | 62 / 64 | **43 / 64** | 62 / 64 |

A bank scores near-perfectly on its own exemplars and collapses off them. Any
number from a bank-backed stack on these boards is an upper bound, and the harness
prints a warning saying so whenever a bank is active.

The same ten boards, same harness, the two configurations side by side
(`run_e`, deployed 2026-08-06):

| | trained alone | deployed ensemble |
|---|---|---|
| Perfect boards | 0 / 10 | 9 / 10 |
| Per-square | 599 / 640 | 638 / 640 |
| Errors | 41 | 2 |
| ECE | 0.026 (overconfident mid-range) | 0.142 (**under**confident throughout) |
| Board confidence, perfect vs not | — · 0.950 | 0.874 · 0.687 |

Nearly all of those perfect boards are the bank recognising its own exemplars. The
one board it never saw, `up_20260718_0001`, is the ensemble's *worst* at 62/64, and
the only board whose confidence drops below 0.8. That single honest board is the
closest thing to a generalisation measurement this project currently has — and note
what it says about the trained-alone column: 0/10 perfect is a real number, and the
whole-board reading of an unseen position is still the unsolved part.

> [!warning] Trained-alone perfect-board rate went *down* at the last model swap
> `d16` scored 2/10 perfect alone; `run_e` scores 0/10, while beating `d16` by 53
> squares on dim boards and by 3 squares on `_0001`. Two perfect boards out of ten
> is within noise of zero on a set this small — but it is a reminder that the
> per-square gains this project measures have never yet converted into reliable
> whole-board reads. See [[training-and-data#How run_e was decided 2026-08-06]].

**There is currently no uncontaminated eval data.** Every labeled board this project
owns went into the bank. The confirmed-label path exists to fix that: boards
confirmed through normal use of the snapshot app are, by construction, boards the
bank was not built from.

> [!tip] Split by session, not by photo
> A different photo of the same set in the same light is nearly as contaminating as
> being in the bank. When there are enough confirmed labels to hold some out, hold
> out whole *sessions*.

## Running it

```bash
uv run board-reader-inference eval                    # data/eval + data/uploads
BR_MODEL=models/run_e/model.onnx uv run board-reader-inference eval
uv run board-reader-inference eval --json report.json --autorotate
uv run board-reader-inference eval --no-prior           # the raw estimator alone
```

The harness builds its estimator from the environment through the same
`_default_estimator()` the gateway uses, so it measures **what actually serves** —
ensemble, blend weight and all — and prints that configuration above every report.
`--autorotate` folds the orientation guess back in; without it the score isolates
recognition from orientation.

The **hard-constraint decode is on by default**, for the same "measure what serves"
reason: `/estimate` now returns a repaired board beside every read, so a harness
that skipped the repair would stop scoring the product. It is reported as a second
block rather than folded into the headline — the estimator's own numbers stay
first, and the decode's **fixed** and **broken** squares are printed apart, because
a repair that trades a right square for a wrong one is not an improvement. Pass
`--no-prior` to isolate the raw estimator.

Contrast with `tools/train/evaluate.py`, which scores the trained model alone inside
the training venv and never sees the ensemble. That one answers "did the training
run improve"; this one answers "can the app read a board".

## Related

- [[recognition-pipeline]] — what produces the distributions being scored
- [[calibration-and-the-bank]] — why the bank is both the accuracy win and the eval hazard
- [[operating-envelope]] — the conditions under which these numbers hold
- [[TESTING]] — Option I is the hands-on version of this note
