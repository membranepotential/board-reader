---
aliases: [eval, evaluation, perfect-board rate, ground truth]
tags: [board-reader/recognition, board-reader/evaluation]
---

# Measuring accuracy

How well the reader works is a harder question than it looks, and for most of this
project's life we answered it with a number that flattered us. This note is what
we measure now, why, and where the truth comes from.

> [!warning] The number that was wrong
> The old headline was **633/640 per-square (98.9%)**. The exemplar bank that produced
> it was built *from the eval set*. On the one board it had never seen it scored 43/64.
> See [[#History: the bank era]].

## Perfect boards, not per-square

Per-square accuracy is the natural metric and the wrong headline. 94.5% of 64
squares is ~3.5 wrong squares per board, and a position with three wrong squares
is not a position — you cannot play from it, analyse it, or send it to Lichess.

ChessQ Lite V6 on every chess-frame label (`eval data/eval data/uploads --no-prior`,
2026-11-24, [[2026-11-24-chessq-v6-replay-gate]]):

| Metric | Value |
|---|---|
| **Perfect boards** | **41 / 47 (87%)** |
| Per-square | 3002 / 3008 (99.8%) |

The first row is what a person experiences; the second is what we used to report.
Both are in the harness output, in that order. The replay gate states the stream's
version of the same number: perfect boards per take, raw and settled.

> [!info] Why the CV layer is still allowed to be wrong
> DESIGN [[DESIGN#D1 — The CV layer is a stateless per-frame position estimator|D1]] puts recovery in the *application* layer: dense streaming
> plus per-square confidences let a consumer smooth across frames. The snapshot app
> is that architecture with the recovery half missing — one frame, nothing
> downstream. See [[operating-envelope]].

## The error structure

Errors are counted in four kinds, because they have different fixes:

| Kind | Meaning | Usually caused by |
|---|---|---|
| `hallucination` | empty square read as a piece | a neighbour leaning in, a hand in frame |
| `missed` | piece read as empty | occlusion, flat camera angle |
| `colour` | right type, wrong colour | lighting, exposure |
| `type` | wrong piece type | the hard one — see [[recognition-pipeline]] |

On the earlier crop classifier, queen/king confusion (`Q->K`, `q->k`) was the single
largest failure: ten of 35 errors on the ten curated boards. On ChessQ the raw errors at
the replay gate are mid-move frames and hands read as empty squares, and the settled
errors are mostly settle lag
([[2026-11-24-chessq-v6-replay-gate#What the errors are]]).

## Confidence and temperature scaling

The harness prints a reliability diagram: for each confidence bin, what the reader
*claimed* against what it actually achieved, plus the mean board confidence of
perfect and imperfect boards. DESIGN §9 lists confidence calibration as a top risk,
because the downstream smoothing plan consumes confidences.

`--temperature` fits a single temperature `T` and rescales each square's distribution
as `q ∝ p^(1/T)`. Two properties make it safe to reason about:

- **It cannot change an answer.** The transform is monotone per square, so the argmax —
  and therefore `placement` — is bit-identical. The report prints `answers changed`, which
  must always read `0`; anything else is a bug in the implementation, not a finding.
- **It is fitted leave-one-**board**-out.** Each board is scaled by a `T` fitted on the
  *other* boards only. Fitting and scoring on the same boards would flatter itself in
  exactly the way the bank incident below already taught us once.

On the earlier ensemble (23 labelled boards) it took ECE from 0.255 to 0.004 with
`T = 0.251`, stable across folds (0.235–0.257). It has not been re-measured on ChessQ.

> [!warning] The gap is the wrong statistic, and it says the opposite of the truth
> On that ensemble the perfect−imperfect board-confidence **gap** fell from +0.136 to
> +0.017, which reads as if scaling destroyed the board-level signal. It did not.
> Scaling squeezes *every* confidence toward 1.0, so any difference of means shrinks
> mechanically. Only a **rank**-based statistic survives a monotone squeeze — and the
> AUC (probability a perfect board outranks an imperfect one) slightly *improved*,
> 0.862 → 0.877. Report AUC alongside any confidence-gap claim.

Scaling is measured, not served: turning it on for the wire is a deliberate decision, and
this is the number that decision should be made against.

## Where truth comes from

Two sources, both read by `board_reader_inference.labels`, both keyed to a stated
frame rather than to any model output. Every label carries a `frame`
(DESIGN D10): `"chess"` — the board as a player names it, a8 being Black's
queen-side corner, corners optional — or `"corners"`, the frame the photo's corners
define (every label written before D10, and any file without the key). The
recognizer reads the chess frame, so `eval` scores chess-frame labels only and skips
the others with a warning; `tools/migrate_labels_to_chess_frame.py` converts them.
Every board also gets a **frame diagnostic**: the quarter-turn under which the read
agrees best with the truth. `frame: n/n read in the truth's frame` is the healthy
line; a `TURNED` board is a misread orientation or a label in the wrong frame.
`eval-session` prints the same check over a recording's raw reads.

### Confirmed labels (`*.label.json`)

The snapshot page's correction panel is the only place a human ever states what a
board really holds. **Save labelling**, under the board, writes that statement next
to the retained photo (`POST /labels/{upload}`), in the chess frame. Saving a
flawless read counts just as much as fixing four squares — the perfect-board rate is
made of both.

> [!warning] A confirmed label can inherit the *decoder's* answer
> Since the repaired board is what the page displays, saving a read with no
> corrections can hand back truth that came from the hard-constraint decoder rather
> than from the board. The label therefore records `prior_flips` — the squares the
> decoder relabelled — so an eval can find these boards later. It does not make the
> label wrong; it makes the risk **detectable**. When labelling, read the repaired
> squares off the board deliberately.

> [!danger] Labels only score
> Nothing reads a label back into the reader. A label's only job is to score; a reader
> fitted on labels would make the score meaningless again.

### The stills (`data/eval/`)

`data/eval/ground_truth.json` holds ten hand-labeled boards (`img_0565`–`img_0569` from
2026-07, five more from 2026-08-31), with their photos inside the set
(`data/eval/images/`), so every number from it is reproducible. `heldout-20260831/`
holds eleven more photos with `*.label.json` truth. All 21 are marked `frame: "chess"`;
the pre-migration originals sit in `.pre-chess-frame/` beside each.

## History: the bank era

Until 2026-11-24 the reader was a crop classifier blended with a per-set exemplar bank
(removed; the last commit with that stack is `457cf96`). Its eval lessons still hold:

- **Contamination.** The bank's base was built from nine of the ten curated boards. It
  scored 64/64 on each of those and **43/64** on the one it had never seen. Any number
  from a reader that has seen the eval boards is an upper bound.
- **Split by session, not by photo.** A different photo of the same set in the same
  light is nearly as contaminating as the same photo. Hold out whole *sessions*.
- **Quote the population you mean.** A mixed set (real photos, pre-rectified warps, dim
  frames) has a meaningless mean.
- **Say which composition produced a number.** The old reader had several (cascade,
  fine-tune average, bank blend); a number without its composition was not comparable.

The full numbers are in [[calibration-and-the-bank]], [[training-and-data]] and
`docs/reports/`.

## Running it

```bash
uv run board-reader-inference eval                    # data/eval + data/uploads
uv run board-reader-inference eval --json report.json
uv run board-reader-inference eval --no-prior           # the raw recognizer alone
uv run board-reader-inference eval --temperature        # + temperature scaling
```

The harness builds its recognizer from the environment through the same
`recognizer.build_from_env()` the gateway uses, so it measures **what actually serves**
and prints that configuration above every report. With `BR_CHESSQ` unset or its file
missing, that is the stub — the banner says so.

The **hard-constraint decode is on by default**, for the same "measure what serves"
reason: `/estimate` returns a repaired board beside every read, so a harness that
skipped the repair would stop scoring the product. It is reported as a second block
rather than folded into the headline — the recognizer's own numbers stay first, and
the decode's **fixed** and **broken** squares are printed apart, because a repair that
trades a right square for a wrong one is not an improvement. Pass `--no-prior` to
isolate the raw recognizer.

`--temperature` adds the block described above. It is off by default because it costs
a leave-one-out fit per board and, unlike the decoder, it does not serve.

**The replay gate.** `make gate GATE_NAME=<new>` replays the four gate takes
through `BR_CHESSQ`, scores each, and prints `tools/gate_table.py`'s table against
`cq-v6`: raw per-square and raw / prior / settled perfect boards per take, with a
pooled total. The reference is 92/101 raw and 84/101 settled perfect boards
([[2026-11-24-chessq-v6-replay-gate]]). A change that should not move the reader must
print every cell once.

## Related

- [[recognition-pipeline]] — what produces the distributions being scored
- [[operating-envelope]] — the conditions under which these numbers hold
- [[TESTING]] — Option I is the hands-on version of this note
