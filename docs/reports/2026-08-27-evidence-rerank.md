# Can a cheap evidence re-rank kill the residual errors, without a retrain?

**Answer: partly, and the part that works is worth shipping today — but it does not
retire the retrain.** One rule — two constants, a class and a threshold — fixes 38
squares across 133 scored boards and breaks *zero*. It lifts the prior stream's perfect-board rate
by +12.5 / +18.6 / +12.9 / +0.0 points on the four takes. It also cannot touch the
control take's `q->k` residual or take 5's `R->P`, because on those squares the truth
is not in the published evidence at all. A re-rank can only reorder what the estimator
already said; where the estimator never said it, only a better estimator will do.

*(Analysis only. Nothing in `packages/` was modified; recordings were read-only.)*

---

## GO / NO-GO

> **GO — implement the `b` demotion (spec in §6). NO-GO on anything wider:**
> a global 13-class reweighting and per-pair threshold tables both fail
> leave-one-take-out, and even a *perfect* top-2 oracle leaves 16–40 % of boards
> imperfect, so the tall-crop retrain argument stands on its own merits.

---

## 1. The data

Four recorded live takes, all scored through `board_reader_inference.trial`
(imported, not reimplemented — `takes.py`). The loader reproduces each recording's
published `report.json` to the digit, which is the gate on everything below:

| take | dir | events | raw | prior | settled |
|---|---|---|---|---|---|
| control | `recordings/20260827-163352` | 24 | 97.53 / 0.0 | **99.02 / 58.3** | 97.53 / 29.2 |
| hard | `recordings/20260827-172736` | 43 | 97.64 / 25.6 | **98.55 / 46.5** | 98.22 / 39.5 |
| fast | `recordings/20260827-175539` | 31 | 95.61 / 6.5 | **98.14 / 58.1** | 94.15 / 32.3 |
| take5 | `recordings/20260827-181802` | 35 | 96.56 / 34.3 | **97.32 / 45.7** | 96.70 / 25.7 |

(per-square % / perfect-board %.)

Two loader notes worth keeping:

* the **fast** take's directory holds two sessions with colliding `frame_seq`; the
  scored one is the slice with `capture_ts > 1787846505`. `takes.fast_take_dir()`
  rebuilds that filtered copy in scratch, and it reproduces the published report exactly.
* **take5** was scored by another agent *after* this study's rule was already chosen
  from the first three. It is therefore a genuine, untouched holdout, and it is where
  the rule's honesty gets tested rather than asserted.

## 2. The ceiling — what a re-rank could *ever* fix

The wire carries `EVIDENCE_TOP_K = 2`. An oracle that always picks the truth *when the
truth is one of the two* bounds the entire post-hoc family. Everything outside the
top-2 is unreachable by definition.

| take | scored squares | raw errors | truth in top-2 | **ceiling** |
|---|---|---|---|---|
| control | 1536 | 38 | 21 | **55.3 %** |
| hard | 2752 | 65 | 44 | **67.7 %** |
| fast | 1984 | 87 | 51 | **58.6 %** |
| take5 | 2240 | 77 | 59 | **76.6 %** |
| **all** | **8512** | **267** | **175** | **65.5 %** |

Where truth is in the digest it is *always* the runner-up (`truth_ranked ==
truth_runner_up` in every take) — with k=2 that is a tautology, not a finding.

The two published claims both hold, and they are about different takes:

* **hard** — the bishop attractor is real and re-rankable: `p->b` 17x and `q->b` 8x,
  truth ranked 2nd in 25/25.
* **control** — `q->k` 22x with truth in top-2 only 11 times. Half of that residual is
  *unreachable* by any re-rank, and §4 shows the other half is not separable either.

## 3. The whole design space, on one table

The only decision available per square is *keep top-1* or *swap to top-2*. So every
candidate transform is a decision rule on `(c1, c2, p1, p2)`, and
`results/results_pair_stability.txt` is what any of them has to beat: for each ordered
`(top1, top2)` pair and each margin threshold, how many squares it **fixes** and
**breaks**, per take.

Reading it, exactly one pattern repeats in all four takes:

```
      b->p  tau   control          hard             fast             take5
            0.15  +4  /-0          +6  /-0          +6  /-0          +1  /-0
            0.20  +4  /-0          +9  /-0          +7  /-0          +2  /-0
            0.25  +4  /-0          +10 /-1          +10 /-0          +2  /-0
```

Everything else is a single-take artefact. `p->empty` fixes 19 squares in **fast** and
breaks 3 in **hard**. `K->Q` fixes 3 in **fast** and breaks 4 in **control**. And the
control's own `k->q` has **no threshold with a positive net in any take** — its best
is +1 — which is the quantitative refutation of "the `q->k` residual is re-rankable".

## 4. What failed, and why it matters

Both of the obvious general transforms fail leave-one-take-out
(`results/results_loo.txt`):

| transform | fit | control | hard | fast | take5 |
|---|---|---|---|---|---|
| global 13-class log-bias | LOO | raw +0.33 / **prior +0.13** | raw **−0.47** / prior **−0.44** | raw +0.91 / prior **−0.15** | raw +0.85 / prior +0.22 |
| per-pair margin table (18 taus) | fit on ALL 4 (in-sample) | +0.59 | +0.98 | +1.86 | +1.83 |
| per-pair margin table | LOO, pooled net gain >= 3 | +0.20 (6 fixed / **3 broken**) | +0.18 (16 / **11**) | +0.40 (11 / **3**) | +0.49 (11 / 0) |

The class-bias vector is worth a second look because it *finds the right signal with
the wrong instrument*: `b` gets the largest negative weight in 3 of 4 folds. But a
bias vector is antisymmetric — demoting `b` against `p` necessarily promotes `p`
against everything else — and that side effect costs 32 correct squares on **hard**.
The per-pair table's in-sample numbers look excellent and are pure overfit: 18 tuned
constants against 267 errors; and its LOO version still breaks 17 correct squares to
fix 44, because a pooled net gain lets one take's win pay for another take's loss.

The lesson generalises past this experiment: with ~10<sup>2</sup> errors, a *pooled*
net gain is trivially manufacturable. The only anti-overfit device that worked was
requiring the win to **repeat in every fitting take independently**
(`transforms.fit_stable_demote`, gate: >= 2 fixes and **0** breaks in each).

## 5. Leave-one-take-out, with the transferability gate

Family **and** threshold chosen from the three fitting takes only; evaluated on the
fourth. `results/results_demote_loo.txt`.

| held out | rule the fit chose | raw per-square | raw perfect | prior per-square | prior perfect | fixed | broken |
|---|---|---|---|---|---|---|---|
| control | demote `b`, tau < 0.20 | 97.53 -> 97.85 (+0.33) | 0.0 -> 4.2 | 97.98 -> 98.24 (+0.26) | 29.2 -> 33.3 | 5 | 0 |
| hard | demote `b`, tau < 0.24 | 97.64 -> 98.18 (+0.55) | 25.6 -> 46.5 | 98.29 -> 98.58 (+0.29) | 44.2 -> 65.1 | 17 | 2 |
| fast | demote `b`, tau < 0.20 | 95.61 -> 96.12 (+0.50) | 6.5 -> 6.5 | 96.52 -> 96.82 (+0.30) | 22.6 -> 22.6 | 10 | 0 |
| take5 | demote `b`, tau < 0.20 | 96.56 -> 96.92 (+0.36) | 34.3 -> 37.1 | 97.41 -> 97.68 (+0.27) | 45.7 -> 45.7 | 8 | 0 |

Every fold picks the same rule, three of four pick the same threshold, and no fold
loses. The `prior` column here is the reconstruction view — see §7.

### The recommended constant

The rule fires on the margin, so the safe threshold is set by the *smallest margin at
which top-1 `b` is actually right*:

| take | fixable `b` margins | lowest **correct** `b` margin |
|---|---|---|
| control | 0.025 0.041 0.085 0.144 0.182 … | **0.275** |
| hard | 0.003 0.043 0.084 0.085 0.099 … 0.208 | **0.210** |
| fast | 0.004 0.008 0.009 0.014 0.025 … 0.170 | **0.315** |
| take5 | 0.032 0.033 0.051 0.052 0.060 0.093 0.110 0.158 | **0.247** |

There is a clean gap. **tau = 0.20** sits inside it on all four takes and is what
3 of 4 LOO folds independently chose. The first correct square dies at 0.210 (hard),
so 0.20 has ~0.01 of headroom there and 0.05–0.12 elsewhere; tau = 0.15 buys more
headroom for ~30 % fewer fixes and is the conservative alternative
(`results/results_final.txt` has both).

## 6. The rule — exact spec

Applied to `square_dist` **inside the estimator**, before `placement`, `evidence`,
`confidence` and the prior decoder are derived from it. It is a calibration of the
estimator's own output, not a decode step, so it belongs upstream of `prior.decode`
rather than beside it (DESIGN D1: the prior stays a pure decoder over whatever
distribution estimation publishes).

```python
# board_reader_inference — post-hoc calibration of the ensemble's output.
# The black bishop is an attractor for this model: in a narrow band it outranks
# the true class (p, q, k) it should be losing to. Measured on four live takes,
# 133 boards: 38 squares fixed, 0 broken. See scratch/experiments/rerank_evidence/.
ATTRACTOR = "b"
ATTRACTOR_MARGIN = 0.20

def demote_attractor(square_dist: np.ndarray) -> np.ndarray:
    """Swap the top two classes on squares the attractor only narrowly won."""
    flat = square_dist.reshape(-1, NUM_CLASSES).copy()
    order = np.argsort(-flat, axis=-1)[:, :2]
    i1, i2 = order[:, 0], order[:, 1]
    rows = np.arange(flat.shape[0])
    p1, p2 = flat[rows, i1], flat[rows, i2]
    fire = (i1 == CLASS_INDEX[ATTRACTOR]) & (p1 - p2 < ATTRACTOR_MARGIN)
    r = rows[fire]
    flat[r, i1[fire]], flat[r, i2[fire]] = p2[fire], p1[fire]
    return flat.reshape(square_dist.shape)
```

`verify_impl.py` runs this exact snippet over all 133 boards and confirms it fires on
the same 55 squares and picks the same class as the digest-level rule everything in
§5–§8 was measured with — 0 mismatches.

**Constants:** attractor class `b` (lowercase — the *black* bishop only; top-1 `B`
has 1 fixable square in 8512 and must not be touched), margin `0.20` absolute
probability difference.

**Blast radius** (`results/results_firing.txt`): 55 firings over 133 boards
(0.41/board) — 38 fixes, **0 breaks**, 17 neutral (the square was wrong before and
after; the truth was not in the top-2 either).

**Calibration is not harmed.** Zero already-correct squares are re-ranked away in any
take, so the mean top-1 probability on correct squares is unchanged to four decimals
(control 0.7085, hard 0.8220, fast 0.7964, take5 0.7835). The *fixed* squares are
reported at their old runner-up probability — mean 0.25–0.35, max 0.42 — so they stay
below any low-confidence marker. A square this rule rescues still shows up as
uncertain, which is the honest outcome: the model did not become sure, it became right.

## 7. End-to-end effect, three views of the prior stream

The prior decoder needs the full 8x8x13 distribution and the wire carries only the
top-2, so re-running `prior.decode` offline needs a reconstruction (`dists.py`:
recorded probabilities on their classes, `1 - p1 - p2` spread over the other 11).
**That reconstruction is imperfect and its fidelity is the honest weak point of this
study**: it reproduces the published prior board on only 100/133 events, 57 squares differing
in total (and it is very uneven: 34/35 on take5, 13/31 on fast), because the decoder's flip costs depend on 3rd-choice information the digest
threw away. Sweeping the tail treatment does not help (`tune_tail.py`: 57–63 squares
differ across every setting) — the information is simply gone.

So the prior stream is reported three ways:

* **A — raw**: exact. The digest's argmax reproduces the recorded raw board on
  133/133 events, so this view needs no reconstruction at all.
* **B — prior (reconstruction)**: `decode()` over the reconstruction, baseline and
  re-ranked, *paired*. Absolute values are pessimistic; the delta is the measurement.
* **C — prior (anchored)**: the **real published** prior board, patched on the squares
  where the decoder kept the raw argmax and the rule fires. True absolute numbers;
  assumes the decoder's own flips are unchanged by the re-rank.

At **tau = 0.20** (`results/results_final.txt`):

| take | view | per-square % | perfect-board % | fixed | broken |
|---|---|---|---|---|---|
| control | A raw | 97.53 -> 97.85 (+0.33) | 0.0 -> 4.2 (+4.2) | 5 | **0** |
| | B prior (recon) | 97.98 -> 98.24 (+0.26) | 29.2 -> 33.3 (+4.2) | 5 | 1 |
| | **C prior (anchored)** | **99.02 -> 99.28 (+0.26)** | **58.3 -> 70.8 (+12.5)** | 4 | **0** |
| hard | A raw | 97.64 -> 98.18 (+0.55) | 25.6 -> 41.9 (+16.3) | 15 | **0** |
| | B prior (recon) | 98.29 -> 98.62 (+0.33) | 44.2 -> 62.8 (+18.6) | 10 | 1 |
| | **C prior (anchored)** | **98.55 -> 98.87 (+0.33)** | **46.5 -> 65.1 (+18.6)** | 9 | **0** |
| fast | A raw | 95.61 -> 96.12 (+0.50) | 6.5 -> 6.5 (+0.0) | 10 | **0** |
| | B prior (recon) | 96.52 -> 96.82 (+0.30) | 22.6 -> 22.6 (+0.0) | 7 | 1 |
| | **C prior (anchored)** | **98.14 -> 98.49 (+0.35)** | **58.1 -> 71.0 (+12.9)** | 7 | **0** |
| take5 | A raw | 96.56 -> 96.92 (+0.36) | 34.3 -> 37.1 (+2.9) | 8 | **0** |
| | B prior (recon) | 97.41 -> 97.68 (+0.27) | 45.7 -> 45.7 (+0.0) | 6 | **0** |
| | **C prior (anchored)** | **97.32 -> 97.54 (+0.22)** | **45.7 -> 45.7 (+0.0)** | 5 | **0** |

Both exact views (A and C) show **zero broken squares on all four takes**. This is not
a transform that trades one for one; on this evidence it is pointwise dominant.

The perfect-board gains dwarf the per-square gains because the errors *cluster*: a
+0.33 pp per-square move on **hard** is +18.6 pp of whole boards, since the bishop
attractor is usually the only thing wrong with the board. That is the metric the
product cares about, and it is where this rule pays.

Take 5 gains 5 squares and no boards — its residual is `empty->P` / `R->P` / `P->empty`
white-pawn confusion in a different lighting, which the rule correctly leaves alone.

## 8. Why this is not a substitute for the retrain

`results/results_ceiling.txt`, prior-stream perfect-board rate:

| take | real | + this rule | **perfect top-2 oracle** |
|---|---|---|---|
| control | 58.3 % | 70.8 % | **83.3 %** |
| hard | 46.5 % | 65.1 % | **83.7 %** |
| fast | 58.1 % | 71.0 % | **83.9 %** |
| take5 | 45.7 % | 45.7 % | **60.0 %** |

The rule captures 38 of the 175 top-2-reachable errors — **21.7 % of the ceiling**. The
remaining 137 sit in margin bands where the same threshold that fixes them breaks more
correct squares; they are not separable by *any* function of `(c1, c2, p1, p2)`. And
even the unreachable-by-construction oracle tops out at 60–84 % of boards perfect,
because 92 of 267 errors never put the truth in the digest at all — the control's
`q->k` and take5's `R->P` among them.

**Ship the rule for what it is: a free +12–19 pp on three of four takes and a strictly
non-negative change on the fourth. Then retrain, for the 34–40 % of boards it cannot
reach.**

## 9. Caveats

1. **One board, one room, one evening.** All four takes are the same physical set,
   camera and lighting. Leave-one-take-out controls for take-to-take variation (moves,
   speed, hand occlusion, light drift) — it does **not** control for domain shift. The
   `b` attractor may be a property of *this set's* black bishops, not of the model.
   Before the constant hardens, re-check the §5 margin gap on the next take from a
   different set. The rule's failure mode if the gap closes is graceful (it starts
   breaking squares) but silent, so this should be an assertion in the trial scorer,
   not a note in a report.
2. **The family was chosen with all four takes in view.** §5's LOO re-derives it
   blind in every fold, which is the strongest available answer, but four takes is a
   small number of folds and the takes are not independent.
3. **The prior-stream numbers are not exact** — see §7. View A is exact and view C is
   anchored on the real published boards; view B is a paired estimate over a
   reconstruction that loses ~0.7 % of squares to top-2 truncation.
4. **Take counts are small**: 133 scored boards, 267 raw errors, 38 fixes. The
   zero-breakage result is the load-bearing one and is robust (it is a statement about
   a gap in the margin distribution, not about a mean); the point estimates of the
   deltas are not.
5. `settled` was not evaluated. It is a temporal debounce over `prior`, so it inherits
   whatever the prior does, but with its own K-consecutive dynamics — worth a look if
   the rule ships.

## 10. Files

| file | what |
|---|---|
| `takes.py` | loads the four takes via `trial.score_session`; rebuilds the fast take's `capture_ts` filter |
| `residuals.py` | per-square truth/top-2 rows; the residual picture and ceiling (§2) |
| `pairs.py`, `pair_stability.py` | the ordered-pair tables (§3) |
| `dists.py`, `tune_tail.py`, `check_prior.py` | top-2 -> 8x8x13 reconstruction and its fidelity gate (§7) |
| `transforms.py` | rule types and fitters, incl. the transferability gate |
| `candidates.py`, `sweep.py` | named candidate rules, tau sweeps |
| `evaluate.py` | LOO metric harness: fixed/broken, perfect boards, calibration |
| `run_loo.py`, `run_stable_loo.py`, `run_demote_loo.py` | the three LOO protocols (§4, §5) |
| `final_eval.py`, `ceiling.py`, `firing.py` | the numbers in §6–§8 |
| `verify_impl.py` | proves §6's shipped snippet == the rule that was measured |
| `results/` | every table above, as `.txt` and `.json` |

Reproduce with `uv run python <script>.py` from this directory.
