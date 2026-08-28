# Two-kings decode — investigated, and the claim refuted

_2026-08-28 night · closes the "settled coherence demotes the CORRECT king"
follow-up filed by the [[2026-08-28-improvement-report|improvement report]].
Code: `8c42b83` (docstring + 2 pinning tests, **no behavior change**)._

## The claim, and what was wrong with it

The improvement report filed: on the 08-13 take, raw reads d8 (truth `q`) as
`k` beside the real e8 king; the kings-constraint repair demotes the *real*
king on 31/31 events, doubling one wrong square into a symmetric q↔k swap —
fix by preferring to demote the lower-confidence duplicate.

Both halves were stale:

- **The rule already does that.** `prior._kings_flip` has demoted the
  lowest-`p(king)` duplicate since `faaa4ed` (2026-08-13), tested.
- **The 31/31 figure came from the wrong row.** The shipped config
  (e6f-gray) splits **17 d8 / 17 e8** across two-king events and carries the
  swap on only 9/31 settled boards; 31/31-wrong belongs to the *rejected*
  e7 masked-retrained pair.

## Why the choice is a coin flip — the evidence genuinely crosses

Captured full `square_dist` streams show the bank-locked **false** king d8
reading `k` at a stable 0.33–0.41 with a fat 0.17–0.24 margin, while the
**true** king e8 ranges 0.20–0.45 with margins as thin as 0.003 (one event:
p(k) gap 0.0001). Both raw probability *and* margin favor the wrong square —
the estimator simply believes d8 more. No per-frame tie-break can recover
information that is not in the frame.

## Rule search: 4 rules × 4 gate takes (offline, decode+stabilize are pure)

Baseline reproduced the shipped e6f-gray rows byte-identically (37/37 events)
before scoring alternatives. Settled perfect / per-square:

| rule | 08-13 | 175539 | 111421 | 181802 |
|---|---|---|---|---|
| **p(king) (shipped)** | **22/31 · .9899** | **5/12 · .9609** | **5/23 · .9266** | 3/43 · .9113 |
| margin (pre-`faaa4ed`) | 2/31 · .9698 | 5/12 · .9609 | 5/23 · .9266 | 3/43 · .9117 |
| max-margin | 27/31 · .9970 | 2/12 · .9440 | 5/23 · **.9096** | 3/43 · .9113 |
| neg-entropy | 22/31 · .9899 | 2/12 · .9427 | 5/23 · .9192 | 3/43 · .9113 |
| _oracle (always d8)_ | _29/31 · .9990_ | — | — | — |

`max-margin` nearly matches the oracle on 08-13 — by demoting the *more*
confident king, which is anti-MAP and wins only because this take's bank lock
makes the wrong king the confident one. It breaks two other takes: a fitted
constant, not a decoder. **The shipped rule is Pareto-best.**

## The ceiling

Even the oracle only reaches 29/31 settled vs the shipped 22/31 — the
stabilizer already recovers most of the coin flip, so **~7 settled boards
total** are reachable by *any* per-frame tie-break, across all takes. The
productive lever on 08-13's residual is the d8 recognition itself — a session
bank nearest-exemplar lock — i.e. the bank re-embedding work, not the decoder.

## What landed

- `prior.py` — `_kings_flip` docstring now carries this 4-take re-measurement
  and an explicit "do not re-open on the 08-13 swap alone".
- `test_prior.py` — 2 new tests: the deterministic index tie-break (the take
  really produces 1e-4 gaps), and that a demoted king takes its best class
  *among those that relieve the excess* (picks `q` over a 3:1-preferred `n`
  when the promotion budget is spent — claimed by the docstring, previously
  untested).
- Bonus find, fixed separately (`4d6ce3e`): `make dev` had served **unmasked**
  since the masking ship (`38c40ac` set compose.yaml only) — the deploy-parity
  guard was red on main and is green again.

## Method note

One GPU replay per take captured every `square_dist`; rules were then scored
offline against the captures (decode and stabilize are pure functions). That
made the 4×4 sweep essentially free after the captures, and reproducing the
shipped rows byte-identically first is what makes the comparison trustworthy.
