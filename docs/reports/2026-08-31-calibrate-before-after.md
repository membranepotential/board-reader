# Calibrate-before/after — the write helps the new style, hurts the old take, and every per-square metric hides it

_2026-08-31 · closes STATUS Next 11(a) and gives the "bank writes are not
monotone" open question its sharpest measurement yet. Candidate bank built
through the live `/calibrate` path verbatim (all five 08-31 boards accepted,
1024 → 2304 exemplars); deployed bank never touched. Full detail:
`scratch/experiments/calibrate_before_after/`._

## The Q→K verdict — killed on the wire, untouched in the raw stream

Calibration does **not** stop the model reading two kings on the fresh style
(raw surplus-king confusions 13 → 11 on the held-out 11). What it changes is
the *evidence margin*: with the five boards' exemplars in the bank, the
hard-constraint decoder demotes the right king on **every** held-out board —
post-decode k/q errors **4 → 0**. Held-out prior score 690 → 692/704, perfect
6 → 7/11. Confidence also improves markedly: **ECE 0.345 → 0.229**.

## The regression — non-monotone, and invisible to per-square metrics

The 4-take replay gate (`cba-candidate` beside the `tkr-shipped` baselines):

| take | raw per-sq | prior perfect | **settled perfect** |
|---|---|---|---|
| 08-13 | .9597→.9647 | 15/31 → **7/31** | 22/31 → **12/31** |
| 175539 | .9219→.9492 | 4/12 → 7/12 | 5/12 → **7/12** |
| 111421 | .9164→.9321 | 3/23 → 4/23 | 5/23 → **3/23** |
| 181802 | .8848→.8914 | 3/43 → 3/43 | 3/43 → 3/43 |
| **total** | .9169→**.9272** | 25→21 | **35 → 25** |

**Per-square accuracy rises on all four takes in all three streams; perfect
settled boards fall by 10, all on 08-13.** A per-square gate — or a stills
gate — would have waved this bank through.

**Mechanism (measured):** on 08-13's 34 two-king reads, the shipped tie-break
demoted the *true* king e8 on 17/34 before → **26/34 after**. The 08-31
exemplars pull e8's `p(k)` down uniformly (~0.42 → ~0.37) while false-d8's
stays flat — the same queen-mass-polarity effect
[[2026-08-31-two-kings-reopen]] found for *rules*, here produced by a *bank
write*. Cross-style contamination of the king/queen margin, consistent with
the standing "mixing styles into one bank actively hurts" finding. Not a
debounce artifact: the rate-independent prior row regressed too.

One held-out board is also clearly damaged: `img_0757` loses 4 squares net
(five regressions, all rank 1/2 + c8/h7) — worth its own look.

## Verdicts

1. **Calibrating a fresh style works for that style** — decoded Q→K dead,
   +1 perfect board, ECE −0.116 — **but is not safe to do blind** across
   styles: it cost 10 settled-perfect boards on the other set's take.
2. **Bank writes must be gated on the 4-take settled-perfect rows.** Every
   aggregate per-square metric moved up while the deployment metric moved
   down. This is the regression gate the open question has been asking for,
   now with proof of why weaker gates are insufficient.
3. The product-shaped implication: a bank that **mixes physical sets degrades
   the set it did not just calibrate** — session-scoped (or style-scoped)
   banks, rather than one ever-growing global bank, is the design this points
   at.

Leave-one-out over the five boards was skipped deliberately: the e8 `p(k)`
drop is uniform across all 34 reads, not one board's doing; run it only if a
partial write ever needs salvaging.
