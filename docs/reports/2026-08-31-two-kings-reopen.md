# Two-kings tie-break reopened — the fresh challenger dies in the gate, the ceiling is confirmed

_2026-08-31 · reopens [[2026-08-28-two-kings-decode]] on the fresh evidence its
docstring asked for: the 08-31 batch's 23 new surplus-king events
([[2026-08-31-new-style-batch]]) vs the 7-board ceiling that closed it. Full
detail: `scratch/experiments/two_kings_reopen/report.md`._

## What was run

1. **Stills sweep** — the deployed stack's `square_dist` captured for all 21
   labeled eval boards (byte-exact reproduction of the 08-31 report first:
   305/303/680, ECE 0.345, the three repair-log lines verbatim), then 14
   tie-break rules scored offline through the real `prior.decode` with a
   pluggable kings key, identity-checked against shipped before comparing.
2. **Replay gate** — the 08-28 captures turned out to be gone (never
   committed; `events.ndjson`'s top-2 digest cannot feed a 13-class decoder),
   so four fresh GPU replays reproduced all four 08-28 baseline rows exactly
   (22/31·.9899, 5/12·.9609, 5/23·.9266, 3/43·.9113), with a byte-identity
   control: the cache replay of the shipped rule regenerates the GPU pass's
   `events.ndjson` byte-for-byte on all four takes.

## Result

On the stills, shipped `p(king)` was beatable: **`p(K) − p(Q)`** fixed the
white-pair failures (+5 squares, +1 perfect board, no curated loss,
parameter-free). In the replay gate it **collapsed 08-13 from 22/31 to 0/31
settled perfect** — the worst result of any rule ever tried on that take:

| rule | 08-13 (31) | 175539 (12) | 111421 (23) | 181802 (43) | 08-31 stills |
|---|---|---|---|---|---|
| **`p(king)` — shipped** | **22/31** | 5/12 | 5/23 | 3/43 | 1000/1024 |
| `p(K) − p(Q)` | **0/31** | 5/12 | 5/23 | 3/43 | 1005/1024 |
| cascade `p(K)` | **0/31** | 5/12 | 5/23 | 3/43 | 1006/1024 (=oracle) |

**Mechanism: the queen-mass polarity inverts between datasets.** On the 08-31
boards the false king carries the queen mass (subtracting it fixes the
demotion); on 08-13 the *true* king e8 reads more queen-like (p(Q) 0.16–0.26)
than the false d8 (0.09–0.17), so the same subtraction demotes the real king
31/34 times. A λ sweep of `p(K) − λ·p(Q)` degrades monotonically from λ=0.1 —
there is no safe amount.

## What this settles

- **The shipped rule stands, now against 14 rules and three datasets.**
  `max-margin` and `entropy` are confirmed dead a third independent time.
- **The per-frame ceiling is real.** A rule can reach the oracle on one dataset
  and score 0 on another because the tie-break's error rate is
  *dataset-polarity-dependent* — a recognition property, not a decoder one.
  The lever remains d8 recognition or a temporal identity prior (which the
  planned [[2026-08-31-autolabel-engine|Viterbi engine]] provides offline).
- **A docstring correction landed:** 08-13's d8 misread is the *trained
  cascade* being confidently wrong (`k` at 0.97–0.99), not a session-bank
  exemplar lock as the 08-28 note claimed.
- **Diagnostic with legs:** the deployed bank reads **73 %** on the 08-31
  style and **66 %** on curated stills while carrying 0.70 of the blend — the
  cascade alone beats the full blend on stills. In-session the bank
  auto-calibrates into domain, so this is not a deployment verdict, but it is
  the sharpest framing yet for the calibrate-before/after experiment.

## Follow-ups filed

- **`replay.CONFIG_ENV` omits `BR_CROP_MASK`** — a masked and an unmasked
  replay share a `config_digest` (measured: `e6-control` vs `e6-gray` on
  181802, both `cc530662`, scoring 0.9139 vs 0.9121). The digest exists to
  prevent exactly this ambiguity. → STATUS Next.
- The four `tkr-shipped` GPU baseline replays are kept under
  `recordings/*/replays/`; the 20 cache-replay dirs were deleted after
  scoring.
