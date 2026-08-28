# The 08-28 proof trial — the night wave's stack, live, zero-tap

_Analysis-only. Recording: `recordings/20260828-111421` (frames + events +
`truth.json` + `report.json` inside it). Replays:
`recordings/20260828-111421/replays/{as-shipped,demotion-off}`. Client story:
`client-logs/20260828-111419.ndjson`._

This is the take the night wave existed for: the first live session over the
full 08-27 stack — burst-end scheduling, framing re-detect, settled coherence,
per-session recorder, b-demotion — run zero-tap from a phone on `/camera`.

## 0. Setup

`make dev` (2026-08-28 11:14, GPU visible, arena capped): cascade
`run_e` ⊕ `run_f` at τ 0.95, blended with the session bank; `serve`'s default
`burst-end:burst_end_reads=3` schedule; recorder on. The user played the Italian
from the start position; the take ran 91.5 s, 243 frames.

**One correction to the operator's stated line.** The move list as reported had
`Re1 Re8` *before* castling — impossible (the kings sit on e1/e8). The recorded
stream confirms the physically played order:
**e4 e5 Nf3 Nf6 Bc4 Bc5 O-O O-O Re1 Re8 d3 Qe7 Nc3 Qd8**. Truth was authored
against that order.

## 1. The zero-tap checklist — verdict

Everything the night wave shipped fired on its own; nothing needed a tap until
the operator's optional re-detect at the very end.

| Check | Result |
|---|---|
| Auto-calibrate | **fired, attempt 1**, `label_rotation 0` — bank grew 768 → 1024 exemplars mid-take (see §5) |
| Framing re-detect | baselining/steady all take, **zero false adoptions**; one true adoption when real drift arrived (§3) |
| Burst-end scheduling | 36 reads / 243 frames = **1 per 6.8** (`unscheduled` 132, `estimated` 36) |
| Backstop / refusals | `backstopped 0`, `refused 0`, `failed 0` |
| Client frame loss | **0 dropped** across every heartbeat; 243 sent, 242 acked at close |
| Latency capture→publish | median **1.55 s**, p90 1.87 s, max 2.41 s |
| OOM / crashes | none |
| Recording | complete; scoreable (this report) |

## 2. Scores

Truth: 12 segments (merged plies collapse — the scheduler read nothing between
them), `guard_seqs=3`; 36 events published, 22 scored, 14 guard-excluded.
Boundaries by monotone alignment of the expected placements against
`events.ndjson`.

| stream | per-square | perfect boards |
|---|---|---|
| raw | 1280/1408 · 90.9 % | 2/22 |
| prior | 1291/1408 · 91.7 % | 2/22 |
| settled | 1294/1408 · **91.9 %** | **4/22 · 18.2 %** |

Per-segment accuracy **declines monotonically ~100 % → ~88 %** through the take.
That shape is not the recognizer getting tired — it is §3.

## 3. The drift — watched end to end, for the first time

The client's re-detector logged its shift (stamped quad vs. detector, in square
widths) every ~11 s:

```
44 s  steady  0.321
55 s  arming  0.506
66 s  steady  0.434
77 s  arming  0.629
78 s  ADOPT   0.596      <- automatic re-anchor
89 s  baselining again
```

Drift onset ~44 s in, creeping to ~0.6 of a square — and the per-segment
accuracy decline tracks it. The re-detector armed twice, declined the
borderline dip at 0.434 (exactly the false-adoption discipline it shipped
with), and adopted at 78 s. The operator also pressed manual re-detect at the
very end, after the automatic adoption.

**The limit of the fix:** post-adoption accuracy recovers only to ~88 %, not to
the opening's 100 %. The residual after re-anchoring is *recognition*, not
framing — and §4 says which recognition.

## 4. The attractor guard fired — and the A/B says what it means

`eval-session`'s attractor guard warned twice on this take: the lowest margin at
which a top-1 `b` was *right* is now **0.208** — headroom **+0.008** over the
0.20 firing threshold (the rule shipped on a 0.210 floor, "0 broken") — and up
to 14 truth-`b` squares went to the runner-up inside the band.

The digest is post-rule, so only a replay A/B can split "the demotion ate them"
from "already wrong with `b` second". Same take, same 1024 bank, burst-end
policy, CPU, only `ATTRACTOR_MARGIN` differs (0.20 vs 0.0 = off):

| stream | as-shipped | demotion-off | Δ (shipped) |
|---|---|---|---|
| raw | 1338/1472 · `b->P` ×9 | 1342/1472 · `b->P` ×1, **`empty->b` ×8** | −4 |
| prior | 1357 (perfect 3) | 1358 (perfect 4) | −1 |
| settled | **1373** (perfect 5) | 1369 (perfect 5) | **+4** |

**Verdict.** The "0 broken" guarantee is dead on this take: the rule now *eats*
correct bishops (the c5 bishop demoted to `P`, nine raw events) — but switching
it off replaces them one-for-one with `empty->b` hallucinations on the
neighbouring square. It is a trade, roughly a wash, and still **+4 on the
settled stream, which is what ships**. Both sides of the trade have the same
root: the c5 bishop's crop bleeding into b5 (the bounding-prism attribution
problem, STATUS Next). So: **keep the rule**, and revisit it with the tall-crop
retrain / crop-attribution work — the rule is a bandage over exactly the wound
that work is meant to close.

## 5. Two things the take taught about the bank

- **Auto-calibrate wrote mid-take** (768 → 1024 exemplars) — and 1024 is the
  maturity threshold, so the blend flipped 50/50 → 30/70 *during the session*.
  Working as designed, but it means any bank-backed number quoted "against the
  bank" is quoting a moving target. The curated baseline now pins its own bank
  snapshot for exactly this reason ([[measuring-accuracy#The curated set
  dataevalground_truthjson]]).
- The re-scored curated baseline against the pinned 1024 bank: **304/320,
  0/5 perfect** (prior 303/320: fixed 0, broken 1 — see [[measuring-accuracy]]).

## 6. Did we record the full video? (operator question)

No — and yes to everything the wire saw. The recording holds **every frame the
client sent**: 243 JPEGs, 960×783, ~2.7 fps, 52 MB, per-frame corners; the
client dropped nothing (heartbeats), the node refused nothing. The take is
fully reproducible *from the wire's point of view* — the replays in §4 are that
claim, exercised.

What is **not** captured: the continuous camera stream between sends — the
frames the sharpness-preferring selector never picked. For slow drift, 2.7 fps
stills suffice (§3 was reconstructed from them). What full video would add:
edge/detector debugging at moments the selector skipped, training crops from
unsent frames, and — the sleeper benefit — **trivial truth-authoring** (watch
the video, note the move times, done).

Proposal (registered in STATUS Next, not built): client-side `MediaRecorder`
on the camera track, hardware-encoded, uploaded in chunks *during* the session
(a `timeslice` blob per POST — an end-of-session upload dies with the page),
landing beside the recording. ~25–50 MB for a take this length.
