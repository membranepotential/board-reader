"""Final numbers for the recommended rule, three views of the prior stream.

view A  raw            — exact. The digest argmax reproduces the recorded raw
                         board on 98/98 events, so this needs no reconstruction.
view B  prior (recon)  — decode() over a top-2 reconstruction of square_dist,
                         baseline and re-ranked, *paired*. Absolute values are
                         pessimistic (the reconstruction loses the decoder's
                         3rd-choice information); the delta is the measurement.
view C  prior (anchor) — the REAL published prior board, patched on the squares
                         where the decoder kept the raw argmax and the rule
                         fires. Anchored on true absolute numbers; assumes the
                         decoder's own flips are unchanged by the re-rank.
"""
from __future__ import annotations

import json
from pathlib import Path

from board_reader_core.protocol import encode_placement, parse_placement

import residuals
import takes
from candidates import DemoteTop1
from evaluate import confidence_effect, evaluate

HERE = Path(__file__).resolve().parent
NAMES = list(takes.ALL)
ROWS = {n: residuals.squares(n) for n in NAMES}


def anchored_prior(name: str, rule) -> tuple[int, int, int, int, int, int, int]:
    rep = takes.load(name)
    rows = {(r.event, r.square): r for r in ROWS[name]}
    from board_reader_inference.trial import SQUARE_INDEX

    order = sorted(SQUARE_INDEX, key=lambda s: SQUARE_INDEX[s])
    sq = cb = cn = fixed = broken = pb = pn = 0
    for i, read in enumerate(rep.reads):
        truth = [c for row in parse_placement(rep.truth.segments[read.segment].placement) for c in row]
        prior = [c for row in parse_placement(read.prior) for c in row]
        new = list(prior)
        for j, name_sq in enumerate(order):
            r = rows[(i, name_sq)]
            if prior[j] == r.c1 and rule.swap(r.c1, r.p1, r.c2, r.p2):
                new[j] = r.c2
        for t, x, y in zip(truth, prior, new):
            sq += 1
            cb += x == t
            cn += y == t
            fixed += (x != t) and (y == t)
            broken += (x == t) and (y != t)
        pb += prior == truth
        pn += new == truth
    return sq, cb, cn, fixed, broken, pb, pn


out = {}
for tau in (0.10, 0.15, 0.18, 0.20, 0.21):
    rule = DemoteTop1("b", tau)
    print(f"\n=================== demote top-1 'b' when (p1 - p2) < {tau:.2f} ===================")
    print(f"{'take':8s} {'view':14s}  per-square %              perfect-board %          fixed  broken")
    per = {}
    for n in NAMES:
        d = evaluate(n, rule)
        for key, label in (("raw", "A raw"), ("prior", "B prior(recon)")):
            x = d[key]
            print(f"{n:8s} {label:14s}  {x.correct_base/x.squares*100:6.2f} -> "
                  f"{x.correct_new/x.squares*100:6.2f} ({(x.correct_new-x.correct_base)/x.squares*100:+5.2f})   "
                  f"{x.perfect_base/x.events*100:5.1f} -> {x.perfect_new/x.events*100:5.1f} "
                  f"({(x.perfect_new-x.perfect_base)/x.events*100:+5.1f})   "
                  f"{x.fixed:4d}  {x.broken:4d}")
        sq, cb, cn, f, b, pb, pn = anchored_prior(n, rule)
        ev = len(takes.load(n).reads)
        print(f"{'':8s} {'C prior(anchor)':14s}  {cb/sq*100:6.2f} -> {cn/sq*100:6.2f} "
              f"({(cn-cb)/sq*100:+5.2f})   {pb/ev*100:5.1f} -> {pn/ev*100:5.1f} "
              f"({(pn-pb)/ev*100:+5.1f})   {f:4d}  {b:4d}")
        swp, c0, c1, nr = confidence_effect(n, rule)
        print(f"{'':8s} calibration:    {swp}/{nr} already-correct squares re-ranked away; "
              f"mean top-1 prob on correct squares {c0:.4f} -> {c1:.4f}")
        per[n] = {
            "raw": vars(d["raw"]), "prior_recon": vars(d["prior"]),
            "prior_anchor": {"squares": sq, "correct_base": cb, "correct_new": cn,
                             "fixed": f, "broken": b, "perfect_base": pb,
                             "perfect_new": pn, "events": ev},
            "correct_squares_reranked_away": swp,
        }
    out[f"tau={tau}"] = per

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "final.json").write_text(json.dumps(out, indent=2))
