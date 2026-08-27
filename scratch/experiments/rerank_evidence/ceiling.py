"""The re-ranking CEILING: what a perfect top-2 oracle could do, vs the rule.

A post-hoc re-rank can never see past the digest, so an oracle that always picks
the truth *when it is one of the two* bounds the entire family. What that oracle
leaves on the table is what only a retrain (or richer evidence) can reach.
"""
from __future__ import annotations
from board_reader_core.protocol import parse_placement
from board_reader_inference.trial import SQUARE_INDEX
import residuals, takes
from candidates import DemoteTop1

RULE = DemoteTop1("b", 0.20)
order = sorted(SQUARE_INDEX, key=lambda s: SQUARE_INDEX[s])

print(f"{'take':8s} {'sq':>5s} {'raw err':>8s} {'in top2':>8s} {'ceiling':>8s} "
      f"{'rule fix':>9s} {'of ceiling':>11s} || prior-stream perfect-board %: "
      f"real / rule / top-2 oracle")
T = [0, 0, 0, 0]
for n in takes.ALL:
    rep = takes.load(n)
    rows = {(r.event, r.square): r for r in residuals.squares(n)}
    wrong = [r for r in rows.values() if r.wrong]
    fixable = [r for r in wrong if r.fixable]
    ruled = [r for r in wrong if RULE.swap(r.c1, r.p1, r.c2, r.p2) and r.truth == r.c2]
    pb = pr = po = 0
    for i, read in enumerate(rep.reads):
        truth = [c for row in parse_placement(rep.truth.segments[read.segment].placement) for c in row]
        prior = [c for row in parse_placement(read.prior) for c in row]
        rule_b, orc_b = list(prior), list(prior)
        for j, sq in enumerate(order):
            r = rows[(i, sq)]
            if prior[j] != r.c1:
                continue  # the decoder overrode this square; leave it alone
            if RULE.swap(r.c1, r.p1, r.c2, r.p2):
                rule_b[j] = r.c2
            if r.truth in (r.c1, r.c2):
                orc_b[j] = r.truth
        pb += prior == truth
        pr += rule_b == truth
        po += orc_b == truth
    e = len(rep.reads)
    print(f"{n:8s} {len(rows):5d} {len(wrong):8d} {len(fixable):8d} "
          f"{100*len(fixable)/len(wrong):7.1f}% {len(ruled):9d} "
          f"{100*len(ruled)/len(fixable):10.1f}% || "
          f"{100*pb/e:6.1f} / {100*pr/e:6.1f} / {100*po/e:6.1f}")
    for i, v in enumerate((len(rows), len(wrong), len(fixable), len(ruled))):
        T[i] += v
print(f"\nALL      {T[0]:5d} {T[1]:8d} {T[2]:8d} {100*T[2]/T[1]:7.1f}% {T[3]:9d} "
      f"{100*T[3]/T[2]:10.1f}%")
