"""Per-pair, per-take: does a swap rule on this pair transfer between takes?

A rule is only implementable if the same tau helps everywhere. This prints, for
each ordered pair worth considering, the fixed/broken count in each take at a
sweep of margin thresholds.
"""
from __future__ import annotations
from collections import defaultdict
import residuals

ROWS = {n: residuals.squares(n) for n in residuals.takes.ALL}
TAUS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.01)

cand = defaultdict(int)
for n, rs in ROWS.items():
    for r in rs:
        if r.truth == r.c2 and r.wrong:
            cand[(r.c1, r.c2)] += 1

print(f"{'pair':>10s} {'tau':>5s}  " + "  ".join(f"{n:>14s}" for n in ROWS) + "    total")
for pair, total in sorted(cand.items(), key=lambda kv: -kv[1]):
    if total < 3:
        continue
    print()
    for tau in TAUS:
        cells, tf, tb = [], 0, 0
        for n, rs in ROWS.items():
            sel = [r for r in rs if (r.c1, r.c2) == pair and r.margin < tau]
            f = sum(1 for r in sel if r.truth == pair[1])
            b = sum(1 for r in sel if r.truth == pair[0])
            cells.append(f"+{f:<3d}/-{b:<3d} ={f-b:+3d}")
            tf += f; tb += b
        print(f"{pair[0]+'->'+pair[1]:>10s} {tau:5.2f}  " + "  ".join(cells)
              + f"    +{tf}/-{tb} = {tf-tb:+d}")
