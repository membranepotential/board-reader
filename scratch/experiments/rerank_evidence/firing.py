"""Blast radius and calibration side-effects of the recommended rule."""
from __future__ import annotations
import numpy as np
import residuals, takes
from candidates import DemoteTop1

TAU = 0.20
rule = DemoteTop1("b", TAU)
print(f"rule: demote top-1 'b' when p1 - p2 < {TAU}\n")
print(f"{'take':8s} {'events':>7s} {'fires':>6s} {'per-board':>10s} "
      f"{'fixes':>6s} {'breaks':>7s} {'neutral':>8s}   conf on fixed squares (p1 -> p2)")
tot = [0, 0, 0, 0, 0]
for n in takes.ALL:
    rows = residuals.squares(n)
    ev = len({r.event for r in rows})
    fires = [r for r in rows if rule.swap(r.c1, r.p1, r.c2, r.p2)]
    fx = [r for r in fires if r.truth == r.c2]
    bk = [r for r in fires if r.truth == r.c1]
    nt = len(fires) - len(fx) - len(bk)
    p1 = np.mean([r.p1 for r in fx]) if fx else float("nan")
    p2 = np.mean([r.p2 for r in fx]) if fx else float("nan")
    print(f"{n:8s} {ev:7d} {len(fires):6d} {len(fires)/ev:10.2f} {len(fx):6d} {len(bk):7d} "
          f"{nt:8d}   {p1:.3f} -> {p2:.3f}  (max after: "
          f"{max((r.p2 for r in fx), default=float('nan')):.3f})")
    for i, v in enumerate((ev, len(fires), len(fx), len(bk), nt)):
        tot[i] += v
print(f"\ntotal: {tot[0]} events, {tot[1]} firings ({tot[1]/tot[0]:.2f}/board), "
      f"{tot[2]} fixes, {tot[3]} breaks, {tot[4]} neutral (square was wrong either way)")
