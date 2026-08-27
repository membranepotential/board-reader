"""Step 1b — the ordered (top1, top2) pair table.

For a post-hoc re-rank the only decision available per square is *keep top-1* or
*swap to top-2*. So the whole design space is: for which ordered pairs, and at
which margin, do we swap? This table is what any such rule has to beat.
"""
from __future__ import annotations
from collections import defaultdict
import residuals

rows = [r for rs in residuals.all_squares().values() for r in rs]
by: dict[tuple[str, str], list] = defaultdict(list)
for r in rows:
    by[(r.c1, r.c2)].append(r)

print(f"{'pair(top1,top2)':>16s} {'n':>5s} {'right':>6s} {'fixable':>8s} {'other':>6s} "
      f"{'med_m_right':>12s} {'med_m_fix':>10s}  gain@swap-all")
tot_fix = tot_break = 0
def med(xs): 
    xs = sorted(xs); return xs[len(xs)//2] if xs else float('nan')
for (c1, c2), rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    right = [r for r in rs if r.truth == c1]
    fix = [r for r in rs if r.truth == c2]
    other = len(rs) - len(right) - len(fix)
    if not fix and len(rs) < 5:
        continue
    gain = len(fix) - len(right)
    if len(fix):
        tot_fix += len(fix); tot_break += len(right)
    print(f"{c1+','+c2:>16s} {len(rs):5d} {len(right):6d} {len(fix):8d} {other:6d} "
          f"{med([r.margin for r in right]):12.4f} {med([r.margin for r in fix]):10.4f}  "
          f"{gain:+d}" + ("   <-- net win" if gain > 0 else ""))
print(f"\ntotal over pairs that have any fixable: fix {tot_fix}, break {tot_break}")
