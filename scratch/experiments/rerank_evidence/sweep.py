from __future__ import annotations
import candidates, residuals, takes
from evaluate import evaluate

TAUS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
NAMES = list(takes.ALL)

for fam, make in candidates.FAMILIES.items():
    print(f"\n### {fam}")
    print(f"{'tau':>5s}  " + "".join(f"{n+' raw':>26s}{n+' prior':>26s}" for n in NAMES))
    for tau in TAUS:
        rule = make(tau)
        cells = []
        for n in NAMES:
            d = evaluate(n, rule)
            for s in ("raw", "prior"):
                x = d[s]
                cells.append(f"  {(x.correct_new-x.correct_base):+3d}sq {x.fixed:2d}/-{x.broken:<2d} "
                             f"pf{x.perfect_new-x.perfect_base:+2d}")
        print(f"{tau:5.2f}  " + "".join(f"{c:>26s}" for c in cells))
