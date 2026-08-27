"""Leave-one-take-out over the CLASS-level demote rule (the shape we would ship)."""
from __future__ import annotations
import json
from pathlib import Path

import residuals, takes
from evaluate import confidence_effect, evaluate
from transforms import Rule, fit_stable_demote

HERE = Path(__file__).resolve().parent
NAMES = list(takes.ALL)
ROWS = {n: residuals.squares(n) for n in NAMES}
TAUS = tuple(round(0.02 * i, 2) for i in range(1, 21))


class Demote(Rule):
    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        self.name = "demote"

    def swap(self, c1, p1, c2, p2) -> bool:
        t = self.table.get(c1)
        return t is not None and (p1 - p2) < t


out = {}
for min_fix in (2, 3):
    print(f"\n\n######## class-level demote, gate: fixes>={min_fix}, breaks<=0 "
          f"in EACH fitting take ########")
    for held in NAMES:
        fit = {n: ROWS[n] for n in NAMES if n != held}
        table = fit_stable_demote(fit, taus=TAUS, min_fix=min_fix)
        rule = Demote(table)
        spec = ", ".join(f"demote {c!r} when margin < {t:.2f}" for c, t in sorted(table.items()))
        print(f"\n  held {held}   fitted on {sorted(fit)}")
        print(f"  rule: {spec or '(empty)'}")
        d = evaluate(held, rule)
        for s in ("raw", "prior"):
            print(f"    {s:6s} {d[s].row()}")
        sw, c0, c1, nr = confidence_effect(held, rule)
        print(f"    conf: {sw}/{nr} already-correct squares swapped; mean p1 {c0:.4f} -> {c1:.4f}")
        out[f"minfix{min_fix}/{held}"] = {"rule": spec, **{s: vars(d[s]) for s in d}}

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "demote_loo.json").write_text(json.dumps(out, indent=2))
