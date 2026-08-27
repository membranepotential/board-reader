"""Leave-one-take-out with the transferability gate: family AND tau are chosen
from the two fitting takes only."""
from __future__ import annotations
import json
from pathlib import Path

import residuals, takes
from candidates import PairSet
from evaluate import confidence_effect, evaluate
from transforms import fit_stable_pair_margin

HERE = Path(__file__).resolve().parent
NAMES = list(takes.ALL)
ROWS = {n: residuals.squares(n) for n in NAMES}
TAUS = tuple(round(0.02 * i, 2) for i in range(1, 21))  # 0.02 .. 0.40

out = {}
for max_broken, min_fix in ((0, 1), (0, 2), (0, 3), (1, 3)):
    print(f"\n\n######## gate: fixes>={min_fix} and breaks<={max_broken} in EACH fitting take ########")
    for held in NAMES:
        fit = {n: ROWS[n] for n in NAMES if n != held}
        pairs = fit_stable_pair_margin(fit, taus=TAUS, max_broken=max_broken, min_fix=min_fix)
        rule = PairSet(frozenset(pairs), 0.0)
        # per-pair taus differ; use a PairSet-with-map
        class Mapped(PairSet):
            def swap(self, c1, p1, c2, p2):
                t = pairs.get((c1, c2))
                return t is not None and (p1 - p2) < t
        rule = Mapped(frozenset(pairs), 0.0)
        spec = ", ".join(f"{a}<-{b} tau<{t:.2f}" for (a, b), t in sorted(pairs.items()))
        print(f"\n  held {held}  fitted on {sorted(fit)}")
        print(f"  rule: {spec or '(empty)'}")
        d = evaluate(held, rule)
        for s in ("raw", "prior"):
            print(f"    {s:6s} {d[s].row()}")
        sw, cb, ca, nr = confidence_effect(held, rule)
        print(f"    conf: {sw}/{nr} already-correct squares swapped; mean p1 {cb:.4f} -> {ca:.4f}")
        out[f"gate_b{max_broken}_f{min_fix}/{held}"] = {"rule": spec, **{s: vars(d[s]) for s in d}}

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "stable_loo.json").write_text(json.dumps(out, indent=2))
