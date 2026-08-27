"""The experiment: oracle (fit-on-all) ceiling, then leave-one-take-out."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import residuals
import takes
from evaluate import confidence_effect, evaluate
from transforms import ClassBias, Identity, PairMargin, fit_class_bias, fit_pair_margin

HERE = Path(__file__).resolve().parent
NAMES = list(takes.ALL)
ROWS = {n: residuals.squares(n) for n in NAMES}


def show(tag: str, rule, names, *, with_prior=True) -> dict:
    out = {}
    print(f"\n--- {tag} ---")
    print(f"{'take':8s} {'stream':6s}  per-square %          perfect %             fixed/broken")
    for n in names:
        d = evaluate(n, rule, with_prior=with_prior)
        for s in ("raw", "prior") if with_prior else ("raw",):
            print(f"{n:8s} {s:6s}  {d[s].row()}")
        sw, cb, ca, nr = confidence_effect(n, rule)
        print(f"{'':8s} conf:   {sw}/{nr} already-correct squares swapped away; "
              f"mean top-1 prob on correct squares {cb:.4f} -> {ca:.4f}")
        out[n] = {s: vars(d[s]) for s in d}
        out[n]["correct_swapped_away"] = sw
    return out


def describe(rule) -> str:
    if isinstance(rule, PairMargin):
        return "pair-margin tau: " + ", ".join(
            f"{a}->{b}: {t:.4f}" for (a, b), t in sorted(rule.tau.items(), key=lambda kv: -kv[1])
        )
    if isinstance(rule, ClassBias):
        return "class-bias b: " + ", ".join(f"{c}:{v:+.2f}" for c, v in rule.b.items() if abs(v) > 1e-9)
    return rule.name


results: dict = {}

# ---------------------------------------------------------------- oracle ----
all_rows = [r for n in NAMES for r in ROWS[n]]
for tag, fitter in (("ORACLE pair-margin (fit on ALL 3 takes)", lambda rs: fit_pair_margin(rs, min_gain=1)),
                    ("ORACLE class-bias  (fit on ALL 3 takes)", fit_class_bias)):
    rule = fitter(all_rows)
    print(f"\n### {tag}\n{describe(rule)}")
    results[tag] = {"rule": describe(rule), "per_take": show(tag, rule, NAMES)}

# ------------------------------------------------------------- leave-one ----
for min_gain in (1, 2, 3):
    tag = f"LOO pair-margin (min_gain={min_gain})"
    print(f"\n\n### {tag}")
    per = {}
    for held in NAMES:
        fit_rows = [r for n in NAMES if n != held for r in ROWS[n]]
        rule = fit_pair_margin(fit_rows, min_gain=min_gain)
        print(f"\n  held out: {held}   fitted on {[n for n in NAMES if n != held]}")
        print(f"  {describe(rule)}")
        per[held] = {"rule": describe(rule), **show(f"{tag} / held {held}", rule, [held])}
    results[tag] = per

tag = "LOO class-bias"
print(f"\n\n### {tag}")
per = {}
for held in NAMES:
    fit_rows = [r for n in NAMES if n != held for r in ROWS[n]]
    rule = fit_class_bias(fit_rows)
    print(f"\n  held out: {held}")
    print(f"  {describe(rule)}")
    per[held] = {"rule": describe(rule), **show(f"{tag} / held {held}", rule, [held])}
results[tag] = per

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "loo.json").write_text(json.dumps(results, indent=2, default=str))
