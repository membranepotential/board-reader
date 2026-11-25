"""Tables of the auto-frame probe from the run outputs.

    uv run python tools/probes/autocrop_analyze.py OUT_DIR

Reads OUT_DIR/chessq.jsonl (autocrop_run.py chessq), classical.jsonl
(autocrop_classical.py) and classical_e2e.jsonl (autocrop_run.py classical).

Terms used in the tables:

* **good** proposal: the mean corner error is at most 5 % of the board width.
  With a 10 % margin the crop then still holds the whole board.
* **trusted** grid: the homography fit of the 64 square centroids has
  R^2 >= ``MIN_R2``. Anything else is refused ("no board found").
* **e2e**: ChessQ's per-square accuracy and perfect boards on the crop the
  proposal gives at the original (canvas) resolution, against the same
  numbers on the hand-drawn crop. Only images with a truth placement count; a
  refused proposal counts as no read and is excluded (the column "refused"
  says how many).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from autocrop_lib import bbox, corner_error_any, iou

#: Good final grids (coarse to fine, tiles) measured R^2 >= 0.90 down to 15 %;
#: bad ones <= 0.89 (the timeboxed run, docs/reports/2026-11-25-autocrop-probe.md).
MIN_R2 = 0.90
GOOD = 5.0  # % of board width
MAX_ENERGY = (
    -12.4
)  # DESIGN D16's obscured threshold: a verify read above it is no clean board
CONDS = ["as-is", "room-35", "room-25", "room-15"]


def trusted(g: dict | None) -> bool:
    return bool(g and g.get("found") and g["r2"] >= MIN_R2)


def verified(v: dict | None) -> bool:
    """A crop read that itself shows a clean, whole board."""
    return bool(
        v
        and v.get("read")
        and v["r2"] is not None
        and v["r2"] >= MIN_R2
        and v["energy"] <= MAX_ENERGY
    )


def load(path: Path) -> list[dict]:
    return (
        [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        if path.exists()
        else []
    )


def pct(xs: list[bool]) -> str:
    return f"{100 * np.mean(xs):.0f}" if xs else "-"


def med(xs: list[float]) -> str:
    return f"{np.median(xs):.1f}" if xs else "-"


def e2e_cell(pairs: list[tuple[int, int]]) -> str:
    """(strategy correct, hand correct) per image -> 'sq% / perf% (hand sq% / perf%)'."""
    if not pairs:
        return "-"
    s = np.array(pairs, float)
    return (
        f"{100 * s[:, 0].mean() / 64:.2f} / {100 * (s[:, 0] == 64).mean():.0f} "
        f"({100 * s[:, 1].mean() / 64:.2f} / {100 * (s[:, 1] == 64).mean():.0f})"
    )


def strategies(rec: dict, cls: dict | None, cls_e2e: dict | None) -> dict[str, dict]:
    """Each strategy's final proposal: the grid summary it ends on, whether it
    proposes (trusted) or refuses, the e2e read of its 10 % crop, its
    localisation reads."""
    e2e = rec["e2e"]
    p1, c2f, tiles = rec["pass1"], rec["c2f"], rec["tiles"]
    out = {
        "pass1": {
            "g": p1,
            "ok": trusted(p1),
            "v": e2e.get("pass1", {}).get("0.1"),
            "reads": 1,
        },
        "c2f": {
            "g": c2f,
            "ok": trusted(c2f),
            "v": e2e.get("c2f", {}).get("0.1"),
            "reads": 2,
        },
    }
    # Coarse to fine; when its grid is untrusted, the five tiles and a refine.
    if trusted(c2f):
        out["c2f|tiles"] = {**out["c2f"]}
    else:
        out["c2f|tiles"] = {
            "g": tiles,
            "ok": trusted(tiles),
            "v": e2e.get("tiles", {}).get("0.1"),
            "reads": 8,
        }
    if cls is not None:
        for ladder in ("base", "wide"):
            q = cls.get(ladder)
            g = None
            if q is not None:
                g = {"found": True, "quad": q}
            v = (cls_e2e or {}).get(ladder)
            out[f"classical-{ladder}"] = {"g": g, "ok": q is not None, "v": v, "reads": 0}
        # Classical proposes, one ChessQ read of its crop verifies; ChessQ
        # localisation (c2f, tiles when pass 1 is untrusted) only as the fallback.
        w = out["classical-wide"]
        if w["ok"] and verified(w["v"]):
            out["classical+verify"] = {**w, "reads": 1}
        else:
            fb = out["c2f|tiles"]
            out["classical+verify"] = {
                **fb,
                "reads": 1 + fb["reads"] if w["ok"] else fb["reads"],
            }
    return out


def main(out_dir: Path) -> None:
    recs = load(out_dir / "chessq.jsonl")
    cls_rows = load(out_dir / "classical.jsonl")
    cls: dict[tuple[str, str], dict] = defaultdict(dict)
    cls_ms: dict[tuple[str, str], dict] = defaultdict(dict)
    for r in cls_rows:
        cls[(r["key"], r["cond"])][r["ladder"]] = r["quad"]
        cls_ms[(r["key"], r["cond"])][r["ladder"]] = r["ms"]
    cls_e2e = {
        (r["key"], r["cond"]): r["e2e"] for r in load(out_dir / "classical_e2e.jsonl")
    }

    # Truth outline per (key, cond), recomputed from the deterministic layout.
    from autocrop_run import canvas_quad, conditions, in_run, load_items

    index = {it.key: (n, it) for n, it in enumerate(load_items())}
    # Score each condition on exactly its targeted images (earlier, wider runs
    # left extra records behind).
    recs = [r for r in recs if in_run(r["cond"], index[r["key"]][1])]
    cls_ms = {
        k: v for k, v in cls_ms.items() if k[1] in CONDS and in_run(k[1], index[k[0]][1])
    }
    truth_quad = {
        (r["key"], r["cond"]): canvas_quad(
            index[r["key"]][1], index[r["key"]][0], conditions()[r["cond"]]
        )
        for r in recs
    }

    # A scale-1 canvas has no hand read of its own: it is the as-is read.
    as_is = {r["key"]: r["hand"] for r in recs if r["cond"] == "as-is"}
    for r in recs:
        r.setdefault("hand", as_is.get(r["key"]))
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_cond[r["cond"]].append(r)

    names = [
        "pass1",
        "c2f",
        "c2f|tiles",
        "classical-base",
        "classical-wide",
        "classical+verify",
    ]
    print(f"good = corner error <= {GOOD} %; trusted = grid R2 >= {MIN_R2}")
    print("\n## Per size and strategy")
    print(
        "| condition | n | strategy | proposes % | good of proposed % | bad proposed (n) | missed good (n) "
        "| corner err med % | IoU med | reads | e2e @10 % sq / perfect (hand) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    summary_rows = []
    for c in CONDS:
        rows = by_cond.get(c, [])
        if not rows:
            continue
        for name in names:
            prop, good, bad_n, miss_n, errs, ious, reads, pairs = (
                [],
                [],
                0,
                0,
                [],
                [],
                [],
                [],
            )
            for r in rows:
                k = (r["key"], c)
                s = strategies(r, cls.get(k) if k in cls else None, cls_e2e.get(k)).get(
                    name
                )
                if s is None:
                    continue
                tq = truth_quad[k]
                g = s["g"]
                err = (
                    corner_error_any(np.array(g["quad"], np.float32), tq)
                    if g and g.get("found")
                    else 999.0
                )
                is_good = err <= GOOD
                prop.append(s["ok"])
                reads.append(s["reads"])
                if s["ok"]:
                    good.append(is_good)
                    errs.append(err)
                    ious.append(iou(bbox(np.array(g["quad"], np.float32)), bbox(tq)))
                    bad_n += not is_good
                    v, h = s["v"], r["hand"]
                    if v and v.get("read") and v.get("correct") is not None:
                        pairs.append((v["correct"], h["correct"]))
                elif is_good:
                    miss_n += 1
            if not prop:
                continue
            print(
                f"| {c} | {len(prop)} | {name} | {pct(prop)} | {pct(good)} | {bad_n} | {miss_n} "
                f"| {med(errs)} | {np.median(ious):.3f} | {np.mean(reads):.1f} | {e2e_cell(pairs)} |"
                if ious
                else f"| {c} | {len(prop)} | {name} | {pct(prop)} | - | 0 | {miss_n} | - | - | {np.mean(reads):.1f} | - |"
            )
            summary_rows.append((c, name, pairs))

    print("\n## Margin sweep (coarse to fine, trusted proposals only)")
    print("| condition | 5 % | 10 % | 15 % | hand |")
    print("|---|---|---|---|---|")
    for c in CONDS:
        cells, hand = [], []
        for m in ("0.05", "0.1", "0.15"):
            pairs = []
            for r in by_cond.get(c, []):
                if not trusted(r["c2f"]):
                    continue
                v = r["e2e"].get("c2f", {}).get(m)
                if v and v.get("read") and v.get("correct") is not None:
                    pairs.append((v["correct"], r["hand"]["correct"]))
            cells.append(e2e_cell(pairs).split(" (")[0] if pairs else "-")
            if m == "0.1":
                hand = pairs
        if hand:
            h = np.array(hand, float)
            print(
                f"| {c} | {' | '.join(cells)} | {100 * h[:, 1].mean() / 64:.2f} / {100 * (h[:, 1] == 64).mean():.0f} |"
            )

    print("\n## Failure detection: does 'trusted' separate good from bad grids?")
    print("| grid | condition group | good: trusted % | bad: trusted % | n good / n bad |")
    print("|---|---|---|---|---|")
    groups = {
        "as-is + 50": ["as-is", "grey-50", "room-50", "prev480-50"],
        "35 + 25": [c for c in CONDS if c.endswith(("35", "25"))],
        "15": [c for c in CONDS if c.endswith("15")],
    }
    signals = defaultdict(lambda: ([], []))
    for gname, cs in groups.items():
        for grid in ("pass1", "c2f", "tiles"):
            tg, tb, ng, nb = [], [], 0, 0
            for c in cs:
                for r in by_cond.get(c, []):
                    g = r[grid]
                    if not g.get("found"):
                        continue
                    is_good = g["corner_err"] <= GOOD
                    (tg if is_good else tb).append(trusted(g))
                    ng += is_good
                    nb += not is_good
                    signals[grid][0].append(
                        [g["inliers"], -g["rms_sq"], g["r2"], -g["energy"], g["mass"]]
                    )
                    signals[grid][1].append(is_good)
            print(f"| {grid} | {gname} | {pct(tg)} | {pct(tb)} | {ng} / {nb} |")
    print("\nAUC of single signals for good vs bad grids (all conditions):")
    for grid, (x, y) in signals.items():
        x, y = np.array(x, float), np.array(y, bool)
        if y.all() or (~y).all():
            continue
        aucs = []
        for j, name in enumerate(["inliers", "-rms", "r2", "-energy", "mass"]):
            aucs.append(f"{name} {auc(x[:, j], y):.3f}")
        print(f"  {grid}: " + ", ".join(aucs) + f"  (good {y.sum()}, bad {(~y).sum()})")

    print("\n## Classical detector time (ms, CPU Python, one process)")
    print(
        "| condition | base: found % / median / p90 ms | wide: found % / median / p90 ms |"
    )
    print("|---|---|---|")
    for c in CONDS:
        cells = []
        for ladder in ("base", "wide"):
            ms = [v[ladder] for (k, cc), v in cls_ms.items() if cc == c and ladder in v]
            found = [
                cls[(k, cc)][ladder] is not None
                for (k, cc) in cls_ms
                if cc == c and ladder in cls[(k, cc)]
            ]
            cells.append(
                f"{pct(found)} / {med(ms)} / {np.percentile(ms, 90):.0f}" if ms else "-"
            )
        print(f"| {c} | {' | '.join(cells)} |")

    print("\n## ChessQ read time (session.run ms, this run)")
    ms = [r["pass1"]["ms"] for r in recs]
    print(f"median {np.median(ms):.1f}, p90 {np.percentile(ms, 90):.1f} (n={len(ms)})")

    print("\n## Per group, room-25, c2f|tiles and classical+verify")
    for c in ("room-35", "room-25", "room-15"):
        for grp in ("still", "gate", "live"):
            rows = [r for r in by_cond.get(c, []) if r["group"] == grp]
            for name in ("c2f|tiles", "classical-wide", "classical+verify"):
                ok, good = [], []
                for r in rows:
                    k = (r["key"], c)
                    s = strategies(r, cls.get(k), cls_e2e.get(k)).get(name)
                    if s is None:
                        continue
                    ok.append(s["ok"])
                    if s["ok"] and s["g"] and s["g"].get("found"):
                        good.append(
                            corner_error_any(
                                np.array(s["g"]["quad"], np.float32), truth_quad[k]
                            )
                            <= GOOD
                        )
                print(
                    f"  {c} {grp:5s} {name:17s} n={len(ok):3d} proposes {pct(ok)} % good {pct(good)} %"
                )


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC AUC that a good grid scores higher than a bad one (ties count half)."""
    pos, neg = score[label], score[~label]
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
