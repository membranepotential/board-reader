"""Tables for the preview follow-up of the auto-frame probe
(docs/reports/2026-11-25-autocrop-probe.md): locating on the 480 px preview, the
still jitter, the moved displacements, and a drift rule's false/miss rates.

    uv run python tools/probes/autocrop_preview_analyze.py OUT_DIR

Reads ``preview_locate.jsonl``, ``preview_drift.jsonl`` and ``preview_cost.json``
written by ``autocrop_preview.py``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from autocrop_lib import bbox, iou

MIN_R2 = 0.90
GOOD = 5.0  # corner error, % of the board width
#: A preview whose truth outline moved less than this against the reference
#: counts as still (the outlines' own noise is ~0.5 %).
STILL_TRUTH = 1.5
PERSIST = 2
#: A consecutive pair counts for the jitter when its outline moved less than this.
PAIR_TRUTH = 1.0


def pct(a: list[float], q: float) -> float:
    return float(np.percentile(a, q)) if a else float("nan")


def auc(neg: list[float], pos: list[float]) -> float:
    """P(a positive scores above a negative), ties half."""
    n, p = np.array(neg)[:, None], np.array(pos)[None, :]
    return float(((p > n) + 0.5 * (p == n)).mean())


def load(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def trusted(g: dict | None) -> bool:
    return g is not None and g["r2"] >= MIN_R2


# ------------------------------------------------------------------- locate


def locate_table(recs: list[dict]) -> None:
    print("\n## Locate on the preview (R² >= 0.90 = found; good = corner err <= 5 %)")
    print(
        "| share | n | board px | strategy | found | good of found | bad | err med | R² p10 / med |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in recs:
        by[r["share"]].append(r)
    for share, rs in by.items():
        px = np.median([r["board_px"] for r in rs])
        for s in ("pass1", "c2f"):
            gs = [r[s] for r in rs]
            found = [g for g in gs if trusted(g)]
            good = [g for g in found if g["corner_err"] <= GOOD]
            r2 = [g["r2"] for g in gs if g is not None]
            err = [g["corner_err"] for g in found]
            print(
                f"| {share} | {len(rs)} | {px:.0f} | {s} | {len(found) / len(rs):.0%} "
                f"| {len(good) / max(1, len(found)):.0%} | {len(found) - len(good)} "
                f"| {np.median(err) if err else float('nan'):.1f} % "
                f"| {pct(r2, 10):.3f} / {pct(r2, 50):.3f} |"
            )


# -------------------------------------------------------------------- drift


def disp(a: np.ndarray, b: np.ndarray, ref_w: float) -> float:
    """Max corner displacement in % of the reference board width."""
    return float(np.linalg.norm(a - b, axis=1).max() / ref_w * 100)


def width(q: np.ndarray) -> float:
    x0, _, x1, _ = bbox(q)
    return x1 - x0


def drift_tables(recs: list[dict], strategy: str) -> None:
    still = defaultdict(list)
    moved = defaultdict(list)
    for r in recs:
        key = (r["take"], r["share"])
        (still if r["move"] is None else moved)[key].append(r)
    # Reference per take: the median of the first three trusted previews (the
    # board where the crop was set), and the truth outline at those previews.
    refs = {}
    for key, rs in still.items():
        rs.sort(key=lambda r: r["i"])
        first = [r for r in rs if trusted(r[strategy])][:3]
        if len(first) < 3:
            continue
        lq = np.median([r[strategy]["quad"] for r in first], axis=0)
        tq = np.median([r["truth"] for r in first], axis=0)
        refs[key] = (lq, tq)

    def d_loc(r: dict, key) -> float | None:
        if key not in refs or not trusted(r[strategy]):
            return None
        lq = refs[key][0]
        return disp(np.array(r[strategy]["quad"]), lq, width(lq))

    def d_iou(r: dict, key) -> float | None:
        if key not in refs or not trusted(r[strategy]):
            return None
        return 1 - iou(bbox(np.array(r[strategy]["quad"])), bbox(refs[key][0]))

    def d_truth(r: dict, key) -> float:
        tq = refs[key][1]
        return disp(np.array(r["truth"]), tq, width(tq))

    print(f"\n## Drift, strategy {strategy}")
    print(
        "| share | previews | untrusted | outline moved >= 1.5 % | jitter med / p99 / max | "
        "vs ref med / p99 / max | 1-IoU p99 |"
    )
    print("|---|---|---|---|---|---|---|")
    vs_ref_all = {}
    for share in sorted({k[1] for k in still}):
        jit, vs, ious, n, untr, real = [], [], [], 0, 0, 0
        for key, rs in still.items():
            if key[1] != share or key not in refs:
                continue
            prev = None  # (located quad, truth quad) of the last trusted preview
            for r in rs:
                n += 1
                if not trusted(r[strategy]):
                    untr += 1
                    continue
                q, t = np.array(r[strategy]["quad"]), np.array(r["truth"])
                # Jitter: consecutive trusted previews whose outline did not move.
                if prev is not None and disp(t, prev[1], width(t)) < PAIR_TRUTH:
                    jit.append(disp(q, prev[0], width(prev[0])))
                prev = (q, t)
                # Against the reference: only while the outline is still near it
                # (the 2026-08 tripods creep by ~2 % over a take).
                if d_truth(r, key) >= STILL_TRUTH:
                    real += 1
                    continue
                vs.append(d_loc(r, key))
                ious.append(d_iou(r, key))
        vs_ref_all[share] = vs
        print(
            f"| {share} | {n} | {untr / max(1, n):.1%} | {real} "
            f"| {pct(jit, 50):.2f} / {pct(jit, 99):.2f} / {max(jit, default=float('nan')):.2f} % "
            f"| {pct(vs, 50):.2f} / {pct(vs, 99):.2f} / {max(vs, default=float('nan')):.2f} % "
            f"| {pct(ious, 99):.3f} |"
        )

    # Moved: per (kind, magnitude) the located and the true displacement.
    rows = defaultdict(
        lambda: {"loc": [], "truth": [], "iou": [], "pairs": [], "untr": 0, "n": 0}
    )
    events = defaultdict(list)  # (kind, mag) -> list of per-event D sequences
    for key, rs in moved.items():
        if key not in refs:
            continue
        seq = defaultdict(list)
        for r in sorted(rs, key=lambda r: r["i"]):
            kind, mag, e = r["move"]
            row = rows[(kind, mag)]
            row["n"] += 1
            row["truth"].append(d_truth(r, key))
            d = d_loc(r, key)
            if d is None:
                row["untr"] += 1
            else:
                row["loc"].append(d)
                row["iou"].append(d_iou(r, key))
                row["pairs"].append((d_truth(r, key), d, d_iou(r, key)))
            seq[(kind, mag, e)].append(d)
        for (kind, mag, _), ds in seq.items():
            events[(kind, mag)].append(ds)
    print("\n| move | n | untrusted | true disp med | located disp p10 / med | 1-IoU med |")
    print("|---|---|---|---|---|---|")
    for (kind, mag), row in rows.items():
        print(
            f"| {kind} {mag:+g} | {row['n']} | {row['untr'] / row['n']:.0%} "
            f"| {np.median(row['truth']):.1f} % | {pct(row['loc'], 10):.1f} / "
            f"{pct(row['loc'], 50):.1f} % | {pct(row['iou'], 50):.3f} |"
        )

    # The rule: fire when PERSIST consecutive trusted previews exceed theta;
    # an untrusted preview neither counts nor breaks the streak.
    def fires(ds: list[float | None], theta: float) -> int:
        streak, n = 0, 0
        for d in ds:
            if d is None:
                continue
            streak = streak + 1 if d > theta else 0
            if streak >= PERSIST:
                n += 1
                streak = 0
        return n

    print(
        f"\n### Rule: max corner displacement > θ on {PERSIST} consecutive trusted previews"
    )
    print(
        "| θ | still: previews > θ | still: false fires (per hour) | "
        "miss shift 3/5/10 % | miss rot 3/5/10° | miss zoom ±5/±10/±20 % |"
    )
    print("|---|---|---|---|---|---|")
    still_seqs = []
    hours = 0.0
    for key, rs in still.items():
        if key[1] != "50" or key not in refs:
            continue
        ds = [None if d_truth(r, key) >= STILL_TRUTH else d_loc(r, key) for r in rs]
        still_seqs.append(ds)
        hours += len(rs) * 1.5 / 3600
    for theta in (2, 3, 4, 5, 6, 8):
        vs = vs_ref_all["50"]
        over = sum(d > theta for d in vs) / len(vs)
        false = sum(fires(ds, theta) for ds in still_seqs)

        def miss(kind: str, mags: tuple, theta: float = theta) -> str:
            out = []
            for m in mags:
                evs = events.get((kind, m), [])
                if kind == "zoom":
                    evs = evs + events.get((kind, -m), [])
                out.append(f"{sum(fires(e, theta) == 0 for e in evs)}/{len(evs)}")
            return " · ".join(out)

        print(
            f"| {theta} % | {over:.1%} | {false} ({false / hours:.1f}) "
            f"| {miss('shift', (3, 5, 10))} | {miss('rotate', (3, 5, 10))} "
            f"| {miss('zoom', (5, 10, 20))} |"
        )
    print(f"(still: {sum(len(s) for s in still_seqs)} previews, {hours * 60:.0f} min)")
    # Which metric separates still previews from moves of >= 5 % true displacement?
    big_loc = [d for row in rows.values() for t, d, _ in row["pairs"] if t >= 5]
    big_iou = [u for row in rows.values() for t, _, u in row["pairs"] if t >= 5]
    still_iou = [
        d_iou(r, key)
        for key, rs in still.items()
        if key in refs
        for r in rs
        if trusted(r[strategy]) and d_truth(r, key) < STILL_TRUTH
    ]
    print(
        f"AUC still vs true >= 5 %: max corner {auc(vs_ref_all['50'], big_loc):.3f}, "
        f"1-IoU {auc(still_iou, big_iou):.3f}"
    )


def cost(recs: list[dict], path: Path) -> None:
    ms = [r["ms"] for r in recs]
    fit = [r["fit_ms"] for r in recs]
    print(
        f"\n## Cost: fp32 GPU read median {np.median(ms):.0f} ms (p90 {pct(ms, 90):.0f}), "
        f"fit {np.median(fit):.1f} ms"
    )
    if path.exists():
        c = json.loads(path.read_text())
        print(
            f"int8 CPU read median {np.median(c['int8_cpu_ms']):.0f} ms, "
            f"fit {np.median(c['fit_ms']):.1f} ms"
        )


def main(out: Path) -> None:
    loc = out / "preview_locate.jsonl"
    if loc.exists():
        locate_table(load(loc))
    dr = out / "preview_drift.jsonl"
    if dr.exists():
        recs = load(dr)
        drift_tables(recs, "pass1")
        cost(recs, out / "preview_cost.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
