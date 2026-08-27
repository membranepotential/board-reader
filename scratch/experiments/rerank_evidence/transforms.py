"""Post-hoc re-rank transforms over the top-2 evidence, and their fitters.

A transform sees only what the wire carries: per square, ``(c1, p1)`` and
``(c2, p2)``. Its only move is *keep* or *swap*. Everything below is therefore a
decision rule on ``(c1, c2, p1, p2)``, implementable in a few lines at estimate
time.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import log

from board_reader_core.protocol import CLASSES


class Rule:
    """swap(c1, p1, c2, p2) -> bool."""

    def swap(self, c1: str, p1: float, c2: str, p2: float) -> bool:  # pragma: no cover
        raise NotImplementedError

    def apply(self, rows):
        """Yield the re-ranked top class for each row."""
        for r in rows:
            yield (r.c2 if self.swap(r.c1, r.p1, r.c2, r.p2) else r.c1)


@dataclass
class Identity(Rule):
    name: str = "identity"

    def swap(self, c1, p1, c2, p2) -> bool:
        return False


@dataclass
class ClassBias(Rule):
    """Global per-class log-bias: argmax over the top-2 of ``log p + b[c]``."""

    b: dict[str, float] = field(default_factory=lambda: dict.fromkeys(CLASSES, 0.0))
    name: str = "class-bias"

    def swap(self, c1, p1, c2, p2) -> bool:
        if p2 <= 0.0:
            return False
        return log(p2) + self.b.get(c2, 0.0) > log(p1) + self.b.get(c1, 0.0)


@dataclass
class PairMargin(Rule):
    """Per ordered pair ``(top1, top2)``: swap when ``p1 - p2 < tau``."""

    tau: dict[tuple[str, str], float] = field(default_factory=dict)
    name: str = "pair-margin"

    def swap(self, c1, p1, c2, p2) -> bool:
        t = self.tau.get((c1, c2))
        return t is not None and (p1 - p2) < t


# --------------------------------------------------------------------------- #
# Fitters
# --------------------------------------------------------------------------- #


def fit_pair_margin(rows, *, min_gain: int = 1, min_support: int = 1) -> PairMargin:
    """Per pair, the margin threshold maximising (fixed - broken).

    ``min_gain`` is the anti-overfit gate: a pair only enters the rule when its
    best threshold nets at least that many squares on the fitting data.
    """
    by: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        by[(r.c1, r.c2)].append(r)
    tau: dict[tuple[str, str], float] = {}
    for pair, rs in by.items():
        cand = sorted({r.margin for r in rs if r.truth == pair[1]})
        if len(cand) < min_support:
            continue
        best_gain, best_t = 0, None
        for t in cand:
            t = t + 1e-9
            fixed = sum(1 for r in rs if r.margin < t and r.truth == pair[1])
            broken = sum(1 for r in rs if r.margin < t and r.truth == pair[0])
            if fixed - broken > best_gain:
                best_gain, best_t = fixed - broken, t
        if best_t is not None and best_gain >= min_gain:
            tau[pair] = best_t
    return PairMargin(tau=tau)


def fit_class_bias(rows, *, rounds: int = 40, grid: int = 81) -> ClassBias:
    """Coordinate ascent on the 13 log-biases, maximising per-square accuracy.

    The objective is piecewise constant, so each coordinate is swept over a dense
    grid rather than differentiated. Ties keep the smaller |b| (i.e. identity).
    """
    b = dict.fromkeys(CLASSES, 0.0)
    pre = [(r.c1, r.c2, log(r.p1) if r.p1 > 0 else -60.0,
            log(r.p2) if r.p2 > 0 else -60.0, r.truth) for r in rows]

    def score(bb) -> int:
        n = 0
        for c1, c2, l1, l2, t in pre:
            pick = c2 if l2 + bb.get(c2, 0.0) > l1 + bb.get(c1, 0.0) else c1
            n += pick == t
        return n

    best = score(b)
    values = [(-3.0 + 6.0 * i / (grid - 1)) for i in range(grid)]
    for _ in range(rounds):
        improved = False
        for c in CLASSES:
            keep, cur = b[c], best
            for v in values:
                b[c] = v
                s = score(b)
                if s > cur or (s == cur and abs(v) < abs(keep)):
                    cur, keep = s, v
            b[c] = keep
            if cur > best:
                best, improved = cur, True
        if not improved:
            break
    return ClassBias(b=b)


def fit_stable_pair_margin(
    rows_by_take: dict[str, list], *, taus, max_broken: int = 0, min_fix: int = 1
):
    """Fit a PairSet-style rule using only *transferable* evidence.

    A pair enters the rule at threshold ``tau`` only if, in **every** fitting
    take independently, it fixes at least ``min_fix`` squares and breaks at most
    ``max_broken``. Requiring the win to repeat across takes — rather than to be
    large once — is the whole anti-overfit device here; with ~190 errors total,
    a pooled net gain is far too easy to manufacture.
    """
    pairs: dict[tuple[str, str], float] = {}
    cand = {(r.c1, r.c2) for rs in rows_by_take.values() for r in rs
            if r.wrong and r.truth == r.c2}
    for pair in cand:
        best = None
        for tau in taus:
            ok = True
            gain = 0
            for rs in rows_by_take.values():
                sel = [r for r in rs if (r.c1, r.c2) == pair and r.margin < tau]
                f = sum(1 for r in sel if r.truth == pair[1])
                b = sum(1 for r in sel if r.truth == pair[0])
                if f < min_fix or b > max_broken:
                    ok = False
                    break
                gain += f - b
            if ok and (best is None or gain > best[1]):
                best = (tau, gain)
        if best is not None:
            pairs[pair] = best[0]
    return pairs


def fit_stable_demote(rows_by_take: dict[str, list], *, taus, max_broken: int = 0,
                      min_fix: int = 1) -> dict[str, float]:
    """Same transferability gate, one level coarser: keyed on *top-1 class only*.

    The runner-up is not part of the condition, so the rule reads "class ``c`` is
    an attractor — when it wins narrowly, believe the runner-up instead".
    """
    out: dict[str, float] = {}
    classes = {r.c1 for rs in rows_by_take.values() for r in rs
               if r.wrong and r.truth == r.c2}
    for c in classes:
        best = None
        for tau in taus:
            ok, gain = True, 0
            for rs in rows_by_take.values():
                sel = [r for r in rs if r.c1 == c and r.margin < tau]
                f = sum(1 for r in sel if r.truth == r.c2)
                b = sum(1 for r in sel if r.truth == c)
                if f < min_fix or b > max_broken:
                    ok = False
                    break
                gain += f - b
            if ok and (best is None or gain > best[1]):
                best = (tau, gain)
        if best is not None:
            out[c] = best[0]
    return out
