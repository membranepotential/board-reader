"""Leave-one-take-out evaluation of a re-rank rule.

Metrics, per take and per stream:
  * per-square accuracy and perfect-board rate, raw and re-ranked;
  * squares FIXED vs squares BROKEN, counted separately;
  * the effect on the confidence of squares that were already right.

The RAW numbers are exact: the digest's argmax reproduces the recorded raw board
on 98/98 events. The PRIOR numbers go through a top-2 reconstruction of the
distribution and are therefore *paired* estimates — see report.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from board_reader_core.protocol import (
    CLASS_INDEX,
    GRID,
    NUM_CLASSES,
    encode_placement,
    parse_placement,
)
from board_reader_inference.prior import decode
from board_reader_inference.trial import square_errors

import dists
import residuals
import takes
from transforms import Identity, Rule


@dataclass
class StreamDelta:
    squares: int = 0
    correct_base: int = 0
    correct_new: int = 0
    fixed: int = 0
    broken: int = 0
    events: int = 0
    perfect_base: int = 0
    perfect_new: int = 0

    def row(self) -> str:
        a0 = self.correct_base / self.squares
        a1 = self.correct_new / self.squares
        p0 = self.perfect_base / self.events
        p1 = self.perfect_new / self.events
        return (f"{a0*100:6.2f} -> {a1*100:6.2f}  ({(a1-a0)*100:+5.2f})   "
                f"{p0*100:5.1f} -> {p1*100:5.1f}  ({(p1-p0)*100:+5.1f})   "
                f"fixed {self.fixed:3d}  broken {self.broken:3d}  net {self.fixed-self.broken:+4d}")


def _reranked_dist(digest, rule: Rule) -> np.ndarray:
    """Reconstruction with the rule applied as a value-swap of the top two."""
    arr = dists.dist_from_digest(digest).reshape(-1, NUM_CLASSES)
    for i, e in enumerate(digest):
        c1, p1 = e[0]
        c2, p2 = e[1] if len(e) > 1 else ("empty", 0.0)
        if rule.swap(c1, p1, c2, p2):
            arr[i, CLASS_INDEX[c1]] = p2
            arr[i, CLASS_INDEX[c2]] = p1
    return arr.reshape(GRID, GRID, NUM_CLASSES)


def evaluate(name: str, rule: Rule, *, with_prior: bool = True) -> dict[str, StreamDelta]:
    rep = takes.load(name)
    rows = residuals.squares(name)
    by_event: dict[int, list] = {}
    for r in rows:
        by_event.setdefault(r.event, []).append(r)

    raw = StreamDelta()
    prior = StreamDelta()
    for i, read in enumerate(rep.reads):
        truth = rep.truth.segments[read.segment].placement
        ev = by_event[i]
        # --- raw stream (exact) ---
        raw.events += 1
        raw.squares += 64
        ok_b = ok_n = 0
        for r in ev:
            new = r.c2 if rule.swap(r.c1, r.p1, r.c2, r.p2) else r.c1
            b, n = (r.c1 == r.truth), (new == r.truth)
            ok_b += b
            ok_n += n
            raw.fixed += (not b) and n
            raw.broken += b and (not n)
        raw.correct_base += ok_b
        raw.correct_new += ok_n
        raw.perfect_base += ok_b == 64
        raw.perfect_new += ok_n == 64
        if not with_prior:
            continue
        # --- prior stream (reconstruction, paired) ---
        base = decode(dists.dist_from_digest(read.event.evidence)).placement
        new_b = decode(_reranked_dist(read.event.evidence, rule)).placement
        tb = [c for row in parse_placement(truth) for c in row]
        gb = [c for row in parse_placement(base) for c in row]
        gn = [c for row in parse_placement(new_b) for c in row]
        prior.events += 1
        prior.squares += 64
        for t, x, y in zip(tb, gb, gn):
            prior.correct_base += x == t
            prior.correct_new += y == t
            prior.fixed += (x != t) and (y == t)
            prior.broken += (x == t) and (y != t)
        prior.perfect_base += base == truth
        prior.perfect_new += new_b == truth
    return {"raw": raw, "prior": prior}


def confidence_effect(name: str, rule: Rule) -> tuple[int, float, float, int]:
    """(n correct squares touched, mean conf before/after on already-right squares,
    n correct squares whose reported top-1 prob dropped)."""
    rows = residuals.squares(name)
    right = [r for r in rows if r.c1 == r.truth]
    swapped = [r for r in right if rule.swap(r.c1, r.p1, r.c2, r.p2)]
    before = float(np.mean([r.p1 for r in right])) if right else 0.0
    after = float(np.mean([(r.p2 if rule.swap(r.c1, r.p1, r.c2, r.p2) else r.p1)
                           for r in right])) if right else 0.0
    return len(swapped), before, after, len(right)
