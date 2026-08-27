"""Verify the report's implementation snippet reproduces the measured rule exactly.

Runs the proposed numpy transform over every reconstructed distribution and checks,
square by square, that the resulting argmax equals the class the digest-level rule
(``DemoteTop1('b', 0.20)``) picks.
"""
from __future__ import annotations

import numpy as np

from board_reader_core.protocol import CLASS_INDEX, CLASSES, NUM_CLASSES

import dists
import residuals
import takes
from candidates import DemoteTop1
from board_reader_inference.trial import SQUARE_INDEX

ATTRACTOR = "b"
ATTRACTOR_MARGIN = 0.20


def demote_attractor(square_dist: np.ndarray) -> np.ndarray:
    """Swap the top two classes on squares the attractor only narrowly won."""
    flat = square_dist.reshape(-1, NUM_CLASSES).copy()
    order = np.argsort(-flat, axis=-1)[:, :2]
    i1, i2 = order[:, 0], order[:, 1]
    rows = np.arange(flat.shape[0])
    p1, p2 = flat[rows, i1], flat[rows, i2]
    fire = (i1 == CLASS_INDEX[ATTRACTOR]) & (p1 - p2 < ATTRACTOR_MARGIN)
    r = rows[fire]
    flat[r, i1[fire]], flat[r, i2[fire]] = p2[fire], p1[fire]
    return flat.reshape(square_dist.shape)


rule = DemoteTop1(ATTRACTOR, ATTRACTOR_MARGIN)
order = sorted(SQUARE_INDEX, key=lambda s: SQUARE_INDEX[s])
mismatch = fired = 0
for name in takes.ALL:
    rep = takes.load(name)
    rows = {(r.event, r.square): r for r in residuals.squares(name)}
    for i, read in enumerate(rep.reads):
        out = demote_attractor(dists.dist_from_digest(read.event.evidence))
        got = [CLASSES[j] for j in out.reshape(-1, NUM_CLASSES).argmax(-1)]
        for j, sq in enumerate(order):
            r = rows[(i, sq)]
            want = r.c2 if rule.swap(r.c1, r.p1, r.c2, r.p2) else r.c1
            fired += want != r.c1
            mismatch += got[j] != want
print(f"{fired} firings, {mismatch} mismatches between the shipped snippet and the "
      f"digest-level rule — the snippet is exactly what §5-§8 measured."
      if mismatch == 0 else f"MISMATCH: {mismatch}")
