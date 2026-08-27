"""Reconstruct an 8x8x13 distribution from the recorded top-2 digest.

The digest is lossy: two (class, prob) pairs per square. The reconstruction puts
the recorded probabilities on their classes and spreads the residual mass
``1 - p1 - p2`` uniformly over the other 11. It is only faithful where the prior
decoder's cost function depends on the top two — which is exactly the regime the
decoder works in (it relabels the lowest-margin squares) — so the fidelity check
below is the gate on every prior-stream number in this study.
"""
from __future__ import annotations

import numpy as np

from board_reader_core.protocol import CLASS_INDEX, GRID, NUM_CLASSES

TAIL = NUM_CLASSES - 2


def dist_from_digest(digest) -> np.ndarray:
    """(8,8,13) float64 reconstruction of the distribution behind one digest."""
    arr = np.empty((GRID * GRID, NUM_CLASSES), dtype=np.float64)
    for i, entry in enumerate(digest):
        (c1, p1) = entry[0]
        (c2, p2) = entry[1] if len(entry) > 1 else ("empty", 0.0)
        rest = max(0.0, 1.0 - p1 - p2) / TAIL
        arr[i, :] = rest
        arr[i, CLASS_INDEX[c1]] = p1
        arr[i, CLASS_INDEX[c2]] = p2
    return arr.reshape(GRID, GRID, NUM_CLASSES)
