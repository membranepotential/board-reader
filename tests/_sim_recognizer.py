"""A test-side recognizer that reads :func:`board_reader_core.sim.render_board` renders.

The e2e tests need a recognizer that reads *pixels*, so that a position travelling
through the pipe is a fact about the frames rather than a stub's authored answer.
ChessQ cannot be it: it reads 0 of 14 sim renders perfectly (plan step 5, D3), and
it needs weights a test run does not have. The renders are exact, though — a flat
top-down board, each square one fill colour with a centred glyph — so the square
can be matched against a template of every class, rendered by the same function at
the same size. That is exact on an unperturbed render and nothing more: it is the
pipe that is on trial here, never recognition.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from board_reader_core.protocol import CLASSES, GRID, NUM_CLASSES
from board_reader_core.sim import render_board

#: Probability on the matched class; the rest is spread evenly over the others.
CONFIDENCE = 0.95


@cache
def _templates(square_px: int) -> np.ndarray:
    """``(2, 13, s, s, 3)``: each class on a light (0) and a dark (1) square.

    a8 is light and b8 dark in :func:`render_board`, so one render per class with
    that class on both squares gives both templates.
    """
    out = np.empty((2, NUM_CLASSES, square_px, square_px, 3), dtype=np.float32)
    for k, name in enumerate(CLASSES):
        rank8 = "8" if name == "empty" else f"{name}{name}6"
        board = render_board(f"{rank8}/8/8/8/8/8/8/8", square_px=square_px)
        for parity in (0, 1):
            x0 = parity * square_px
            out[parity, k] = board[:square_px, x0 : x0 + square_px]
    return out


class SimRecognizer:
    """Reads a whole-frame sim render into its chess-frame ``(8, 8, 13)`` distribution."""

    name = "sim"
    model_path: str | None = None

    def read(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h != w or h % GRID:
            raise ValueError(f"not a sim render: {w}x{h} is not a square board of 8 squares")
        s = h // GRID
        templates = _templates(s)
        dist = np.full(
            (GRID, GRID, NUM_CLASSES), (1 - CONFIDENCE) / (NUM_CLASSES - 1), np.float32
        )
        for r in range(GRID):
            for c in range(GRID):
                cell = image[r * s : (r + 1) * s, c * s : (c + 1) * s].astype(np.float32)
                cost = ((templates[(r + c) % 2] - cell) ** 2).sum(axis=(1, 2, 3))
                dist[r, c, int(cost.argmin())] = CONFIDENCE
        return dist
