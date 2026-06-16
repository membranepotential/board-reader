"""64-square cropping from a rectified board (DESIGN.md WP-B1).

The edge delivers a canonical top-down crop (``out_size`` x ``out_size``, the board
filling the frame). This splits it into 64 square images in the **same order as the
L3 protocol grid** — row 0 = rank 8, column 0 = file a — so the recognizer's output
lines up with :data:`board_reader_core.protocol` without any re-indexing.

A center ``inset`` keeps only the middle of each square, suppressing the grid lines
and the neighbouring-square spill that would otherwise pollute a piece embedding.
"""

from __future__ import annotations

import cv2
import numpy as np

from board_reader_core.protocol import GRID


def crop_squares(
    board: np.ndarray, *, inset: float = 0.0, size: int | None = None
) -> list[np.ndarray]:
    """Split a rectified ``board`` into 64 square crops (rank 8→1, file a→h).

    Parameters
    ----------
    board:
        Canonical top-down BGR crop; should be (near) square.
    inset:
        Fraction of each square trimmed off every edge (0 = whole square, 0.1 = keep
        the central 80%). Useful to drop grid lines / spill before embedding.
    size:
        If given, each crop is resized to ``size`` x ``size`` (square embedders want a
        fixed input). If ``None``, crops keep their native pixel size.

    Returns a list of 64 images in row-major order: index ``r*8 + c`` is rank ``8-r``,
    file ``chr(ord('a')+c)``.
    """
    if not 0.0 <= inset < 0.5:
        raise ValueError("inset must be in [0, 0.5)")
    h, w = board.shape[:2]
    sh, sw = h / GRID, w / GRID
    crops: list[np.ndarray] = []
    for r in range(GRID):
        for c in range(GRID):
            y0, y1 = r * sh, (r + 1) * sh
            x0, x1 = c * sw, (c + 1) * sw
            dy, dx = (y1 - y0) * inset, (x1 - x0) * inset
            sq = board[
                int(round(y0 + dy)) : int(round(y1 - dy)),
                int(round(x0 + dx)) : int(round(x1 - dx)),
            ]
            if size is not None:
                sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
            crops.append(sq)
    return crops
