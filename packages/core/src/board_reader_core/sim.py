"""Synthetic board rendering for the sim harness (DESIGN.md §6).

A dependency-free way to produce board frames without a camera or a downloaded
clip: render a top-down board image from a placement string. Used by the edge
synthetic feed and by tests. This is sim/test scaffolding, intentionally simple —
not the real capture path. Lives in core so both the edge and tests can use it.
"""

from __future__ import annotations

import cv2
import numpy as np

from board_reader_core.protocol import START_PLACEMENT, parse_placement

_LIGHT_SQ = (181, 217, 240)  # BGR, a "light" board square
_DARK_SQ = (85, 138, 181)


def render_board(
    placement: str = START_PLACEMENT,
    *,
    square_px: int = 48,
) -> np.ndarray:
    """Render a top-down BGR board image from a placement (rank 8 top, file a left)."""
    grid = parse_placement(placement)
    n = 8 * square_px
    img = np.zeros((n, n, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in range(8):
        for c in range(8):
            y0, x0 = r * square_px, c * square_px
            color = _LIGHT_SQ if (r + c) % 2 == 0 else _DARK_SQ
            img[y0 : y0 + square_px, x0 : x0 + square_px] = color
            cell = grid[r][c]
            if cell == "empty":
                continue
            glyph = cell.upper()
            glyph_color = (245, 245, 245) if cell.isupper() else (20, 20, 20)
            scale = square_px / 48.0
            (tw, th), _ = cv2.getTextSize(glyph, font, scale, 2)
            cv2.putText(
                img,
                glyph,
                (x0 + (square_px - tw) // 2, y0 + (square_px + th) // 2),
                font,
                scale,
                glyph_color,
                2,
                cv2.LINE_AA,
            )
    return img


def render_frames(placements: list[str], *, square_px: int = 48) -> list[np.ndarray]:
    """Render a sequence of placements into board images."""
    return [render_board(p, square_px=square_px) for p in placements]
