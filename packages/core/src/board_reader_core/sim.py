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


# --- Perspective simulation (the "angled broadcast view", DESIGN D8) -----------
#
# The real edge sees the board at an angle and must rectify it back to a canonical
# top-down crop (WP-A). To develop and test that offline without a camera, we take a
# canonical board image and warp it onto a larger canvas as a foreshortened
# quadrilateral. Corner order is **TL, TR, BR, BL** everywhere — for a canonical
# board (rank 8 top, file a left) that is a8, h8, h1, a1.

# Corner order shared by the warp helpers and the edge rectifier.
CORNER_ORDER: tuple[str, ...] = ("a8", "h8", "h1", "a1")


def board_corners(size: int) -> np.ndarray:
    """The 4 corners (TL,TR,BR,BL) of a canonical ``size``x``size`` board image."""
    s = float(size)
    return np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)


def warp_to_quad(
    board: np.ndarray,
    dst_quad: np.ndarray,
    *,
    canvas: tuple[int, int] | None = None,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Warp a canonical square ``board`` onto ``dst_quad`` (TL,TR,BR,BL) of a canvas.

    Returns a BGR image of size ``canvas`` (h, w); defaults to a square canvas just
    big enough to hold the quad. The area outside the quad is filled with
    ``background`` (black by default) so the board reads as a clear foreground
    quadrilateral — exactly what the detector keys on.
    """
    h, w = board.shape[:2]
    if h != w:
        raise ValueError("board image must be square")
    src = board_corners(w)
    dst = np.asarray(dst_quad, dtype=np.float32)
    if dst.shape != (4, 2):
        raise ValueError("dst_quad must be 4 (x, y) corners")
    if canvas is None:
        side = int(np.ceil(dst.max())) + 1
        canvas = (side, side)
    out_h, out_w = canvas
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        board,
        matrix,
        (out_w, out_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )


def simulate_angled_view(
    board: np.ndarray,
    *,
    tilt: float = 0.25,
    yaw: float = 0.06,
    margin: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Render ``board`` as a plausible angled view; return ``(image, dst_quad)``.

    Deterministic (no RNG). ``tilt`` foreshortens the far (top) edge, ``yaw`` skews
    left/right, and ``margin`` insets the board within the canvas so it sits fully
    inside the frame. ``dst_quad`` is the ground-truth board outline in image
    coordinates (TL,TR,BR,BL) — handy for tests that exercise manual corners.
    """
    side = board.shape[0]
    canvas = int(round(side * (1.0 + 2 * margin)))
    m = side * margin
    top = side * tilt / 2.0  # how much the far edge shrinks inward on each side
    sk = side * yaw  # horizontal skew
    dst = np.array(
        [
            [m + top + sk, m],                 # TL (a8, far)
            [m + side - top + sk, m],          # TR (h8, far)
            [m + side, m + side],              # BR (h1, near)
            [m, m + side],                     # BL (a1, near)
        ],
        dtype=np.float32,
    )
    img = warp_to_quad(board, dst, canvas=(canvas, canvas))
    return img, dst
