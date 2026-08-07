"""Sanity checks for an operator-supplied board quad (DESIGN §4.5, WP-A).

``--corners`` is four points the operator taps or types **once**, and everything
downstream trusts them completely: on the ``--forward-original`` path they are
stamped verbatim on every wire frame and *are* the streamed placement's
orientation; on the rectify path they are the homography every crop is warped
through. Nothing later in the pipe can tell a good quad from a typo'd one — a
transposed digit or a swapped pair of points produces confidently-wrong geometry
for as long as the run lasts, with no error anywhere. On real footage that is
hours of it, discovered afterwards.

So the quad is checked at the boundary, once, before any frame moves. These are
pure predicates over four points — no CLI, no socket, no frame — except
:func:`check_quad_in_frame`, which also needs the source's resolution and so runs
on the first frame instead: a video file, an image directory and a camera all
only reveal their frame size once a frame has actually been read.

Convexity is the load-bearing check. The four points are a *walk around the
board* in a8, h8, h1, a1 order, so an ordering slip — the classic one is entering
the corners as a8,h8,a1,h1 — folds the quad into a bowtie, which is exactly what
a sign-consistent turn test refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from board_reader_core.boarddetect import LOW_ANGLE_ASPECT, quad_aspect

# Two corners closer than this are one tap entered twice, not a quad.
_MIN_SEPARATION_PX = 1.0

# A playing surface smaller than this cannot be cut into 64 usable squares — it
# would put each square under 8x8 px. Anything this small is a typo, not a board.
_MIN_AREA_PX = 64.0 * 64.0

# Enclosed area as a fraction of the quad's own longest span, squared. A square
# scores 0.5 and even a brutally foreshortened board (aspect 0.2) still scores
# ~0.19, so below this the points are near-collinear: a sliver, not a board.
_MIN_FILL = 0.05

# How far outside the source frame a corner may sit before we call it wrong.
# Deliberately not zero: a board whose near corner is cropped by the frame edge is
# a real field case, and extrapolating that corner slightly off-frame is the right
# thing for the operator to do. The failures this catches — a quad measured on a
# different resolution, a transposed digit — overshoot by far more than a tenth.
_FRAME_MARGIN = 0.10

QuadDefectKind = Literal[
    "not_a_quad",
    "coincident_points",
    "self_intersecting",
    "not_convex",
    "degenerate",
    "out_of_frame",
]


@dataclass(frozen=True)
class QuadDefect:
    """Why a set of points cannot be a board outline.

    ``message`` is operator-facing and reads as a predicate of the quad ("is not
    convex — ...", "lies outside the 1280x720 source frame: ..."), so a caller can
    prefix it with whatever named the quad (``--corners``, "the board quad").
    """

    kind: QuadDefectKind
    message: str


class OperatorQuadError(ValueError):
    """An operator-declared quad that cannot be right.

    Raised where the defect is only detectable mid-stream (the in-frame check on
    the first frame); the startup checks report a :class:`QuadDefect` instead and
    let the CLI turn it into a clean exit.
    """


def _turn_signs(quad: np.ndarray) -> np.ndarray:
    """Cross product at each vertex: the direction the walk turns there."""
    edges = np.roll(quad, -1, axis=0) - quad
    nxt = np.roll(edges, -1, axis=0)
    return edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0]


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Do segments ``ab`` and ``cd`` properly cross? (Touching does not count.)"""
    d1, d2 = _orient(c, d, a), _orient(c, d, b)
    d3, d4 = _orient(a, b, c), _orient(a, b, d)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def _area(quad: np.ndarray) -> float:
    """Shoelace area, unsigned — winding is the caller's problem, not ours."""
    x, y = quad[:, 0], quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _point(p: np.ndarray) -> str:
    return f"({p[0]:g}, {p[1]:g})"


def check_quad_shape(pts: np.ndarray) -> QuadDefect | None:
    """Everything about a board quad that is knowable without a frame.

    Returns the first defect found, or ``None`` if the points could plausibly be a
    board outline. The order is chosen so each defect gets the message that names
    its actual cause: a repeated point makes every later predicate meaningless; a
    bowtie's shoelace area nearly cancels, so it must be caught before the area
    test calls it degenerate; and four collinear points fail as a sliver rather
    than as the "not convex" they technically also are.
    """
    flat = np.asarray(pts, dtype=np.float64).ravel()
    if flat.size != 8:
        return QuadDefect(
            "not_a_quad", f"needs exactly 4 (x, y) points; got {flat.size} numbers"
        )
    if not np.isfinite(flat).all():
        return QuadDefect("not_a_quad", "contains a value that is not a finite number")
    quad = flat.reshape(4, 2)

    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)):
        if float(np.linalg.norm(quad[i] - quad[j])) < _MIN_SEPARATION_PX:
            return QuadDefect(
                "coincident_points",
                f"repeats a point: {i} and {j} are both {_point(quad[i])}",
            )

    if _segments_cross(quad[0], quad[1], quad[2], quad[3]) or _segments_cross(
        quad[1], quad[2], quad[3], quad[0]
    ):
        return QuadDefect(
            "self_intersecting",
            "crosses itself — the four points walk around the board in a8,h8,h1,a1 "
            "order; a bowtie usually means the last two were swapped",
        )

    area = _area(quad)
    if area < _MIN_AREA_PX:
        return QuadDefect(
            "degenerate",
            f"encloses only {area:.0f} px^2 — far too small to be a board "
            f"(64 squares need at least {_MIN_AREA_PX:.0f} px^2)",
        )
    span = float(
        max(
            np.linalg.norm(quad[i] - quad[j])
            for i in range(4)
            for j in range(i + 1, 4)
        )
    )
    if area < _MIN_FILL * span * span:
        return QuadDefect(
            "degenerate",
            f"is a sliver: {area:.0f} px^2 across a {span:.0f} px span, i.e. the "
            "points are nearly collinear",
        )

    signs = _turn_signs(quad)
    if not (bool((signs > 0).all()) or bool((signs < 0).all())):
        return QuadDefect(
            "not_convex",
            "is not convex — a board outline turns the same way at all four "
            "corners, so one of the coordinates is wrong",
        )
    return None


def check_quad_in_frame(
    pts: np.ndarray, *, width: int, height: int
) -> QuadDefect | None:
    """The one check that needs a frame: are the corners actually on the picture?

    Not knowable at startup — see the module docstring — so callers run this on the
    first frame the source yields, before anything goes on the wire.
    """
    quad = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    mx, my = _FRAME_MARGIN * width, _FRAME_MARGIN * height
    outside = [
        i
        for i, (x, y) in enumerate(quad)
        if not (-mx <= x <= width + mx and -my <= y <= height + my)
    ]
    if not outside:
        return None
    where = ", ".join(f"{i}={_point(quad[i])}" for i in outside)
    return QuadDefect(
        "out_of_frame",
        f"lies outside the {width}x{height} source frame: {where} — corners "
        "measured on a different resolution are the usual cause",
    )


def low_angle_warning(pts: np.ndarray) -> str | None:
    """Camera-too-flat warning for a quad, or ``None`` when the angle is usable.

    Warn, never block: the aspect is a proxy for camera elevation, the operator may
    know exactly why theirs is unusual, and blocking a run on a proxy is worse than
    an honest heads-up. Same call the gateway makes on a posted photo, which tags
    the estimate ``very_low_angle`` rather than refusing it. The point of saying it
    here is *timing* — at startup, not after a bad run.
    """
    aspect = quad_aspect(np.asarray(pts, dtype=np.float32).reshape(4, 2))
    if aspect >= LOW_ANGLE_ASPECT:
        return None
    return (
        f"the corners project to a board aspect of {aspect:.2f}, below "
        f"{LOW_ANGLE_ASPECT:.2f}: the camera is near board level, where pieces "
        "occlude the rank behind them and recognition collapses. Raise the camera "
        "if you can (docs/operating-envelope.md)."
    )
