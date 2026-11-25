"""What locate found, as plain dataclasses, and how far a board moved.

The locate results are plain dataclasses here, because ``control`` imports nothing
from ``recognition`` (.claude/rules/code-organization.md); the package root converts
``locate``'s ``Grid`` and ``Proposal`` into them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from board_reader_core.control import CropRect

#: A point in fractions of the whole camera frame: x of its width, y of its height.
Point = tuple[float, float]
#: The board's outer corners a8, h8, h1, a1 (the order does not matter here).
Quad = tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class Located:
    """The board in a whole camera frame: its corners and its grid fit's R^2."""

    corners: Quad
    r2: float


@dataclass(frozen=True)
class FrameProposal[R]:
    """``autoframe`` found the board in a full frame: the crop to push (already
    snapped and valid), the board it was built from, and the last read when it saw
    the board large enough to publish (``Proposal.dominant``), else ``None``."""

    crop: CropRect
    board: Located
    read: R | None


def displacement(anchor: Located, board: Located, aspect: float) -> float:
    """How far ``board`` lies from ``anchor``, in widths of ``anchor``: the symmetric
    Hausdorff distance between the two corner sets in pixels, over the anchor's
    bounding-box width in pixels. ``aspect`` is the frame's width over its height."""

    def scaled(quad: Quad) -> list[Point]:
        return [(x * aspect, y) for x, y in quad]

    a, b = scaled(anchor.corners), scaled(board.corners)

    def farthest(src: list[Point], dst: list[Point]) -> float:
        return max(min(math.dist(p, q) for q in dst) for p in src)

    xs = [x for x, _ in a]
    width = max(max(xs) - min(xs), 1e-9)
    return max(farthest(a, b), farthest(b, a)) / width
