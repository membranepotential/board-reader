"""Where the board is in a frame, from ChessQ's own square attention (Auto-frame).

ChessQ's decoder has one query per square. Three of its cross-attention heads
(L1H0, L2H0, L3H0) put each query's attention on that square, so the 64 attention
centroids lie on the board's 8x8 grid. A homography from the grid's square centres
to those centroids gives the board's four corners. The served model files output
these heads as ``square_attention`` (``make models``,
``tools/chessq_export/attention.py``), so every ChessQ read carries its grid at no
extra model call (:attr:`~.recognizer.Recognition.grid`).
docs/reports/2026-11-25-autocrop-probe.md measured all of this.

Coordinates: a :class:`Grid` is in **fractions of the image it was read from**
(x of its width, y of its height), the same convention as
:class:`~board_reader_core.control.CropRect`. :meth:`Grid.within` moves a grid read
inside a region of a frame into that frame's fractions. For the crop the phone
sends, that is ``grid.within(crop.x, crop.y, crop.w, crop.h)``. Corners are in chess
order a8, h8, h1, a1: ChessQ knows which corner is which.

API, for the control loop that frames a camera:

* :func:`square_centroids` and :func:`fit_grid`: the pure grid fit. ``fit_grid``
  returns ``None`` only for a fit that has no board shape at all; a grid that is
  merely bad has a low ``r2``. Trust a grid only when :attr:`Grid.trusted` holds
  (``r2 >= MIN_R2``): in the probe every bad final grid had R^2 <= 0.89.
* :func:`autoframe` ``(read, frame) -> Proposal | NoBoard``: propose the crop for a
  whole full-resolution frame. It calls ``read`` (a recognizer's ``read``) 2 to 8
  times: the whole frame, then its rough board box grown by :data:`GROW` on every
  side. When that grid is untrusted, it reads five tiles and refines the best one.
  :class:`Proposal` carries the crop (:func:`proposed_crop`, snapped like the
  /control editor), the grid in the frame's fractions, the number
  of reads, and the last read's :class:`~.recognizer.Recognition`. The caller may
  publish that read as a position read when :attr:`Proposal.dominant` holds (the
  board fills at least :data:`BOARD_DOMINANT` of that read's width,
  docs/reports/2026-11-24-chessq-framing.md).
* :func:`locate_preview` ``(read, preview) -> Grid | None``: the board in the
  phone's whole-frame preview (480 px long side, JPEG q 0.6), in the full frame's
  fractions (the preview is the whole frame, downscaled), or ``None`` when no
  trusted grid fits. One read, no coarse to fine: for comparing the board's
  position with the current crop (camera moved?). How well a single pass works at
  480 px is measured separately; this is the place to change if it does not.
* :func:`proposed_crop` ``(grid, width, height) -> CropRect``: the crop Auto-frame
  proposes for a grid in the fractions of a ``width`` x ``height`` frame: the
  board's box plus :data:`MARGIN`, and on top as much more as a king on the far
  rank needs (:data:`KING_SQUARES`). Use it to re-frame from a grid you already
  have.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from board_reader_core.control import MIN_CROP_SIDE, CropRect

if TYPE_CHECKING:
    # Runtime would be a cycle: recognizer.Recognition holds a Grid.
    from board_reader_inference.recognition.recognizer import Recognition

#: A full-resolution BGR frame in, its read out: a recognizer's ``read``.
ImageRead = Callable[[np.ndarray], "Recognition"]

#: The smallest R^2 of a grid fit that is trusted (the probe's failure signal).
MIN_R2 = 0.90

#: The proposed crop: the board's box grown by this share of its own width and
#: height on every side (the probe measured 10 %; take 10 % or more).
MARGIN = 0.10

#: A king's height in squares, for the crop's top edge. The grid is the board's
#: playing surface, so a far-rank piece rises above it toward the top of the
#: image. A FIDE tournament set has a 95 mm king on 50-65 mm squares (1.5-1.9);
#: the probe's outdoor set needs about 1.75, so take the top of the range. Up to
#: 1.9 it cost no accuracy (docs/reports/2026-11-25-autocrop-probe.md, top margin).
KING_SQUARES = 1.9

#: Coarse to fine: the rough box of the first read, grown by this share on every
#: side, is read again. It puts the board at about half of that read's width.
GROW = 0.50

#: A read whose board spans at least this share of its width reads as well as a
#: board-dominant crop (docs/reports/2026-11-24-chessq-framing.md).
BOARD_DOMINANT = 0.50

#: The fallback when coarse to fine finds no trusted grid: four 60 % corner tiles
#: and the 60 % centre, as ``(x0, y0, x1, y1)`` fractions of the frame. A board at
#: 15 % of the frame is about 25 % of a tile.
TILES = (
    (0.0, 0.0, 0.6, 0.6),
    (0.4, 0.0, 1.0, 0.6),
    (0.0, 0.4, 0.6, 1.0),
    (0.4, 0.4, 1.0, 1.0),
    (0.2, 0.2, 0.8, 0.8),
)

#: A square centre counts as an inlier within this many squares of the fit.
INLIER_SQUARES = 0.35

#: Half the side, in patches, of the window around a query's attention peak that
#: its centroid is taken over (5x5 patches).
WINDOW = 2

#: The smallest region, in pixels a side, worth a read.
MIN_READ_PX = 32

#: Side of a model patch in input pixels; the input is ``PATCH * patches`` square.
PATCH = 14

#: Square ``i`` (FEN order: a8..h8, a7..h1) has its centre here in board
#: coordinates, which put a8's outer corner at (0, 0) and h1's at (8, 8).
SQUARE_CENTRES = np.array([[i % 8 + 0.5, i // 8 + 0.5] for i in range(64)], np.float32)
#: The board's outer corners in board coordinates: a8, h8, h1, a1.
BOARD_CORNERS = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32)

#: A crop edge is a whole number of these units of the frame (as /control snaps).
CROP_UNITS = 10_000


@dataclass(frozen=True, eq=False)
class Grid:
    """ChessQ's board in one image: its 8x8 grid fitted to the square centroids.

    ``corners`` is ``(4, 2)`` float, a8 h8 h1 a1, in fractions of the image (they
    may lie outside ``[0, 1]`` when the board leaves the picture). ``r2`` is the
    fit's R^2 over all 64 centroids, in the image's pixels; ``inliers`` counts the
    centroids within :data:`INLIER_SQUARES` of the fit.
    """

    corners: np.ndarray
    r2: float
    inliers: int

    @property
    def trusted(self) -> bool:
        """Whether the fit is good enough to act on (``r2 >= MIN_R2``)."""
        return self.r2 >= MIN_R2

    @property
    def box(self) -> tuple[float, float, float, float]:
        """The corners' bounding box ``(x0, y0, x1, y1)``, in the same fractions."""
        x0, y0 = self.corners.min(axis=0)
        x1, y1 = self.corners.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    def within(self, x: float, y: float, w: float, h: float) -> Grid:
        """This grid, read from the region ``(x, y, w, h)`` of a frame (fractions of
        it), in that frame's fractions. The fit's ``r2`` and ``inliers`` stay."""
        corners = self.corners * np.array([w, h], np.float32) + np.array([x, y], np.float32)
        return Grid(corners, self.r2, self.inliers)


def square_centroids(attention: np.ndarray, width: int, height: int) -> np.ndarray:
    """The 64 square centroids in a ``width`` x ``height`` image's pixels, ``(64, 2)``.

    ``attention`` is ``(heads, 64, patches, patches)``, one map per head and square
    query over the model's patch grid. Per head and query the centroid is taken
    in the :data:`WINDOW` around the map's peak; the heads' centroids are averaged.
    The model squashes the image to a square, so the patch grid stretches back
    onto the image's width and height separately. A query with no attention in its
    window has a NaN centroid.
    """
    heads, queries, side, _ = attention.shape
    n = 2 * WINDOW + 1
    peak = attention.reshape(heads, queries, -1).argmax(axis=-1)
    py, px = np.divmod(peak, side)
    offsets = np.arange(n) - WINDOW
    # Window rows/columns in patch units; outside the map they read zero weight.
    rows = py[..., None] + offsets  # (heads, queries, n)
    cols = px[..., None] + offsets
    padded = np.pad(attention, ((0, 0), (0, 0), (WINDOW, WINDOW), (WINDOW, WINDOW)))
    win = padded[
        np.arange(heads)[:, None, None, None],
        np.arange(queries)[None, :, None, None],
        (rows + WINDOW)[..., :, None],
        (cols + WINDOW)[..., None, :],
    ].astype(np.float64)  # (heads, queries, n, n)
    mass = win.sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = (win.sum(axis=-2) * cols).sum(axis=-1) / mass
        cy = (win.sum(axis=-1) * rows).sum(axis=-1) / mass
    patches = np.stack([cx.mean(axis=0), cy.mean(axis=0)], axis=-1)  # (queries, 2)
    # A patch's centre is at (index + 0.5) * PATCH in the square input.
    scale = np.array([width, height], np.float64) / (side * PATCH)
    return ((patches + 0.5) * PATCH * scale).astype(np.float32)


def fit_grid(centroids: np.ndarray, width: int, height: int) -> Grid | None:
    """The 8x8 grid through ``centroids`` (``(64, 2)``, pixels of a ``width`` x
    ``height`` image, FEN order), or ``None`` when no board shape fits at all.

    The homography from square centres to centroids is fitted with LMedS, which
    ignores up to half the squares (a hand, a square outside the frame); R^2 is
    over all 64, so such squares still lower it. ``None`` means a non-finite
    centroid, no homography, or corners that are not a convex quadrilateral.
    """
    pts = np.asarray(centroids, np.float32)
    if pts.shape != (64, 2) or not np.isfinite(pts).all():
        return None
    homography, _ = cv2.findHomography(SQUARE_CENTRES, pts, cv2.LMEDS)
    if homography is None:
        return None
    proj = cv2.perspectiveTransform(SQUARE_CENTRES[:, None], homography)[:, 0]
    quad = cv2.perspectiveTransform(BOARD_CORNERS[:, None], homography)[:, 0]
    if not (np.isfinite(quad).all() and cv2.isContourConvex(quad)):
        return None
    square = float(np.sqrt(abs(cv2.contourArea(quad)))) / 8
    residual = np.linalg.norm(proj - pts, axis=1)
    ss_tot = float(((pts - pts.mean(axis=0)) ** 2).sum())
    if ss_tot <= 0 or square <= 0:
        return None
    r2 = 1 - float((residual**2).sum()) / ss_tot
    inliers = int((residual < INLIER_SQUARES * square).sum())
    corners = quad / np.array([width, height], np.float32)
    return Grid(corners.astype(np.float32), r2, inliers)


def proposed_crop(grid: Grid, width: int, height: int) -> CropRect:
    """The crop for ``grid`` (fractions of a ``width`` x ``height`` frame): its box
    grown by :data:`MARGIN` on every side, and its top raised further to hold a
    king on any square (:func:`_king_top`), inside the frame, edges snapped to
    whole :data:`CROP_UNITS` and both sides at least ``MIN_CROP_SIDE``, exactly as
    the /control editor builds a rectangle (``lib/control/crop.ts``)."""
    x0, y0, x1, y1 = grid.box
    mx, my = MARGIN * (x1 - x0), MARGIN * (y1 - y0)
    left, right = _span(x0 - mx, x1 + mx)
    top, bottom = _span(min(y0 - my, _king_top(grid, width / height)), y1 + my)
    return CropRect(
        x=left / CROP_UNITS,
        y=top / CROP_UNITS,
        w=(right - left) / CROP_UNITS,
        h=(bottom - top) / CROP_UNITS,
    )


def _king_top(grid: Grid, aspect: float) -> float:
    """The highest point, in fractions of the frame's height, that the top of a
    king (:data:`KING_SQUARES`) standing on the centre of any square reaches.

    Locally the camera sees the board in weak perspective: the homography's
    Jacobian at a square's centre has singular values ``s1 >= s2``, where ``s1``
    is a square's side seen face on and ``s2 / s1`` is the sine of the camera's
    elevation. A vertical post of height ``k`` squares then shows as
    ``k * s1 * cos(elevation)`` along the foreshortened direction (the second
    left singular vector), and the king rises by that length's vertical part. A
    top-down camera sees no rise; a low one sees nearly the king's full height.
    The fit works in units of the frame's height, so x is scaled by ``aspect``.
    """
    corners = grid.corners * np.array([aspect, 1], np.float32)
    h = cv2.getPerspectiveTransform(BOARD_CORNERS, corners.astype(np.float32))
    u, v = SQUARE_CENTRES[:, 0].astype(np.float64), SQUARE_CENTRES[:, 1].astype(np.float64)
    p = h @ np.stack([u, v, np.ones_like(u)])  # (3, 64)
    points = (p[:2] / p[2]).T  # (64, 2): x, y
    # d(x, y) / d(u, v) per square, (64, 2, 2): (h[r, c] - point[r] * h[2, c]) / w.
    jacobian = (h[:2, :2] - points[:, :, None] * h[2, :2]) / p[2][:, None, None]
    left, s, _ = np.linalg.svd(jacobian)
    cos = np.sqrt(np.clip(1 - (s[:, 1] / s[:, 0]) ** 2, 0, 1))
    rise = KING_SQUARES * s[:, 0] * cos * np.abs(left[:, 1, 1])
    return float((points[:, 1] - rise).min())


def _span(lo: float, hi: float) -> tuple[int, int]:
    """One axis in whole units, inside the frame and at least the minimum side."""
    least = round(MIN_CROP_SIDE * CROP_UNITS)
    start = min(max(math.floor(lo * CROP_UNITS + 0.5), 0), CROP_UNITS - least)
    end = min(max(math.floor(hi * CROP_UNITS + 0.5), start + least), CROP_UNITS)
    return start, end


@dataclass(frozen=True, eq=False)
class Proposal:
    """Auto-frame found the board.

    ``crop`` is the proposed crop (:func:`proposed_crop` of ``grid``); ``grid`` the
    final, trusted grid in the frame's fractions; ``reads`` how many reads it
    took. ``recognition`` is the last read, of the region around the board, and
    ``board_share`` the share of that read's width the board spans.
    """

    crop: CropRect
    grid: Grid
    reads: int
    recognition: Recognition
    board_share: float

    @property
    def dominant(self) -> bool:
        """Whether ``recognition`` saw the board large enough to count as a
        position read (:data:`BOARD_DOMINANT`)."""
        return self.board_share >= BOARD_DOMINANT


@dataclass(frozen=True)
class NoBoard:
    """Auto-frame found no trusted grid: the UI says "no board found".

    ``r2`` is the best R^2 of any grid it fitted, ``None`` when no read fitted one."""

    reads: int
    r2: float | None


#: A region of the frame in pixels, ``(x0, y0, x1, y1)``.
_Box = tuple[int, int, int, int]


@dataclass(frozen=True, eq=False)
class _Look:
    """One read of a region of the frame."""

    recognition: Recognition
    grid: Grid | None  # in the frame's fractions


def autoframe(read: ImageRead, frame: np.ndarray) -> Proposal | NoBoard:
    """Propose the crop for the board in ``frame``, a whole full-resolution BGR
    camera frame (module docstring)."""
    height, width = frame.shape[:2]
    looks: list[_Look] = []

    def look(box: _Box) -> _Look:
        x0, y0, x1, y1 = box
        recognition = read(frame[y0:y1, x0:x1])
        grid = recognition.grid
        if grid is not None:
            grid = grid.within(
                x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height
            )
        looks.append(_Look(recognition, grid))
        return looks[-1]

    def refine(grid: Grid | None) -> _Look | None:
        box = None if grid is None else _grown(grid, GROW, width, height)
        return None if box is None else look(box)

    def found(last: _Look | None) -> Proposal | None:
        if last is None or last.grid is None or not last.grid.trusted:
            return None
        local = last.recognition.grid
        assert local is not None  # the read's own grid, before within()
        x0, _, x1, _ = local.box
        return Proposal(
            proposed_crop(last.grid, width, height),
            last.grid,
            len(looks),
            last.recognition,
            x1 - x0,
        )

    first = look((0, 0, width, height))
    fine = refine(first.grid)
    if proposal := found(fine):
        return proposal
    tiles = [
        look((round(a * width), round(b * height), round(c * width), round(d * height)))
        for a, b, c, d in TILES
    ]
    best = max(
        (t.grid for t in tiles if t.grid is not None),
        key=lambda g: (g.inliers, g.r2),
        default=None,
    )
    if proposal := found(refine(best)):
        return proposal
    r2s = [t.grid.r2 for t in looks if t.grid is not None]
    return NoBoard(len(looks), max(r2s, default=None))


def locate_preview(read: ImageRead, preview: np.ndarray) -> Grid | None:
    """The trusted grid of one read of ``preview``, a downscaled whole camera frame,
    in the full frame's fractions; ``None`` when there is none (module docstring)."""
    grid = read(preview).grid
    return grid if grid is not None and grid.trusted else None


def _grown(grid: Grid, share: float, width: int, height: int) -> _Box | None:
    """``grid``'s box grown by ``share`` of its size on every side, in the frame's
    pixels and clipped to it; ``None`` when too small to read."""
    x0, y0, x1, y1 = grid.box
    mx, my = share * (x1 - x0), share * (y1 - y0)
    box = (
        max(0, math.floor((x0 - mx) * width)),
        max(0, math.floor((y0 - my) * height)),
        min(width, math.ceil((x1 + mx) * width)),
        min(height, math.ceil((y1 + my) * height)),
    )
    if box[2] - box[0] < MIN_READ_PX or box[3] - box[1] < MIN_READ_PX:
        return None
    return box
