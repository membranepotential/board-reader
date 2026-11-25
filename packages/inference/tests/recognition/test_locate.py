"""Locating the board from ChessQ's square attention (:mod:`.locate`).

Synthetic attention maps stand in for the model: a Gaussian blob per square query
at the patch its square centre falls on, for a board placed by a known homography.
The Auto-frame driver runs against a fake read that "sees" a board only when it is
large enough in the region read, the way ChessQ does (the probe's coarse-to-fine
finding).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from board_reader_core.control import CropRect

from board_reader_inference.recognition.chessq import INPUT_PX
from board_reader_inference.recognition.locate import (
    BOARD_CORNERS,
    MIN_R2,
    SQUARE_CENTRES,
    Grid,
    NoBoard,
    Proposal,
    autoframe,
    fit_grid,
    locate_preview,
    proposed_crop,
    square_centroids,
)
from board_reader_inference.recognition.recognizer import Recognition

PATCHES = INPUT_PX // 14  # 46


def _attention(quad: np.ndarray, width: int, height: int, sigma: float = 0.7) -> np.ndarray:
    """(3, 64, 46, 46) maps for a board whose outer corners a8 h8 h1 a1 lie at
    ``quad`` (pixels of a ``width`` x ``height`` image)."""
    homography = cv2.getPerspectiveTransform(BOARD_CORNERS, quad.astype(np.float32))
    centres = cv2.perspectiveTransform(SQUARE_CENTRES[:, None], homography)[:, 0]
    # Image pixels -> patch units of the squashed 644 px input (centre of patch k at k).
    patch = centres * np.array([INPUT_PX / width, INPUT_PX / height]) / 14 - 0.5
    yy, xx = np.mgrid[0:PATCHES, 0:PATCHES]
    maps = np.exp(
        -((xx - patch[:, 0, None, None]) ** 2 + (yy - patch[:, 1, None, None]) ** 2)
        / (2 * sigma**2)
    )
    maps /= maps.sum(axis=(1, 2), keepdims=True)
    return np.stack([maps, maps, maps]).astype(np.float32)


def _grid(quad: np.ndarray, width: int, height: int) -> Grid | None:
    return fit_grid(
        square_centroids(_attention(quad, width, height), width, height), width, height
    )


def _corner_error(grid: Grid, quad: np.ndarray, width: int, height: int) -> float:
    """Mean corner distance in pixels, chess order compared as given."""
    return float(np.linalg.norm(grid.corners * [width, height] - quad, axis=1).mean())


def test_a_clean_grid_gives_the_board_corners_in_chess_order():
    w, h = 800, 600
    quad = np.array([[200, 100], [600, 100], [600, 500], [200, 500]], np.float32)
    grid = _grid(quad, w, h)
    assert grid is not None and grid.trusted
    assert grid.r2 > 0.999 and grid.inliers == 64
    assert _corner_error(grid, quad, w, h) < 2.0
    assert grid.box == pytest.approx((0.25, 1 / 6, 0.75, 5 / 6), abs=0.005)


def test_a_rotated_board_in_perspective_keeps_a8_where_it_is():
    """A quarter turn (a8 at the top right) seen at a slant on a wide frame."""
    w, h = 1280, 720
    quad = np.array([[900, 150], [980, 600], [300, 650], [420, 120]], np.float32)
    grid = _grid(quad, w, h)
    assert grid is not None and grid.trusted
    assert _corner_error(grid, quad, w, h) < 3.0


def test_attention_with_no_board_shape_fits_no_grid():
    uniform = np.full((3, 64, PATCHES, PATCHES), 1 / PATCHES**2, np.float32)
    assert fit_grid(square_centroids(uniform, 640, 480), 640, 480) is None
    empty = np.zeros_like(uniform)  # no mass anywhere: NaN centroids
    assert fit_grid(square_centroids(empty, 640, 480), 640, 480) is None


def test_diffuse_heads_fit_no_trusted_grid():
    noise = np.random.default_rng(0).random((3, 64, PATCHES, PATCHES)).astype(np.float32)
    grid = fit_grid(square_centroids(noise, 640, 480), 640, 480)
    assert grid is None or not grid.trusted


def test_the_r2_gate_rejects_a_grid_whose_squares_scatter():
    """Three ranks of squares attend somewhere else entirely: LMedS still fits the
    other five, but R^2 over all 64 says the grid is not a board."""
    w, h = 800, 600
    quad = np.array([[200, 100], [600, 100], [600, 500], [200, 500]], np.float32)
    attention = _attention(quad, w, h)
    rng = np.random.default_rng(1)
    for q in range(40, 64):
        attention[:, q] = np.roll(attention[:, q], rng.integers(-20, 20, 2), axis=(1, 2))
    grid = fit_grid(square_centroids(attention, w, h), w, h)
    assert grid is not None and grid.r2 < MIN_R2 and not grid.trusted


def test_within_moves_a_grid_from_a_crop_into_the_frame():
    grid = Grid(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), 0.99, 64)
    moved = grid.within(0.2, 0.1, 0.5, 0.4)
    np.testing.assert_allclose(
        moved.corners, [[0.2, 0.1], [0.7, 0.1], [0.7, 0.5], [0.2, 0.5]]
    )
    assert (moved.r2, moved.inliers) == (0.99, 64)


def _square(x0: float, y0: float, x1: float, y1: float) -> Grid:
    return Grid(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), 0.99, 64)


def test_a_top_down_board_gets_the_box_plus_ten_percent_snapped():
    """Seen from straight above, a piece shows no height: 10 % on every side."""
    crop = proposed_crop(_square(0.3, 0.2, 0.7, 0.6), 1000, 1000)
    assert (crop.x, crop.y, crop.w, crop.h) == pytest.approx((0.26, 0.16, 0.48, 0.48))
    for edge in (crop.x, crop.y, crop.w, crop.h):
        assert edge * 10_000 == pytest.approx(round(edge * 10_000), abs=1e-6)


#: A board seen at a low angle on a 1920 x 1080 frame, a8 h8 h1 a1: the far side
#: (a8-h8) is narrower and on top, the ranks are foreshortened.
_LOW = np.array([[0.35, 0.30], [0.65, 0.30], [0.75, 0.60], [0.25, 0.60]], np.float32)


def _margins(crop: CropRect, grid: Grid) -> tuple[float, float, float, float]:
    """The crop's margins left, top, right, bottom, in fractions of the frame."""
    x0, y0, x1, y1 = grid.box
    return x0 - crop.x, y0 - crop.y, crop.x + crop.w - x1, crop.y + crop.h - y1


def test_a_low_board_gets_room_on_top_for_a_far_rank_king():
    """A far-rank square is 74 px wide and its rank 26 px deep, so the camera's
    elevation has a sine of 0.35: a king (1.9 squares) shows 1.9 * 74 * 0.94 =
    132 px tall, from a base 12 px inside the edge, so it tops out 120 px (0.11
    of the frame) above the board. The other sides keep 10 % of the box."""
    grid = Grid(_LOW, 0.99, 64)
    left, top, right, bottom = _margins(proposed_crop(grid, 1920, 1080), grid)
    assert (left, right, bottom) == pytest.approx((0.05, 0.05, 0.03), abs=1e-4)
    assert top == pytest.approx(0.11, abs=0.004)
    assert top > 2 * bottom


@pytest.mark.parametrize("turn", [1, 2, 3])
def test_the_top_margin_does_not_depend_on_which_side_is_the_far_rank(turn: int):
    """White to the left, the right or at the far side: the same quad in the image
    gives the same crop, whichever chess side lies on top."""
    rotated = Grid(np.roll(_LOW, turn, axis=0), 0.99, 64)
    assert proposed_crop(rotated, 1920, 1080) == proposed_crop(
        Grid(_LOW, 0.99, 64), 1920, 1080
    )


def test_the_proposed_crop_stays_inside_the_frame():
    """A board running off the right edge: the crop ends at the frame's edge. A low
    board near the top: the king's room stops at the frame's top."""
    crop = proposed_crop(_square(0.6, -0.05, 1.08, 0.5), 1000, 1000)
    assert crop.x + crop.w <= 1 and crop.y == 0
    assert crop.x == pytest.approx(0.552)
    near_top = proposed_crop(Grid(_LOW - [0, 0.25], 0.99, 64), 1920, 1080)
    assert near_top.y == 0 and near_top.y + near_top.h <= 1


def test_a_tiny_box_still_proposes_the_smallest_crop():
    crop = proposed_crop(_square(0.5, 0.5, 0.51, 0.51), 1000, 1000)
    assert crop.w == crop.h == 0.05


# --------------------------------------------------------------------------- #
# The Auto-frame driver against a fake read
# --------------------------------------------------------------------------- #


def _frame(width: int, height: int) -> np.ndarray:
    """A frame whose pixels hold their own coordinates, so the fake read can tell
    which region of it it was handed."""
    yy, xx = np.mgrid[0:height, 0:width]
    return np.dstack([xx, yy]).astype(np.int32)


class _FakeChessQ:
    """Reads a board at ``quad`` (frame fractions) the way ChessQ does at a given
    size: no grid below ``lost`` of the region's width, a rough untrusted one
    below ``sharp``, a trusted one at or above it."""

    def __init__(self, quad: np.ndarray, *, lost: float, sharp: float) -> None:
        self.quad, self.lost, self.sharp = quad, lost, sharp
        self.regions: list[tuple[int, int, int, int]] = []

    def __call__(self, image: np.ndarray) -> Recognition:
        x0, y0 = (int(v) for v in image[0, 0])
        h, w = image.shape[:2]
        self.regions.append((x0, y0, w, h))
        frame_w, frame_h = self.width, self.height
        local = (self.quad * [frame_w, frame_h] - [x0, y0]) / [w, h]
        inside = (local >= -0.01).all() and (local <= 1.01).all()
        share = float(local[:, 0].max() - local[:, 0].min())
        grid = None
        if inside and share >= self.lost:
            grid = Grid(local.astype(np.float32), 0.99 if share >= self.sharp else 0.6, 50)
        return Recognition(np.zeros((8, 8, 13)), grid=grid)

    def on(self, frame: np.ndarray) -> _FakeChessQ:
        self.height, self.width = frame.shape[:2]
        return self


def _board(x0: float, y0: float, side: float, aspect: float) -> np.ndarray:
    """A square board ``side`` of the frame's width wide, top-left at (x0, y0)."""
    return np.array(
        [
            [x0, y0],
            [x0 + side, y0],
            [x0 + side, y0 + side * aspect],
            [x0, y0 + side * aspect],
        ],
        np.float32,
    )


def test_coarse_to_fine_proposes_after_two_reads():
    """A board at 30 % of the frame: the first read gives only a rough box, the
    read of that box grown by half sees it at half its width, sharp."""
    frame = _frame(1920, 1080)
    quad = _board(0.55, 0.3, 0.3, 16 / 9)
    read = _FakeChessQ(quad, lost=0.2, sharp=0.45).on(frame)
    result = autoframe(read, frame)
    assert isinstance(result, Proposal)
    assert result.reads == 2 == len(read.regions)
    np.testing.assert_allclose(result.grid.corners, quad, atol=2e-3)
    assert result.board_share == pytest.approx(0.5, abs=0.01)
    assert result.dominant
    assert result.recognition.grid is not None and result.recognition.grid.trusted
    crop = result.crop
    assert crop.x == pytest.approx(0.55 - 0.03, abs=1e-3)
    assert crop.w == pytest.approx(0.36, abs=1e-3)


def test_the_tiles_find_a_board_too_small_for_the_first_read():
    """A board at 15 %: the whole frame shows nothing, a 60 % tile a rough box,
    and the refine of that box a trusted grid. 1 + 5 + 1 reads."""
    frame = _frame(1920, 1080)
    quad = _board(0.1, 0.1, 0.15, 16 / 9)
    read = _FakeChessQ(quad, lost=0.2, sharp=0.45).on(frame)
    result = autoframe(read, frame)
    assert isinstance(result, Proposal)
    assert result.reads == 7
    assert read.regions[1] == (0, 0, 1152, 648)  # the first tile, which holds the board
    np.testing.assert_allclose(result.grid.corners, quad, atol=2e-3)


def test_no_trusted_grid_anywhere_is_no_board():
    frame = _frame(1920, 1080)
    rough = _FakeChessQ(_board(0.3, 0.25, 0.25, 16 / 9), lost=0.2, sharp=2.0).on(frame)
    result = autoframe(rough, frame)
    assert isinstance(result, NoBoard)
    assert result.reads == 8  # whole frame, refine, five tiles, refine of the best
    assert result.r2 == pytest.approx(0.6)

    blind = _FakeChessQ(_board(0.3, 0.3, 0.3, 16 / 9), lost=2.0, sharp=2.0).on(frame)
    assert autoframe(blind, frame) == NoBoard(reads=6, r2=None)


def test_locate_preview_is_one_read_gated_on_r2():
    preview = _frame(480, 270)
    quad = _board(0.3, 0.2, 0.4, 16 / 9)
    sharp = _FakeChessQ(quad, lost=0.2, sharp=0.3).on(preview)
    grid = locate_preview(sharp, preview)
    assert grid is not None and len(sharp.regions) == 1
    np.testing.assert_allclose(grid.corners, quad, atol=1e-6)
    rough = _FakeChessQ(quad, lost=0.2, sharp=0.9).on(preview)
    assert locate_preview(rough, preview) is None
