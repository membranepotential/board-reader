"""Tests for the synthetic board renderer (sim harness)."""

from __future__ import annotations

import numpy as np

from board_reader_core.protocol import EMPTY_PLACEMENT, START_PLACEMENT
from board_reader_core.sim import (
    board_corners,
    render_board,
    render_frames,
    simulate_angled_view,
    warp_to_quad,
)


def test_render_board_shape_and_dtype():
    img = render_board(START_PLACEMENT, square_px=32)
    assert img.shape == (8 * 32, 8 * 32, 3)
    assert img.dtype == np.uint8


def test_render_board_draws_pieces_only_where_present():
    """The start position has more non-background ink than the empty board."""
    start = render_board(START_PLACEMENT)
    empty = render_board(EMPTY_PLACEMENT)
    # Pieces add light/dark glyph pixels; the populated board differs from empty.
    assert not np.array_equal(start, empty)
    assert int(np.abs(start.astype(int) - empty.astype(int)).sum()) > 0


def test_render_frames_returns_one_image_per_placement():
    frames = render_frames([START_PLACEMENT, EMPTY_PLACEMENT])
    assert len(frames) == 2
    assert all(f.shape == frames[0].shape for f in frames)


def test_warp_to_quad_places_the_board_at_the_given_corners():
    board = render_board(square_px=32)
    side = board.shape[0]
    # Identity quad on a same-size canvas reproduces the board (corner-to-corner).
    img = warp_to_quad(board, board_corners(side), canvas=(side, side))
    assert img.shape == board.shape
    assert int(np.abs(img.astype(int) - board.astype(int)).mean()) < 5


def test_simulate_angled_view_returns_image_and_ground_truth_quad():
    board = render_board(square_px=32)
    img, quad = simulate_angled_view(board)
    assert img.ndim == 3 and img.shape[0] == img.shape[1]
    assert quad.shape == (4, 2)
    # Foreshortening: the far (top) edge is shorter than the near (bottom) edge.
    top_w = np.linalg.norm(quad[1] - quad[0])
    bottom_w = np.linalg.norm(quad[2] - quad[3])
    assert top_w < bottom_w
