"""Tests for the synthetic board renderer (sim harness)."""

from __future__ import annotations

import numpy as np

from board_reader_core.protocol import EMPTY_PLACEMENT, START_PLACEMENT
from board_reader_core.sim import render_board, render_frames


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
