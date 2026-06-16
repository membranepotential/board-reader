"""Tests for 64-square cropping (WP-B1)."""

from __future__ import annotations

import numpy as np

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_inference.squares import crop_squares


def test_crops_64_squares_in_protocol_order():
    board = render_board(START_PLACEMENT, square_px=48)
    crops = crop_squares(board)
    assert len(crops) == 64
    # Index r*8+c is rank 8-r, file a+c. Top-left (index 0) is a8 = a black rook square,
    # which differs from the empty middle (index 4*8 = a4).
    a8, a4 = crops[0], crops[32]
    assert a8.shape[0] > 0 and a4.shape[0] > 0
    assert not np.array_equal(a8, a4)


def test_resize_and_inset_change_crop_size():
    board = render_board(square_px=48)
    full = crop_squares(board)[0]
    inset = crop_squares(board, inset=0.2)[0]
    sized = crop_squares(board, size=32)[0]
    assert inset.shape[0] < full.shape[0]  # inset trims edges
    assert sized.shape[:2] == (32, 32)


def test_inset_out_of_range_rejected():
    board = render_board(square_px=24)
    for bad in (-0.1, 0.5, 0.9):
        try:
            crop_squares(board, inset=bad)
        except ValueError:
            continue
        raise AssertionError(f"inset={bad} should have raised")
