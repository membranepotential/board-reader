"""Tests for /estimate debug visualizations (draw_boxes, crop_mosaic, to_b64_jpeg)."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from board_reader_core.protocol import GRID, NUM_CLASSES, START_PLACEMENT, dist_from_placement
from board_reader_core.sim import render_board
from board_reader_inference.diagnostics import crop_mosaic, draw_boxes, to_b64_jpeg
from board_reader_inference.squares import (
    BboxGeometry,
    crop_squares_photo,
    full_frame_corners,
    square_boxes,
)


def _square_dist():
    return dist_from_placement(START_PLACEMENT, confidence=0.9)


def test_draw_boxes_returns_downscaled_shape():
    board = render_board(square_px=32)  # small enough that no downscale kicks in
    boxes = square_boxes(full_frame_corners(board), BboxGeometry(0.0, 0.0), board.shape)
    out = draw_boxes(board, boxes, _square_dist())
    assert out.shape == board.shape


def test_draw_boxes_downscales_large_images():
    board = render_board(square_px=200)  # long side > 1200
    boxes = square_boxes(full_frame_corners(board), BboxGeometry(0.0, 0.0), board.shape)
    out = draw_boxes(board, boxes, _square_dist())
    assert max(out.shape[:2]) <= 1200
    assert max(out.shape[:2]) < max(board.shape[:2])


def test_crop_mosaic_returns_8_col_grid():
    board = render_board(START_PLACEMENT, square_px=32)
    crops = crop_squares_photo(
        board, full_frame_corners(board), geometry=BboxGeometry(0.0, 0.0), size=64
    )
    mosaic = crop_mosaic(crops, _square_dist())
    assert mosaic.ndim == 3 and mosaic.shape[2] == 3
    # 8 cells wide, each MOSAIC_CELL_WIDTH (96) px.
    assert mosaic.shape[1] == 8 * 96


def test_to_b64_jpeg_roundtrips():
    board = render_board(square_px=16)
    b64 = to_b64_jpeg(board)
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == board.shape
