"""Tests for 64-square cropping (WP-B1), rectified and full-piece photo variants."""

from __future__ import annotations

import numpy as np

from board_reader_core.protocol import GRID, START_PLACEMENT
from board_reader_core.sim import render_board, simulate_angled_view
from board_reader_inference.squares import (
    crop_squares,
    crop_squares_photo,
    full_frame_corners,
    photo_square_boxes,
    rectify_photo,
)


def _near_white(crop: np.ndarray) -> bool:
    """Does the crop contain the white glyph colour (245,245,245) of sim pieces?"""
    return bool((crop > 235).all(axis=-1).any())


def test_crop_squares_photo_geometry_and_content():
    board = render_board(START_PLACEMENT, square_px=48)
    photo, quad = simulate_angled_view(board)  # quad = a8,h8,h1,a1 ground truth
    crops = crop_squares_photo(photo, quad, size=224)
    assert len(crops) == 64
    assert all(c.shape == (224, 224, 3) for c in crops)
    # e2 holds a white pawn glyph; e4's tall box looks over empty e4..e6 only
    # (headroom extends upward = toward rank 8 in the angled sim view).
    e2, e4 = crops[6 * 8 + 4], crops[4 * 8 + 4]
    assert _near_white(e2), "the piece glyph must land in its own square's crop"
    assert not _near_white(e4), "an empty square's crop must not contain a glyph"


def test_crop_squares_photo_clips_at_image_edges():
    board = render_board(square_px=32)
    h, w = board.shape[:2]
    # Board fills the whole frame: rank-8 boxes' headroom pokes above the image.
    crops = crop_squares_photo(board, full_frame_corners(board), size=64)
    assert len(crops) == 64 and all(c.shape == (64, 64, 3) for c in crops)


def test_rectify_photo_roundtrips_the_sim_warp():
    board = render_board(square_px=48)
    photo, quad = simulate_angled_view(board)
    rect = rectify_photo(photo, quad, out_size=board.shape[0])
    # The un-warped board should closely match the original rendering.
    diff = np.abs(rect.astype(int) - board.astype(int)).mean()
    assert diff < 12, f"rectified board deviates too much (mean abs diff {diff:.1f})"


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


def test_photo_square_boxes_shape():
    board = render_board(square_px=32)
    boxes = photo_square_boxes(full_frame_corners(board))
    assert boxes.shape == (64, 4, 2)
    assert boxes.dtype == np.float32


def test_photo_square_boxes_are_the_flat_squares_at_zero_headroom_and_pad():
    """On a flat full-frame image, headroom=0/side_pad=0 boxes ARE the 64 squares."""
    board = render_board(square_px=32)
    h, w = board.shape[:2]
    sh, sw = h / GRID, w / GRID
    boxes = photo_square_boxes(
        full_frame_corners(board), headroom=0.0, side_pad=0.0
    )
    # index 0 = a8 = top-left square; index 32 = a4 (row 4, col 0).
    expected0 = np.array(
        [[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32
    )
    expected32 = np.array(
        [[0, 4 * sh], [sw, 4 * sh], [sw, 5 * sh], [0, 5 * sh]], dtype=np.float32
    )
    assert np.allclose(boxes[0], expected0, atol=1e-3)
    assert np.allclose(boxes[32], expected32, atol=1e-3)


def test_photo_square_boxes_from_shifted_corners_diverges_from_rot90_with_padding():
    """A pitfall the /estimate autorotate debug fix avoids: both ``headroom``
    (upward) and ``side_pad`` (sideways) extend a box along fixed IMAGE axes,
    scaled by a board-relative width — independent of which corner is labeled
    a8. So re-deriving boxes from a corner order shifted by k is NOT the same
    as rotating the already-computed boxes by k: the extension direction
    rotates with the (wrong) re-derivation too, so it samples the wrong pixels
    for a 90/270-degree turn. Confirmed here for headroom alone, side_pad
    alone, and both together (each ~1.5-2px on a 32px-square board — the
    divergence grows with the padding, e.g. ~6px at headroom=0.6). Only with
    *both* paddings at zero (no extension to misdirect) do the two agree —
    the special case that made this easy to miss."""
    board = render_board(square_px=32)
    photo, quad = simulate_angled_view(board)  # a8,h8,h1,a1, perspective-warped
    for headroom, side_pad in [(0.2, 0.0), (0.0, 0.15), (0.2, 0.15)]:
        boxes = photo_square_boxes(quad, headroom=headroom, side_pad=side_pad)
        grid = boxes.reshape(8, 8, 4, 2)
        for k in (1, 3):
            shifted = quad[[(i + k) % 4 for i in range(4)]]
            recomputed = photo_square_boxes(shifted, headroom=headroom, side_pad=side_pad)
            rotated = np.rot90(grid, k, axes=(0, 1)).reshape(-1, 4, 2)
            assert not np.allclose(recomputed, rotated, atol=1e-2)

    boxes0 = photo_square_boxes(quad, headroom=0.0, side_pad=0.0).reshape(8, 8, 4, 2)
    shifted1 = quad[[(i + 1) % 4 for i in range(4)]]
    recomputed0 = photo_square_boxes(shifted1, headroom=0.0, side_pad=0.0)
    assert np.allclose(
        recomputed0, np.rot90(boxes0, 1, axes=(0, 1)).reshape(-1, 4, 2), atol=1e-2
    )


def test_crop_squares_photo_behavior_unchanged():
    """crop_squares_photo's output is unaffected by exposing photo_square_boxes."""
    board = render_board(START_PLACEMENT, square_px=48)
    photo, quad = simulate_angled_view(board)
    crops = crop_squares_photo(photo, quad, size=224)
    assert len(crops) == 64
    assert all(c.shape == (224, 224, 3) for c in crops)
