"""Tests for edge board detection & rectification (WP-A1/A2/A3, DESIGN §4.4.1)."""

from __future__ import annotations

import numpy as np

from board_reader_core.boarddetect import detect_board_quad
from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board, simulate_angled_view
from board_reader_edge.rectify import (
    STATUS_NEEDS_SETUP,
    STATUS_OK,
    STATUS_RECALIBRATING,
    Rectifier,
    homography_from_corners,
    order_quad,
    rectify,
)


def _gray_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two same-size images (grayscale)."""
    import cv2

    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    return float(np.corrcoef(ga, gb)[0, 1])


def test_order_quad_canonicalises_any_input_order():
    # a square given out of order -> TL, TR, BR, BL
    pts = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], dtype=np.float32)
    ordered = order_quad(pts)
    np.testing.assert_allclose(
        ordered, [[10, 10], [90, 10], [90, 90], [10, 90]], atol=1e-3
    )


def test_homography_maps_corners_to_canonical_square():
    corners = np.array([[5, 7], [105, 9], [110, 110], [3, 108]], dtype=np.float32)
    h = homography_from_corners(corners, out_size=80)
    # The TL corner must land on (0,0) and BR on (80,80) under the homography.
    pts = np.array([corners[0], corners[2]], dtype=np.float32).reshape(-1, 1, 2)
    import cv2

    mapped = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
    np.testing.assert_allclose(mapped[0], [0, 0], atol=1e-3)
    np.testing.assert_allclose(mapped[1], [80, 80], atol=1e-3)


def test_detect_board_quad_recovers_the_angled_outline():
    board = render_board(START_PLACEMENT, square_px=48)
    img, gt = simulate_angled_view(board)
    found = detect_board_quad(img)
    assert found is not None
    # Detected corners are within a few pixels of the ground-truth outline.
    err = np.linalg.norm(found - gt, axis=1).mean()
    assert err < 0.03 * img.shape[1], f"corner error too large: {err:.1f}px"


def test_rectify_round_trip_recovers_the_board():
    board = render_board(START_PLACEMENT, square_px=48)
    img, _ = simulate_angled_view(board)
    found = detect_board_quad(img)
    assert found is not None
    crop = rectify(img, found, out_size=board.shape[0])
    # The un-warped crop strongly resembles the original top-down board. Residual
    # sub-pixel corner error leaves a thin border, so this auto-detect path bars at
    # 0.85; the exact-corner path (test_manual_corners_*) clears 0.9.
    assert _gray_corr(crop, board) > 0.85


def test_manual_corners_rectify_directly_and_stay_ok():
    board = render_board(START_PLACEMENT, square_px=48)
    img, gt = simulate_angled_view(board)
    rect = Rectifier(out_size=board.shape[0], manual_corners=gt)
    result = rect.process(img)
    assert result.status == STATUS_OK
    assert result.ok and result.crop is not None
    assert _gray_corr(result.crop, board) > 0.9


def test_auto_rectifier_locks_then_holds_ok_when_steady():
    board = render_board(START_PLACEMENT, square_px=48)
    img, _ = simulate_angled_view(board)
    rect = Rectifier(out_size=board.shape[0])
    first = rect.process(img)
    assert first.status == STATUS_OK and rect.locked
    # Same frame again -> no drift -> still ok.
    again = rect.process(img)
    assert again.status == STATUS_OK


def test_auto_rectifier_flags_recalibrating_on_a_bump_and_suppresses_crop():
    board = render_board(START_PLACEMENT, square_px=48)
    img, _ = simulate_angled_view(board, tilt=0.2, yaw=0.0)
    rect = Rectifier(out_size=board.shape[0])
    assert rect.process(img).status == STATUS_OK
    # A markedly different view (camera bumped) -> recalibrating, crop suppressed.
    bumped, _ = simulate_angled_view(board, tilt=0.45, yaw=0.18)
    result = rect.process(bumped)
    assert result.status == STATUS_RECALIBRATING
    assert result.crop is None
    # Re-locked on the new geometry: a steady follow-up frame returns to ok.
    assert rect.process(bumped).status == STATUS_OK


def test_auto_rectifier_needs_setup_when_no_board_present():
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    rect = Rectifier()
    result = rect.process(blank)
    assert result.status == STATUS_NEEDS_SETUP
    assert result.crop is None and not rect.locked


def test_cadence_skips_detection_between_ticks(monkeypatch):
    """Auto mode re-detects at most every redetect_interval_s of capture time."""
    import board_reader_edge.rectify as rmod

    board = render_board(START_PLACEMENT, square_px=48)
    img, _ = simulate_angled_view(board)
    calls = []
    real = rmod.detect_board_quad
    monkeypatch.setattr(rmod, "detect_board_quad", lambda im: calls.append(1) or real(im))

    rect = Rectifier(out_size=board.shape[0], redetect_interval_s=1.0)
    assert rect.process(img, 0.0).status == STATUS_OK  # first lock: detects
    assert rect.process(img, 0.5).status == STATUS_OK  # within cadence: cached
    assert rect.process(img, 0.9).status == STATUS_OK  # still cached
    assert rect.process(img, 1.1).status == STATUS_OK  # tick: detects again
    assert len(calls) == 2


def test_cadence_zero_interval_or_no_ts_detects_every_frame(monkeypatch):
    import board_reader_edge.rectify as rmod

    board = render_board(START_PLACEMENT, square_px=48)
    img, _ = simulate_angled_view(board)
    calls = []
    real = rmod.detect_board_quad
    monkeypatch.setattr(rmod, "detect_board_quad", lambda im: calls.append(1) or real(im))

    for _ in range(3):  # no ts -> the old per-frame behaviour
        Rectifier(out_size=board.shape[0]).process(img)
    assert len(calls) == 3

    calls.clear()
    rect = Rectifier(out_size=board.shape[0], redetect_interval_s=0)
    rect.process(img, 0.0)
    rect.process(img, 0.01)
    assert len(calls) == 2


def test_cadence_reanchors_on_backwards_ts(monkeypatch):
    """A clock step backwards re-anchors the cadence instead of stalling it."""
    import board_reader_edge.rectify as rmod

    board = render_board(START_PLACEMENT, square_px=48)
    img, _ = simulate_angled_view(board)
    calls = []
    real = rmod.detect_board_quad
    monkeypatch.setattr(rmod, "detect_board_quad", lambda im: calls.append(1) or real(im))

    rect = Rectifier(out_size=board.shape[0], redetect_interval_s=1.0)
    rect.process(img, 10.0)
    rect.process(img, 2.0)  # went backwards -> due again, cadence re-anchored at 2.0
    assert len(calls) == 2
    rect.process(img, 2.5)  # within the re-anchored window -> cached
    assert len(calls) == 2


def test_lost_board_redetects_every_frame_until_it_returns():
    """A failed detection must not advance the cadence: the gap stays honest."""
    board = render_board(START_PLACEMENT, square_px=48)
    img, _ = simulate_angled_view(board)
    blank = np.zeros_like(img)

    rect = Rectifier(out_size=board.shape[0], redetect_interval_s=1.0)
    assert rect.process(img, 0.0).status == STATUS_OK
    # Board vanishes at the next tick: needs_setup, and the cadence does NOT advance,
    # so every following frame is examined rather than riding stale geometry.
    assert rect.process(blank, 1.5).status == STATUS_NEEDS_SETUP
    assert rect.process(blank, 1.6).status == STATUS_NEEDS_SETUP
    # The board comes back within what would have been a skip window -> seen at once.
    assert rect.process(img, 1.7).status == STATUS_OK
