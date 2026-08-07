"""Automatic board detection (boarddetect.detect_board_quad)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from board_reader_core.boarddetect import (
    _alternation,
    _validate,
    detect_board_quad,
)
from board_reader_core.sim import render_board, simulate_angled_view

_REPO = Path(__file__).resolve().parents[3]
_EVAL = _REPO / "data" / "eval"
_CCBF = _REPO / "data" / "streams" / "frames" / "ccbf-r5_2sRJ4JuL_m8"


def _corner_errors(quad: np.ndarray, truth: np.ndarray) -> list[float]:
    """Distance from each detected corner to its nearest ground-truth corner
    (order-free: detected order is geometric, truth order is chess-canonical)."""
    return [float(min(np.linalg.norm(p - t) for t in truth)) for p in quad]


def test_synthetic_angled_view() -> None:
    image, truth = simulate_angled_view(render_board())
    quad = detect_board_quad(image)
    assert quad is not None
    errs = _corner_errors(quad, np.asarray(truth))
    # Synthetic truth is exact; detection should be within a few pixels.
    assert max(errs) < 0.01 * image.shape[1]


def test_returns_ordered_convex_quad() -> None:
    image, _ = simulate_angled_view(render_board())
    quad = detect_board_quad(image)
    assert quad is not None
    assert quad.shape == (4, 2) and quad.dtype == np.float32
    tl, tr, br, bl = quad
    assert tl[0] < tr[0] and bl[0] < br[0]  # left corners left of right corners
    assert tl[1] < bl[1] and tr[1] < br[1]  # top corners above bottom corners


def test_no_board_returns_none() -> None:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    assert detect_board_quad(noise) is None
    flat = np.full((600, 800, 3), 128, np.uint8)
    assert detect_board_quad(flat) is None


@pytest.mark.parametrize("key", ["img_0565", "img_0566"])
def test_real_photo(key: str) -> None:
    """Real phone photos: detection within ~3% of image width of the operator's
    hand-tapped corners (which are themselves only ~1-2% accurate — on img_0566
    the detector's rectification visibly aligns better than the ground truth)."""
    gt_file = _EVAL / "ground_truth.json"
    if not gt_file.exists():
        pytest.skip("eval assets not present")
    meta = json.loads(gt_file.read_text())[key]
    photo = Path(meta["source"]).expanduser()
    if not photo.exists():
        pytest.skip(f"source photo {photo} not present")
    image = cv2.imread(str(photo))
    quad = detect_board_quad(image)
    assert quad is not None
    truth = np.array(meta["corners_a8_h8_h1_a1"], np.float32)
    errs = _corner_errors(quad, truth)
    assert np.mean(errs) < 0.03 * image.shape[1]


@pytest.mark.parametrize("name", ["IMG_0567", "IMG_0568", "IMG_0569"])
def test_hard_photo_detected_or_none(name: str) -> None:
    """Dim / warm-cast / extreme-angle photos (no hand ground truth): a returned
    quad must be sane — convex (checked inside detect), plausibly board-sized,
    and fully inside the frame with a small tolerance. None is acceptable; a
    wildly wrong quad is not."""
    photo = Path("~/Downloads").expanduser() / f"{name}.JPEG"
    if not photo.exists():
        pytest.skip(f"{photo} not present")
    image = cv2.imread(str(photo))
    quad = detect_board_quad(image)
    if quad is None:
        return
    h, w = image.shape[:2]
    area = cv2.contourArea(quad)
    assert 0.15 * w * h < area < 0.95 * w * h
    tol = 0.02 * w
    assert (quad[:, 0] > -tol).all() and (quad[:, 0] < w + tol).all()
    assert (quad[:, 1] > -tol).all() and (quad[:, 1] < h + tol).all()


def test_phase_scoring_survives_offscreen_square_centers() -> None:
    """A phase hypothesis whose homography projects a few square centers out of
    frame must still score on the in-frame squares (the coarse homography is
    fitted on a sub-grid; at steep angles extrapolating it overshoots even for
    the CORRECT phase) — but a hypothesis mostly off-image stays unscoreable."""
    board = render_board(square_px=32)  # flat top-down board, ideal alternation
    sq = 32
    # Identity-ish lattice->image map: lattice (0,0) at the first inner corner.
    h = np.array([[sq, 0, sq], [0, sq, sq], [0, 0, 1]], np.float32)
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    assert _alternation(gray, h, 0.0, 0.0) > 4.0
    # Shift the mapping so a full column of centers leaves the image: still
    # scoreable (56 in-frame squares clear the 24-per-colour floor).
    h_shift = h.copy()
    h_shift[0, 2] += 1.2 * sq
    assert _alternation(gray, h_shift, 0.0, 0.0) > 4.0
    # Push most of the board out: below the floor -> unscoreable.
    h_gone = h.copy()
    h_gone[0, 2] += 6 * sq
    assert _alternation(gray, h_gone, 0.0, 0.0) == -1.0


def test_one_square_shifted_quad_is_rejected_by_the_outside_ring() -> None:
    """The guard that makes shifted quads unrepresentable: a quad moved one
    square off the surface leaves a true board line just outside itself, and
    acceptance must reject that even though the shifted INSIDE still alternates
    (deterministic synthetic pin — the broadcast-frame regression test that
    also covers this skips on machines without the local footage)."""
    from board_reader_core.boarddetect import _acceptance

    sq = 48
    # A 12x12-square scene: a faintly checkered "table" (so a shifted quad's
    # off-board line still alternates weakly and sign-consistently — a flat
    # surround would fail sign-consistency and never reach the ring guard),
    # with a strong 8x8 board two squares in from each edge.
    img = np.zeros((12 * sq, 12 * sq), np.uint8)
    for r in range(12):
        for c in range(12):
            img[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = (
                131 if (r + c) % 2 == 0 else 124
            )
    for r in range(8):
        for c in range(8):
            v = 220 if (r + c) % 2 == 0 else 40
            img[(r + 2) * sq : (r + 3) * sq, (c + 2) * sq : (c + 3) * sq] = v
    board = np.array(
        [[2 * sq, 2 * sq], [10 * sq, 2 * sq], [10 * sq, 10 * sq], [2 * sq, 10 * sq]],
        np.float32,
    )
    assert _acceptance(img, board) is not None
    shifted = board + np.array([sq, 0], np.float32)  # one square sideways
    assert _acceptance(img, shifted) is None


def test_validate_accepts_crowded_low_min_when_median_is_strong() -> None:
    """The crowded-board acceptance: one nearly-dead (but sign-consistent)
    rank with a strong median passes; a sign-flipped rank never does."""
    sq = 64
    quad = np.array([[0, 0], [8 * sq, 0], [8 * sq, 8 * sq], [0, 8 * sq]], np.float32)

    def board_with_weak_row(row: int, lo: int, hi: int, flip: bool = False):
        img = np.zeros((8 * sq, 8 * sq), np.uint8)
        for r in range(8):
            for c in range(8):
                light = (r + c) % 2 == 0
                if r == row:
                    v = (lo if flip else hi) if light else (hi if flip else lo)
                else:
                    v = 200 if light else 50
                img[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = v
        return img

    # A back rank so piece-covered its alternation is ~4 gray levels: accepted.
    assert _validate(board_with_weak_row(0, 97, 104), quad)
    # The same rank alternating the WRONG way (e.g. a border row of the table):
    # sign consistency kills it regardless of the median.
    assert not _validate(board_with_weak_row(0, 80, 124, flip=True), quad)


def test_ccbf_r5_broadcast_frames_detect() -> None:
    """Regression on the real degraded-angle broadcast (frames are machine-local;
    skipped where absent). The corner finder is internally nondeterministic, so
    each frame gets two chances — the cadenced re-detect in production has many
    more. Before the FN fixes this footage detected 6/7."""
    if not _CCBF.exists():
        pytest.skip("ccbf-r5 frames not present")
    files = sorted(_CCBF.glob("*.png"))
    assert len(files) >= 7
    truth = np.array(
        [[266.7, 104.3], [1031.9, 117.1], [1332.5, 719.5], [26.3, 698.3]],
        np.float32,
    )
    detected = 0
    for f in files:
        image = cv2.imread(str(f), cv2.IMREAD_COLOR)
        quad = next(
            (q for q in (detect_board_quad(image) for _ in range(2)) if q is not None),
            None,
        )
        if quad is None:
            continue
        detected += 1
        # Ordered comparison: the camera is fixed and both quads are TL,TR,BR,BL,
        # so nearest-match would let a collapsed quad slip through.
        errs = np.linalg.norm(quad - truth, axis=1)
        assert max(errs) < 60, f"{f.name}: detection {errs} px off the fixed camera"
    assert detected == len(files)


def test_rectified_crop_detects_full_frame() -> None:
    """An already-rectified board crop is itself detectable; the quad is
    (close to) the full image."""
    crop_file = _EVAL / "img_0565_rect.png"
    if not crop_file.exists():
        pytest.skip("eval assets not present")
    image = cv2.imread(str(crop_file))
    quad = detect_board_quad(image)
    assert quad is not None
    w = image.shape[1]
    corners = np.array([[0, 0], [w, 0], [w, w], [0, w]], np.float32)
    errs = _corner_errors(quad, corners)
    assert max(errs) < 0.05 * w
