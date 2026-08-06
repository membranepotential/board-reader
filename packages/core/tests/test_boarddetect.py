"""Automatic board detection (boarddetect.detect_board_quad)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from board_reader_core.boarddetect import detect_board_quad
from board_reader_core.sim import render_board, simulate_angled_view

_REPO = Path(__file__).resolve().parents[3]
_EVAL = _REPO / "data" / "eval"


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
