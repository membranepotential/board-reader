"""64-square cropping from a rectified board or an original photo (DESIGN.md WP-B1).

Two cropping regimes feed the recognizer, both in the **L3 protocol order** (row 0 =
rank 8, column 0 = file a):

* :func:`crop_squares` — split a canonical top-down crop into 64 flat squares. Fine
  for occupancy, but a warped square *decapitates* pieces: the top-down projection
  smears a piece across neighbouring squares and the square cell keeps only its base.
* :func:`crop_squares_photo` — Chesscog-style tall boxes taken from the ORIGINAL
  photo: each square's base quad is projected into the image and extended upward so
  the crop contains the whole standing piece. This is what an embedding backbone
  should actually see; per-square accuracy on real photos is markedly better.

A center ``inset`` (rectified path) keeps only the middle of each square,
suppressing grid lines and neighbouring-square spill.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from board_reader_core.protocol import GRID

# Canonical frame side used for square geometry; matches the snapshot crop size.
CANONICAL = 512.0


def crop_squares(
    board: np.ndarray, *, inset: float = 0.0, size: int | None = None
) -> list[np.ndarray]:
    """Split a rectified ``board`` into 64 square crops (rank 8→1, file a→h).

    Parameters
    ----------
    board:
        Canonical top-down BGR crop; should be (near) square.
    inset:
        Fraction of each square trimmed off every edge (0 = whole square, 0.1 = keep
        the central 80%). Useful to drop grid lines / spill before embedding.
    size:
        If given, each crop is resized to ``size`` x ``size`` (square embedders want a
        fixed input). If ``None``, crops keep their native pixel size.

    Returns a list of 64 images in row-major order: index ``r*8 + c`` is rank ``8-r``,
    file ``chr(ord('a')+c)``.
    """
    if not 0.0 <= inset < 0.5:
        raise ValueError("inset must be in [0, 0.5)")
    h, w = board.shape[:2]
    sh, sw = h / GRID, w / GRID
    crops: list[np.ndarray] = []
    for r in range(GRID):
        for c in range(GRID):
            y0, y1 = r * sh, (r + 1) * sh
            x0, x1 = c * sw, (c + 1) * sw
            dy, dx = (y1 - y0) * inset, (x1 - x0) * inset
            sq = board[
                int(round(y0 + dy)) : int(round(y1 - dy)),
                int(round(x0 + dx)) : int(round(x1 - dx)),
            ]
            if size is not None:
                sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
            crops.append(sq)
    return crops


def _canonical_to_image(corners: np.ndarray) -> np.ndarray:
    """Homography mapping the canonical ``CANONICAL``² frame onto image ``corners``.

    ``corners`` is the playing surface in image pixels, order **a8, h8, h1, a1**
    (= TL,TR,BR,BL of the canonical frame, :data:`board_reader_core.sim.CORNER_ORDER`).
    """
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    src = np.array(
        [[0, 0], [CANONICAL, 0], [CANONICAL, CANONICAL], [0, CANONICAL]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, c)


def rectify_photo(image: np.ndarray, corners: np.ndarray, out_size: int = 512) -> np.ndarray:
    """Warp an original photo to the canonical top-down crop (server-side fallback).

    Deliberately mirrors the edge's rectify; duplicated here because the inference
    node must never import the edge package (layer separation) but the snapshot
    HTTP ingress accepts original photos + corners.
    """
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]], dtype=np.float32
    )
    return cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(c, dst), (out_size, out_size)
    )


def full_frame_corners(image: np.ndarray) -> np.ndarray:
    """Corners of a whole (already-rectified) image, a8,h8,h1,a1 order."""
    h, w = image.shape[:2]
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def photo_square_boxes(
    corners: np.ndarray, *, headroom: float = 1.6, side_pad: float = 0.15
) -> np.ndarray:
    """The 64 axis-aligned sampling boxes :func:`crop_squares_photo` cuts (rank 8→1, file a→h).

    For each square, its 4 base corners are projected into the image via the
    corner homography; the box is the projected quad's bounding box extended
    **upward** by ``headroom``× the square's projected width (piece tops point up
    in a normal photo) and sideways by ``side_pad``×. Exposed on its own (from the
    geometry :func:`crop_squares_photo` itself uses) so diagnostics can draw the
    exact boxes recognition sampled, without re-deriving the homography.

    ``corners``: playing-surface corners in image pixels, order **a8, h8, h1, a1**.
    Returns ``(64, 4, 2)`` float32 quads (TL, TR, BR, BL), **not** clipped to any
    image size — callers that slice pixels (like :func:`crop_squares_photo`) clip
    themselves; callers that only draw (like diagnostics) don't need to.
    """
    if headroom < 0 or side_pad < 0:
        raise ValueError("headroom and side_pad must be >= 0")
    H = _canonical_to_image(corners)
    s = CANONICAL / GRID
    # Project all 81 lattice points once; index [r, c] = canonical (c*s, r*s).
    lat = np.array(
        [[[c * s, r * s]] for r in range(GRID + 1) for c in range(GRID + 1)],
        dtype=np.float32,
    )
    pts = cv2.perspectiveTransform(lat, H).reshape(GRID + 1, GRID + 1, 2)
    boxes = np.empty((GRID * GRID, 4, 2), dtype=np.float32)
    for r in range(GRID):
        for c in range(GRID):
            quad = np.array(
                [pts[r, c], pts[r, c + 1], pts[r + 1, c + 1], pts[r + 1, c]],
                dtype=np.float32,
            )
            w = 0.5 * (
                float(np.linalg.norm(quad[1] - quad[0]))
                + float(np.linalg.norm(quad[2] - quad[3]))
            )
            x0 = float(quad[:, 0].min()) - side_pad * w
            x1 = float(quad[:, 0].max()) + side_pad * w
            y0 = float(quad[:, 1].min()) - headroom * w
            y1 = float(quad[:, 1].max())
            boxes[r * GRID + c] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return boxes


def crop_squares_photo(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    headroom: float = 1.6,
    side_pad: float = 0.15,
    size: int = 224,
) -> list[np.ndarray]:
    """Cut 64 *tall* per-square boxes out of the ORIGINAL photo (rank 8→1, file a→h).

    Each box (:func:`photo_square_boxes`) is clipped to the image, padded to a
    square (edge replication), and resized to ``size``².

    Compared to slicing the top-down warp (:func:`crop_squares`), the classifier
    sees the whole standing piece instead of a decapitated, smeared base — the
    Chesscog-style trick that recovers real-photo accuracy.

    ``corners``: playing-surface corners in image pixels, order **a8, h8, h1, a1**.
    """
    h_img, w_img = image.shape[:2]
    boxes = photo_square_boxes(corners, headroom=headroom, side_pad=side_pad)
    crops: list[np.ndarray] = []
    for box in boxes:
        x0, y0 = box[0]
        x1, y1 = box[2]
        xi0, yi0 = max(0, math.floor(x0)), max(0, math.floor(y0))
        xi1, yi1 = min(w_img, math.ceil(x1)), min(h_img, math.ceil(y1))
        patch = image[yi0:yi1, xi0:xi1]
        if patch.size == 0:  # corner off-frame; keep the slot with a blank
            patch = np.zeros(
                (8, 8, 3) if image.ndim == 3 else (8, 8), dtype=image.dtype
            )
        ph, pw = patch.shape[:2]
        if ph != pw:  # pad to square so the resize doesn't change aspect
            d = abs(ph - pw)
            top, bot = (0, 0) if ph > pw else (d // 2, d - d // 2)
            left, right = (d // 2, d - d // 2) if ph > pw else (0, 0)
            patch = cv2.copyMakeBorder(
                patch, top, bot, left, right, cv2.BORDER_REPLICATE
            )
        crops.append(cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA))
    return crops
