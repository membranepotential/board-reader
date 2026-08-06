"""Automatic board detection on real photos (DESIGN.md WP-A1, D4).

Locates the chess board's **playing surface** in an arbitrary photo and returns its
outer quadrilateral. Per DESIGN D4 / decision 6, detection *runs on the edge* in the
live pipe; the function lives in core so the inference node can also run it for the
phone-snapshot ingress, where the "edge" is a phone browser that cannot run OpenCV
cheaply. Core stays dependency-light: numpy + cv2 only.

Algorithm (classic CV, no ML, piece-occlusion tolerant):

1. **Inner-corner grid.** ``cv2.findChessboardCornersSBWithMeta`` with
   ``CALIB_CB_LARGER`` finds a sub-grid of the board's interior x-corners — pieces
   occlude some of the 7x7, but any (>=5x5)-ish sub-grid is enough. A small retry
   ladder (downscale sizes x optional CLAHE) covers dim/low-contrast photos where
   the first pass finds nothing. The grid may also come back *larger* than 7 wide
   when the board's outer border happens to produce lattice-spaced intersections;
   those supersets are handled by the phase search below.
2. **Lattice homography + phase.** A homography maps the found sub-grid to integer
   lattice coordinates. Where the sub-grid sits inside the true 7x7 interior
   lattice is ambiguous when rows/columns are missing (or extra). Each candidate
   offset is scored by the 8x8 light/dark **alternation** of the implied squares
   *plus* **x-corner evidence at the hypothesised missing corners** — the true
   offset predicts real (if faint) corners there, a wrong offset predicts them on
   the uniform border wood. Aggregate alternation alone cannot separate an
   exactly-one-square shift; the missing-corner evidence can.
3. **Rectified re-snap.** The image is warped flat with the coarse quad; all 49
   interior corners are then at *known* grid positions, so ``cornerSubPix`` snaps
   them precisely. Points that drifted (occluded by a piece) or lack an x-corner
   contrast signature are rejected; a RANSAC homography over the survivors gives
   the refined board frame. Falls back to the coarse quad if too few survive.
4. **Validation.** The final quad must show a *consistent* light/dark alternation
   in **every** row and column of its 8x8 surface (ring-median square luminance,
   so pieces standing on squares don't spoil it). A quad that is off by one
   square, or latched onto a false lattice, fails this and the next ladder rung /
   offset hypothesis is tried; if none passes, ``None`` is returned.

Orientation is **geometric only** (TL,TR,BR,BL by the sum/diff rule) — which corner
is a8 cannot be known from geometry (decision #15); the caller/operator resolves it.
"""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

# Interior x-corners of an 8x8 board form a 7x7 lattice.
_INNER = 7
# Minimum sub-grid pattern findChessboardCornersSB grows from.
_SEED_PATTERN = (5, 5)
# Largest grid dimension we still interpret (superset grids include border rows).
_MAX_GRID = 9
# Rectified-space square size (px) for the re-snap pass.
_SQ = 100
# Minimum aggregate light/dark alternation (gray levels) for a phase hypothesis.
_MIN_ALTERNATION = 4.0
# Minimum per-row/column ring-median alternation for final acceptance (gray
# levels). Correct quads measure ~25-40 on real photos; one-square-shifted or
# false-lattice quads measure < 3.
_MIN_ROWCOL_ALTERNATION = 8.0
# Retry ladder: (downscale max dimension, apply CLAHE).
_LADDER: tuple[tuple[int, bool], ...] = (
    (1200, False),
    (1000, False),
    (1600, False),
    (1200, True),
    (1000, True),
)
# Phase hypotheses tried per ladder rung (best-scored first).
_MAX_PHASES = 2


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL,TR,BR,BL by the classic sum/difference rule."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()  # y - x
    return np.array(
        [p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]],
        dtype=np.float32,
    )


def _alternation(gray: np.ndarray, h_inv: np.ndarray, x0: float, y0: float) -> float:
    """|mean(light) - mean(dark)| over the 8x8 squares whose inner-corner lattice
    starts at ``(x0, y0)``; -1 if any square center falls outside the image."""
    cs, rs = np.meshgrid(np.arange(8), np.arange(8))
    centers = np.stack([x0 - 1 + cs + 0.5, y0 - 1 + rs + 0.5], axis=-1)
    proj = cv2.perspectiveTransform(
        centers.reshape(-1, 1, 2).astype(np.float32), h_inv
    ).reshape(8, 8, 2)
    h, w = gray.shape
    means = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            x, y = int(round(proj[r, c, 0])), int(round(proj[r, c, 1]))
            if not (2 <= x < w - 2 and 2 <= y < h - 2):
                return -1.0
            means[r, c] = gray[y - 2 : y + 3, x - 2 : x + 3].mean()
    a = np.concatenate([means[::2, ::2].ravel(), means[1::2, 1::2].ravel()])
    b = np.concatenate([means[::2, 1::2].ravel(), means[1::2, ::2].ravel()])
    return float(abs(a.mean() - b.mean()))


def _xcorner_response(gray_f: np.ndarray, p: np.ndarray, r: int) -> float:
    """Quadrant-contrast x-corner-ness at point ``p`` (opposite quadrants alike,
    adjacent quadrants different)."""
    x, y = int(round(float(p[0]))), int(round(float(p[1])))
    h, w = gray_f.shape
    if x < r or y < r or x >= w - r or y >= h - r:
        return 0.0
    tl = gray_f[y - r : y, x - r : x].mean()
    tr = gray_f[y - r : y, x : x + r].mean()
    bl = gray_f[y : y + r, x - r : x].mean()
    br = gray_f[y : y + r, x : x + r].mean()
    return float(abs(tl + br - tr - bl) - 0.5 * (abs(tl - br) + abs(tr - bl)))


def _phase_scores(
    gray: np.ndarray, h_inv: np.ndarray, gh: int, gw: int, found: set[tuple[int, int]]
) -> list[tuple[float, int, int]]:
    """Score every candidate phase (dx, dy) of the found sub-grid inside the true
    7x7 interior lattice; return (score, dx, dy) tuples, best first.

    Interior corner (i, j) sits at found-lattice coordinate (i - dx, j - dy);
    superset grids (gw or gh > 7) make the offset range extend below zero.
    """
    gray_f = gray.astype(np.float32)
    unit = float(
        np.linalg.norm(
            (
                cv2.perspectiveTransform(np.array([[[1.0, 0.0]]], np.float32), h_inv)
                - cv2.perspectiveTransform(np.array([[[0.0, 0.0]]], np.float32), h_inv)
            ).ravel()
        )
    )
    radius = max(4, int(unit * 0.12))
    out: list[tuple[float, int, int]] = []
    for dy in range(min(0, _INNER - gh), max(0, _INNER - gh) + 1):
        for dx in range(min(0, _INNER - gw), max(0, _INNER - gw) + 1):
            alt = _alternation(gray, h_inv, -dx, -dy)
            if alt <= _MIN_ALTERNATION:
                continue
            missing = [
                (i - dx, j - dy)
                for j in range(_INNER)
                for i in range(_INNER)
                if (i - dx, j - dy) not in found
            ]
            evidence = 0.0
            if missing:
                pts = cv2.perspectiveTransform(
                    np.array(missing, np.float32).reshape(-1, 1, 2), h_inv
                ).reshape(-1, 2)
                evidence = float(
                    np.mean([_xcorner_response(gray_f, p, radius) for p in pts])
                )
            out.append((alt + 0.5 * evidence, dx, dy))
    out.sort(reverse=True)
    return out


def _coarse_candidates(image: np.ndarray) -> Iterator[np.ndarray]:
    """Yield coarse outer-quad hypotheses (full-resolution coords), best first,
    across the preprocessing retry ladder."""
    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_LARGER
    )
    for max_dim, use_clahe in _LADDER:
        scale = min(1.0, max_dim / max(image.shape[:2]))
        small = (
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else image
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        if use_clahe:
            gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        ret, corners, meta = cv2.findChessboardCornersSBWithMeta(
            gray, _SEED_PATTERN, flags
        )
        if not ret or corners is None or len(corners) < 15:
            continue
        gh, gw = meta.shape[:2]
        if gh > _MAX_GRID or gw > _MAX_GRID:
            continue
        pts = corners.reshape(-1, 2).astype(np.float32)
        lattice = np.array([[c, r] for r in range(gh) for c in range(gw)], np.float32)
        if len(pts) >= 20:
            # RANSAC so superset rows (border intersections slightly off-lattice)
            # don't skew the fit.
            H, _ = cv2.findHomography(pts, lattice, cv2.RANSAC, 0.08)
        else:
            H, _ = cv2.findHomography(pts, lattice, 0)
        if H is None:
            continue
        h_inv = np.linalg.inv(H).astype(np.float32)
        found = set(map(tuple, np.round(lattice).astype(int)))
        for _score, dx, dy in _phase_scores(gray, h_inv, gh, gw, found)[:_MAX_PHASES]:
            x0, y0 = -1 - dx, -1 - dy
            ext = np.array(
                [[x0, y0], [x0 + 8, y0], [x0 + 8, y0 + 8], [x0, y0 + 8]], np.float32
            )
            quad = cv2.perspectiveTransform(ext.reshape(-1, 1, 2), h_inv).reshape(-1, 2)
            yield quad / scale


def _refine_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp flat, snap all 49 interior corners at their known rectified positions,
    refit. Returns the refined quad, or ``quad`` unchanged."""
    margin = _SQ  # one full square of context around the surface
    size = 8 * _SQ
    dst = np.array(
        [[margin, margin], [size + margin, margin],
         [size + margin, size + margin], [margin, size + margin]],
        np.float32,
    )
    Hr = cv2.getPerspectiveTransform(_order_quad(quad), dst)
    rect = cv2.warpPerspective(image, Hr, (size + 2 * margin, size + 2 * margin))
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY) if rect.ndim == 3 else rect
    grid = np.array(
        [[margin + _SQ * (c + 1), margin + _SQ * (r + 1)]
         for r in range(_INNER) for c in range(_INNER)],
        np.float32,
    )
    snapped = cv2.cornerSubPix(
        gray,
        grid.reshape(-1, 1, 2).copy(),
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
    ).reshape(-1, 2)
    moved = np.linalg.norm(snapped - grid, axis=1)
    gray_f = gray.astype(np.float32)
    resp = np.array([_xcorner_response(gray_f, p, r=10) for p in snapped])
    lattice = np.array(
        [[c, r] for r in range(_INNER) for c in range(_INNER)], np.float32
    )
    accept = (moved < 0.2 * _SQ) & (resp > 8.0)
    if int(accept.sum()) < 12:
        return quad
    H2, _ = cv2.findHomography(snapped[accept], lattice[accept], cv2.RANSAC, 3.0)
    if H2 is None:
        return quad
    proj = cv2.perspectiveTransform(
        snapped[accept].reshape(-1, 1, 2), H2
    ).reshape(-1, 2)
    if int((np.linalg.norm(proj - lattice[accept], axis=1) < 0.05).sum()) < 12:
        return quad
    ext = np.array([[-1, -1], [7, -1], [7, 7], [-1, 7]], np.float32)
    q_rect = cv2.perspectiveTransform(
        ext.reshape(-1, 1, 2), np.linalg.inv(H2).astype(np.float32)
    )
    return cv2.perspectiveTransform(
        q_rect, np.linalg.inv(Hr).astype(np.float32)
    ).reshape(-1, 2)


def _validate(image: np.ndarray, quad: np.ndarray) -> bool:
    """Accept only quads whose 8x8 surface alternates consistently in *every* row
    and column (ring-median luminance per square, robust to pieces)."""
    if not cv2.isContourConvex(quad.astype(np.int32)):
        return False
    h, w = image.shape[:2]
    area = cv2.contourArea(quad)
    if not (0.03 * w * h < area < 1.2 * w * h):
        return False
    sq = 64
    dst = np.array([[0, 0], [8 * sq, 0], [8 * sq, 8 * sq], [0, 8 * sq]], np.float32)
    Hr = cv2.getPerspectiveTransform(_order_quad(quad), dst)
    rect = cv2.warpPerspective(image, Hr, (8 * sq, 8 * sq))
    gray = (
        cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY) if rect.ndim == 3 else rect
    ).astype(np.float32)
    med = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            s = gray[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq]
            ring = np.concatenate(
                [
                    s[4:12, 4:-4].ravel(),
                    s[-12:-4, 4:-4].ravel(),
                    s[12:-12, 4:12].ravel(),
                    s[12:-12, -12:-4].ravel(),
                ]
            )
            med[r, c] = np.median(ring)
    sgn = np.array(
        [[1 if (r + c) % 2 == 0 else -1 for c in range(8)] for r in range(8)]
    )
    signed = med * sgn
    d = np.concatenate([signed.mean(axis=1), signed.mean(axis=0)])
    if np.abs(d).min() < _MIN_ROWCOL_ALTERNATION:
        return False
    return bool((np.sign(d) == np.sign(np.median(d))).all())


def detect_board_quad(image: np.ndarray) -> np.ndarray | None:
    """Locate the chess board playing surface in a real photo.

    Returns 4x2 float32 corners in TL,TR,BR,BL *geometric* order (orientation
    unknown — which corner is a8 is the caller's/operator's concern), or ``None``
    if no board is found.
    """
    for coarse in _coarse_candidates(image):
        quad = _order_quad(_refine_quad(image, coarse))
        if _validate(image, quad):
            return quad
    return None
