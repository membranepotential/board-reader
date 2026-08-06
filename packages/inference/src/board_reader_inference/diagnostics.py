"""Debug visualizations for ``/estimate?debug=1`` (opt-in, DESIGN §4.5).

Pure functions: given the photo/crop an estimate ran against, the per-square
sampling boxes, and the ``square_dist`` an estimate produced, render two
diagnostic images so an operator can see what recognition actually looked at.
HTTP-response-only — never part of the position event itself (DESIGN D1's
stateless-estimator contract is unaffected).
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from board_reader_core.protocol import CLASSES, GRID

# Debug image budgets: keep base64 payloads reasonable.
MAX_BOXES_SIDE = 1200
MOSAIC_CELL_WIDTH = 96
_LABEL_BAR_HEIGHT = 16

# Confidence -> color thresholds shared by both diagnostic images.
_GREEN = (0, 170, 0)
_YELLOW = (0, 200, 220)
_RED = (0, 0, 220)


def _confidence_color(p: float) -> tuple[int, int, int]:
    """BGR color for a per-square confidence: red (<0.5) -> yellow -> green (>=0.8)."""
    p = max(0.0, min(1.0, p))
    if p >= 0.8:
        return _GREEN
    if p <= 0.5:
        return _RED
    t = (p - 0.5) / 0.3  # 0..1 across the [0.5, 0.8] band
    lo, hi = (_RED, _YELLOW) if t < 0.5 else (_YELLOW, _GREEN)
    u = t / 0.5 if t < 0.5 else (t - 0.5) / 0.5
    return tuple(int(lo[i] + (hi[i] - lo[i]) * u) for i in range(3))


def _label_and_confidence(cell_dist: np.ndarray) -> tuple[str, float]:
    """A square's 13-class distribution -> (predicted-class letter, confidence)."""
    idx = int(np.argmax(cell_dist))
    cls = CLASSES[idx]
    # cv2.putText is ASCII-only (non-ASCII renders as "??"), so empty is ".".
    return ("." if cls == "empty" else cls), float(cell_dist[idx])


def draw_boxes(image: np.ndarray, boxes: np.ndarray, square_dist: np.ndarray) -> np.ndarray:
    """Draw the 64 per-square sampling ``boxes`` over ``image``, color-coded by confidence.

    ``boxes`` is ``(64, 4, 2)`` quads and ``square_dist`` is ``(8, 8, 13)``, both in
    rank 8→1, file a→h order. ``image`` is downscaled to a max
    :data:`MAX_BOXES_SIDE`-px long side first (the boxes are scaled to match) so
    debug payloads stay small; thin lines since the 64 boxes overlap on purpose
    (tall piece boxes).
    """
    h, w = image.shape[:2]
    scale = min(1.0, MAX_BOXES_SIDE / max(h, w))
    out = (
        cv2.resize(image, (round(w * scale), round(h * scale)))
        if scale < 1.0
        else image.copy()
    )
    flat = np.asarray(square_dist).reshape(GRID * GRID, -1)
    for i in range(GRID * GRID):
        quad = np.round(boxes[i] * scale).astype(np.int32)
        label, conf = _label_and_confidence(flat[i])
        color = _confidence_color(conf)
        cv2.polylines(out, [quad], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        base_x, base_y = quad[2]  # box's base corner (bottom, nearest the camera)
        cv2.putText(
            out, label, (int(base_x), int(base_y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    return out


def crop_mosaic(crops: list[np.ndarray], square_dist: np.ndarray) -> np.ndarray:
    """Tile the 64 embedder-input ``crops`` into an 8x8 grid, each labeled with its
    predicted class + confidence.

    Rank 8 row first, file a leftmost — matching ``square_dist`` order. Each cell
    is resized to :data:`MOSAIC_CELL_WIDTH` wide (keeping its crop's aspect) and
    topped with a small color-coded label bar (e.g. ``"N 0.94"``, ``". 0.99"``
    for empty).
    """
    flat = np.asarray(square_dist).reshape(GRID * GRID, -1)
    w = MOSAIC_CELL_WIDTH
    cells: list[np.ndarray] = []
    for i, crop in enumerate(crops):
        ch, cw = crop.shape[:2]
        h = max(1, round(ch * w / cw))
        cell = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        if cell.ndim == 2:
            cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
        label, conf = _label_and_confidence(flat[i])
        bar = np.zeros((_LABEL_BAR_HEIGHT, w, 3), dtype=np.uint8)
        cv2.putText(
            bar, f"{label} {conf:.2f}", (2, _LABEL_BAR_HEIGHT - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, _confidence_color(conf), 1, cv2.LINE_AA,
        )
        cells.append(np.vstack([bar, cell]))
    row_h = max(c.shape[0] for c in cells)
    padded = [
        cv2.copyMakeBorder(c, 0, row_h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        if c.shape[0] < row_h
        else c
        for c in cells
    ]
    rows = [
        np.hstack(padded[r * GRID : (r + 1) * GRID]) for r in range(GRID)
    ]
    return np.vstack(rows)


def to_b64_jpeg(image: np.ndarray, quality: int = 80) -> str:
    """Encode ``image`` (BGR) as a JPEG and base64-encode the bytes (no ``data:`` prefix)."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
