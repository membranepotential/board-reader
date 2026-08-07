"""Board detection & rectification on the edge (DESIGN.md WP-A1/A2/A3, §4.4.1).

The edge's job before forwarding is to turn whatever the camera sees — a board at
the angled broadcast view (DESIGN D8) — into a **canonical top-down crop** in a
fixed orientation, so the inference node can crop 64 squares from a known frame and
never has to reason about geometry. This module owns that:

Detection itself is **not** implemented here — it is
:func:`board_reader_core.boarddetect.detect_board_quad`, shared with the inference
node's ``/detect`` so the browser edge and a real Pi edge locate boards identically.
(An earlier Canny -> contours -> largest-convex-quad detector lived here; it missed
real photos this one finds, and having two detectors meant the streaming path was
quietly worse than the snapshot path.)

* :func:`homography_from_corners` / :func:`rectify` — map a board quad to the
  canonical square crop.
* :class:`Rectifier` — stateful per-board helper: caches the homography, accepts
  manual corners, and reports drift status (``ok`` / ``recalibrating`` /
  ``needs_setup``) per the camera-movement model in DESIGN §4.4.1.

Corner order is **TL, TR, BR, BL** throughout. For the canonical ``a1_bottom_left``
frame (rank 8 top, file a left, viewed from White's side) that is **a8, h8, h1, a1**
— matching :data:`board_reader_core.sim.CORNER_ORDER`.

Orientation note (DESIGN decision #15): detection recovers *geometry* only; the
board's 4-fold rotational ambiguity (which physical corner is a1) is **not** inferred
from the pieces. The automatic path assumes the configured default view; the operator
declares the true mapping by passing corners to :meth:`set_corners` in a8,h8,h1,a1
order. Auto-orientation from board-intrinsic cues (DESIGN §4.4.1 b.1) is later work.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from board_reader_core.boarddetect import detect_board_quad

# Status values this module can report; a subset of protocol.VALID_STATUS. The edge
# never emits a position event itself, but it tags each crop so the forwarder (and,
# later, status events on the wire) can act on a bump/reposition.
STATUS_OK = "ok"
STATUS_RECALIBRATING = "recalibrating"
STATUS_NEEDS_SETUP = "needs_setup"


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points into TL, TR, BR, BL by the classic sum/difference trick.

    TL has the smallest x+y, BR the largest; TR has the smallest y-x, BL the largest.
    Robust to any input ordering as long as the points form a convex quad.
    """
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if p.shape != (4, 2):
        raise ValueError("need exactly 4 points")
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()  # y - x
    return np.array(
        [p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]],
        dtype=np.float32,
    )


def homography_from_corners(corners: np.ndarray, out_size: int) -> np.ndarray:
    """3x3 homography mapping the board ``corners`` (TL,TR,BR,BL) to a square crop."""
    dst = np.array(
        [[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), dst)


def rectify(image: np.ndarray, corners: np.ndarray, out_size: int = 384) -> np.ndarray:
    """Warp ``image`` so the board ``corners`` fill a canonical ``out_size`` crop."""
    h = homography_from_corners(corners, out_size)
    return cv2.warpPerspective(image, h, (out_size, out_size))


@dataclass
class RectifyResult:
    """Outcome of :meth:`Rectifier.process` for one frame.

    ``crop`` is the canonical top-down board (``None`` when suppressed — i.e. while
    recalibrating or when setup is needed). ``corners`` is the board outline used
    (image space, TL,TR,BR,BL), or ``None`` if the board was not located.
    """

    crop: np.ndarray | None
    status: str
    corners: np.ndarray | None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and self.crop is not None


class Rectifier:
    """Stateful board localiser + rectifier for one board (DESIGN §4.4.1).

    Two modes:

    * **Manual** — the operator taps the four corners once (``set_corners`` or the
      ``manual_corners=`` ctor arg). This declares both geometry *and* orientation,
      so every frame is rectified directly against the cached homography. This is the
      primary sim path.
    * **Auto** — no corners supplied; each frame is detected with
      :func:`board_reader_core.boarddetect.detect_board_quad`. The first successful
      detection locks the cache; later frames re-detect and the corner displacement
      vs. the cache classifies the situation: small -> ``ok``; large (a bump) ->
      ``recalibrating`` with the crop **suppressed** for that frame (we re-lock to the
      new geometry so the next stable frame returns ``ok``); board not locatable ->
      ``needs_setup``.

    Suppression during the transient is deliberate (DESIGN §4.4.1): a mis-localized
    crop yields confidently-wrong squares, which is worse for downstream smoothing
    than an honest gap.

    **Detection cadence.** Detection costs ~90-100 ms on a 1200x1600 photo — payable
    at the at-rest keepalive rate, unpayable at every frame of a 10-15 fps burst
    (D4). So auto mode re-detects at most every ``redetect_interval_s`` of *capture
    time*; frames in between are rectified against the cached homography, which is
    what D4's "re-runs only on ROI drift" asks for. The cadence is driven by the
    ``ts`` passed to :meth:`process` (the frame's ``capture_ts``, DESIGN D5) — a
    caller that passes no ``ts`` gets the old per-frame detection, and
    ``redetect_interval_s=0`` forces it too. A board that has never been located
    is always detected, cadence or not: there is no cache to fall back on.
    """

    def __init__(
        self,
        out_size: int = 384,
        *,
        manual_corners: np.ndarray | None = None,
        drift_tol: float = 0.04,
        redetect_interval_s: float = 1.0,
    ) -> None:
        if redetect_interval_s < 0:
            raise ValueError("redetect_interval_s must be non-negative")
        self.out_size = out_size
        self.drift_tol = drift_tol
        self.redetect_interval_s = redetect_interval_s
        self._corners: np.ndarray | None = None
        self._manual = False
        self._last_detect_ts: float | None = None
        if manual_corners is not None:
            self.set_corners(manual_corners)

    @property
    def locked(self) -> bool:
        """Whether a homography is currently cached."""
        return self._corners is not None

    @property
    def corners(self) -> np.ndarray | None:
        return None if self._corners is None else self._corners.copy()

    def set_corners(self, corners: np.ndarray) -> None:
        """Operator corner-tap: declare the board frame (geometry + orientation)."""
        self._corners = order_quad(corners)
        self._manual = True

    def _displacement(self, a: np.ndarray, b: np.ndarray) -> float:
        """Mean per-corner shift between two quads, normalised by frame width."""
        return float(np.linalg.norm(a - b, axis=1).mean())

    def _detect_due(self, ts: float | None) -> bool:
        """Should this frame pay for a detection, or ride the cached homography?

        Only a locked board may skip: without a cache there is nothing to rectify
        against. A ``ts`` that moved backwards (clock step, D5) re-anchors the
        cadence rather than stalling it.
        """
        if self._corners is None or self.redetect_interval_s <= 0:
            return True
        if ts is None or self._last_detect_ts is None:
            return True
        elapsed = ts - self._last_detect_ts
        return not 0.0 <= elapsed < self.redetect_interval_s

    def process(self, image: np.ndarray, ts: float | None = None) -> RectifyResult:
        """Rectify one frame; ``ts`` is its ``capture_ts`` and drives the cadence."""
        width = float(image.shape[1])
        if self._manual:
            # Operator-declared frame: trust it, rectify directly every frame.
            assert self._corners is not None
            return RectifyResult(
                rectify(image, self._corners, self.out_size), STATUS_OK, self._corners
            )

        if not self._detect_due(ts):
            # Between cadence ticks: the cached geometry stands (D4).
            assert self._corners is not None
            return RectifyResult(
                rectify(image, self._corners, self.out_size), STATUS_OK, self._corners
            )

        detected = detect_board_quad(image)
        if detected is None:
            # Lost the board (or never had it). Needs operator setup either way.
            # The cadence is deliberately *not* advanced: a board we cannot see is
            # re-detected every frame until it comes back, so a lost board reads as
            # a continuous gap rather than a second of crops off stale geometry.
            return RectifyResult(None, STATUS_NEEDS_SETUP, None)

        self._last_detect_ts = ts
        if self._corners is None:
            self._corners = detected  # first lock
            return RectifyResult(
                rectify(image, detected, self.out_size), STATUS_OK, detected
            )

        drift = self._displacement(detected, self._corners) / width
        self._corners = detected  # auto re-lock on current geometry (bump recovery)
        if drift > self.drift_tol:
            # A bump: geometry moved. Suppress this frame; re-locked for the next.
            return RectifyResult(None, STATUS_RECALIBRATING, detected)
        return RectifyResult(rectify(image, detected, self.out_size), STATUS_OK, detected)
