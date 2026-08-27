"""Motion meter -> stability score on the edge (DESIGN.md D3, WP-C2).

Every frame the edge forwards carries ``stability`` in [0,1] (1.0 = at rest). Per
DESIGN D3 it is an **advisory hint** the consumer may ignore, so it must be cheap
above all: we downscale to a tiny grayscale thumbnail and measure the mean absolute
inter-frame difference. No optical flow, no background model, no per-square work.

The mapping from "pixels changed" to "how still is the board" is two steps:

* a **gain** curve ``exp(-gain * d)`` turns the mean abs difference ``d`` into an
  instantaneous stability — smooth, saturating, and zero-cost;
* an **asymmetric smoother** tracks that instant value with a fast attack and a
  slower release, so a hand entering frame drops stability immediately (the
  forwarder must burst *now*, DESIGN D4) while the recovery back to rest takes a
  few frames rather than declaring "at rest" between two strides of one move.

Deterministic and clock-free: the meter sees only pixels, so offline replay of a
clip scores identically to the same clip live.
"""

from __future__ import annotations

import cv2
import numpy as np

FLOOR_NEAR = 4.0
FLOOR_FAR_DIVISOR = 5.0


class MotionMeter:
    """Low-res inter-frame differencing -> stability in [0,1], 1.0 = at rest.

    Parameters
    ----------
    downscale:
        Side length of the square grayscale thumbnail each frame is reduced to.
        32x32 keeps a piece-sized move well above the sensor-noise floor while
        costing a single ``cv2.resize``.
    gain:
        Steepness of the (difference / learned floor) -> stability curve, in RATIO
        units — the meter scores how far above its own resting difference a frame
        sits, not an absolute brightness change.
    attack / release:
        Smoothing weights for a falling and a rising stability respectively. Attack
        is deliberately the faster of the two (see module docstring).
    tolerance:
        Ratios up to this score a full 1.0 — the resting scatter band. Motion has
        to exceed ``tolerance x`` the learned floor before stability drops at all.
    floor_up / floor_down:
        Chase rates of the learned floor when a frame differences above / below it.
        Down is the faster (finds genuine rest quickly); up is slow enough to ride
        out a move without learning it as the new rest.
    noise_floor:
        The floor's absolute lower bound and its seed — the meter never learns a
        resting difference below plain sensor noise, and it starts there rather
        than from the first observed diff so a camera opened onto motion cannot
        learn motion as its floor.

    The adaptive floor replaced a fixed one on 2026-08-27: on the deployment phone
    the resting difference measured ~6x the assumed constant, stability lived
    under 0.35 all day, and everything thresholded on it was dead. Values tuned by
    replaying the three recorded takes (see the mirrored TS meter's constants).
    """

    def __init__(
        self,
        *,
        downscale: int = 32,
        gain: float = 1.5,
        attack: float = 0.7,
        release: float = 0.4,
        tolerance: float = 2.0,
        floor_up: float = 0.02,
        floor_down: float = 0.08,
        noise_floor: float = 0.0015,
    ) -> None:
        if downscale < 2:
            raise ValueError("downscale must be >= 2")
        if not 0.0 < attack <= 1.0 or not 0.0 < release <= 1.0:
            raise ValueError("attack and release must be in (0, 1]")
        self.downscale = downscale
        self.gain = gain
        self.attack = attack
        self.release = release
        self.tolerance = tolerance
        self.floor_up = floor_up
        self.floor_down = floor_down
        self.noise_floor = noise_floor
        self._prev: np.ndarray | None = None
        self._floor: float | None = None
        self._stability = 1.0

    @property
    def stability(self) -> float:
        """The current smoothed stability, without consuming a frame."""
        return self._stability

    def _thumbnail(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        small = cv2.resize(
            gray, (self.downscale, self.downscale), interpolation=cv2.INTER_AREA
        )
        return small.astype(np.float32) / 255.0

    def update(self, image: np.ndarray) -> float:
        """Fold one captured frame in and return the stability to stamp on it.

        The first frame has nothing to difference against and scores 1.0 — an
        unknown board is reported at rest, matching D3's "``stability`` is a hint,
        never a gate". Frames are compared as fixed-size thumbnails, so a capture
        resolution change costs one noisy reading rather than an exception.
        """
        thumb = self._thumbnail(image)
        prev, self._prev = self._prev, thumb
        if prev is None:
            self._stability = 1.0
            return self._stability

        diff = float(np.abs(thumb - prev).mean())
        if self._floor is None:
            self._floor = self.noise_floor
        # The upward chase is gated (mirrors the TS meter's FLOOR_NEAR): a frame
        # far above the floor is a clear move and teaches it at a fifth of the
        # rate, so multi-second continuous motion is not adopted as rest.
        if diff < self._floor:
            step = self.floor_down
        elif diff / self._floor <= FLOOR_NEAR:
            step = self.floor_up
        else:
            step = self.floor_up / FLOOR_FAR_DIVISOR
        self._floor = max(self.noise_floor, self._floor + step * (diff - self._floor))
        instant = float(np.exp(-self.gain * max(diff / self._floor - self.tolerance, 0.0)))
        weight = self.attack if instant < self._stability else self.release
        self._stability = float(
            np.clip((1.0 - weight) * self._stability + weight * instant, 0.0, 1.0)
        )
        return self._stability

    def reset(self) -> None:
        """Forget the reference frame — the next ``update`` scores 1.0 again."""
        self._prev = None
        self._floor = None
        self._stability = 1.0
