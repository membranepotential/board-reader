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


class MotionMeter:
    """Low-res inter-frame differencing -> stability in [0,1], 1.0 = at rest.

    Parameters
    ----------
    downscale:
        Side length of the square grayscale thumbnail each frame is reduced to.
        32x32 keeps a piece-sized move well above the sensor-noise floor while
        costing a single ``cv2.resize``.
    gain:
        Steepness of the difference -> stability curve. At the default, a 1% mean
        brightness change over the thumbnail scores ~0.6 and a 3% change ~0.1.
        Area-averaging into the thumbnail already damps small motion considerably,
        so the curve has to be steep to keep a real piece move above the noise.
    attack / release:
        Smoothing weights for a falling and a rising stability respectively. Attack
        is deliberately the faster of the two (see module docstring).
    noise_floor:
        Mean absolute difference (in [0,1] intensity units) treated as sensor noise
        and subtracted before the gain curve, so a static board reads exactly 1.0
        rather than drifting just under it.
    """

    def __init__(
        self,
        *,
        downscale: int = 32,
        gain: float = 80.0,
        attack: float = 0.7,
        release: float = 0.4,
        noise_floor: float = 0.004,
    ) -> None:
        if downscale < 2:
            raise ValueError("downscale must be >= 2")
        if not 0.0 < attack <= 1.0 or not 0.0 < release <= 1.0:
            raise ValueError("attack and release must be in (0, 1]")
        self.downscale = downscale
        self.gain = gain
        self.attack = attack
        self.release = release
        self.noise_floor = noise_floor
        self._prev: np.ndarray | None = None
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
        instant = float(np.exp(-self.gain * max(diff - self.noise_floor, 0.0)))
        weight = self.attack if instant < self._stability else self.release
        self._stability = float(
            np.clip((1.0 - weight) * self._stability + weight * instant, 0.0, 1.0)
        )
        return self._stability

    def reset(self) -> None:
        """Forget the reference frame — the next ``update`` scores 1.0 again."""
        self._prev = None
        self._stability = 1.0
