"""Auto-frame's thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class Tuning:
    """Auto-frame's thresholds. The defaults are placeholders from the auto-crop
    probe (docs/reports/2026-11-25-autocrop-probe.md: corners jitter 1-2 % of the
    board's width while the camera is still); retune them from the drift
    measurement."""

    #: The smallest R^2 of a preview's grid that is trusted. Mirrors
    #: ``recognition.locate.MIN_R2``; the root passes that value.
    min_r2: float = 0.90
    #: A preview is off the anchor when it is farther than this share of the
    #: anchor's width (:func:`displacement`).
    drift: float = 0.06
    #: Two off-anchor previews in a row within this share of the board's width of
    #: each other: the camera has stopped moving.
    settle: float = 0.03
    #: How many off-anchor, settled previews in a row mean the camera moved.
    drift_previews: int = 2
    #: A re-frame for drift waits at least this long after the previous request (s).
    min_reframe_interval: float = 10.0
    #: Retry delays after the 1st, 2nd, 3rd ... failed locate in a row (s); the
    #: last one repeats.
    backoff: tuple[float, ...] = (3.0, 10.0, 30.0)
    #: A full-frame request unanswered for this long has failed (s).
    full_frame_timeout: float = 15.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_r2 <= 1:
            raise ValueError("min_r2 must lie in [0, 1]")
        if self.drift <= 0 or self.settle <= 0:
            raise ValueError("drift and settle must be positive")
        if self.drift_previews < 1:
            raise ValueError("drift_previews must be at least 1")
        if self.min_reframe_interval < 0 or self.full_frame_timeout <= 0:
            raise ValueError("min_reframe_interval must be >= 0, full_frame_timeout > 0")
        delays = self.backoff
        if not delays or delays[0] <= 0 or any(a > b for a, b in pairwise(delays)):
            raise ValueError("backoff must be positive, non-empty and non-decreasing")

    def retry_delay(self, failures: int) -> float:
        """The wait after ``failures`` (>= 1) failed locates in a row."""
        return self.backoff[min(failures, len(self.backoff)) - 1]


TUNING = Tuning()
