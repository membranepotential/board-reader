"""Auto-frame's thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class Tuning:
    """Auto-frame's thresholds. The drift rule and the preview limits are the ones
    measured on the phone's preview (docs/reports/2026-11-25-autocrop-probe.md,
    "Follow-up: the phone's preview"): 0 false fires on 421 still previews, every
    move of 6 % or more of the board's width caught within 3 previews."""

    #: The smallest R^2 of a preview's grid that counts for drift. Stricter than the
    #: full frame's (``recognition.locate.MIN_R2``, 0.90), because most still jumps
    #: over 5 % came from grids at R^2 0.92-0.96 pulled by a hand.
    preview_min_r2: float = 0.97
    #: A counted preview is off the anchor when it is farther than this share of the
    #: anchor's width (:func:`~.geometry.displacement`): the still previews' p99.
    drift: float = 0.05
    #: How many counted previews in a row off the anchor mean the camera moved. A
    #: preview that does not count neither adds to the run nor breaks it.
    drift_previews: int = 2
    #: This many previews in a row that do not count (a person in view, a camera
    #: knocked off the board) say "board not visible"; they never re-frame.
    unseen_previews: int = 10
    #: At most one preview is located per this many seconds: each locate is a
    #: ChessQ read (~195 ms on the GPU), and the phone sends one every 1.5 s.
    preview_every: float = 3.0
    #: Below this share of the frame's width the board is too small to locate on a
    #: 480 px preview (66 % found at 35 %, 23 % at 25 %): previews are not watched.
    min_preview_share: float = 0.35
    #: A re-frame for drift waits at least this long after the previous request (s).
    min_reframe_interval: float = 10.0
    #: Retry delays after the 1st, 2nd, 3rd ... failed locate in a row (s); the
    #: last one repeats.
    backoff: tuple[float, ...] = (3.0, 10.0, 30.0)
    #: A full-frame request unanswered for this long has failed (s).
    full_frame_timeout: float = 15.0

    def __post_init__(self) -> None:
        if not 0 <= self.preview_min_r2 <= 1:
            raise ValueError("preview_min_r2 must lie in [0, 1]")
        if not 0 <= self.min_preview_share <= 1:
            raise ValueError("min_preview_share must lie in [0, 1]")
        if self.drift <= 0:
            raise ValueError("drift must be positive")
        if self.drift_previews < 1 or self.unseen_previews < 1:
            raise ValueError("drift_previews and unseen_previews must be at least 1")
        if self.preview_every < 0 or self.min_reframe_interval < 0:
            raise ValueError("preview_every and min_reframe_interval must be >= 0")
        if self.full_frame_timeout <= 0:
            raise ValueError("full_frame_timeout must be positive")
        delays = self.backoff
        if not delays or delays[0] <= 0 or any(a > b for a, b in pairwise(delays)):
            raise ValueError("backoff must be positive, non-empty and non-decreasing")

    def retry_delay(self, failures: int) -> float:
        """The wait after ``failures`` (>= 1) failed locates in a row."""
        return self.backoff[min(failures, len(self.backoff)) - 1]


TUNING = Tuning()
