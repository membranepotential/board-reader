"""Adaptive frame forwarding on the edge (DESIGN.md D4, WP-C3).

The camera runs at full rate; the wire does not. D4 replaces the old "only send one
settled frame" gate with two rates:

* **at rest** — a low keepalive (~1-2 fps) so the consumer keeps seeing the board;
* **during motion** — a **burst** (10-15 fps) so the whole transition is streamed
  and the application layer can segment the move itself (D3).

Motion is measured by :class:`~board_reader_edge.motion.MotionMeter`, and the
at-rest <-> burst transition is hysteretic: entry is eager (one dip below
``enter_burst``), exit is patient (sustained calm above ``exit_burst`` for
``exit_hold_s``). A single noisy frame therefore cannot flap the state, and the
pause between two strides of the same move does not split one burst into two.

**Time base.** Every rate decision is made against ``frame.capture_ts`` (DESIGN D5),
never a wall clock: replaying a recorded clip flat out must forward exactly the
frames the live edge would have. There is deliberately no ``time`` import here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Iterator

from board_reader_core.wire import Frame

from board_reader_edge.motion import MotionMeter

# Slack on the capture-time due comparison, so a frame that lands exactly on its
# due time is not dropped by float rounding. Microseconds are far below any rate
# we forward at.
_DUE_EPS = 1e-6


@dataclass(frozen=True)
class ForwarderConfig:
    """Rates and hysteresis thresholds for :class:`AdaptiveForwarder` (DESIGN D4)."""

    keepalive_fps: float = 1.0  # at-rest forward rate
    burst_fps: float = 12.0  # during-motion forward rate
    enter_burst: float = 0.7  # stability below this -> enter burst
    exit_burst: float = 0.85  # stability above this...
    exit_hold_s: float = 0.7  # ...for this long (capture-time) -> back to at rest

    def __post_init__(self) -> None:
        if self.keepalive_fps <= 0 or self.burst_fps <= 0:
            raise ValueError("forward rates must be positive")
        if not self.enter_burst <= self.exit_burst:
            raise ValueError("enter_burst must not exceed exit_burst (hysteresis)")
        if self.exit_hold_s < 0:
            raise ValueError("exit_hold_s must be non-negative")


class AdaptiveForwarder:
    """Keepalive + burst frame forwarding for one board (DESIGN D4, WP-C3).

    Feed it **every** captured frame — the motion meter needs the full stream to
    difference against, and dropping frames upstream would make stability lie.
    :meth:`decide` returns the frame to forward (stamped with ``stability`` and,
    inside a burst, ``burst_id``) or ``None`` to drop it.
    """

    def __init__(
        self,
        config: ForwarderConfig = ForwarderConfig(),
        *,
        meter: MotionMeter | None = None,
    ) -> None:
        self.config = config
        self.meter = meter if meter is not None else MotionMeter()
        self._burst_id: str | None = None
        self._calm_since: float | None = None  # capture_ts calm began, in a burst
        self._last_forward_ts: float | None = None

    @property
    def bursting(self) -> bool:
        return self._burst_id is not None

    @property
    def burst_id(self) -> str | None:
        """Id of the burst in progress, or ``None`` at rest."""
        return self._burst_id

    def _track_burst(self, frame: Frame, stability: float) -> None:
        """Apply the D4 hysteresis to the at-rest <-> burst state for one frame."""
        if self._burst_id is None:
            if stability < self.config.enter_burst:
                # Eager entry: the transition starts *now*, so the burst is named
                # after the frame that detected it and every frame of the burst
                # carries that id.
                self._burst_id = f"{frame.board_id}:{frame.seq}"
                self._calm_since = None
            return

        if stability <= self.config.exit_burst:
            self._calm_since = None  # still moving; the hold restarts on next calm
        elif self._calm_since is None:
            self._calm_since = frame.capture_ts
        elif frame.capture_ts - self._calm_since >= self.config.exit_hold_s - _DUE_EPS:
            self._burst_id = None
            self._calm_since = None

    def _due(self, capture_ts: float) -> bool:
        """Has the current forward period elapsed since the last forwarded frame?

        The period is read at decision time, so entering a burst raises the rate
        immediately instead of waiting out the keepalive interval already pending.
        """
        if self._last_forward_ts is None:
            return True  # the first frame ever is always forwarded
        elapsed = capture_ts - self._last_forward_ts
        if elapsed < 0:
            return True  # clock stepped backwards (D5): re-anchor rather than stall
        fps = self.config.burst_fps if self.bursting else self.config.keepalive_fps
        return elapsed >= 1.0 / fps - _DUE_EPS

    def decide(self, frame: Frame) -> Frame | None:
        """Score one captured frame and return it to forward, or ``None`` to drop."""
        stability = self.meter.update(frame.image)
        self._track_burst(frame, stability)
        if not self._due(frame.capture_ts):
            return None
        self._last_forward_ts = frame.capture_ts
        return dataclasses.replace(frame, stability=stability, burst_id=self._burst_id)

    def forward(self, frames: Iterable[Frame]) -> Iterator[Frame]:
        """Filter a capture stream down to the frames that go on the wire."""
        for frame in frames:
            forwarded = self.decide(frame)
            if forwarded is not None:
                yield forwarded
