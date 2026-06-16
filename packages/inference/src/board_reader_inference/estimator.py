"""Per-frame position estimators (DESIGN.md L2 / D1).

The estimator is a *stateless per-frame position estimator*: frame in -> a full
8x8x13 distribution out, assembled into an L3 :class:`PositionEvent`. It applies no
chess rules, no legality, no move tracking (DESIGN.md D1).

M0 ships :class:`StubEstimator`, which ignores pixels and emits a fixed configured
placement so the end-to-end pipe can be proven before the real DINOv2 + classifier
head (M1 / WP-B) lands behind this same :class:`Estimator` interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from board_reader_core.protocol import (
    START_PLACEMENT,
    ClockState,
    PositionEvent,
    dist_from_placement,
    iso_utc,
)
from board_reader_core.wire import Frame


@runtime_checkable
class Estimator(Protocol):
    """Frame -> L3 position event. The seam the real CV model plugs into (M1)."""

    def estimate(self, frame: Frame) -> PositionEvent:
        ...


class StubEstimator:
    """Walking-skeleton estimator: emits a fixed placement, ignoring pixels.

    Proves the transport + protocol + viewer path. The ``confidence`` is configurable
    so the viewer's confidence indicators have something non-trivial to render.
    """

    def __init__(
        self,
        placement: str = START_PLACEMENT,
        *,
        confidence: float = 0.95,
        orientation: str = "a1_bottom_left",
    ) -> None:
        self.placement = placement
        self.confidence = confidence
        self.orientation = orientation
        # Pre-build the distribution once; the stub returns the same board each frame.
        self._dist = dist_from_placement(placement, confidence)

    def estimate(self, frame: Frame) -> PositionEvent:
        return PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc=iso_utc(frame.capture_ts),
            square_dist=self._dist,
            stability=frame.stability,
            clock=ClockState(frame.clock),
            status="ok",
            orientation=self.orientation,
        )
