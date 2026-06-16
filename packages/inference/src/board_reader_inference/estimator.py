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

from board_reader_inference.centroid import CentroidClassifier
from board_reader_inference.embed import Embedder
from board_reader_inference.squares import crop_squares


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


class CentroidEstimator:
    """Real per-frame estimator: embed 64 squares, classify by nearest centroid (M1).

    Stateless per DESIGN D1 — it reads the rectified crop the edge forwarded, crops
    the 64 squares (WP-B1), embeds them (WP-B2, any :class:`~board_reader_inference.embed.Embedder`),
    and turns the per-square centroid distances into the canonical 8x8x13
    ``square_dist`` (WP-B3). No chess rules, legality, or move tracking.

    The embedder and classifier are injected so the same estimator runs with the
    lightweight M1 embedder, the DINOv2 backbone, or a fine-tuned head — and with
    centroids calibrated from any labeled position.
    """

    def __init__(
        self,
        embedder: Embedder,
        classifier: CentroidClassifier,
        *,
        inset: float = 0.1,
        square_size: int | None = None,
        orientation: str = "a1_bottom_left",
    ) -> None:
        self.embedder = embedder
        self.classifier = classifier
        self.inset = inset
        self.square_size = square_size
        self.orientation = orientation

    def estimate(self, frame: Frame) -> PositionEvent:
        crops = crop_squares(frame.image, inset=self.inset, size=self.square_size)
        embeddings = self.embedder.embed(crops)
        square_dist = self.classifier.classify_board(embeddings)
        return PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc=iso_utc(frame.capture_ts),
            square_dist=square_dist,
            stability=frame.stability,
            clock=ClockState(frame.clock),
            status="ok",
            orientation=self.orientation,
        )
