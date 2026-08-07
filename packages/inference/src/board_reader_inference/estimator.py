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

import numpy as np

from board_reader_core.protocol import (
    START_PLACEMENT,
    ClockState,
    PositionEvent,
    dist_from_placement,
    iso_utc,
)
from board_reader_core.wire import Frame

from board_reader_inference.centroid import CentroidClassifier, ExemplarClassifier
from board_reader_inference.embed import Embedder
from board_reader_inference.squares import (
    DEFAULT_GEOMETRY,
    BboxGeometry,
    CropGeometry,
    crop_squares,
    crop_squares_photo,
)


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
    centroids (or the validated-stronger kNN exemplar bank) calibrated from any
    labeled position.
    """

    def __init__(
        self,
        embedder: Embedder,
        classifier: CentroidClassifier | ExemplarClassifier,
        *,
        inset: float = 0.1,
        square_size: int | None = None,
        orientation: str = "a1_bottom_left",
        geometry: BboxGeometry = DEFAULT_GEOMETRY,
    ) -> None:
        self.embedder = embedder
        self.classifier = classifier
        self.inset = inset
        self.square_size = square_size
        self.orientation = orientation
        # Photo-path box geometry; must match what calibration used (see
        # squares.crop_squares_photo). The exemplar bank is defined in terms of
        # image-axis boxes, so this path is bbox geometry by construction — the
        # type says so, and a pose-geometry bank would need a different bank.
        self.geometry: BboxGeometry = geometry

    @property
    def headroom(self) -> float:
        return self.geometry.headroom

    @property
    def side_pad(self) -> float:
        return self.geometry.side_pad

    def estimate(self, frame: Frame) -> PositionEvent:
        crops = crop_squares(frame.image, inset=self.inset, size=self.square_size)
        return self._event_from_crops(frame, crops)

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        """Estimate from an ORIGINAL photo + playing-surface corners (a8,h8,h1,a1).

        Uses tall full-piece boxes (:func:`crop_squares_photo`) instead of flat
        squares of a top-down warp — exact for centroids calibrated with
        ``crop_mode="photo"``, approximate otherwise (the gateway matches modes).
        """
        crops = crop_squares_photo(
            frame.image,
            corners,
            geometry=self.geometry,
            size=self.square_size or 224,
        )
        return self._event_from_crops(frame, crops)

    def _event_from_crops(
        self, frame: Frame, crops: list[np.ndarray]
    ) -> PositionEvent:
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


class EnsembleEstimator:
    """Blend a zero-shot trained classifier with a per-set calibration bank (§4.6).

    Calibration-as-booster: the exemplar bank is near-perfect under conditions it
    has seen but brittle outside them; the trained model is the reverse. Averaging
    their square distributions tracks the better member square-by-square (measured
    2026-07-18 on the 7-board labeled eval: trained 415/448, bank 430/448, 50/50
    blend 439/448). Both members must expose the photo path (``estimate_photo``).
    """

    crop_mode = "photo"

    def __init__(
        self,
        primary: Estimator,
        booster: Estimator,
        *,
        weight: float = 0.5,
    ) -> None:
        if not 0.0 < weight < 1.0:
            raise ValueError("weight must be in (0, 1) — at 0 or 1 use the member directly")
        self.primary = primary
        self.booster = booster
        self.weight = weight

    # The gateway's debug/refresh plumbing reads box geometry off the active
    # estimator; delegate to the primary (the trained model's geometry). Members
    # crop independently, so a pose-geometry primary and a bbox booster coexist.
    @property
    def geometry(self) -> CropGeometry:
        return self.primary.geometry

    @property
    def headroom(self) -> float:
        return self.primary.headroom

    @property
    def side_pad(self) -> float:
        return self.primary.side_pad

    @property
    def square_size(self) -> int | None:
        return getattr(self.primary, "square_size", None)

    @property
    def orientation(self) -> str:
        return self.primary.orientation

    def estimate(self, frame: Frame) -> PositionEvent:
        from board_reader_inference.squares import full_frame_corners

        return self.estimate_photo(frame, full_frame_corners(frame.image))

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        a = self.primary.estimate_photo(frame, corners)
        b = self.booster.estimate_photo(frame, corners)
        dist = self.weight * np.asarray(a.square_dist, dtype=np.float32) + (
            1.0 - self.weight
        ) * np.asarray(b.square_dist, dtype=np.float32)
        return PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc=iso_utc(frame.capture_ts),
            square_dist=dist,
            stability=frame.stability,
            clock=ClockState(frame.clock),
            status="ok",
            orientation=self.orientation,
        )
