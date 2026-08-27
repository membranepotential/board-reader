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
    CLASS_INDEX,
    NUM_CLASSES,
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


class PhotoDistReader(Protocol):
    """An estimator that hands back its RAW per-square distribution for a photo.

    The *blend* seam. A composite (:class:`EnsembleEstimator`) must average its
    members' distributions rather than their events, because the calibration in
    :func:`demote_attractor` belongs to the blend a consumer sees — applied once,
    downstream of every member — and a member that had already swapped its own
    top two would change what gets averaged.
    """

    geometry: CropGeometry
    square_size: int | None
    orientation: str

    def photo_dist(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        ...


# --------------------------------------------------------------------------- #
# Post-hoc calibration of the estimator's own output (the black-bishop demotion)
# --------------------------------------------------------------------------- #

#: The class the deployed stack over-predicts: the BLACK bishop is an attractor,
#: and when it wins top-1 only narrowly the runner-up (``p``/``q``/``k``) is the
#: right answer. Lowercase only — white's ``B`` is not an attractor (1 fixable
#: square in 8512).
ATTRACTOR = "b"

#: Top1-minus-top2 probability difference below which the attractor's win is not
#: believed. Measured over four live takes, 133 boards, 8512 squares
#: (docs/reports/2026-08-27-evidence-rerank.md §5-6, leave-one-take-out
#: validated): 55 firings, **38 squares fixed, 0 broken**. What makes 0.20 safe
#: is a gap, not a mean — the lowest margin at which a top-1 ``b`` was actually
#: RIGHT is 0.210, so the threshold sits below every correct read in every take.
#: That gap is a property of the takes so far, so the trial scorer measures it on
#: every session and warns when a new take starts closing it
#: (:func:`board_reader_inference.trial.score_attractor_guard`).
ATTRACTOR_MARGIN = 0.20


def demote_attractor(square_dist: np.ndarray) -> np.ndarray:
    """Swap the top two classes on squares the attractor only narrowly won.

    A calibration of the estimator's own output, so it lands here: downstream of
    every blend, upstream of ``placement``/``evidence``/``confidence`` and of
    :func:`board_reader_inference.prior.decode`, which stays a pure decoder over
    whatever distribution estimation publishes (DESIGN D1).

    The two entries trade values, so this is a permutation: the row still sums to
    what it summed to, and the square's max probability — hence the event's
    ``confidence`` — is *unchanged*. A rescued square therefore reports at the
    probability the attractor had won it with (mean 0.34-0.47 per take), which is
    low by construction: a narrow win means ``p1 < p2 + 0.20``. The model did not
    become sure, it became right. (The study's §6 prose says such a square reports
    at the old *runner-up* probability, ~0.1 lower; that is its own accounting
    slip — ``results_firing.txt`` reads the "after" value off ``p2`` — and it is
    not reachable without inventing a redistribution nothing measured. The
    implementation here is §6's snippet, the one ``verify_impl.py`` validated.)

    Idempotent: after a swap the attractor is no longer top-1, so a second pass
    cannot fire on the same square.
    """
    arr = np.asarray(square_dist, dtype=np.float32)
    flat = arr.reshape(-1, NUM_CLASSES).copy()
    order = np.argsort(-flat, axis=-1)[:, :2]
    i1, i2 = order[:, 0], order[:, 1]
    rows = np.arange(flat.shape[0])
    p1, p2 = flat[rows, i1], flat[rows, i2]
    fire = (i1 == CLASS_INDEX[ATTRACTOR]) & (p1 - p2 < ATTRACTOR_MARGIN)
    r = rows[fire]
    flat[r, i1[fire]], flat[r, i2[fire]] = p2[fire], p1[fire]
    return flat.reshape(arr.shape)


def read_event(
    frame: Frame,
    square_dist: np.ndarray,
    *,
    orientation: str,
    status: str = "ok",
) -> PositionEvent:
    """The one place a raw distribution becomes a *calibrated* read.

    Every read the measured stack produces ends here — the trained model and both
    of its composites (:mod:`board_reader_inference.trained`) and the
    :class:`EnsembleEstimator` blend on top of one — so
    :func:`demote_attractor` is applied exactly once per read, whoever asked:
    ``POST /estimate`` and the live ingest worker share it through
    :meth:`board_reader_inference.routing.PhotoRead.read`, and ``replay``,
    ``eval`` and a calibration re-read all come through the same estimators. No
    consumer carries a copy, so no consumer can double-apply it — and composites
    blend :class:`PhotoDistReader` distributions rather than member events, so a
    member cannot apply it early either.

    The two estimators that do NOT end here are the ones the calibration was not
    measured on: :class:`StubEstimator` (an authored placement, no model output)
    and a bare :class:`CentroidEstimator` (see :meth:`CentroidEstimator._event`).
    """
    return PositionEvent.from_square_dist(
        board_id=frame.board_id,
        seq=frame.seq,
        utc=iso_utc(frame.capture_ts),
        square_dist=demote_attractor(square_dist),
        stability=frame.stability,
        clock=ClockState(frame.clock),
        status=status,
        orientation=orientation,
    )


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
        # Deliberately NOT through :func:`read_event`: there is no model output to
        # calibrate here, only an authored placement, and a stub that quietly
        # served a different board than the one it was configured with would make
        # the walking skeleton lie about the pipe.
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
        return self._event(frame, self._dist_from_crops(crops))

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        """Estimate from an ORIGINAL photo + playing-surface corners (a8,h8,h1,a1).

        Uses tall full-piece boxes (:func:`crop_squares_photo`) instead of flat
        squares of a top-down warp — exact for centroids calibrated with
        ``crop_mode="photo"``, approximate otherwise (the gateway matches modes).
        """
        return self._event(frame, self.photo_dist(frame.image, corners))

    def _event(self, frame: Frame, square_dist: np.ndarray) -> PositionEvent:
        # Deliberately NOT :func:`read_event`: a bank *alone* is outside the
        # attractor calibration's domain. ATTRACTOR_MARGIN is an ABSOLUTE
        # probability difference fitted on the trained stack's distribution, and a
        # kNN bank's is flatter — on the LiteEmbedder fixture this bank's own
        # CORRECT black bishops read at margins 0.105 and 0.116, i.e. inside the
        # firing band, so demoting here would eat squares the study never
        # measured (its four takes are all trained ⊕ bank). The blend below is
        # what was measured, and it does calibrate.
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

    def photo_dist(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """This bank's RAW distribution for a photo (:class:`PhotoDistReader`)."""
        crops = crop_squares_photo(
            image,
            corners,
            geometry=self.geometry,
            size=self.square_size or 224,
        )
        return self._dist_from_crops(crops)

    def _dist_from_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        return self.classifier.classify_board(self.embedder.embed(crops))


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
        primary: PhotoDistReader,
        booster: PhotoDistReader,
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

    def photo_dist(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """The members' RAW distributions, blended (:class:`PhotoDistReader`).

        Members are asked for distributions, never for their events: the
        attractor calibration is applied once, by :func:`read_event`, to the
        blend that leaves this composite.
        """
        a = np.asarray(self.primary.photo_dist(image, corners), dtype=np.float32)
        b = np.asarray(self.booster.photo_dist(image, corners), dtype=np.float32)
        return self.weight * a + (1.0 - self.weight) * b

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        return read_event(
            frame,
            self.photo_dist(frame.image, corners),
            orientation=self.orientation,
        )
