"""One routing decision per frame — shared by ``POST /estimate`` and the live stream.

A frame reaches recognition through exactly one of four branches, decided by what
the active estimator can read and by whether the frame carries the board's corners:

1. :class:`PhotoRead` — the estimator reads ORIGINAL photos (tall full-piece boxes
   instead of a decapitating warp), so it gets the pixels as shot. Without corners
   the body is a rectified crop and the whole frame is the board (approximate, but
   it keeps the old rectified-crop contract working).
2. :class:`RectifiedRead` — corners against a rectified-mode estimator: warp
   server-side, because a top-down crop is what such a bank was calibrated on (and
   what the M0 stub's centroids assume).
3. :class:`WholeFrameRead` — no corners, no photo path: the JPEG *is* the board.
4. :class:`Uncalibrated` — the §4.6 zero-setup state (an occupancy estimator is
   loaded but no piece-type calibration is active): occupancy is all that can
   honestly be read, and there is NO position event to publish.

Branch 4 is why this module exists. The snapshot path and the stream path used to
decide this separately, and the stream's copy was written with three branches: it
published the M0 stub's fixed start placement at 0.95 confidence at stream rate — a
fabricated position on the hub, for as long as nobody calibrated. The types make
that unrepresentable: :class:`Uncalibrated` has no way to produce a
:class:`PositionEvent`, and both callers get their route from the same function
(:func:`route_for`, via :meth:`~board_reader_inference.session.EstimatorSession.route`),
so a fifth branch cannot reach one path and miss the other.

A route is the decision for *one* frame, with the frame in hand: deciding is cheap
and happens on the event loop, while :meth:`PhotoRead.read` and friends do the
pixel work and are meant to run in a worker thread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, cast, runtime_checkable

import numpy as np

from board_reader_core.protocol import PositionEvent
from board_reader_core.wire import Frame

from board_reader_inference.estimator import CentroidEstimator, Estimator
from board_reader_inference.squares import (
    BboxGeometry,
    CropGeometry,
    crop_squares_photo,
    full_frame_corners,
    rectify_photo,
)

# Square crop size when the estimator has no opinion — the same default
# :func:`crop_squares_photo` uses.
DEFAULT_SQUARE_PX = 224

# A rectified crop is a top-down board with no headroom to give: the boxes are the
# squares themselves.
RECTIFIED_GEOMETRY = BboxGeometry(headroom=0.0, side_pad=0.0)

PhotoEstimate = Callable[[Frame, np.ndarray], PositionEvent]


@runtime_checkable
class PhotoReader(Protocol):
    """An estimator that can read an ORIGINAL photo given the board's corners.

    Declared by the classifier (kNN/centroid banks carry ``crop_mode``) or by the
    estimator itself (the trained crop classifier), and resolved by
    :func:`photo_estimator` — never assumed.
    """

    geometry: CropGeometry
    square_size: int | None

    def estimate(self, frame: Frame) -> PositionEvent:
        ...

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        ...


def photo_estimator(est: Estimator) -> PhotoReader | None:
    """``est`` as a photo reader, if it is calibrated/trained for original photos.

    Photo-mode is declared either by the classifier (kNN/centroid banks) or by the
    estimator itself (the trained crop classifier). One implementation, so the
    streamed and the posted photo can never be routed differently.
    """
    photo_mode = (
        getattr(getattr(est, "classifier", None), "crop_mode", None) == "photo"
        or getattr(est, "crop_mode", None) == "photo"
    )
    if not photo_mode:
        return None
    if not callable(getattr(est, "estimate_photo", None)):
        return None
    return cast(PhotoReader, est)


@dataclass(frozen=True, slots=True)
class ReadView:
    """What recognition actually saw: the pixels, the quad, and the box geometry.

    ``?debug=1`` draws from this rather than re-deriving it, so the visualisation
    cannot drift from the branch that produced the answer.
    """

    image: np.ndarray
    corners: np.ndarray
    geometry: CropGeometry
    size: int


@dataclass(frozen=True, slots=True)
class PhotoRead:
    """Branch 1: original pixels + corners -> the estimator's tall-box photo path."""

    estimator: PhotoReader
    frame: Frame
    corners: np.ndarray  # the board quad, a8,h8,h1,a1 (the whole frame if unstated)

    def read(self) -> PositionEvent:
        return self.estimator.estimate_photo(self.frame, self.corners)

    def view(self) -> ReadView:
        return ReadView(
            image=self.frame.image,
            corners=self.corners,
            geometry=self.estimator.geometry,
            size=self.estimator.square_size or DEFAULT_SQUARE_PX,
        )


@dataclass(frozen=True, slots=True)
class RectifiedRead:
    """Branch 2: corners against a rectified-mode estimator -> warp, then estimate."""

    estimator: Estimator
    frame: Frame
    corners: np.ndarray

    def read(self) -> PositionEvent:
        return self.estimator.estimate(self._rectified())

    def view(self) -> ReadView:
        image = self._rectified().image
        return ReadView(
            image=image,
            corners=full_frame_corners(image),
            geometry=RECTIFIED_GEOMETRY,
            size=DEFAULT_SQUARE_PX,
        )

    def _rectified(self) -> Frame:
        return replace(
            self.frame,
            image=rectify_photo(self.frame.image, self.corners),
            corners=None,
        )


@dataclass(frozen=True, slots=True)
class WholeFrameRead:
    """Branch 3: no corners and no photo path — the JPEG is the board, unchanged."""

    estimator: Estimator
    frame: Frame

    def read(self) -> PositionEvent:
        return self.estimator.estimate(self.frame)

    def view(self) -> ReadView:
        return ReadView(
            image=self.frame.image,
            corners=full_frame_corners(self.frame.image),
            geometry=RECTIFIED_GEOMETRY,
            size=DEFAULT_SQUARE_PX,
        )


@dataclass(frozen=True, slots=True)
class OccupancyRead:
    """An occupancy-only reading: which squares hold *something*, and the crops.

    Deliberately not a :class:`PositionEvent`: piece type does not transfer across
    piece styles on frozen features (DESIGN §4.6), so there is no position here to
    publish — only a readout, and the crops a calibration could be built from.
    """

    dist: np.ndarray  # (8,8,13) from the occupancy bank
    crops: list[np.ndarray]


@dataclass(frozen=True, slots=True)
class Uncalibrated:
    """Branch 4: occupancy estimator loaded, no piece-type calibration active (§4.6).

    ``read()`` yields occupancy, never a position — the hub carries positions
    someone actually estimated, never a placeholder. The snapshot path answers with
    the occupancy readout (and may auto-calibrate off it); a stream has no response
    to answer with, so it counts the frame and skips it.
    """

    occupancy: CentroidEstimator
    frame: Frame
    corners: np.ndarray

    def read(self) -> OccupancyRead:
        crops = crop_squares_photo(
            self.frame.image,
            self.corners,
            geometry=self.occupancy.geometry,
            size=self._size(),
        )
        dist = self.occupancy.classifier.classify_board(
            self.occupancy.embedder.embed(crops)
        )
        return OccupancyRead(dist=dist, crops=crops)

    def view(self) -> ReadView:
        return ReadView(
            image=self.frame.image,
            corners=self.corners,
            geometry=self.occupancy.geometry,
            size=self._size(),
        )

    def _size(self) -> int:
        return self.occupancy.square_size or DEFAULT_SQUARE_PX


#: A branch that can produce a position event. ``Uncalibrated`` is deliberately not
#: one of them.
PositionRead = PhotoRead | RectifiedRead | WholeFrameRead
Route = PositionRead | Uncalibrated


def board_corners(frame: Frame) -> np.ndarray:
    """The frame's board quad — the whole image when the frame states none."""
    return frame.corners if frame.corners is not None else full_frame_corners(frame.image)


def route_for(est: Estimator, frame: Frame) -> PositionRead:
    """Branches 1–3: how ``est`` should read ``frame`` (§4.5).

    Branch 4 is not decidable from an estimator alone — it is a property of the
    session (:meth:`board_reader_inference.session.EstimatorSession.route`).
    """
    photo = photo_estimator(est)
    if photo is not None:
        return PhotoRead(photo, frame, board_corners(frame))
    if frame.corners is not None:
        return RectifiedRead(est, frame, frame.corners)
    return WholeFrameRead(est, frame)
