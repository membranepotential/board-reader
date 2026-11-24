"""One routing decision per frame — shared by ``POST /estimate`` and the live stream.

A frame reaches recognition through exactly one of three branches, decided by what
the active estimator can read and by whether the frame carries the board's corners:

1. :class:`PhotoRead` — the estimator reads ORIGINAL photos (tall full-piece boxes
   instead of a decapitating warp), so it gets the pixels as shot. Without corners
   the body is a rectified crop and the whole frame is the board (approximate, but
   it keeps the old rectified-crop contract working).
2. :class:`RectifiedRead` — corners against a rectified-mode estimator: warp
   server-side, because a top-down crop is what such an estimator reads.
3. :class:`WholeFrameRead` — no corners, no photo path: the JPEG *is* the board.

Both callers get their route from the same function (:func:`route_for`), so a
photo posted to ``/estimate`` and the same photo streamed can never be read
through different branches.

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

from board_reader_inference.estimator import Estimator
from board_reader_inference.squares import (
    BboxGeometry,
    CropGeometry,
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


#: A branch that produces a position event.
Route = PhotoRead | RectifiedRead | WholeFrameRead


def board_corners(frame: Frame) -> np.ndarray:
    """The frame's board quad — the whole image when the frame states none."""
    return frame.corners if frame.corners is not None else full_frame_corners(frame.image)


def route_for(est: Estimator, frame: Frame) -> Route:
    """How ``est`` should read ``frame`` (§4.5)."""
    photo = photo_estimator(est)
    if photo is not None:
        return PhotoRead(photo, frame, board_corners(frame))
    if frame.corners is not None:
        return RectifiedRead(est, frame, frame.corners)
    return WholeFrameRead(est, frame)
