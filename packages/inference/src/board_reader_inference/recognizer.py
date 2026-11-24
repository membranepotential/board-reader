"""The recognizer seam (DESIGN D10): a whole frame in, a chess-frame distribution out.

A :class:`Recognizer` reads one BGR frame and returns the ``(8, 8, 13)``
``square_dist`` of the board in it, **in the chess frame** (rank 8 first, file a
first, as in FEN). It finds the board and its orientation by itself, so it takes
no corners, crops nothing and is never calibrated. ChessQ Lite
(:mod:`.chessq`) is the core implementation; another model is one more file
implementing this protocol.

:func:`event_from_dist` is the one place a recognizer's distribution becomes a
:class:`~board_reader_core.protocol.PositionEvent`.

The gateway still speaks :class:`~.estimator.Estimator`, so a recognizer serves
through :class:`RecognizerEstimator`.

**Missing weights degrade, loudly** (plan decision D2). ``BR_CHESSQ`` is set by
both entry points, but the weights are never in git, so a fresh clone has none.
Then the stub recognizer serves — the walking skeleton's fixed start position —
with a startup banner and a ``fallback`` reason on ``/healthz``, so ``docker
compose up`` still works and nobody mistakes the stub for a read. ``make models``
fetches the weights. ``BR_CHESSQ`` unset serves the stub the same way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from board_reader_core.protocol import (
    START_PLACEMENT,
    ClockState,
    PositionEvent,
    dist_from_placement,
    iso_utc,
)
from board_reader_core.wire import Frame

from board_reader_inference.chessq import ChessQRecognizer
from board_reader_inference.estimator import Estimator
from board_reader_inference.labels import BoardFrame
from board_reader_inference.routing import RECTIFIED_GEOMETRY
from board_reader_inference.squares import CropGeometry

#: The ``orientation`` every recognizer event carries. Reserved since D10: the
#: placement is always in the chess frame, so this never varies.
ORIENTATION = "a1_bottom_left"


class Recognizer(Protocol):
    """A whole frame -> its board's ``(8, 8, 13)`` distribution, chess frame."""

    @property
    def name(self) -> str: ...

    @property
    def frame(self) -> Literal["chess"]: ...

    @property
    def model_path(self) -> str | None: ...

    def read(self, image: np.ndarray) -> np.ndarray: ...


class StubRecognizer:
    """Ignores the pixels and reads one authored placement (the walking skeleton)."""

    name = "stub"
    frame: Literal["chess"] = "chess"
    model_path: str | None = None

    def __init__(self, placement: str = START_PLACEMENT, *, confidence: float = 0.95) -> None:
        self.placement = placement
        self._dist = dist_from_placement(placement, confidence)

    def read(self, image: np.ndarray) -> np.ndarray:
        return self._dist


def event_from_dist(frame: Frame, square_dist: np.ndarray) -> PositionEvent:
    """A recognizer's distribution for ``frame`` as its (always ``ok``) event.

    The distribution is published as the recognizer produced it, uncalibrated.
    ``status`` is ``ok`` whatever the board looks like (DESIGN D1).
    """
    return PositionEvent.from_square_dist(
        board_id=frame.board_id,
        seq=frame.seq,
        utc=iso_utc(frame.capture_ts),
        square_dist=square_dist,
        stability=frame.stability,
        clock=ClockState(frame.clock),
        status="ok",
        orientation=ORIENTATION,
    )


class RecognizerEstimator:
    """A :class:`Recognizer` behind the gateway's estimator interface.

    It declares the photo path (``crop_mode``) so routing hands it the ORIGINAL
    pixels rather than a server-side warp, and it ignores any corners the frame
    carries.
    """

    crop_mode = "photo"
    # What ``?debug=1`` draws over the frame: the flat squares of the quad the
    # frame carries. The recognizer cut no boxes, so this is the operator's view
    # of where the corners say the board is, labeled with what was read there.
    geometry: CropGeometry = RECTIFIED_GEOMETRY
    square_size: int | None = None
    orientation = ORIENTATION

    def __init__(self, recognizer: Recognizer, *, fallback: str | None = None) -> None:
        self.recognizer = recognizer
        #: Why this recognizer serves instead of the one the environment asked
        #: for (missing weights), or ``None`` when it is the one asked for.
        self.fallback = fallback

    def estimate(self, frame: Frame) -> PositionEvent:
        return event_from_dist(frame, self.recognizer.read(frame.image))

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        """The photo path routing sends original pixels to; ``corners`` are unused."""
        return self.estimate(frame)


def read_frame(estimator: Estimator) -> BoardFrame:
    """The frame ``estimator``'s placements are in: the recognizer's own (chess), or
    the frame the frame's corners define for any other estimator."""
    if isinstance(estimator, RecognizerEstimator):
        return estimator.recognizer.frame
    return "corners"


def _stub(reason: str) -> RecognizerEstimator:
    """The stub recognizer, announced: a fixed start position is not a read."""
    print(
        f"!! {reason}\n!! serving the STUB recognizer: a fixed start position, "
        "not a read of the board",
        file=sys.stderr,
    )
    return RecognizerEstimator(StubRecognizer(), fallback=reason)


def build_from_env() -> RecognizerEstimator:
    """The recognizer the environment asks for.

    ``BR_CHESSQ=<path>`` -> ChessQ Lite; the stub, loudly, when the variable is
    unset or names no file (module docstring).
    """
    path = os.environ.get("BR_CHESSQ")
    if not path:
        return _stub("BR_CHESSQ is unset; set it to a model file (`make models`)")
    if not Path(path).is_file():
        return _stub(f"BR_CHESSQ={path!r} does not name a model file; run `make models`")
    print(f"serving ChessQ from BR_CHESSQ={path!r}")
    return RecognizerEstimator(ChessQRecognizer(path))
