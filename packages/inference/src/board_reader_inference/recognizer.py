"""The recognizer seam (DESIGN D10): a whole frame in, a chess-frame distribution out.

A :class:`Recognizer` reads one BGR frame and returns a :class:`Recognition`: the
``(8, 8, 13)`` ``square_dist`` of the board in it, **in the chess frame** (rank 8
first, file a first, as in FEN), and the board's ``direction`` in the image when
the model reports one. It finds the board and its orientation by itself, so it
takes no corners, crops nothing and is never calibrated. ChessQ Lite
(:mod:`.chessq`) is the core implementation; another model is one more file
implementing this protocol.

:func:`event_from_dist` is the one place a recognizer's distribution becomes a
:class:`~board_reader_core.protocol.PositionEvent`, and :func:`frame_reader` the
one read the gateway, the ingest stream and replay all run.

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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from board_reader_core.protocol import (
    START_PLACEMENT,
    ClockState,
    PositionEvent,
    dist_from_placement,
    iso_utc,
)
from board_reader_core.wire import Frame

#: The ``orientation`` every recognizer event carries. Reserved since D10: the
#: placement is always in the chess frame, so this never varies.
ORIENTATION = "a1_bottom_left"



@dataclass(frozen=True, slots=True)
class Recognition:
    """What a recognizer read off one frame.

    ``direction`` is the bearing of the e1 -> e8 vector in the image, in degrees
    clockwise from straight up, in ``(-180, 180]``: 0 means White sits at the
    bottom of the picture, 90 that White sits on the left. ``None`` when the
    recognizer does not report it (the stub) or reported nothing usable.
    """

    square_dist: np.ndarray
    direction: float | None = None


@dataclass(frozen=True, slots=True)
class FrameReading:
    """One frame's read: its position event and the board's direction (see
    :class:`Recognition`), which rides beside the event, never in it."""

    event: PositionEvent
    direction: float | None = None


#: One frame in, its read out — what a read *is*, for every caller.
FrameRead = Callable[[Frame], FrameReading]


class Recognizer(Protocol):
    """A whole frame -> its board's chess-frame :class:`Recognition`."""

    @property
    def name(self) -> str: ...

    @property
    def model_path(self) -> str | None: ...

    def read(self, image: np.ndarray) -> Recognition: ...


class StubRecognizer:
    """Ignores the pixels and reads one authored placement (the walking skeleton).

    ``fallback`` says why the stub serves in place of the recognizer the
    environment asked for (``/healthz`` reports it); ``None`` when a test or a
    caller asked for the stub itself.
    """

    name = "stub"
    model_path: str | None = None

    def __init__(
        self,
        placement: str = START_PLACEMENT,
        *,
        confidence: float = 0.95,
        fallback: str | None = None,
    ) -> None:
        self.placement = placement
        self.fallback = fallback
        self._recognition = Recognition(dist_from_placement(placement, confidence))

    def read(self, image: np.ndarray) -> Recognition:
        return self._recognition


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


def frame_reader(recognizer: Recognizer) -> FrameRead:
    """``recognizer`` as a read: the frame's whole image in, its event out.

    Any ``corners`` the frame carries (older edges and recordings send them) are
    ignored — the recognizer finds the board itself.
    """

    def read(frame: Frame) -> FrameReading:
        recognition = recognizer.read(frame.image)
        return FrameReading(
            event_from_dist(frame, recognition.square_dist), recognition.direction
        )

    return read


def _stub(reason: str) -> StubRecognizer:
    """The stub recognizer, announced: a fixed start position is not a read."""
    print(
        f"!! {reason}\n!! serving the STUB recognizer: a fixed start position, "
        "not a read of the board",
        file=sys.stderr,
    )
    return StubRecognizer(fallback=reason)


def build_from_env() -> Recognizer:
    """The recognizer the environment asks for.

    ``BR_CHESSQ=<path>`` -> ChessQ Lite; the stub, loudly, when the variable is
    unset or names no file (module docstring).
    """
    path = os.environ.get("BR_CHESSQ")
    if not path:
        return _stub("BR_CHESSQ is unset; set it to a model file (`make models`)")
    if not Path(path).is_file():
        return _stub(f"BR_CHESSQ={path!r} does not name a model file; run `make models`")
    # Here, not at the top: chessq builds this module's Recognition.
    from board_reader_inference.chessq import ChessQRecognizer

    print(f"serving ChessQ from BR_CHESSQ={path!r}")
    return ChessQRecognizer(path)
