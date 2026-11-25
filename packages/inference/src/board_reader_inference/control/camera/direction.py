"""Which way each board lies in the camera image, from the recognizer's reads.

The registry follows each board's **direction** (:meth:`CameraRegistry.observe_direction`):
which side White sits on, and whether the camera moved since the angle settled.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field

from board_reader_core.control import BoardDirection, WhiteSide

#: The running angle is the median of this many latest reads, so one or two odd
#: reads (a hand over the board) do not move it.
DIRECTION_READS = 5
#: The angle settles once that many reads span at least this many seconds of
#: capture time, so a burst of reads of one scene cannot settle it alone.
DIRECTION_SETTLE_S = 2.0
#: A running angle this far from the settled one means the camera moved. ChessQ
#: reads a fixed camera within about 2 degrees.
CAMERA_MOVED_DEGREES = 15.0

#: The angle at which White sits on each side (DESIGN D11; :class:`BoardDirection`).
_SIDES: dict[WhiteSide, float] = {"bottom": 0.0, "left": 90.0, "top": 180.0, "right": -90.0}


def _wrap(degrees: float) -> float:
    """``degrees`` folded into (-180, 180]."""
    folded = math.remainder(degrees, 360.0)
    return 180.0 if folded == -180.0 else folded


def white_side(degrees: float) -> WhiteSide:
    """The side White sits on for a board direction: the nearest of the four."""
    return min(_SIDES, key=lambda side: abs(_wrap(degrees - _SIDES[side])))


@dataclass(frozen=True, slots=True)
class _Read:
    degrees: float
    ts: float


@dataclass(slots=True)
class DirectionTrack:
    """One board's direction: the latest reads and the angle they settled on."""

    reads: deque[_Read] = field(default_factory=lambda: deque(maxlen=DIRECTION_READS))
    settled: float | None = None

    def observe(self, degrees: float, ts: float) -> None:
        self.reads.append(_Read(_wrap(degrees), ts))
        if self.settled is None and len(self.reads) == DIRECTION_READS:
            stamps = [read.ts for read in self.reads]
            if max(stamps) - min(stamps) >= DIRECTION_SETTLE_S:
                self.settled = self.running()

    def running(self) -> float | None:
        """The median of the latest reads, taken around the newest so it never wraps."""
        if not self.reads:
            return None
        newest = self.reads[-1].degrees
        offset = statistics.median(_wrap(read.degrees - newest) for read in self.reads)
        return _wrap(newest + offset)

    def view(self) -> BoardDirection | None:
        running = self.running()
        if running is None:
            return None
        moved = (
            self.settled is not None
            and abs(_wrap(running - self.settled)) > CAMERA_MOVED_DEGREES
        )
        return BoardDirection(
            degrees=round(running, 1), white_side=white_side(running), camera_moved=moved
        )
