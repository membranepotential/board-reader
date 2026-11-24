"""Back-date ``changed_utc`` with pixels from the frames between two reads.

The node reads a sparse subset of the frames that reach it (:mod:`.schedule`), so the
first read that shows a new board can come seconds after the board changed. The
frames in between did arrive; they were just not worth a recognizer read. This module
keeps a small grey copy of each of them (:class:`FrameRing`) and, when the settled
board changes, asks of those frames only one question, by pixels: from which frame on
do the changed squares already look like they do in the read that first showed the new
board? (:func:`first_new_frame`). No recognizer read, no GPU: a few hundred
microseconds per move.

**Which pixels.** There are no square coordinates here: ChessQ finds the board by
itself and reports none. The changed squares are instead where the two reference
frames differ: the last read that still showed the old board on the changed squares,
and the read that first showed the new one (:class:`ChangeClock` finds both in its
read log). Both are reads the stabilizer accepted as a board, so neither holds a hand
over those squares in the common case, and their difference is the moved pieces
(plus shadows and anything else that changed, which the share threshold absorbs).

**A hand is neither.** A frame counts as new when at least :data:`MATCH_SHARE` of
those pixels sit within :data:`MATCH_LEVELS` of the new reference. A hand still over
the square matches neither reference, so it is not new. The search walks back from the
new reference and stops at the first frame that is not new, so the answer is the start
of the unbroken run of new-looking frames that ends at the read: a frame that flickers
to "new" while the hand is still moving does not count unless every frame after it
agrees.

**Brightness.** The phone's auto-exposure shifts the whole frame from one frame to
the next (a hand entering darkens it), so each grey frame is scaled to a fixed median
before it is compared. The comparison is on a downscaled, blurred grey copy
(:data:`GREY_WIDTH` pixels wide), which also absorbs sub-pixel shake.

**Where it runs.** Inline in the read (:func:`~.ingest._read_and_settle`), right after
the stabilizer, so the event that publishes the new board already carries the
back-dated time and the field's meaning stays "when the current settled board first
appeared" with no second event. Its cost is microseconds per held frame against a
~200 ms recognizer read; replay runs the same call, so the gate measures it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from board_reader_core.protocol import GRID, parse_placement

#: The grey copy a held frame keeps is about this wide: the frame is shrunk by the
#: whole factor that gets closest (OpenCV's area resize is ~3x faster at a whole
#: factor), so the phone's 960 x 520 crop keeps 160 x 86 = 14 kB.
GREY_WIDTH = 160

#: How far back the ring holds frames, in seconds of capture time. A move whose first
#: new read comes later than this after the last old read is not back-dated. At the
#: forwarder's 10 fps tail that is at most ~100 frames, ~1.4 MB per board.
HOLD_SECONDS = 10.0

#: And never more than this many frames, whatever the edge's rate (a cap on memory
#: for an edge that sends faster than the shipped one): ~2.2 MB per board.
HOLD_FRAMES = 160

#: How many reads the clock remembers, to find the two references of a commit. The
#: shipped schedule reads ~3 times per move plus a 2 s keepalive, so 64 reads cover
#: far more than :data:`HOLD_SECONDS`.
READ_LOG = 64

#: Grey levels (after the median scaling to 128) by which the two references must
#: differ for a pixel to count as part of the change.
CHANGE_LEVELS = 25.0

#: Grey levels within which a frame's pixel matches the new reference.
MATCH_LEVELS = 15.0

#: The share of the change's pixels that must match the new reference for a frame
#: to count as showing the new board.
MATCH_SHARE = 0.7

#: Fewer changed pixels than this and the references are too alike to judge by.
MIN_CHANGE_PIXELS = 12

#: The median a grey copy is scaled to before any comparison (auto-exposure).
_MEDIAN_LEVEL = 128.0


@dataclass(frozen=True, slots=True)
class HeldFrame:
    """One frame as the ring keeps it: its identity and a small grey copy.

    ``grey`` is already blurred and brightness-normalised (:func:`grey_of`), uint8.
    """

    seq: int
    ts: float
    grey: np.ndarray


def grey_of(image: np.ndarray) -> np.ndarray:
    """The comparable copy of a BGR frame: grey, about :data:`GREY_WIDTH` wide,
    blurred, scaled to a median of 128, as uint8. ~0.5 ms for a 960 px frame."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = max(1, round(grey.shape[1] / GREY_WIDTH))
    h, w = grey.shape[0] // k, grey.shape[1] // k
    small = cv2.resize(grey[: h * k, : w * k], (w, h), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (3, 3), 0)
    scale = _MEDIAN_LEVEL / max(float(np.median(blurred)), 1.0)
    return cv2.convertScaleAbs(blurred, alpha=scale)


class FrameRing:
    """The recent frames of one board stream, bounded by :data:`HOLD_SECONDS` of
    capture time and :data:`HOLD_FRAMES` frames.

    Live, :meth:`~.ingest.FrameIngest.submit` pushes every arriving frame (the
    latest-wins inbox drops most of them before a read) and the worker takes a
    :meth:`snapshot` on the event loop before each read; replay pushes every recorded
    frame. A ``seq`` that does not advance starts afresh: that is a new streaming
    session (the recorder and the orientation guard roll on the same rule).
    """

    def __init__(self) -> None:
        self._frames: deque[HeldFrame] = deque(maxlen=HOLD_FRAMES)

    def push(self, seq: int, ts: float, image: np.ndarray) -> None:
        if self._frames and seq <= self._frames[-1].seq:
            self._frames.clear()
        self._frames.append(HeldFrame(seq, ts, grey_of(image)))
        while self._frames and ts - self._frames[0].ts > HOLD_SECONDS:
            self._frames.popleft()

    def snapshot(self) -> tuple[HeldFrame, ...]:
        """The held frames, oldest first, as an immutable copy for the worker thread."""
        return tuple(self._frames)


def first_new_frame(
    window: Sequence[HeldFrame], old: HeldFrame, new: HeldFrame
) -> HeldFrame | None:
    """The earliest frame of ``window`` from which every frame up to ``new`` shows the
    new board on the changed pixels, or ``None`` when the frame right before ``new``
    does not (or nothing can be judged).

    ``window`` is the frames strictly between the references, oldest first. A frame
    of another size than the references (an edge that changed its crop) ends the
    search, as does a frame that is not new.
    """
    if old.grey.shape != new.grey.shape:
        return None
    change = cv2.absdiff(old.grey, new.grey) > CHANGE_LEVELS
    if int(change.sum()) < MIN_CHANGE_PIXELS:
        return None
    target = new.grey[change].astype(np.int16)
    earliest = None
    for frame in reversed(window):
        if frame.grey.shape != new.grey.shape:
            break
        matching = np.abs(frame.grey[change].astype(np.int16) - target) < MATCH_LEVELS
        if matching.mean() < MATCH_SHARE:
            break
        earliest = frame
    return earliest


@dataclass(frozen=True, slots=True)
class _Read:
    seq: int
    ts: float
    grid: list[list[str]]


class ChangeClock:
    """The ``changed_utc`` a board publishes: the stabilizer's, back-dated by pixels.

    One per board stream, like the stabilizer and the orientation guard, and fed every
    published read in order (:meth:`observe`). When the settled board changes it finds
    the two references in its read log (see the module docstring) and the held frames
    between them, and publishes the earliest new-looking frame's capture time if that
    is earlier than the stabilizer's; otherwise, and on every read that commits
    nothing, it publishes what it published before. No clock: replay stays
    deterministic.
    """

    def __init__(self) -> None:
        self._reads: deque[_Read] = deque(maxlen=READ_LOG)
        self._settled: str | None = None
        self._published: float | None = None

    def observe(
        self,
        seq: int,
        ts: float,
        read: str,
        settled: str,
        changed_ts: float,
        held: Sequence[HeldFrame],
    ) -> float:
        """Log this read and return the ``changed_utc`` (as capture seconds) to publish.

        ``read`` is the read's own board, ``settled`` the stabilizer's board after it,
        ``changed_ts`` the stabilizer's :attr:`~.stabilize.BoardStabilizer.changed_ts`,
        and ``held`` the board's :meth:`FrameRing.snapshot` taken for this read.
        """
        if self._reads and seq <= self._reads[-1].seq:
            self._reads.clear()  # a new streaming session: nothing before it is a reference
        self._reads.append(_Read(seq, ts, parse_placement(read)))
        before, self._settled = self._settled, settled
        if self._published is None or before != settled:
            found = None if before is None else self._backdate(before, settled, changed_ts, held)
            self._published = changed_ts if found is None else min(found.ts, changed_ts)
        return self._published

    def _backdate(
        self, before: str, after: str, changed_ts: float, held: Sequence[HeldFrame]
    ) -> HeldFrame | None:
        old_board, new_board = parse_placement(before), parse_placement(after)
        changed = [
            (r, c)
            for r in range(GRID)
            for c in range(GRID)
            if old_board[r][c] != new_board[r][c]
        ]
        reads = list(self._reads)
        first = next((i for i, read in enumerate(reads) if read.ts == changed_ts), None)
        if first is None:
            return None
        last_old = next(
            (
                read
                for read in reversed(reads[:first])
                if all(read.grid[r][c] == old_board[r][c] for r, c in changed)
            ),
            None,
        )
        if last_old is None:
            return None
        by_seq = {frame.seq: frame for frame in held}
        old, new = by_seq.get(last_old.seq), by_seq.get(reads[first].seq)
        if old is None or new is None or old.ts != last_old.ts or new.ts != changed_ts:
            return None
        window = [frame for frame in held if old.ts < frame.ts < new.ts]
        return first_new_frame(window, old, new)
