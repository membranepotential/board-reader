"""Ground truth for a recorded trial: ``truth.json``, a step function over frame seq.

A trial is not a set of photos; it is one board changing a handful of times. So
truth is a list of *segments*, each holding from its ``from_seq`` until the next
one starts::

    {"format": 1,
     "board_id": "phone",
     "segments": [{"from_seq": 0,   "placement": "rnbqkbnr/pppppppp/8/8/..."},
                  {"from_seq": 130, "placement": "rnbqkbnr/pppppppp/8/8/..."}],
     "guard_seqs": 12}

**The guard is the honest part.** A move takes a hand over the board for about a
second, and during it there is no ground truth to speak of — the operator's
segment boundary is one instant, the occlusion is a window. Events whose frame
falls within ``guard_seqs`` frames *after* a boundary are therefore excluded from
scoring rather than counted as errors (12 seqs is ~1.5 s at the 08-13 trial's
7.5 fps). Excluded counts are reported, because a guard that swallows half the
session is a measurement, not a detail.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from board_reader_core.protocol import parse_placement
from board_reader_core.recording import Recording

#: Where a session's ground truth lives, inside the recording it describes.
TRUTH_NAME = "truth.json"
TRUTH_FORMAT = 1

#: Frames after a segment boundary that are not scored — the hand-over-the-board
#: window. ~1.5 s at the 08-13 trial's 7.5 fps.
DEFAULT_GUARD_SEQS = 12


class TruthError(ValueError):
    """A ``truth.json`` that cannot be trusted to score anything."""


@dataclass(frozen=True)
class TruthSegment:
    """One board, and the frame seq it starts holding from."""

    from_seq: int
    placement: str

    def __post_init__(self) -> None:
        if self.from_seq < 0:
            raise TruthError(f"from_seq must be >= 0, got {self.from_seq}")
        parse_placement(self.placement)  # a malformed board fails at load, not mid-score


@dataclass(frozen=True)
class SessionTruth:
    """What was on the board, over the whole session."""

    board_id: str
    segments: tuple[TruthSegment, ...]
    guard_seqs: int = DEFAULT_GUARD_SEQS
    starts: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.board_id:
            raise TruthError("board_id must be a non-empty string")
        if not self.segments:
            raise TruthError("truth has no segments")
        if self.guard_seqs < 0:
            raise TruthError(f"guard_seqs must be >= 0, got {self.guard_seqs}")
        starts = tuple(s.from_seq for s in self.segments)
        if any(b <= a for a, b in pairwise(starts)):
            raise TruthError(f"segment from_seqs must be strictly increasing, got {list(starts)}")
        object.__setattr__(self, "starts", starts)

    @property
    def boundaries(self) -> int:
        """Segment changes — one fewer than the segments, the first is the opening."""
        return len(self.segments) - 1

    def index_for(self, frame_seq: int) -> int:
        """Index of the segment holding ``frame_seq``; ``-1`` before the first."""
        return bisect_right(self.starts, frame_seq) - 1

    def in_guard(self, frame_seq: int) -> bool:
        """Whether ``frame_seq`` sits in the occlusion window after a boundary.

        The *first* segment's start is never guarded: a session opening on a board
        is news, not a move with a hand over it.
        """
        i = self.index_for(frame_seq)
        if i <= 0:
            return False
        return frame_seq - self.starts[i] < self.guard_seqs

    def check_covers(self, first_seq: int) -> None:
        """Reject truth that starts after the stream does.

        A first segment beginning mid-session would silently drop every event
        before it, which reads as "the guard was generous" rather than as the
        authoring mistake it is. ``0`` is always accepted so a truth file can be
        written without looking up the recording's first seq.
        """
        if self.starts[0] not in (0, first_seq):
            raise TruthError(
                f"the first segment starts at seq {self.starts[0]}, but the stream's "
                f"first frame is seq {first_seq} — the first from_seq must be 0 or "
                f"{first_seq}, or the events before it have no truth"
            )

    def to_dict(self) -> dict:
        return {
            "format": TRUTH_FORMAT,
            "board_id": self.board_id,
            "segments": [
                {"from_seq": s.from_seq, "placement": s.placement} for s in self.segments
            ],
            "guard_seqs": self.guard_seqs,
        }


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise TruthError(message)


def load_truth(path: str | Path) -> SessionTruth:
    """Parse and validate a ``truth.json``. Raises :class:`TruthError` on anything off."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise TruthError(f"cannot read {p}: {e}") from e
    except json.JSONDecodeError as e:
        raise TruthError(f"{p} is not valid JSON: {e}") from e
    _require(isinstance(data, dict), f"{p} must hold a JSON object")
    _require(
        data.get("format") == TRUTH_FORMAT,
        f"unsupported truth format {data.get('format')!r} (this reader speaks {TRUTH_FORMAT})",
    )
    board_id = data.get("board_id")
    _require(isinstance(board_id, str) and bool(board_id), "board_id must be a non-empty string")
    raw = data.get("segments")
    _require(isinstance(raw, list) and bool(raw), "segments must be a non-empty list")
    segments: list[TruthSegment] = []
    for i, rec in enumerate(raw):
        _require(isinstance(rec, dict), f"segment {i} must be an object")
        _require("from_seq" in rec and "placement" in rec, f"segment {i} needs from_seq and placement")
        try:
            segments.append(TruthSegment(int(rec["from_seq"]), str(rec["placement"])))
        except (TypeError, ValueError) as e:
            raise TruthError(f"segment {i}: {e}") from e
    guard = data.get("guard_seqs", DEFAULT_GUARD_SEQS)
    try:
        guard_seqs = int(guard)
    except (TypeError, ValueError) as e:
        raise TruthError(f"guard_seqs must be an integer, got {guard!r}") from e
    return SessionTruth(
        board_id=str(board_id), segments=tuple(segments), guard_seqs=guard_seqs
    )


def truth_skeleton(recording: Recording) -> str:
    """A fill-in-the-blanks ``truth.json`` for ``recording``, as JSON text.

    Printed when the file is missing: the two things the operator would otherwise
    have to go dig out of ``frames.ndjson`` — the board id and the first seq — are
    already filled in.
    """
    from board_reader_core.protocol import START_PLACEMENT

    board_id, first_seq = "board", 0
    for rec in recording.frames():
        board_id, first_seq = rec.board_id, rec.seq
        break
    return json.dumps(
        {
            "format": TRUTH_FORMAT,
            "board_id": board_id,
            "segments": [
                {"from_seq": first_seq, "placement": START_PLACEMENT},
                {"from_seq": first_seq + 100, "placement": "<the board after the 1st move>"},
            ],
            "guard_seqs": DEFAULT_GUARD_SEQS,
        },
        indent=2,
    )
