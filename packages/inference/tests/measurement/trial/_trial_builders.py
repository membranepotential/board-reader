"""Builders for the trial tests: a recording on disk, its events, and a truth to score against.

Shared by every ``test_*.py`` here, which import it by name (the same arrangement
as ``tests/_sim_recognizer.py``), so keep the basename unique across the workspace.
"""

from __future__ import annotations

from board_reader_core.protocol import GRID, START_PLACEMENT, parse_placement
from board_reader_core.recording import FrameRecorder, Recording
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.measurement.trial.evidence import SQUARE_INDEX
from board_reader_inference.measurement.trial.truth import SessionTruth, TruthSegment

# Rank 8 read with two black kings — the 08-13 trial's actual raw failure. One
# square wrong (d8), and impossible.
TWO_KINGS = "rnbkkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
# The k/q swap of the same squares: possible, still wrong, and wrong on *two*
# squares.
SWAPPED = "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"


def event_line(
    frame_seq: int,
    placement: str,
    *,
    settled: str | None = None,
    board_id: str = "phone",
    evidence: list | None = None,
) -> dict:
    """One events.ndjson line, in the shape ``server._record_published`` writes."""
    line: dict = {
        "board_id": board_id,
        "seq": frame_seq,
        "utc": "2026-08-13T17:33:47.820Z",
        "placement": placement,
        "confidence": 0.78,
        "status": "ok",
        "frame_seq": frame_seq,
        "capture_ts": 1786642427.0 + frame_seq,
        "published_ts": 1786642429.0 + frame_seq,
    }
    if settled is not None:
        line["settled"] = {"placement": settled, "pending": []}
    if evidence is not None:
        line["evidence"] = evidence
    return line


def digest(placement: str, **rivals: tuple[str, float, float]) -> list:
    """A top-2 digest agreeing with ``placement``, board order.

    Each keyword names a square whose read was close: ``d8=("q", 0.51, 0.47)``
    means the winner (whatever ``placement`` says) took 0.51 and ``q`` 0.47.
    Every other square is a confident, uncontested read.
    """
    grid = parse_placement(placement)
    squares = [
        [[grid[r][c], 0.99], ["empty" if grid[r][c] != "empty" else "P", 0.005]]
        for r in range(GRID)
        for c in range(GRID)
    ]
    for name, (runner_up, won, lost) in rivals.items():
        i = SQUARE_INDEX[name]
        squares[i] = [[squares[i][0][0], won], [runner_up, lost]]
    return squares


def record_session(tmp_path, events: list[dict], frames: list[tuple[dict, bytes]] = ()) -> Recording:
    recorder = FrameRecorder(tmp_path, session_id="s", writer="test")
    for header, jpeg in frames:
        recorder.record(header, jpeg)
    for line in events:
        recorder.record_event(line)
    recorder.stop()
    return Recording(recorder.path)


def photo_frames(n: int, placement: str = START_PLACEMENT) -> list[tuple[dict, bytes]]:
    """``n`` recorded frames of ``placement``, one second apart."""
    photo = render_board(placement)
    return [
        Frame(board_id="phone", seq=seq, capture_ts=1786642427.0 + seq, image=photo).encode()
        for seq in range(n)
    ]


def truth_of(**over) -> SessionTruth:
    return SessionTruth(
        board_id=over.pop("board_id", "phone"),
        segments=over.pop(
            "segments",
            (TruthSegment(0, START_PLACEMENT), TruthSegment(100, E4)),
        ),
        guard_seqs=over.pop("guard_seqs", 5),
    )


#: The 08-13 failure in miniature: a raw board that alternates between right and
#: a two-square-wrong k/q swap, and a move at seq 100 whose first two reads land
#: inside the guard.
FLIPFLOP_EVENTS = [
    event_line(0, START_PLACEMENT),
    event_line(10, SWAPPED),
    event_line(20, START_PLACEMENT),
    event_line(30, SWAPPED),
    event_line(100, E4),  # in the guard
    event_line(103, E4),  # in the guard
    event_line(110, E4),
    event_line(120, E4),
]
