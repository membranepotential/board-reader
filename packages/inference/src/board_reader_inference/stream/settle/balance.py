"""**An obscured read commits a piece that moved, never one that only vanished.** A hand resting
still on the board makes no motion, so the schedule reads it, and ChessQ reads the
covered squares as empty at p >= 0.96: confidence cannot tell. The frame energy can
(:func:`~board_reader_inference.recognition.chessq.frame_energy`), and above
:attr:`~board_reader_inference.stream.settle.config.SettleConfig.obscured_energy` the read is *obscured*. It still starts and
extends challengers, because a mid-move read with the hand on the piece also says
something true (the lifted piece's square is empty). Dropping obscured reads
outright cost 15 gate boards in the probe (docs/reports/2026-11-24-hand-signal-probe.md).

What such a read may commit is a *balanced piece group* (:func:`balanced_squares`):
ripe squares where as many lose piece P to "empty" as gain P. The energy says that
a hand is in the frame, not where: a covered square looks normal. But a hand hides
pieces, which gives losses without gains, and those never balance; a move, a
capture and castling do. So a move in plain view shows while a hand rests
elsewhere on the board (on the live take 20261125-024330 a hand at the far edge
held nine moves back, one of them for 13 s), and a piece still in the hand, a
covered piece and a promotion wait for a clear read, which confirms or resets
each challenger (docs/reports/2026-11-25-partial-commit-under-hand.md).

The groups are judged per piece, so this is not "a whole move": castling shows its
king before its rook when only the king's squares are ripe, and en passant shows
the capturing pawn's move while the captured pawn, which only vanished, stays
until a clear read.
"""

from __future__ import annotations

from board_reader_core.protocol import CLASSES, EMPTY


def balanced_squares(
    settled: list[list[str]], grid: list[list[str]], ripe: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """The ripe squares an obscured read may commit (module docstring).

    A ripe square either loses its settled piece to "empty" or gains the read's
    piece (on an empty square or over another piece, a capture). The squares of
    piece P form a group, and the group is *balanced* when as many squares lose P
    as gain P: every P that left a square stands on another one. The squares of
    the balanced groups are returned, the rest are not.

    This counts pieces and consults no move rules, like the guard: a knight that
    "moves" like a bishop balances too. What it refuses is a piece that only
    vanished (a hand over it, or a piece in the hand) and a piece that only
    appeared (a misread at the hand's edge, a piece set down after its square
    emptied on an earlier commit).

    Committing these squares cannot raise the board's excess, so the caller needs
    no guard: per piece the count stays the same, a capture lowers the captured
    piece's count, and every condition of :func:`~.rules.violation` grows only
    with a count.
    """
    empty = CLASSES[EMPTY]
    lost: dict[str, list[tuple[int, int]]] = {}
    gained: dict[str, list[tuple[int, int]]] = {}
    for r, c in ripe:
        if grid[r][c] == empty:
            lost.setdefault(settled[r][c], []).append((r, c))
        else:
            gained.setdefault(grid[r][c], []).append((r, c))
    return [
        square
        for piece, squares in lost.items()
        if len(squares) == len(gained.get(piece, ()))
        for square in squares + gained[piece]
    ]
