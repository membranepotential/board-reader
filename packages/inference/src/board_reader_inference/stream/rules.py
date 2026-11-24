"""Boards no game of standard chess can reach: the conditions the settle guard counts.

:class:`~board_reader_inference.stream.stabilize.BoardStabilizer` composes its
settled board out of squares that settle one by one, and a naive per-square
debounce can compose a board no read ever claimed (two black kings, from the two
halves of a k/q swap). Its guard refuses a commit that makes the settled board
*more* impossible than it already is, and this module says how impossible a board
is: :func:`violation` counts how far a placement breaks each condition, and
:attr:`Violation.total` is the one integer the guard compares.

The conditions are necessary ones on any position of a standard game, whoever is
to move. They are not legality (DESIGN D1): the guard never asks how a position
was reached, and the raw read on the wire is never changed.

* **Kings** — at most one per colour. Deliberately ``<= 1`` and not ``== 1``: a
  hand or the back-rank occlusion of DESIGN D8 legitimately hides a king, so a
  board with no king is not impossible.
* **Pieces** — at most 16 per colour; a side can never gain one.
* **Promotions** — per colour, every officer beyond the starting set (two knights,
  two bishops, two rooks, one queen) used up a pawn, and a surviving pawn has not
  promoted, so surplus officers plus pawns are at most 8.

The board is a :class:`chess.Board` (python-chess, GPL-3.0-or-later; see
THIRD-PARTY.md). Its own :meth:`chess.Board.status` does not fit the counts:
``TOO_MANY_KINGS`` fires only on three kings in all, so two white kings pass, and
it has no promotion budget. The counts are therefore taken here, off its
bitboards.

The placement is in the chess frame (row 0 is rank 8), as every read is since the
recognizer resolves the orientation (DESIGN D10, locked decision #15).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chess

from board_reader_core.control import Rules

#: A side starts with 16 pieces and can never gain one.
MAX_PIECES = 16

#: A side starts with 8 pawns, so at most 8 promotions can ever have happened.
PROMOTION_BUDGET = 8

#: How many of each officer a side holds without a promotion having happened.
_STARTING_OFFICERS = ((chess.KNIGHT, 2), (chess.BISHOP, 2), (chess.ROOK, 2), (chess.QUEEN, 1))


@dataclass(frozen=True, slots=True)
class Violation:
    """How far a board is from possible: one non-negative excess per condition."""

    kings: int = 0
    pieces: int = 0
    promotions: int = 0

    @property
    def total(self) -> int:
        """The integer the settle guard keeps from rising."""
        return self.kings + self.pieces + self.promotions


def board(placement: str) -> chess.Board:
    """``placement`` (FEN field 1) as a python-chess board, White to move.

    No castling rights and no en-passant square: a placement says neither.
    """
    return chess.Board(f"{placement} w - - 0 1")


def _count(position: chess.Board, piece: chess.PieceType, color: chess.Color) -> int:
    return chess.popcount(position.pieces_mask(piece, color))


def _promotions(position: chess.Board, color: chess.Color) -> int:
    surplus = sum(
        max(0, _count(position, piece, color) - free) for piece, free in _STARTING_OFFICERS
    )
    return surplus + _count(position, chess.PAWN, color)


#: Placements whose :func:`violation` is kept. Building a :class:`chess.Board`
#: from a placement costs about 0.1 ms, the guard asks several times per read,
#: and a board's stream repeats the same few placements over and over.
CACHED_PLACEMENTS = 4096


@lru_cache(maxsize=CACHED_PLACEMENTS)
def violation(placement: str) -> Violation:
    """How far ``placement`` breaks each condition (module docstring)."""
    position = board(placement)
    colors = (chess.WHITE, chess.BLACK)
    return Violation(
        kings=sum(max(0, _count(position, chess.KING, c) - 1) for c in colors),
        pieces=sum(
            max(0, chess.popcount(position.occupied_co[c]) - MAX_PIECES) for c in colors
        ),
        promotions=sum(max(0, _promotions(position, c) - PROMOTION_BUDGET) for c in colors),
    )


def excess(placement: str, rules: Rules) -> int:
    """How impossible ``placement`` is under ``rules``, as the guard compares it.

    ``free`` counts nothing, so the guard never refuses a settle there: the board
    holds whatever the reads converge to, extra material and variants included.
    """
    if rules == "free":
        return 0
    return violation(placement).total
