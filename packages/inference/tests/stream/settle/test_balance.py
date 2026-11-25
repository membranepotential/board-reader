"""Tests for the obscured read's balanced piece groups (:mod:`board_reader_inference.stream.settle.balance`)."""

from __future__ import annotations

import pytest
from _settle_builders import (
    E4,
    E4_D5,
    E4_GONE_A2,
    E4_GONE_B8,
    EXD5,
    GONE_B8,
    SWAP,
)

from board_reader_core.protocol import START_PLACEMENT, encode_placement, parse_placement

from board_reader_inference.stream.settle.balance import balanced_squares
from board_reader_inference.stream.settle.rules import violation

#: White ready to castle short, castled, and the king moved with the rook to follow.
CAN_CASTLE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R"
CASTLED = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1"
KING_ONLY = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ2KR"
#: White's e5 pawn takes d5 en passant, and a white pawn promotes on a8.
EP_BEFORE = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR"
EP_AFTER = "rnbqkbnr/ppp1pppp/3P4/8/8/8/PPPP1PPP/RNBQKBNR"
PROMOTE_BEFORE = "4k3/P7/8/8/8/8/8/4K3"
PROMOTE_AFTER = "Q3k3/8/8/8/8/8/8/4K3"


def _squares(*names: str) -> list[tuple[int, int]]:
    """Algebraic names -> the stabilizer's (row, column) squares."""
    return [(8 - int(name[1]), "abcdefgh".index(name[0])) for name in names]


def _balanced(settled: str, read: str) -> set[tuple[int, int]]:
    """``balanced_squares`` with every differing square ripe."""
    was, now = parse_placement(settled), parse_placement(read)
    ripe = [(r, c) for r in range(8) for c in range(8) if was[r][c] != now[r][c]]
    return set(balanced_squares(was, now, ripe))


@pytest.mark.parametrize(
    ("settled", "read", "squares"),
    [
        (START_PLACEMENT, E4, ("e2", "e4")),  # a move
        (E4_D5, EXD5, ("e4", "d5")),  # a capture: the pawn left e4 and stands on d5
        (CAN_CASTLE, CASTLED, ("e1", "g1", "h1", "f1")),  # castling: two groups
        (START_PLACEMENT, GONE_B8, ()),  # a piece that only vanished
        (GONE_B8, START_PLACEMENT, ()),  # a piece that only appeared
        (START_PLACEMENT, SWAP, ()),  # two pieces misread as each other: no square emptied
        (START_PLACEMENT, E4_GONE_B8, ("e2", "e4")),  # the move, not the hidden knight
        (START_PLACEMENT, E4_GONE_A2, ()),  # two pawns left, one arrived: which one moved?
        (CAN_CASTLE, KING_ONLY, ("e1", "g1")),  # the king's group balances on its own
        (EP_BEFORE, EP_AFTER, ("e5", "d6")),  # en passant: the captured pawn only vanished
        (PROMOTE_BEFORE, PROMOTE_AFTER, ()),  # a promotion: a pawn vanished, a queen appeared
    ],
)
def test_balanced_squares_are_the_pieces_that_left_one_square_for_another(
    settled, read, squares
):
    assert _balanced(settled, read) == set(_squares(*squares))


def test_balanced_squares_looks_at_ripe_squares_only():
    """A group is judged on what is ripe: the move's target alone is a piece that
    only appeared, however clearly the read shows the source empty too."""
    was, now = parse_placement(START_PLACEMENT), parse_placement(E4)
    assert balanced_squares(was, now, _squares("e4")) == []
    assert balanced_squares(was, now, []) == []


def test_committing_balanced_squares_never_raises_the_excess():
    """Why an obscured commit needs no guard: per piece the count stays the same
    and a capture only removes a piece. Seeded boards from a small alphabet, so
    that kings, pawns and queens sit at their limits; each read makes a few moves
    and a few misreads. Adopting the whole read does raise the excess on the same
    boards, so the generator reaches the conditions."""
    import random

    rng = random.Random(20261125)
    alphabet = ["empty"] * 6 + list("KkPpQqNn")
    cells = [(r, c) for r in range(2) for c in range(8)]
    pad = [["empty"] * 8 for _ in range(6)]

    def total(grid: list[list[str]]) -> int:
        return violation(encode_placement(grid)).total

    committed = whole_read_raises = 0
    for _ in range(1000):
        was = [[rng.choice(alphabet) for _ in range(8)] for _ in range(2)] + pad
        now = [row[:] for row in was]
        for _ in range(rng.randint(1, 3)):  # moves and captures
            (r0, c0), (r1, c1) = rng.sample(cells, 2)
            now[r1][c1], now[r0][c0] = now[r0][c0], "empty"
        for _ in range(rng.randint(0, 2)):  # misreads
            r, c = rng.choice(cells)
            now[r][c] = rng.choice(alphabet)
        ripe = [(r, c) for r, c in cells if was[r][c] != now[r][c]]
        taken = balanced_squares(was, now, ripe)
        after = [row[:] for row in was]
        for r, c in taken:
            after[r][c] = now[r][c]
        committed += bool(taken)
        whole_read_raises += total(now) > total(was)
        assert total(after) <= total(was)
    assert committed > 300
    assert whole_read_raises > 20
