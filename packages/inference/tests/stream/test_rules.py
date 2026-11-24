"""Tests for the settle guard's conditions (:mod:`board_reader_inference.stream.rules`)."""

from __future__ import annotations

import pytest

from board_reader_core.protocol import START_PLACEMENT

from board_reader_inference.stream.rules import Violation, excess, violation

MIDGAME = "r2qk2r/1b1pbppp/p1n1p3/1ppnP3/5B2/1BNP1N2/PPP2PPP/R2Q1RK1"


def _army(white: str, black: str = "k") -> str:
    """A placement holding ``white`` on ranks 2-4 and ``black`` on ranks 7-5.

    Each argument is the pieces in order, one symbol per square, filled a2, b2, ...
    (white) and a7, b7, ... (black), so a test names counts rather than squares,
    and no pawn lands on a back rank.
    """
    rows = ["."] * 8
    whites = white.ljust(24, ".")
    blacks = black.ljust(24, ".")
    for i in range(3):
        rows[6 - i] = whites[8 * i : 8 * i + 8]  # rank 2 is row 6
        rows[1 + i] = blacks[8 * i : 8 * i + 8]  # rank 7 is row 1
    return "/".join(_rank_fen(r.ljust(8, ".")) for r in rows)


def _rank_fen(rank: str) -> str:
    out, run = "", 0
    for cell in rank:
        if cell == ".":
            run += 1
            continue
        out += (str(run) if run else "") + cell
        run = 0
    return out + (str(run) if run else "")


def test_a_real_game_breaks_nothing():
    assert violation(START_PLACEMENT) == Violation()
    assert violation(MIDGAME).total == 0
    assert violation("8/8/8/8/8/8/8/8").total == 0


@pytest.mark.parametrize(
    ("placement", "expected"),
    [
        (_army("KK"), Violation(kings=1)),
        (_army("KKK"), Violation(kings=2)),
        (_army("K", "kk"), Violation(kings=1)),
        (_army("K" + "Q" * 3 + "P" * 8), Violation(promotions=2)),  # 2 promotions + 8 pawns
        (_army("K" + "NNNN" + "RRR" + "P" * 8), Violation(promotions=3)),
        (_army("K", "k" + "qq" + "p" * 8), Violation(promotions=1)),
    ],
)
def test_each_condition_counts_its_own_excess(placement, expected):
    assert violation(placement) == expected


def test_two_white_kings_count_although_python_chess_lets_them_pass():
    """``chess.Board.status`` flags only three kings in all; we count per colour."""
    assert violation(_army("KK", "")).kings == 1


def test_the_kings_condition_is_at_most_one_never_exactly_one():
    """A hand or an occluded back rank hides a king (DESIGN D8): no king is fine."""
    assert violation(_army("P" * 8, "k")).total == 0
    assert violation(_army("P" * 8, "")).total == 0


def test_the_piece_count_binds_only_alongside_a_king_error():
    """The promotion budget already implies <=16 pieces when a side has one king,
    so this condition can only bind on its own when the king count is also wrong."""
    # 15 officers + 2 kings = 17 white pieces, spending exactly the 8 promotions
    # the budget allows, so only the count and the king are wrong.
    assert violation(_army("KK" + "NNNN" + "BBBB" + "RRRR" + "QQQ")) == Violation(
        kings=1, pieces=1
    )
    # With one king, 17 pieces cannot be reached without breaking the budget.
    assert violation(_army("K" + "P" * 9 + "NNBBRRQ")).promotions > 0


def test_free_rules_count_nothing():
    """``free`` is the switch that turns the guard off: no board is impossible."""
    two_kings = _army("KK")
    assert excess(two_kings, "standard") == violation(two_kings).total == 1
    assert excess(two_kings, "free") == 0
