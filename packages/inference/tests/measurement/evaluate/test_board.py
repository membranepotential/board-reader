"""Tests for scoring one board (:mod:`board_reader_inference.measurement.evaluate.board`).

One wrong square costs a whole board; errors carry their square and confidence.
"""

from __future__ import annotations

import pytest
from _evaluate_builders import MIDGAME, dist_with, labeled_board

from board_reader_core.protocol import GRID, dist_from_placement, parse_placement

from board_reader_inference.measurement.evaluate.board import score_board


def test_a_flawless_read_is_a_perfect_board():
    result = score_board(labeled_board(), dist_from_placement(MIDGAME))
    assert result.perfect and result.correct == 64 and not result.errors
    assert result.predicted == MIDGAME


def test_one_wrong_square_costs_the_whole_board():
    """The headline metric's entire reason for existing."""
    result = score_board(labeled_board(), dist_with(MIDGAME, {"g1": ("B", 0.9)}))
    assert not result.perfect
    assert result.correct == 63
    assert [str(e) for e in result.errors] == ["g1 K->B"]
    assert result.errors[0].kind == "type"


def test_errors_carry_their_square_and_confidence():
    # d2 is empty in MIDGAME (rank 2 is PPP2PPP) — a pawn there is a hallucination.
    result = score_board(labeled_board(), dist_with(MIDGAME, {"d2": ("P", 0.97)}))
    (err,) = result.errors
    assert err.square == "d2" and err.truth == "empty" and err.predicted == "P"
    assert err.kind == "hallucination"
    assert err.confidence == pytest.approx(0.97, abs=1e-6)


def test_square_naming_matches_the_placement_frame():
    """Row 0 is rank 8 and column 0 is file a — an inverted axis would rename
    every error and make reports point at the wrong squares."""
    truth = parse_placement(MIDGAME)
    assert truth[0][0] == "r"  # a8, first rank listed / first file
    assert truth[GRID - 1][0] == "R"  # a1, last rank listed / first file
    assert truth[GRID - 1][GRID - 2] == "K"  # g1
    result = score_board(labeled_board(), dist_with(MIDGAME, {"a8": ("empty", 0.9)}))
    assert str(result.errors[0]) == "a8 r->empty"
