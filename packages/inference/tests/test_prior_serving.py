"""The hard-constraint decoder on the serving path (:mod:`.prior` via ``/estimate``).

Kept apart from ``test_server.py`` because these assert one narrow property: the
repair is served *beside* the read and never *as* it. The §5 invariant is that
``placement`` is the argmax of ``square_dist``; the decoder must not touch either,
or the raw distribution stops being raw and a downstream legality smoother ends up
double-counting a prior it cannot see (DESIGN D1).
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT, placement_from_dist
from board_reader_core.sim import render_board

from board_reader_inference.estimator import StubEstimator
from board_reader_inference.server import create_app

# The start position with e1 holding a second white king: impossible, and
# impossible in the one way the decoder is best at (``kings``).
TWO_WHITE_KINGS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNK"


def _estimate(placement: str) -> dict:
    client = TestClient(create_app(estimator=StubEstimator(placement)))
    _, buf = cv2.imencode(".jpg", render_board(START_PLACEMENT, square_px=32))
    response = client.post(
        "/estimate", content=buf.tobytes(), headers={"content-type": "image/jpeg"}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_estimate_serves_the_repair_without_touching_the_read():
    body = _estimate(TWO_WHITE_KINGS)

    # The read itself is exactly what the estimator produced, invariant intact.
    assert body["placement"] == TWO_WHITE_KINGS
    assert placement_from_dist(np.asarray(body["square_dist"])) == TWO_WHITE_KINGS

    # The repair rides alongside, and it is a *different*, possible board.
    assert body["prior"]["placement"] != TWO_WHITE_KINGS
    assert body["prior"]["placement"].count("K") == 1


def test_the_repair_names_the_square_it_changed_and_the_rule_that_forced_it():
    """A silent repair is not auditable — every flip says where, what and why."""
    (flip,) = _estimate(TWO_WHITE_KINGS)["prior"]["flips"]

    assert flip["square"] in ("e1", "h1")  # whichever king was the cheaper to give up
    assert flip["from"] == "K"
    assert flip["constraint"] == "kings"


def test_a_possible_board_is_served_unrepaired():
    """The common case: nothing violated, so nothing changed, and it says so."""
    body = _estimate(START_PLACEMENT)

    assert body["prior"]["flips"] == []
    assert body["prior"]["placement"] == body["placement"] == START_PLACEMENT
    assert body["prior"]["residual"] == {}


def test_the_back_rank_constraint_sits_out_when_the_frame_is_only_guessed():
    """Serving never passes ``quarter_turns``; the guesser minimises the very
    quantity the constraint measures, so trusting it deletes correct pawns.

    A pawn on rank 8 is left alone here *by design* — the alternative is acting on
    an orientation nobody established.
    """
    # Black's a8 rook read as a pawn, with a7 empty: 8 black pawns, 15 black
    # pieces, one king, promotion budget exactly met — the ONLY thing wrong with
    # this board is the pawn standing on rank 8.
    pawn_on_rank_8 = "pnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    body = _estimate(pawn_on_rank_8)

    assert body["prior"]["flips"] == []
    assert body["prior"]["placement"] == pawn_on_rank_8
