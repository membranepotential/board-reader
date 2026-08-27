"""The hard-constraint decoder on the serving path (:mod:`.prior` via ``/estimate``).

Kept apart from ``test_server.py`` because these assert one narrow property: the
repair is served *beside* the read and never *as* it. The §5 invariant is that
``placement`` is the argmax of ``square_dist``; the decoder must not touch either,
or the raw distribution stops being raw and a downstream legality smoother ends up
double-counting a prior it cannot see (DESIGN D1).
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from board_reader_core.protocol import (
    CLASS_INDEX,
    NUM_CLASSES,
    START_PLACEMENT,
    parse_placement,
    placement_from_dist,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.estimator import StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.ingest import FrameIngest
from board_reader_inference.routing import route_for
from board_reader_inference.server import create_app
from board_reader_inference.squares import full_frame_corners
from board_reader_inference.trained import TrainedEstimator

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


def test_the_live_stream_carries_the_repair_beside_the_raw_read():
    """The WS path is what the /camera panel displays — the two-kings fix must
    reach it, not just the HTTP endpoints (it shipped without, and a trial's live
    panel showed two kings that no /estimate response would have)."""
    import asyncio

    from board_reader_core.wire import Frame

    from board_reader_inference.hub import PositionHub
    from board_reader_inference.ingest import FrameIngest
    from board_reader_inference.routing import route_for

    est = StubEstimator(TWO_WHITE_KINGS)
    published: list = []

    async def run():
        hub = PositionHub()
        ingest = FrameIngest(
            lambda frame: route_for(est, frame),
            hub,
            on_event=lambda event, frame: published.append((event, frame)),
        )
        ingest.submit(
            Frame(
                board_id="A1",
                seq=7,
                capture_ts=1000.0,
                image=np.zeros((64, 64, 3), dtype=np.uint8),
            )
        )
        await asyncio.wait_for(ingest.drain(), timeout=5.0)
        await ingest.close()
        return hub.latest("A1")

    event = asyncio.run(run())

    # Raw read untouched (the §5 invariant), repair riding beside it.
    assert event.placement == TWO_WHITE_KINGS
    assert event.prior is not None
    assert event.prior.placement.count("K") == 1
    assert event.prior.placement != TWO_WHITE_KINGS
    assert [f.constraint for f in event.prior.flips] == ["kings"]
    # The wire shape a viewer parses.
    assert event.to_dict()["prior"]["flips"][0]["constraint"] == "kings"
    # The on_event hook (the gateway's trial log) saw the same event and its frame.
    assert published and published[0][0] is event and published[0][1].seq == 7


# --- the black-bishop demotion, on the seam both paths share ------------------

#: A midgame board, deliberately NOT the start position: a start-position read
#: would trip the gateway's bank bootstrap (§4.6) and answer through a
#: freshly-calibrated ensemble instead of the stack under test.
MIDGAME = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
#: What that board's raw read is below, before the calibration: no black king.
MIDGAME_RAW = "r1bqbb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"


def _narrow_bishop_estimator() -> TrainedEstimator:
    """A trained stack whose e8 reads 'b' 0.40 over 'k' 0.30 — inside the band.

    Raw, that board has no black king at all (impossible, and the prior decoder
    would have to repair it). Demoted, it is :data:`MIDGAME` exactly. So which
    board the rest of the pipe saw is legible from the outside.
    """
    probs = np.full((64, NUM_CLASSES), 0.001, dtype=np.float64)
    for i, cls in enumerate(c for row in parse_placement(MIDGAME) for c in row):
        probs[i, CLASS_INDEX[cls]] = 0.9
    probs[4] = 0.001
    probs[4, CLASS_INDEX["b"]] = 0.40
    probs[4, CLASS_INDEX["k"]] = 0.30
    probs /= probs.sum(axis=-1, keepdims=True)

    def logits(batch: np.ndarray) -> np.ndarray:
        # Board order, clamped: a tail batch is padded up to batch_size with
        # repeats of its final crop, and those rows are sliced off by the caller.
        n = batch.shape[0]
        rows = [probs[min(logits.i + j, 63)] for j in range(n)]
        logits.i += min(n, 64 - logits.i)
        return np.log(np.stack(rows)).astype(np.float32)

    logits.i = 0
    return TrainedEstimator(logits, batch_size=64)


def _demoted_dist(est: TrainedEstimator, image: np.ndarray) -> np.ndarray:
    """What the served distribution must be: the rule over the RAW read, once."""
    from board_reader_inference.estimator import demote_attractor

    raw = est.photo_dist(image, full_frame_corners(image))
    assert placement_from_dist(raw) == MIDGAME_RAW
    return demote_attractor(raw)


def test_estimate_demotes_the_narrow_bishop_before_the_prior_decodes_it():
    """The order the report specifies: the calibration runs inside the estimator,
    so the decoder is handed the demoted board and finds nothing to repair. Had it
    decoded the raw read instead, it would have had to invent a black king."""
    est = _narrow_bishop_estimator()
    image = render_board(MIDGAME, square_px=32)
    _, buf = cv2.imencode(".jpg", image)
    client = TestClient(create_app(estimator=est))
    body = client.post(
        "/estimate", content=buf.tobytes(), headers={"content-type": "image/jpeg"}
    ).json()

    assert body["placement"] == MIDGAME
    assert body["prior"]["flips"] == []
    assert body["prior"]["placement"] == MIDGAME
    # The swap is a permutation, so e8 now reports the king at exactly the
    # probability the bishop had won it with — a narrow win, so still uncertain.
    raw = _narrow_bishop_estimator().photo_dist(image, full_frame_corners(image))
    assert body["square_dist"][0][4][CLASS_INDEX["k"]] == pytest.approx(
        float(raw[0, 4, CLASS_INDEX["b"]]), abs=1e-3
    )


def test_the_stream_and_the_snapshot_demote_identically_and_exactly_once():
    """``/ingest`` and ``POST /estimate`` reach recognition through the same route
    (:mod:`.routing`), so neither can miss the calibration and neither can add a
    second pass of its own: both serve the rule applied once to the raw read."""
    image = render_board(MIDGAME, square_px=32)
    _, buf = cv2.imencode(".jpg", image)
    expected = _demoted_dist(_narrow_bishop_estimator(), image)

    client = TestClient(create_app(estimator=_narrow_bishop_estimator()))
    snapshot = client.post(
        "/estimate", content=buf.tobytes(), headers={"content-type": "image/jpeg"}
    ).json()

    est = _narrow_bishop_estimator()

    async def run():
        hub = PositionHub()
        ingest = FrameIngest(lambda frame: route_for(est, frame), hub)
        ingest.submit(Frame(board_id="A1", seq=1, capture_ts=1000.0, image=image))
        await asyncio.wait_for(ingest.drain(), timeout=5.0)
        await ingest.close()
        return hub.latest("A1")

    streamed = asyncio.run(run())

    np.testing.assert_allclose(streamed.square_dist, expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(snapshot["square_dist"]), expected, atol=1e-3)
    assert streamed.placement == snapshot["placement"] == MIDGAME
