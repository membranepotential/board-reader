"""The recognizer seam (:mod:`board_reader_inference.recognizer`, DESIGN D10).

ChessQ's own contract is in ``test_chessq.py``; this file pins what every
recognizer gets from the seam: the stub, the event a distribution becomes, the
estimator adapter the gateway serves, and the environment builder.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from board_reader_core.protocol import CLASSES, START_PLACEMENT, parse_placement
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.estimator import Estimator, StubEstimator
from board_reader_inference.recognizer import (
    ORIENTATION,
    RecognizerEstimator,
    StubRecognizer,
    build_from_env,
    event_from_dist,
)
from board_reader_inference.routing import PhotoRead, route_for
from board_reader_inference.squares import full_frame_corners

MIDGAME = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R"


def _frame(**kw: object) -> Frame:
    base = Frame(board_id="B7", seq=41, capture_ts=1_700_000_000.25, image=render_board())
    return replace(base, **kw)


def test_stub_reads_its_authored_placement_whatever_the_pixels():
    stub = StubRecognizer(MIDGAME, confidence=0.8)
    assert (stub.name, stub.frame) == ("stub", "chess")
    for image in (render_board(), np.zeros((50, 90, 3), dtype=np.uint8)):
        dist = stub.read(image)
        assert dist.shape == (8, 8, len(CLASSES))
        np.testing.assert_array_equal(
            np.array(CLASSES)[dist.argmax(-1)], np.array(parse_placement(MIDGAME))
        )
        np.testing.assert_allclose(dist.max(-1), 0.8, atol=1e-6)


def test_event_from_dist_keeps_the_distribution_and_the_frame_stamps():
    dist = StubRecognizer(MIDGAME).read(render_board())
    frame = _frame(stability=0.3, clock="white")
    event = event_from_dist(frame, dist)
    assert (event.board_id, event.seq, event.status) == ("B7", 41, "ok")
    assert event.utc.startswith("2023-11-14T22:13:20.25")
    assert (event.stability, event.clock.running) == (0.3, "white")
    assert event.orientation == ORIENTATION == "a1_bottom_left"
    assert event.placement == MIDGAME
    np.testing.assert_array_equal(event.square_dist, dist)


def test_adapter_is_an_estimator_on_the_photo_path_that_ignores_corners():
    est = RecognizerEstimator(StubRecognizer(MIDGAME))
    assert isinstance(est, Estimator)
    frame = _frame(corners=full_frame_corners(render_board()) * 0.5)
    route = route_for(est, frame)
    assert isinstance(route, PhotoRead)
    assert route.frame.image is frame.image  # original pixels, no warp
    assert route.read().placement == MIDGAME
    assert est.estimate(_frame()).placement == MIDGAME


def test_build_from_env_without_br_chessq_keeps_the_old_tiers(monkeypatch):
    for name in ("BR_CHESSQ", "BR_MODEL", "BR_MODEL_B", "BR_MODEL_TUNED"):
        monkeypatch.delenv(name, raising=False)
    est = build_from_env()
    assert isinstance(est, StubEstimator)
    assert est.placement == START_PLACEMENT
