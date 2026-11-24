"""The recognizer seam (:mod:`board_reader_inference.recognition.recognizer`, DESIGN D10).

ChessQ's own contract is in ``test_chessq.py``; this file pins what every
recognizer gets from the seam: the stub, the event a distribution becomes, the
frame read the gateway runs, and the environment builder.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from board_reader_core.protocol import CLASSES, START_PLACEMENT, parse_placement
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.recognition.recognizer import (
    ORIENTATION,
    Recognition,
    StubRecognizer,
    build_from_env,
    event_from_dist,
    frame_reader,
)

MIDGAME = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R"


def _frame(**kw: object) -> Frame:
    base = Frame(board_id="B7", seq=41, capture_ts=1_700_000_000.25, image=render_board())
    return replace(base, **kw)


def test_stub_reads_its_authored_placement_whatever_the_pixels():
    stub = StubRecognizer(MIDGAME, confidence=0.8)
    assert stub.name == "stub"
    for image in (render_board(), np.zeros((50, 90, 3), dtype=np.uint8)):
        recognition = stub.read(image)
        assert recognition.direction is None  # the stub knows no board direction
        dist = recognition.square_dist
        assert dist.shape == (8, 8, len(CLASSES))
        np.testing.assert_array_equal(
            np.array(CLASSES)[dist.argmax(-1)], np.array(parse_placement(MIDGAME))
        )
        np.testing.assert_allclose(dist.max(-1), 0.8, atol=1e-6)


def test_event_from_dist_keeps_the_distribution_and_the_frame_stamps():
    dist = StubRecognizer(MIDGAME).read(render_board()).square_dist
    frame = _frame(stability=0.3, clock="white")
    event = event_from_dist(frame, dist)
    assert (event.board_id, event.seq, event.status) == ("B7", 41, "ok")
    assert event.utc.startswith("2023-11-14T22:13:20.25")
    assert (event.stability, event.clock.running) == (0.3, "white")
    assert event.orientation == ORIENTATION == "a1_bottom_left"
    assert event.placement == MIDGAME
    np.testing.assert_array_equal(event.square_dist, dist)


def test_a_frame_read_hands_the_recognizer_the_whole_image_and_ignores_corners():
    seen: list[np.ndarray] = []

    class Spy(StubRecognizer):
        def read(self, image: np.ndarray) -> Recognition:
            seen.append(image)
            return Recognition(super().read(image).square_dist, direction=88.5)

    frame = _frame(corners=np.array([[1, 2], [60, 3], [62, 61], [0, 59]], dtype=np.float32))
    reading = frame_reader(Spy(MIDGAME))(frame)
    assert len(seen) == 1 and seen[0] is frame.image  # as sent: no crop, no warp
    assert (reading.event.placement, reading.event.seq) == (MIDGAME, 41)
    assert reading.direction == 88.5  # the recognizer's direction rides beside the event


def test_build_from_env_without_br_chessq_serves_the_stub_loudly(monkeypatch, capsys):
    monkeypatch.delenv("BR_CHESSQ", raising=False)
    recognizer = build_from_env()
    assert isinstance(recognizer, StubRecognizer)
    assert recognizer.placement == START_PLACEMENT
    assert "BR_CHESSQ is unset" in (recognizer.fallback or "")
    assert "STUB recognizer" in capsys.readouterr().err
