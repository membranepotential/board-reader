"""Tests for the inference gateway (DESIGN.md §4.5) via Starlette TestClient."""

from __future__ import annotations

import json

from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame
from board_reader_inference.estimator import StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.server import create_app


def _client(**kw):
    return TestClient(create_app(StubEstimator(START_PLACEMENT), PositionHub()))


def _send_frame(ws, seq=0):
    frame = Frame(board_id="A1", seq=seq, capture_ts=float(seq), image=render_board())
    header, jpeg = frame.encode()
    ws.send_text(json.dumps(header))
    ws.send_bytes(jpeg)


def test_healthz_ok():
    r = _client().get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_boards_404_before_any_frame():
    assert _client().get("/boards/A1").status_code == 404


def test_ingest_frame_then_board_snapshot_available():
    client = _client()
    with client.websocket_connect("/ingest") as ingest:
        _send_frame(ingest, seq=5)
        ack = ingest.receive_json()
        assert ack == {"ack": 5, "board_id": "A1"}
    snap = client.get("/boards/A1")
    assert snap.status_code == 200
    assert snap.json()["placement"] == START_PLACEMENT


def test_end_to_end_ingest_to_viewer_stream():
    """A frame pushed on /ingest reaches a /viewer subscriber as a position event."""
    client = _client()
    with client.websocket_connect("/viewer") as viewer:
        with client.websocket_connect("/ingest") as ingest:
            _send_frame(ingest, seq=1)
            ingest.receive_json()  # ack
            event = json.loads(viewer.receive_text())
    assert event["board_id"] == "A1"
    assert event["placement"] == START_PLACEMENT
    assert event["status"] == "ok"
    assert "square_dist" in event


def test_viewer_replays_retained_latest_on_connect():
    client = _client()
    with client.websocket_connect("/ingest") as ingest:
        _send_frame(ingest, seq=9)
        ingest.receive_json()
    # A viewer connecting AFTER the frame still gets the retained latest immediately.
    with client.websocket_connect("/viewer") as viewer:
        event = json.loads(viewer.receive_text())
    assert event["seq"] == 9
