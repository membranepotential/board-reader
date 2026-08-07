"""Tests for ``GET /boards/{id}/frame`` — the latest raw ingested JPEG per board."""

from __future__ import annotations

import json

from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame, decode_jpeg
from board_reader_inference.estimator import StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.server import create_app


def _client() -> TestClient:
    return TestClient(create_app(StubEstimator(START_PLACEMENT), PositionHub()))


def _send_frame(ws, board_id: str, seq: int) -> bytes:
    """Send one frame on an open ``/ingest`` socket; returns its raw JPEG bytes."""
    frame = Frame(board_id=board_id, seq=seq, capture_ts=float(seq), image=render_board())
    header, jpeg = frame.encode()
    ws.send_text(json.dumps(header))
    ws.send_bytes(jpeg)
    return jpeg


def test_board_frame_404_before_any_ingest():
    r = _client().get("/boards/A1/frame")
    assert r.status_code == 404
    assert "error" in r.json()


def test_board_frame_round_trips_the_last_ingested_frame():
    client = _client()
    with client.websocket_connect("/ingest") as ingest:
        _send_frame(ingest, "A1", seq=1)
        ingest.receive_json()
        last_jpeg = _send_frame(ingest, "A1", seq=2)
        ingest.receive_json()

    r = client.get("/boards/A1/frame")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-seq"] == "2"
    assert r.headers["x-capture-ts"] == "2.0"
    assert r.content == last_jpeg
    # The bytes are exactly what was received — decodable via the shared wire codec.
    decode_jpeg(r.content)


def test_board_frame_is_independent_per_board():
    client = _client()
    with client.websocket_connect("/ingest") as ingest:
        a1_jpeg = _send_frame(ingest, "A1", seq=1)
        ingest.receive_json()
        b2_jpeg = _send_frame(ingest, "B2", seq=1)
        ingest.receive_json()

    r_a1 = client.get("/boards/A1/frame")
    r_b2 = client.get("/boards/B2/frame")
    assert r_a1.status_code == 200
    assert r_b2.status_code == 200
    assert r_a1.content == a1_jpeg
    assert r_b2.content == b2_jpeg

    assert client.get("/boards/other-board/frame").status_code == 404
