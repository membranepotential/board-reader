"""Tests for ``GET /boards/{id}/frame`` — the latest raw ingested JPEG per board."""

from __future__ import annotations

import json

from _socket_reads import receive_json
from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame, decode_jpeg

from board_reader_inference.recognition.recognizer import StubRecognizer
from board_reader_inference.server import create_app
from board_reader_inference.stream.hub import PositionHub


def _client() -> TestClient:
    return TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))


def _send_frame(ws, board_id: str, seq: int, corners=None) -> bytes:
    """Send one frame on an open ``/ingest`` socket; returns its raw JPEG bytes."""
    frame = Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=float(seq),
        image=render_board(),
        corners=corners,
    )
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
        receive_json(ingest)
        last_jpeg = _send_frame(ingest, "A1", seq=2)
        receive_json(ingest)

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
        receive_json(ingest)
        b2_jpeg = _send_frame(ingest, "B2", seq=1)
        receive_json(ingest)

    r_a1 = client.get("/boards/A1/frame")
    r_b2 = client.get("/boards/B2/frame")
    assert r_a1.status_code == 200
    assert r_b2.status_code == 200
    assert r_a1.content == a1_jpeg
    assert r_b2.content == b2_jpeg

    assert client.get("/boards/other-board/frame").status_code == 404


QUAD = [[12.0, 30.0], [480.0, 28.0], [496.0, 500.0], [4.0, 496.0]]  # a8,h8,h1,a1


def test_board_frame_never_reports_the_corners_a_legacy_stream_carried():
    """Older edges still send ``corners``; the node reads the whole frame and no
    longer passes the quad on (``X-Corners`` is gone)."""
    client = _client()
    with client.websocket_connect("/ingest") as ingest:
        _send_frame(ingest, "A1", seq=1, corners=QUAD)
        assert receive_json(ingest) == {"ack": 1, "board_id": "A1"}

    r = client.get("/boards/A1/frame")
    assert r.status_code == 200
    assert "x-corners" not in r.headers


def test_a_malformed_frame_is_refused_without_killing_the_socket():
    """One bad frame (wrong-length corners, truncated JPEG) is refused in the ack
    slot; the SAME socket then carries the next good frame. A buggy browser edge
    must get its error back, not a dead connection to reconnect-loop against."""
    client = _client()
    with client.websocket_connect("/ingest") as ingest:
        bad_header = {
            "type": "frame", "board_id": "A1", "seq": 1, "capture_ts": 1.0,
            "corners": [1.0, 2.0],  # 2 floats, not 8
        }
        ingest.send_text(json.dumps(bad_header))
        ingest.send_bytes(Frame(
            board_id="A1", seq=1, capture_ts=1.0, image=render_board()
        ).encode()[1])
        assert "error" in receive_json(ingest)

        good_header = {
            "type": "frame", "board_id": "A1", "seq": 2, "capture_ts": 2.0,
        }
        ingest.send_text(json.dumps(good_header))
        ingest.send_bytes(b"not a jpeg")
        assert "error" in receive_json(ingest)

        _send_frame(ingest, "A1", seq=3, corners=QUAD)
        assert receive_json(ingest) == {"ack": 3, "board_id": "A1"}
