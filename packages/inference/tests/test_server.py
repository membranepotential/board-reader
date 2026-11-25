"""Tests for the inference gateway (DESIGN.md §4.5) via Starlette TestClient."""

from __future__ import annotations

import json

from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT, dist_from_placement
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.recognition.recognizer import Recognition, StubRecognizer
from board_reader_inference.server import create_app
from board_reader_inference.stream.hub import PositionHub


def _client(**kw):
    return TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))


def _send_frame(ws, seq=0):
    frame = Frame(board_id="A1", seq=seq, capture_ts=float(seq), image=render_board())
    header, jpeg = frame.encode()
    ws.send_text(json.dumps(header))
    ws.send_bytes(jpeg)


def _app_estimator(client: TestClient):
    return client.app.state.estimator


def test_healthz_ok():
    r = _client().get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_boards_404_before_any_frame():
    assert _client().get("/boards/A1").status_code == 404


def test_ingest_frame_then_board_snapshot_available(gateway, drain_ingest):
    # The ack means "received", not "estimated" — hence gateway + drain_ingest
    # rather than a bare client (see conftest.py).
    client = gateway(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))
    with client.websocket_connect("/ingest") as ingest:
        _send_frame(ingest, seq=5)
        ack = ingest.receive_json()
        assert ack == {"ack": 5, "board_id": "A1"}
    drain_ingest(client)
    snap = client.get("/boards/A1")
    assert snap.status_code == 200
    assert snap.json()["placement"] == START_PLACEMENT


def test_end_to_end_ingest_to_viewer_stream(gateway):
    """A frame pushed on /ingest reaches a /viewer subscriber as a position event."""
    # Both sockets must share ONE event loop. On a bare TestClient each
    # `websocket_connect` gets its own portal, so the hub's subscriber queue would
    # live on the viewer's loop while `publish` runs on the ingest loop — waking a
    # future across loops, which strands `receive_text()` below forever.
    client = gateway(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))
    with (
        client.websocket_connect("/viewer") as viewer,
        client.websocket_connect("/ingest") as ingest,
    ):
        _send_frame(ingest, seq=1)
        ingest.receive_json()  # ack
        # The frame's motion update goes out at submit, before its read publishes.
        motion = json.loads(viewer.receive_text())
        event = json.loads(viewer.receive_text())
    assert motion["type"] == "motion"
    assert event["board_id"] == "A1"
    assert event["placement"] == START_PLACEMENT
    assert event["status"] == "ok"
    assert "square_dist" in event


class _ScriptedRecognizer:
    """Reads one authored (board, frame energy) per call, in order."""

    name = "scripted"
    model_path: str | None = None

    def __init__(self, reads: list[tuple[str, float]]) -> None:
        self._reads = iter(reads)

    def read(self, image) -> Recognition:
        placement, energy = next(self._reads)
        return Recognition(dist_from_placement(placement), frame_energy=energy)


def test_a_move_beside_a_resting_hand_is_served_by_the_gateway(gateway, drain_ingest):
    """Through the gateway with the shipped settle window: two obscured reads, one
    second apart, show 1.e4 in plain view and a hand over b8. The board the node
    serves has the move, keeps the knight, and says a hand is in the frame."""
    e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    e4_gone_b8 = "r1bqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    recognizer = _ScriptedRecognizer(
        [(START_PLACEMENT, -14.0), (e4_gone_b8, -10.0), (e4_gone_b8, -10.0)]
    )
    client = gateway(create_app(recognizer, PositionHub()))
    with client.websocket_connect("/ingest") as ingest:
        for seq in range(3):
            _send_frame(ingest, seq=seq)
            assert ingest.receive_json()["ack"] == seq
    drain_ingest(client)
    board = client.get("/boards/A1").json()
    assert board["placement"] == e4
    assert board["obscured"] is True
    assert board["pending"] == ["b8"]


def test_estimate_snapshot_roundtrip():
    """POST /estimate (rectified crop JPEG body) -> position event, retained in hub."""
    from board_reader_core.wire import encode_jpeg

    client = _client()
    r = client.post(
        "/estimate",
        content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    event = r.json()
    assert event["board_id"] == "snapshot"
    assert event["placement"] == START_PLACEMENT
    assert "square_dist" in event
    # The event is retained like any streamed frame's.
    assert client.get("/boards/snapshot").json()["seq"] == event["seq"]


def test_estimate_rejects_garbage_bytes():
    r = _client().post(
        "/estimate", content=b"not a jpeg", headers={"Content-Type": "image/jpeg"}
    )
    assert r.status_code == 400
    assert "error" in r.json()


def test_estimate_reads_the_whole_photo_and_ignores_legacy_query_params():
    """``?corners``, ``?debug`` and ``?autorotate`` were the crop-classifier stack's;
    an old client sending them still gets the recognizer's read of the whole photo."""
    from board_reader_core.wire import encode_jpeg

    r = _client().post(
        "/estimate?corners=1,2,3&debug=1&autorotate=1",
        content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["placement"] == START_PLACEMENT
    assert not {"debug", "autorotated", "warnings"} & body.keys()


def test_viewer_replays_retained_latest_on_connect(gateway, drain_ingest):
    # Draining first is what keeps this test from HANGING the suite: if the frame
    # is never estimated, nothing is retained, and `receive_text()` below waits
    # forever for an event that will never come (see conftest.py).
    client = gateway(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))
    with client.websocket_connect("/ingest") as ingest:
        _send_frame(ingest, seq=9)
        ingest.receive_json()
    drain_ingest(client)
    # A viewer connecting AFTER the frame still gets the retained latest immediately.
    with client.websocket_connect("/viewer") as viewer:
        event = json.loads(viewer.receive_text())
    assert event["seq"] == 9


def test_uploads_saved_when_retention_enabled(tmp_path, monkeypatch):
    from board_reader_core.wire import encode_jpeg

    monkeypatch.setenv("BR_SAVE_UPLOADS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))
    body = encode_jpeg(render_board())

    r = client.post("/estimate", content=body, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200
    jpgs = sorted(tmp_path.glob("*-estimate.jpg"))
    metas = sorted(tmp_path.glob("*-estimate.json"))
    assert len(jpgs) == 1 and len(metas) == 1
    assert jpgs[0].read_bytes() == body
    meta = json.loads(metas[0].read_text())
    assert meta["kind"] == "estimate"
    # The response's placement is annotated back onto the sidecar, so every
    # retained shot doubles as (predicted-)labeled eval data.
    assert meta["placement"] == r.json()["placement"]

    # undecodable bodies are rejected AND not retained
    client.post("/estimate", content=b"not a jpeg", headers={"Content-Type": "image/jpeg"})
    assert len(list(tmp_path.glob("*-estimate.jpg"))) == 1


def test_uploads_not_saved_by_default(tmp_path, monkeypatch):
    from board_reader_core.wire import encode_jpeg

    monkeypatch.delenv("BR_SAVE_UPLOADS", raising=False)
    monkeypatch.chdir(tmp_path)  # any accidental relative write would land here
    client = TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))
    r = client.post(
        "/estimate", content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_low_light_warning_on_dark_photos():
    import numpy as np

    from board_reader_core.wire import encode_jpeg

    client = _client()
    dark = (render_board() * 0.15).astype(np.uint8)
    r = client.post("/estimate", content=encode_jpeg(dark), headers={"Content-Type": "image/jpeg"})
    assert r.json().get("warnings") == ["low_light"]
    r = client.post(
        "/estimate", content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert "warnings" not in r.json()
