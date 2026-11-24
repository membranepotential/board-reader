"""The camera control channel over HTTP (``/boards/{id}/camera``).

The control UI writes a board's config; the edge's report poll reads it back; the
edge's preview and the last streamed frame come back to the control UI. Every
request here is answered before the handler returns, so a plain ``TestClient`` is
right (TESTING.md §0, the ingest rule).
"""

from __future__ import annotations

import json

import numpy as np
from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame, encode_jpeg

from board_reader_inference.hub import PositionHub
from board_reader_inference.recognizer import StubRecognizer
from board_reader_inference.server import create_app

REPORT = {
    "devices": [{"id": "d1", "label": "Back Camera"}, {"id": "d2", "label": "Back Ultra Wide Camera"}],
    "device": "Back Camera",
    "frame": {"width": 1920, "height": 1080},
    "ingest": "idle",
}


def _client() -> TestClient:
    return TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))


def test_an_unknown_board_starts_with_the_default_config_and_no_phone():
    state = _client().get("/boards/phone/camera").json()
    assert state["config"] == {
        "crop": None,
        "device": None,
        "resolution": "1080p",
        "streaming": False,
    }
    assert state["report"] is None
    assert state["preview"] is None
    assert state["frame"] is None
    assert isinstance(state["now"], float)


def test_the_config_the_control_ui_puts_is_what_the_phone_poll_gets_back():
    client = _client()
    config = {
        "crop": {"x": 0.2, "y": 0.1, "w": 0.55, "h": 0.8},
        "device": "Back Ultra Wide Camera",
        "resolution": "2160p",
        "streaming": True,
    }
    put = client.put("/boards/phone/camera", json=config)
    assert put.status_code == 200
    assert put.json()["config"] == config

    polled = client.post("/boards/phone/camera/report", json=REPORT)
    assert polled.status_code == 200
    assert polled.json() == config

    # ... and the poll is what the control UI then sees of the phone.
    state = client.get("/boards/phone/camera").json()
    assert state["report"] == REPORT
    assert state["seen_at"] <= state["now"]


def test_boards_are_configured_independently():
    client = _client()
    client.put("/boards/left/camera", json={"streaming": True})
    assert client.get("/boards/right/camera").json()["config"]["streaming"] is False


def test_a_bad_config_is_refused_and_the_old_one_stands():
    client = _client()
    client.put("/boards/phone/camera", json={"streaming": True})
    bad = client.put("/boards/phone/camera", json={"crop": {"x": 0.9, "y": 0, "w": 0.5, "h": 0.5}})
    assert bad.status_code == 422
    assert "crop" in bad.json()["error"]
    assert client.get("/boards/phone/camera").json()["config"]["streaming"] is True


def test_a_body_that_is_not_json_is_refused_in_words():
    response = _client().put("/boards/phone/camera", content=b"streaming=1")
    assert response.status_code == 422
    assert "JSON" in response.json()["error"]


def test_a_bad_report_is_refused():
    response = _client().post("/boards/phone/camera/report", json={"ingest": "flying"})
    assert response.status_code == 422


def test_the_preview_round_trips_and_its_metadata_reaches_the_control_ui():
    client = _client()
    assert client.get("/boards/phone/camera/preview").status_code == 404
    jpeg = encode_jpeg(render_board()[:120, :200])
    stored = client.post(
        "/boards/phone/camera/preview", content=jpeg, headers={"content-type": "image/jpeg"}
    )
    assert stored.status_code == 204

    served = client.get("/boards/phone/camera/preview")
    assert served.content == jpeg
    assert served.headers["cache-control"] == "no-store"
    preview = client.get("/boards/phone/camera").json()["preview"]
    assert (preview["width"], preview["height"]) == (200, 120)
    assert served.headers["x-preview-seq"] == str(preview["seq"])


def test_a_preview_that_is_not_a_jpeg_is_refused():
    response = _client().post("/boards/phone/camera/preview", content=b"not a picture")
    assert response.status_code == 400


def test_the_last_streamed_frame_shows_up_in_the_camera_state():
    client = _client()
    image = np.full((300, 400, 3), 128, dtype=np.uint8)
    with client.websocket_connect("/ingest") as ws:
        header, jpeg = Frame(board_id="phone", seq=7, capture_ts=1.0, image=image).encode()
        ws.send_text(json.dumps(header))
        ws.send_bytes(jpeg)
        ws.receive_json()
    frame = client.get("/boards/phone/camera").json()["frame"]
    assert (frame["seq"], frame["width"], frame["height"]) == (7, 400, 300)
    assert frame["received_at"] > 0
