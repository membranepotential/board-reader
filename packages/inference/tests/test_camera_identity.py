"""Camera identity across the gateway (DESIGN D18): who may stream under a board id,
and how the live page hears the cameras' names.

The decisions themselves are the registry's (``tests/control/test_camera.py``);
these tests check that ``/ingest`` and ``/viewer`` act on them.
"""

from __future__ import annotations

import json

import numpy as np
from starlette.testclient import TestClient, WebSocketTestSession

from board_reader_core.protocol import START_PLACEMENT, parse_viewer_message
from board_reader_core.wire import Frame

from board_reader_inference.recognition.recognizer import StubRecognizer
from board_reader_inference.server import create_app
from board_reader_inference.stream.hub import PositionHub

PHONE = "kitchen-phone-id"
STALE = "garden-phone-id"


def _client() -> TestClient:
    return TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))


def _send(ingest: WebSocketTestSession, board_id: str, device_id: str | None, seq: int) -> dict:
    frame = Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=float(seq),
        image=np.zeros((64, 64, 3), np.uint8),
        device_id=device_id,
    )
    header, jpeg = frame.encode()
    ingest.send_text(json.dumps(header))
    ingest.send_bytes(jpeg)
    return json.loads(ingest.receive_text())


def _assigned(phone: WebSocketTestSession) -> str:
    assigned = json.loads(phone.receive_text())
    assert assigned["type"] == "assigned"
    return assigned["board_id"]


def test_a_cameras_board_takes_its_frames_only_and_healthz_counts_the_rest():
    with (
        _client() as client,
        client.websocket_connect(f"/cameras/{PHONE}") as phone,
        client.websocket_connect("/ingest") as ingest,
    ):
        assert _assigned(phone) == "board-1"
        assert _send(ingest, "board-1", PHONE, 0) == {"ack": 0, "board_id": "board-1"}
        # A page of a device the node does not know, still streaming as board-1.
        stale = _send(ingest, "board-1", STALE, 1)
        assert stale == {
            "error": "refused frame: camera garden-p is not known to the node "
            "(forgotten?): reload /camera"
        }
        # An edge without a device id: refused on a camera's board, taken elsewhere.
        assert _send(ingest, "board-1", None, 2)["error"].startswith(
            "refused frame: board board-1 belongs to camera board-1"
        )
        assert _send(ingest, "feed", None, 3) == {"ack": 3, "board_id": "feed"}

        counters = client.get("/healthz").json()["ingest"]
        assert (counters["not_owned"], counters["refused"], counters["received"]) == (2, 2, 2)


def test_the_live_page_hears_a_rename_as_names_and_a_late_viewer_hears_them_first():
    with (
        _client() as client,
        client.websocket_connect(f"/cameras/{PHONE}") as phone,
        client.websocket_connect("/cameras") as cameras,
    ):
        assert _assigned(phone) == "board-1"
        with client.websocket_connect("/viewer") as viewer:
            cameras.send_text(json.dumps({"type": "rename", "device_id": PHONE, "name": "kitchen"}))
            heard = parse_viewer_message(viewer.receive_text())
            assert heard.to_dict() == {
                "type": "names", "names": [{"board_id": "board-1", "name": "kitchen"}],
            }
        with client.websocket_connect("/viewer") as late:
            assert parse_viewer_message(late.receive_text()) == heard
