"""Camera identity across the gateway (DESIGN D18): who may stream under a board id,
and how the live page hears the cameras' names.

The decisions themselves are the registry's (``tests/control/camera/test_registry.py``);
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
OTHER = "garden-phone-id"
STALE = "stale-page-id"


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
        client.websocket_connect(f"/cameras/{OTHER}") as other,
        client.websocket_connect("/ingest") as ingest,
    ):
        assert (_assigned(phone), _assigned(other)) == (PHONE, OTHER)
        assert _send(ingest, PHONE, PHONE, 0) == {"ack": 0, "board_id": PHONE}
        # A known camera under another camera's board id.
        assert _send(ingest, PHONE, OTHER, 1) == {
            "error": "refused frame: camera board-2 streams as its device id, not "
            "kitchen-phone-id: reload /camera"
        }
        # A page of a device the node does not know.
        assert _send(ingest, STALE, STALE, 2) == {
            "error": "refused frame: camera stale-pa is not known to the node "
            "(forgotten?): reload /camera"
        }
        # An edge without a device id: refused on a camera's board, taken elsewhere,
        # under a camera's name too (a name is no board id).
        assert _send(ingest, PHONE, None, 3)["error"].startswith(
            "refused frame: board kitchen- belongs to camera board-1"
        )
        assert _send(ingest, "board-1", None, 4) == {"ack": 4, "board_id": "board-1"}

        counters = client.get("/healthz").json()["ingest"]
        assert (counters["not_owned"], counters["refused"], counters["received"]) == (3, 3, 2)


def test_the_live_page_hears_a_rename_as_names_and_a_late_viewer_hears_them_first():
    with (
        _client() as client,
        client.websocket_connect(f"/cameras/{PHONE}") as phone,
        client.websocket_connect("/cameras") as cameras,
    ):
        assert _assigned(phone) == PHONE
        with client.websocket_connect("/viewer") as viewer:
            # A known camera has its default name from the start.
            assert parse_viewer_message(viewer.receive_text()).to_dict() == {
                "type": "names", "names": [{"board_id": PHONE, "name": "board-1"}],
            }
            cameras.send_text(json.dumps({"type": "rename", "device_id": PHONE, "name": "kitchen"}))
            heard = parse_viewer_message(viewer.receive_text())
            assert heard.to_dict() == {
                "type": "names", "names": [{"board_id": PHONE, "name": "kitchen"}],
            }
        with client.websocket_connect("/viewer") as late:
            assert parse_viewer_message(late.receive_text()) == heard
