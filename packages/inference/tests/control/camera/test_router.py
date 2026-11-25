"""The control sockets, driven through the real app.

One ``TestClient`` entered as a context manager, so the phone and the control pages
share one event loop (TESTING.md §0, the ingest rule).
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest
from starlette.testclient import TestClient, WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame, encode_jpeg

from board_reader_inference.control.camera import router
from board_reader_inference.control.camera.direction import DIRECTION_READS
from board_reader_inference.control.camera.registry import CameraRegistry
from board_reader_inference.recognition.recognizer import StubRecognizer
from board_reader_inference.server import create_app
from board_reader_inference.stream.hub import PositionHub

REPORT = {
    "devices": [{"id": "d1", "label": "Back Camera"}, {"id": "d2", "label": "Back Ultra Wide Camera"}],
    "device": "Back Camera",
    "frame": {"width": 1920, "height": 1080},
    "ingest": "idle",
    "kind": "phone",
}
CONFIG = {
    "crop": {"x": 0.2, "y": 0.1, "w": 0.55, "h": 0.8},
    "device": "Back Ultra Wide Camera",
    "resolution": "2160p",
    "streaming": True,
    # A crop drawn on /control: Auto-frame is off for it.
    "auto": False,
}
#: ``CONFIG`` as the node sends it: every key, the defaulted ``rules`` included.
SENT_CONFIG = {**CONFIG, "rules": "standard"}

PHONE = "kitchen-phone-id"


def _client() -> TestClient:
    return TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))


def _until(ws: WebSocketTestSession, kind: str) -> dict:
    """The next message of ``kind``, skipping others (the health tick arrives every
    second on a control socket, so something always arrives and this cannot hang)."""
    for _ in range(50):
        message = json.loads(ws.receive_text())
        if message["type"] == kind:
            return message
    raise AssertionError(f"no {kind} message")


CAMERA = f"/cameras/{PHONE}"


def _hello(phone: WebSocketTestSession) -> str:
    """Read what a phone hears on connect; its board id."""
    assigned = json.loads(phone.receive_text())
    assert assigned["type"] == "assigned"
    assert json.loads(phone.receive_text())["type"] == "config"
    assert json.loads(phone.receive_text())["type"] == "previews"
    assert json.loads(phone.receive_text())["type"] == "send_full_frame"
    return assigned["board_id"]


def test_a_config_set_on_control_is_pushed_to_the_phone():
    with _client() as client, client.websocket_connect(CAMERA) as phone:
        assert json.loads(phone.receive_text()) == {
            "type": "assigned", "board_id": PHONE, "name": "board-1",
        }
        assert json.loads(phone.receive_text())["type"] == "config"
        assert json.loads(phone.receive_text()) == {"type": "previews", "on": False}
        assert json.loads(phone.receive_text()) == {"type": "send_full_frame", "request": 0}
        phone.send_text(json.dumps({"type": "report", "report": REPORT}))

        with client.websocket_connect(f"/boards/{PHONE}/control") as control:
            state = _until(control, "state")
            assert state["report"] == REPORT
            assert state["phone_connected"] is True
            assert json.loads(phone.receive_text()) == {"type": "previews", "on": True}

            control.send_text(json.dumps({"type": "configure", "config": CONFIG}))
            assert json.loads(phone.receive_text()) == {"type": "config", "config": SENT_CONFIG}
            assert _until(control, "state")["config"] == SENT_CONFIG

        # The last page closed: the phone stops spending uplink on previews.
        assert json.loads(phone.receive_text()) == {"type": "previews", "on": False}


def test_a_reconnecting_phone_hears_its_board_and_config_again():
    with _client() as client:
        with client.websocket_connect(CAMERA) as phone:
            assert _hello(phone) == PHONE
        with client.websocket_connect(f"/boards/{PHONE}/control") as control:
            control.send_text(json.dumps({"type": "configure", "config": CONFIG}))
            _until(control, "state")
        with client.websocket_connect(CAMERA) as phone:
            assert json.loads(phone.receive_text()) == {
                "type": "assigned", "board_id": PHONE, "name": "board-1",
            }
            assert json.loads(phone.receive_text()) == {"type": "config", "config": SENT_CONFIG}


def test_a_second_page_of_one_phone_takes_over_and_the_first_is_closed():
    with _client() as client, client.websocket_connect(CAMERA) as first:
        _hello(first)
        with client.websocket_connect(CAMERA) as second:
            assert _hello(second) == PHONE
            assert json.loads(first.receive_text()) == {"type": "replaced"}
            with pytest.raises(WebSocketDisconnect):
                first.receive_text()


def test_the_camera_list_shows_each_phone_renames_and_forgets_it():
    with _client() as client, client.websocket_connect("/cameras") as cameras:
        with client.websocket_connect(CAMERA) as phone:
            _hello(phone)
            listed = _until(cameras, "cameras")["cameras"]
            while not listed:  # the first list may predate the phone
                listed = _until(cameras, "cameras")["cameras"]
            assert listed == [{"device_id": PHONE, "name": "board-1", "connected": True,
                               "report": None, "frame_age_s": None}]

            cameras.send_text(json.dumps({"type": "rename", "device_id": PHONE, "name": "left"}))
            assert json.loads(phone.receive_text()) == {
                "type": "assigned", "board_id": PHONE, "name": "left",
            }
            cameras.send_text(json.dumps({"type": "rename", "device_id": PHONE, "name": "a/b"}))
            assert "is not a camera name" in _until(cameras, "refused")["reason"]
            cameras.send_text(json.dumps({"type": "forget", "device_id": PHONE}))
            assert "is connected" in _until(cameras, "refused")["reason"]
        while _until(cameras, "cameras")["cameras"][0]["connected"]:
            pass  # until the list hears the phone leave
        cameras.send_text(json.dumps({"type": "forget", "device_id": PHONE}))
        while _until(cameras, "cameras")["cameras"]:
            pass  # until the forgotten camera leaves the list


def test_a_phone_without_a_device_id_is_refused_and_an_outdated_page_told_to_reload():
    with _client() as client:
        with client.websocket_connect("/cameras/x") as phone:
            assert json.loads(phone.receive_text()) == {
                "type": "refused", "reason": "'x' is not a device id",
            }
            with pytest.raises(WebSocketDisconnect):
                phone.receive_text()
        with client.websocket_connect("/boards/phone/camera") as old:
            assert "reload" in json.loads(old.receive_text())["reason"]
        assert client.app.state.cameras.cameras(lambda board_id: None).cameras == ()


def test_the_phones_preview_reaches_the_control_page():
    jpeg = encode_jpeg(render_board()[:120, :200])
    with (
        _client() as client, client.websocket_connect(f"/boards/{PHONE}/control") as control,
        client.websocket_connect(CAMERA) as phone,
    ):
        _hello(phone)
        phone.send_bytes(jpeg)
        preview = _until(control, "preview")
        assert (preview["width"], preview["height"]) == (200, 120)
        assert base64.b64decode(preview["jpeg"]) == jpeg


def test_a_bad_message_is_refused_and_the_socket_stays():
    with _client() as client:
        with client.websocket_connect(CAMERA) as phone:
            _hello(phone)
            phone.send_bytes(b"not a picture")
            assert json.loads(phone.receive_text()) == {
                "type": "refused",
                "reason": "preview is not a JPEG",
            }
            phone.send_text(json.dumps({"type": "report", "report": {"ingest": "flying"}}))
            assert json.loads(phone.receive_text())["type"] == "refused"
        with client.websocket_connect(f"/boards/{PHONE}/control") as control:
            bad = {"type": "configure", "config": {"crop": {"x": 0.9, "y": 0, "w": 0.5, "h": 0.5}}}
            control.send_text(json.dumps(bad))
            assert "crop" in _until(control, "refused")["reason"]
            # ... and the old config stands.
            control.send_text(json.dumps({"type": "configure", "config": {"streaming": True}}))
            assert _until(control, "state")["config"]["streaming"] is True


def test_the_health_tick_and_the_last_streamed_frame_reach_the_control_page():
    image = np.full((300, 400, 3), 128, dtype=np.uint8)
    header, jpeg = Frame(board_id="phone", seq=7, capture_ts=1.0, image=image).encode()
    with _client() as client:
        with client.websocket_connect("/ingest") as ws:
            ws.send_text(json.dumps(header))
            ws.send_bytes(jpeg)
            ws.receive_json()
        with client.websocket_connect("/boards/phone/control") as control:
            health = _until(control, "health")
            assert health["recognizer"] == "stub"
            assert health["received"] == 1
            assert health["frame_age_s"] >= 0
            frame = _until(control, "frame")
            assert (frame["seq"], frame["width"], frame["height"]) == (7, 400, 300)
            assert base64.b64decode(frame["jpeg"]) == jpeg


def test_the_health_tick_says_which_way_the_board_lies(monkeypatch):
    monkeypatch.setattr(router, "HEALTH_EVERY_S", 0.01)
    with _client() as client, client.websocket_connect("/boards/phone/control") as control:
        assert _until(control, "health")["direction"] is None
        cameras: CameraRegistry = client.app.state.cameras
        for i in range(DIRECTION_READS):
            cameras.observe_direction("phone", -90.0, 0.5 * i)
        for _ in range(50):
            direction = _until(control, "health")["direction"]
            if direction is not None:
                break
        assert direction == {"degrees": -90.0, "white_side": "right", "camera_moved": False}
