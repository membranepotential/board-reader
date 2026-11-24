"""The camera control channel: the registry's fan-out and its two WebSockets.

The registry is tested directly (its mailboxes are plain objects); the sockets are
driven through one ``TestClient`` entered as a context manager, so the phone and
the control pages share one event loop (TESTING.md §0, the ingest rule).
"""

from __future__ import annotations

import base64
import json

import numpy as np
from starlette.testclient import TestClient, WebSocketTestSession

from board_reader_core.control import (
    CameraConfig,
    CameraReport,
    ConfigMessage,
    CropRect,
    PreviewMessage,
    PreviewsMessage,
    StateMessage,
)
from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame, encode_jpeg

from board_reader_inference.camera_control import (
    CameraRegistry,
    StreamSample,
    frame_rate,
)
from board_reader_inference.hub import PositionHub
from board_reader_inference.recognizer import StubRecognizer
from board_reader_inference.server import create_app

REPORT = {
    "devices": [{"id": "d1", "label": "Back Camera"}, {"id": "d2", "label": "Back Ultra Wide Camera"}],
    "device": "Back Camera",
    "frame": {"width": 1920, "height": 1080},
    "ingest": "idle",
}
CONFIG = {
    "crop": {"x": 0.2, "y": 0.1, "w": 0.55, "h": 0.8},
    "device": "Back Ultra Wide Camera",
    "resolution": "2160p",
    "streaming": True,
}


# ---- the registry -------------------------------------------------------------------


def test_a_phone_hears_its_config_and_that_nobody_watches_on_connect():
    registry = CameraRegistry()
    registry.configure("phone", CameraConfig(streaming=True))
    phone = registry.connect_phone("phone")
    assert phone.take_nowait() == [
        ConfigMessage(CameraConfig(streaming=True)),
        PreviewsMessage(on=False),
    ]


def test_a_config_change_reaches_every_phone_and_every_watcher():
    registry = CameraRegistry()
    phone = registry.connect_phone("phone")
    watchers = [registry.connect_watcher("phone"), registry.connect_watcher("phone")]
    for mailbox in (phone, *watchers):
        mailbox.take_nowait()

    config = CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5), streaming=True)
    registry.configure("phone", config)
    assert phone.take_nowait() == [ConfigMessage(config)]
    for watcher in watchers:
        assert watcher.take_nowait() == [StateMessage(config, None, phone_connected=True)]


def test_previews_are_wanted_exactly_while_a_control_page_watches():
    registry = CameraRegistry()
    phone = registry.connect_phone("phone")
    phone.take_nowait()

    first = registry.connect_watcher("phone")
    assert phone.take_nowait() == [PreviewsMessage(on=True)]
    second = registry.connect_watcher("phone")
    assert phone.take_nowait() == []  # already on
    registry.disconnect_watcher("phone", first)
    assert phone.take_nowait() == []  # one still watches
    registry.disconnect_watcher("phone", second)
    assert phone.take_nowait() == [PreviewsMessage(on=False)]


def test_a_watcher_hears_the_phone_come_and_go_and_its_reports():
    registry = CameraRegistry()
    watcher = registry.connect_watcher("phone")
    assert watcher.take_nowait() == [StateMessage(CameraConfig(), None, phone_connected=False)]

    phone = registry.connect_phone("phone")
    report = CameraReport.from_dict(REPORT)
    registry.report("phone", report)
    # Latest state wins: the connect and the report collapse into one message.
    assert watcher.take_nowait() == [StateMessage(CameraConfig(), report, phone_connected=True)]

    registry.disconnect_phone("phone", phone)
    # The report outlives the phone, so the lens list survives a reload.
    assert watcher.take_nowait() == [StateMessage(CameraConfig(), report, phone_connected=False)]


def test_a_new_watcher_gets_the_last_preview_at_once():
    registry = CameraRegistry()
    preview = PreviewMessage(jpeg=b"jpeg", width=480, height=270)
    registry.preview("phone", preview)
    watcher = registry.connect_watcher("phone")
    assert watcher.take_nowait()[1:] == [preview]


def test_boards_are_independent():
    registry = CameraRegistry()
    left = registry.connect_phone("left")
    left.take_nowait()
    registry.connect_watcher("right")
    registry.configure("right", CameraConfig(streaming=True))
    assert left.take_nowait() == []
    assert registry.board("left").config.streaming is False


def test_the_frame_rate_is_measured_between_two_ticks():
    assert frame_rate(StreamSample(10, 100.0), StreamSample(25, 101.5)) == 10.0
    assert frame_rate(None, StreamSample(25, 101.5)) is None
    # The phone reloaded and its seq began again: nothing to measure.
    assert frame_rate(StreamSample(10, 100.0), StreamSample(2, 101.0)) is None


# ---- the sockets ---------------------------------------------------------------------


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


def test_a_config_set_on_control_is_pushed_to_the_phone():
    with _client() as client, client.websocket_connect("/boards/phone/camera") as phone:
        assert json.loads(phone.receive_text())["type"] == "config"
        assert json.loads(phone.receive_text()) == {"type": "previews", "on": False}
        phone.send_text(json.dumps({"type": "report", "report": REPORT}))

        with client.websocket_connect("/boards/phone/control") as control:
            state = _until(control, "state")
            assert state["report"] == REPORT
            assert state["phone_connected"] is True
            assert json.loads(phone.receive_text()) == {"type": "previews", "on": True}

            control.send_text(json.dumps({"type": "configure", "config": CONFIG}))
            assert json.loads(phone.receive_text()) == {"type": "config", "config": CONFIG}
            assert _until(control, "state")["config"] == CONFIG

        # The last page closed: the phone stops spending uplink on previews.
        assert json.loads(phone.receive_text()) == {"type": "previews", "on": False}


def test_a_reconnecting_phone_hears_the_config_again():
    with _client() as client:
        with client.websocket_connect("/boards/phone/control") as control:
            control.send_text(json.dumps({"type": "configure", "config": CONFIG}))
            _until(control, "state")
        with client.websocket_connect("/boards/phone/camera") as phone:
            assert json.loads(phone.receive_text()) == {"type": "config", "config": CONFIG}


def test_the_phones_preview_reaches_the_control_page():
    jpeg = encode_jpeg(render_board()[:120, :200])
    with (
        _client() as client, client.websocket_connect("/boards/phone/control") as control,
        client.websocket_connect("/boards/phone/camera") as phone,
    ):
        phone.send_bytes(jpeg)
        preview = _until(control, "preview")
        assert (preview["width"], preview["height"]) == (200, 120)
        assert base64.b64decode(preview["jpeg"]) == jpeg


def test_a_bad_message_is_refused_and_the_socket_stays():
    with _client() as client:
        with client.websocket_connect("/boards/phone/camera") as phone:
            phone.receive_text(), phone.receive_text()  # config, previews
            phone.send_bytes(b"not a picture")
            assert json.loads(phone.receive_text()) == {
                "type": "refused",
                "reason": "preview is not a JPEG",
            }
            phone.send_text(json.dumps({"type": "report", "report": {"ingest": "flying"}}))
            assert json.loads(phone.receive_text())["type"] == "refused"
        with client.websocket_connect("/boards/phone/control") as control:
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
