"""The camera control channel: the registry's fan-out and its two WebSockets.

The registry is tested directly (its mailboxes are plain objects); the sockets are
driven through one ``TestClient`` entered as a context manager, so the phone and
the control pages share one event loop (TESTING.md §0, the ingest rule).
"""

from __future__ import annotations

import base64
import json
import logging
import os

import numpy as np
import pytest
from starlette.testclient import TestClient, WebSocketTestSession

from board_reader_core.control import (
    BoardDirection,
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
    CAMERA_CONFIG_ENV,
    DIRECTION_READS,
    CameraRegistry,
    StreamSample,
    frame_rate,
    white_side,
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


# ---- the configs survive a restart ----------------------------------------------------


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "state" / "camera-config.json"
    monkeypatch.setenv(CAMERA_CONFIG_ENV, str(path))
    return path


def test_a_restarted_node_pushes_the_config_it_had(config_file):
    config = CameraConfig.from_dict(CONFIG)
    CameraRegistry().configure("phone", config)
    assert config_file.exists()
    assert not list(config_file.parent.glob(".*.tmp")), "the temporary file was renamed"

    restarted = CameraRegistry()
    phone = restarted.connect_phone("phone")
    assert phone.take_nowait()[0] == ConfigMessage(config)
    assert restarted.board("other").config == CameraConfig()


def test_a_corrupt_config_file_is_ignored_with_a_warning(config_file, caplog):
    config_file.parent.mkdir(parents=True)
    bad = {"boards": {"phone": {**CONFIG, "resolution": "8k"}}}
    for text in ("{not json", json.dumps(bad), json.dumps({"phone": CONFIG})):
        config_file.write_text(text)
        with caplog.at_level(logging.WARNING):
            registry = CameraRegistry()
        assert registry.board("phone").config == CameraConfig()
        assert "corrupt camera config" in caplog.text
        caplog.clear()

    # The next change replaces the corrupt file with a whole one.
    registry.configure("phone", CameraConfig(streaming=True))
    assert CameraRegistry().board("phone").config == CameraConfig(streaming=True)


def test_without_the_env_var_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.delenv(CAMERA_CONFIG_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    CameraRegistry().configure("phone", CameraConfig.from_dict(CONFIG))
    assert list(tmp_path.rglob("*")) == []
    assert CameraRegistry().board("phone").config == CameraConfig()


def test_serve_keeps_the_configs_on_disk_by_default(monkeypatch):
    import uvicorn

    from board_reader_inference import cli

    # A private environment: monkeypatch cannot restore a key the code under test sets.
    env = dict(os.environ)
    env.pop(CAMERA_CONFIG_ENV, None)
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

    cli.cmd_serve(cli.build_parser().parse_args(["serve", "--no-record"]))
    assert env[CAMERA_CONFIG_ENV] == cli.DEFAULT_CAMERA_CONFIG

    env[CAMERA_CONFIG_ENV] = ""
    cli.cmd_serve(cli.build_parser().parse_args(["serve", "--no-record"]))
    assert env[CAMERA_CONFIG_ENV] == "", "an operator's choice (memory only) outranks it"


# ---- which way the board lies --------------------------------------------------------


def _read(registry: CameraRegistry, degrees: list[float], start: float = 0.0) -> float:
    """Feed ``degrees`` as live reads 0.5 s apart; the time after the last one."""
    for i, angle in enumerate(degrees):
        registry.observe_direction("phone", angle, start + 0.5 * i)
    return start + 0.5 * len(degrees)


def test_white_sits_on_the_nearest_side():
    assert [white_side(a) for a in (2.0, 89.0, 91.0, 179.0, -179.0, -88.0, 271.0)] == [
        "bottom", "left", "left", "top", "top", "right", "right",
    ]


def test_the_direction_follows_the_reads_and_ignores_an_odd_one():
    registry = CameraRegistry()
    assert registry.direction("phone") is None
    _read(registry, [88.0, 90.0, 89.0, 200.0, 91.0])
    assert registry.direction("phone") == BoardDirection(90.0, "left", camera_moved=False)


def test_a_camera_turned_after_settling_is_called_moved_until_reframed():
    registry = CameraRegistry()
    now = _read(registry, [89.0] * DIRECTION_READS)
    # Turned by a quarter: once most of the latest reads say so, the camera moved.
    _read(registry, [1.0, -1.0, 0.0], start=now)
    assert registry.direction("phone") == BoardDirection(1.0, "bottom", camera_moved=True)

    # A new crop is a new framing: the angle settles again from here.
    registry.configure("phone", CameraConfig(crop=CropRect(0.1, 0.1, 0.8, 0.8)))
    assert registry.direction("phone") is None
    _read(registry, [0.0] * DIRECTION_READS, start=10.0)
    assert registry.direction("phone") == BoardDirection(0.0, "bottom", camera_moved=False)


def test_streaming_on_and_off_keeps_the_settled_angle():
    registry = CameraRegistry()
    now = _read(registry, [89.0] * DIRECTION_READS)
    registry.configure("phone", CameraConfig(streaming=True))
    _read(registry, [179.0] * 3, start=now)
    view = registry.direction("phone")
    assert view is not None and view.camera_moved


def test_a_burst_of_reads_of_one_scene_does_not_settle_the_angle():
    registry = CameraRegistry()
    for i in range(DIRECTION_READS):
        registry.observe_direction("phone", 89.0, 0.1 * i)  # all within half a second
    registry.observe_direction("phone", 0.0, 0.6)
    registry.observe_direction("phone", 0.0, 0.7)
    registry.observe_direction("phone", 0.0, 0.8)
    view = registry.direction("phone")
    assert view is not None and view.white_side == "bottom" and not view.camera_moved


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


def test_the_health_tick_says_which_way_the_board_lies(monkeypatch):
    import board_reader_inference.camera_control as camera_control

    monkeypatch.setattr(camera_control, "HEALTH_EVERY_S", 0.01)
    with _client() as client, client.websocket_connect("/boards/phone/control") as control:
        assert _until(control, "health")["direction"] is None
        cameras: CameraRegistry = client.app.state.cameras
        _read(cameras, [-90.0] * DIRECTION_READS)
        for _ in range(50):
            direction = _until(control, "health")["direction"]
            if direction is not None:
                break
        assert direction == {"degrees": -90.0, "white_side": "right", "camera_moved": False}
