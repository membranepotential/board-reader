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
from starlette.websockets import WebSocketDisconnect

from board_reader_core.control import (
    AssignedMessage,
    BoardDirection,
    CameraConfig,
    CameraReport,
    ConfigMessage,
    CropRect,
    PreviewMessage,
    PreviewsMessage,
    ReplacedMessage,
    StateMessage,
)
from board_reader_core.protocol import START_PLACEMENT, BoardName, NamesMessage
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame, encode_jpeg

from board_reader_inference.control.camera import (
    CAMERA_CONFIG_ENV,
    DIRECTION_READS,
    CameraRegistry,
    StreamSample,
    frame_rate,
    white_side,
)
from board_reader_inference.recognition.recognizer import StubRecognizer
from board_reader_inference.server import create_app
from board_reader_inference.stream.hub import PositionHub

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
#: ``CONFIG`` as the node sends it: every key, the defaulted ``rules`` included.
SENT_CONFIG = {**CONFIG, "rules": "standard"}


# ---- the registry -------------------------------------------------------------------

PHONE = "kitchen-phone-id"
OTHER = "garden-phone-id"


def test_a_phone_hears_its_board_its_config_and_that_nobody_watches_on_connect():
    registry = CameraRegistry()
    registry.configure("board-1", CameraConfig(streaming=True))  # a saved config keeps its id
    phone = registry.connect_phone(PHONE)
    assert phone.mailbox.take_nowait() == [
        AssignedMessage("board-2"),
        ConfigMessage(CameraConfig()),
        PreviewsMessage(on=False),
    ]


def test_each_new_phone_gets_the_lowest_free_board_and_keeps_it():
    registry = CameraRegistry()
    first = registry.connect_phone(PHONE)
    assert registry.connect_phone(OTHER).mailbox.take_nowait()[0] == AssignedMessage("board-2")
    registry.disconnect_phone(first)
    # The same phone again (a reload, a reconnect) streams as the same board.
    assert registry.connect_phone(PHONE).mailbox.take_nowait()[0] == AssignedMessage("board-1")


def test_a_newer_socket_of_one_phone_replaces_the_older():
    registry = CameraRegistry()
    old = registry.connect_phone(PHONE)
    old.mailbox.take_nowait()
    new = registry.connect_phone(PHONE)
    assert old.mailbox.take_nowait() == [ReplacedMessage()]
    assert new.mailbox.take_nowait()[0] == AssignedMessage("board-1")

    # The replaced socket speaks for nobody: its late report and its close change nothing.
    registry.report(old, CameraReport.from_dict({**REPORT, "ingest": "streaming"}))
    registry.disconnect_phone(old)
    watcher = registry.connect_watcher("board-1")
    assert watcher.take_nowait() == [StateMessage(CameraConfig(), None, phone_connected=True)]


def test_a_config_change_reaches_the_boards_phone_and_every_watcher():
    registry = CameraRegistry()
    phone = registry.connect_phone(PHONE)
    watchers = [registry.connect_watcher("board-1"), registry.connect_watcher("board-1")]
    for mailbox in (phone.mailbox, *watchers):
        mailbox.take_nowait()

    config = CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5), streaming=True)
    registry.configure("board-1", config)
    assert phone.mailbox.take_nowait() == [ConfigMessage(config)]
    for watcher in watchers:
        assert watcher.take_nowait() == [StateMessage(config, None, phone_connected=True)]


def test_previews_are_wanted_exactly_while_a_control_page_watches():
    registry = CameraRegistry()
    phone = registry.connect_phone(PHONE)
    phone.mailbox.take_nowait()

    first = registry.connect_watcher("board-1")
    assert phone.mailbox.take_nowait() == [PreviewsMessage(on=True)]
    second = registry.connect_watcher("board-1")
    assert phone.mailbox.take_nowait() == []  # already on
    registry.disconnect_watcher("board-1", first)
    assert phone.mailbox.take_nowait() == []  # one still watches
    registry.disconnect_watcher("board-1", second)
    assert phone.mailbox.take_nowait() == [PreviewsMessage(on=False)]


def test_a_watcher_hears_the_phone_come_and_go_and_its_reports():
    registry = CameraRegistry()
    watcher = registry.connect_watcher("board-1")
    assert watcher.take_nowait() == [StateMessage(CameraConfig(), None, phone_connected=False)]

    phone = registry.connect_phone(PHONE)
    report = CameraReport.from_dict(REPORT)
    registry.report(phone, report)
    # Latest state wins: the connect and the report collapse into one message.
    assert watcher.take_nowait() == [StateMessage(CameraConfig(), report, phone_connected=True)]

    registry.disconnect_phone(phone)
    # The report outlives the phone, so the lens list survives a reload.
    assert watcher.take_nowait() == [StateMessage(CameraConfig(), report, phone_connected=False)]


def test_a_new_watcher_gets_the_last_preview_at_once():
    registry = CameraRegistry()
    preview = PreviewMessage(jpeg=b"jpeg", width=480, height=270)
    registry.preview(registry.connect_phone(PHONE), preview)
    watcher = registry.connect_watcher("board-1")
    assert watcher.take_nowait()[1:] == [preview]


def test_boards_are_independent():
    registry = CameraRegistry()
    left = registry.connect_phone(PHONE)
    left.mailbox.take_nowait()
    registry.connect_phone(OTHER)
    registry.connect_watcher("board-2")
    registry.configure("board-2", CameraConfig(streaming=True))
    assert left.mailbox.take_nowait() == []
    assert registry.board("board-1").config.streaming is False


# ---- renaming ---------------------------------------------------------------------------


def test_a_rename_sets_a_name_and_nothing_else():
    """The board id is the device's for good: a rename is an alias people read."""
    heard: list[NamesMessage] = []
    registry = CameraRegistry(on_names=heard.append)
    phone = registry.connect_phone(PHONE)
    config = CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5), streaming=True)
    registry.configure("board-1", config)
    page = registry.connect_watcher("board-1")
    for mailbox in (phone.mailbox, page):
        mailbox.take_nowait()

    registry.rename(PHONE, "kitchen table")
    assert registry.board_of(PHONE) == "board-1"
    assert registry.board("board-1").config == config
    assert registry.label("board-1") == "kitchen table"
    # The phone hears its name beside the same board id; the page has nothing new.
    assert phone.mailbox.take_nowait() == [AssignedMessage("board-1", "kitchen table")]
    assert page.take_nowait() == []
    named = NamesMessage(names=(BoardName("board-1", "kitchen table"),))
    assert heard == [NamesMessage(names=()), named]
    assert registry.cameras(lambda board_id: None).cameras[0].name == "kitchen table"

    # The same name again changes nothing; its own board id (or None) clears it.
    registry.rename(PHONE, "kitchen table")
    registry.rename(PHONE, "board-1")
    assert heard[-1] == NamesMessage(names=()) and len(heard) == 3
    assert registry.label("board-1") == "board-1"


def test_a_rename_is_refused_for_a_taken_or_malformed_name_or_an_unknown_camera():
    registry = CameraRegistry()
    registry.connect_phone(PHONE)
    registry.connect_phone(OTHER)
    registry.rename(OTHER, "garden")
    with pytest.raises(ValueError, match="board-2 is taken by camera garden"):
        registry.rename(PHONE, "board-2")  # another camera's board id
    with pytest.raises(ValueError, match="garden is taken by camera garden"):
        registry.rename(PHONE, "garden")  # another camera's name
    with pytest.raises(ValueError, match="is not a camera name"):
        registry.rename(PHONE, " left")
    with pytest.raises(ValueError, match="no camera"):
        registry.rename("unknown-device", "left")
    assert registry.label("board-1") == "board-1"


def test_a_board_id_used_as_a_name_is_not_handed_to_a_new_phone():
    registry = CameraRegistry()
    registry.connect_phone(PHONE)
    registry.rename(PHONE, "board-2")
    assert registry.connect_phone(OTHER).mailbox.take_nowait()[0] == AssignedMessage("board-3")


# ---- forgetting ---------------------------------------------------------------------------


def test_forgetting_a_disconnected_camera_frees_its_id_and_config():
    forgotten: list[str] = []
    heard: list[NamesMessage] = []
    registry = CameraRegistry(on_names=heard.append, on_forget=forgotten.append)
    phone = registry.connect_phone(PHONE)
    registry.configure("board-1", CameraConfig(streaming=True))
    registry.rename(PHONE, "kitchen")
    page = registry.connect_watcher("board-1")
    registry.disconnect_phone(phone)
    page.take_nowait()

    registry.forget(PHONE)
    assert registry.board_of(PHONE) is None
    assert registry.cameras(lambda board_id: None).cameras == ()
    assert forgotten == ["board-1"]
    assert heard[-1] == NamesMessage(names=())
    # The open page stays, on a board with nothing configured.
    assert page.take_nowait() == [StateMessage(CameraConfig(), None, phone_connected=False)]
    # The device that comes back is a new camera: the lowest free id, no config.
    assert registry.connect_phone(OTHER).mailbox.take_nowait()[:2] == [
        AssignedMessage("board-1"), ConfigMessage(CameraConfig()),
    ]
    assert registry.connect_phone(PHONE).mailbox.take_nowait()[0] == AssignedMessage("board-2")


def test_a_connected_or_unknown_camera_cannot_be_forgotten():
    forgotten: list[str] = []
    registry = CameraRegistry(on_forget=forgotten.append)
    registry.connect_phone(PHONE)
    with pytest.raises(ValueError, match="camera board-1 is connected: close its /camera page"):
        registry.forget(PHONE)
    with pytest.raises(ValueError, match="no camera"):
        registry.forget(OTHER)
    assert registry.board_of(PHONE) == "board-1" and forgotten == []


# ---- which frames /ingest takes -------------------------------------------------------


def test_a_cameras_board_takes_frames_from_that_camera_only():
    registry = CameraRegistry()
    registry.connect_phone(PHONE)
    registry.connect_phone(OTHER)
    assert registry.ingest_refusal("board-1", PHONE) is None
    assert registry.ingest_refusal("board-1", OTHER) == (
        "camera garden-p streams as board-2, not board-1: reload /camera"
    )
    # An edge without a device id: only under a board id no camera holds.
    assert registry.ingest_refusal("board-1", None) == (
        "board board-1 belongs to camera board-1; an edge without a device id may "
        "stream only under a board id no camera holds"
    )
    assert registry.ingest_refusal("feed", None) is None
    assert registry.ingest_refusal("board-9", None) is None


def test_a_forgotten_camera_may_not_stream_under_its_old_or_any_board():
    registry = CameraRegistry()
    registry.disconnect_phone(registry.connect_phone(PHONE))
    registry.forget(PHONE)
    registry.connect_phone(OTHER)  # board-1 is the new camera's now
    for board_id in ("board-1", "board-7"):
        assert registry.ingest_refusal(board_id, PHONE) == (
            "camera kitchen- is not known to the node (forgotten?): reload /camera"
        )
    assert registry.ingest_refusal("board-1", OTHER) is None


def test_the_list_names_every_camera_in_board_order_with_its_link_and_frame_age():
    registry = CameraRegistry()
    devices = [f"phone-{n:02d}-id" for n in range(1, 11)]
    links = [registry.connect_phone(device) for device in devices]
    registry.disconnect_phone(links[0])
    listed = registry.cameras(lambda board_id: 2.0 if board_id == "board-10" else None).cameras
    assert [e.board_id for e in listed] == [f"board-{n}" for n in range(1, 11)]
    assert [e.connected for e in listed[:2]] == [False, True]
    assert listed[-1].frame_age_s == 2.0 and listed[0].frame_age_s is None


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


def test_a_restarted_node_gives_each_phone_its_board_name_and_config_back(config_file):
    config = CameraConfig.from_dict(CONFIG)
    registry = CameraRegistry()
    registry.connect_phone(OTHER)
    registry.connect_phone(PHONE)
    registry.rename(PHONE, "left")
    registry.configure("board-2", config)
    assert config_file.exists()
    assert not list(config_file.parent.glob(".*.tmp")), "the temporary file was renamed"

    heard: list[NamesMessage] = []
    restarted = CameraRegistry(on_names=heard.append)
    phone = restarted.connect_phone(PHONE)
    assert phone.mailbox.take_nowait()[:2] == [
        AssignedMessage("board-2", "left"), ConfigMessage(config),
    ]
    assert restarted.board_of(OTHER) == "board-1"
    assert heard == [NamesMessage(names=(BoardName("board-2", "left"),))]


def test_a_corrupt_config_file_is_ignored_with_a_warning(config_file, caplog):
    config_file.parent.mkdir(parents=True)
    bad = {"boards": {"phone": {**CONFIG, "resolution": "8k"}}}
    twice = {"boards": {}, "cameras": {PHONE: "left", OTHER: "left"}}
    stray_name = {"boards": {}, "cameras": {PHONE: "left"}, "names": {OTHER: "garden"}}
    for raw in ("{not json", bad, {"phone": CONFIG}, twice, stray_name):
        config_file.write_text(raw if isinstance(raw, str) else json.dumps(raw))
        with caplog.at_level(logging.WARNING):
            registry = CameraRegistry()
        assert registry.board("phone").config == CameraConfig()
        assert registry.board_of(PHONE) is None
        assert "corrupt camera config" in caplog.text
        caplog.clear()

    # The next change replaces the corrupt file with a whole one.
    registry.connect_phone(PHONE)
    registry.configure("board-1", CameraConfig(streaming=True))
    assert CameraRegistry().board("board-1").config == CameraConfig(streaming=True)


def test_a_file_from_before_the_device_ids_loads_and_drops_its_orphaned_configs(
    config_file, caplog
):
    """Such a file has only ``boards`` (``phone`` was every phone's default id): no
    camera holds them, so they are dropped with one INFO line, and no new camera can
    inherit a crop or ``streaming: true`` from them."""
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"boards": {"phone": CONFIG, "board-1": CONFIG}}))
    with caplog.at_level(logging.INFO):
        registry = CameraRegistry()
    assert [r.getMessage() for r in caplog.records] == [
        "dropping the saved config of board-1, phone: no camera holds them"
    ]
    assert registry.board("phone").config == CameraConfig()
    phone = registry.connect_phone(PHONE)
    assert phone.mailbox.take_nowait()[:2] == [AssignedMessage("board-1"), ConfigMessage(CameraConfig())]
    registry.rename(PHONE, "phone")
    assert registry.board("board-1").config == CameraConfig()


def test_a_file_from_before_the_names_keeps_every_board_id(config_file):
    """A camera renamed before names existed kept the new name as its board id; it
    stays that camera's board id, so nothing recorded under it changes."""
    config_file.parent.mkdir(parents=True)
    config_file.write_text(
        json.dumps({"boards": {"left": CONFIG}, "cameras": {PHONE: "left", OTHER: "board-1"}})
    )
    registry = CameraRegistry()
    assert registry.connect_phone(PHONE).mailbox.take_nowait()[:2] == [
        AssignedMessage("left"), ConfigMessage(CameraConfig.from_dict(CONFIG)),
    ]
    assert (registry.label("left"), registry.label("board-1")) == ("left", "board-1")


def test_a_config_saved_before_the_rules_existed_loads_as_standard(config_file):
    """A config file from before the switch has no ``rules`` key: still standard."""
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"boards": {"board-1": CONFIG}, "cameras": {PHONE: "board-1"}}))
    registry = CameraRegistry()
    assert registry.board("board-1").config == CameraConfig.from_dict(CONFIG)
    assert registry.rules("board-1") == "standard"


def test_the_rules_are_per_board_and_survive_a_restart(config_file):
    registry = CameraRegistry()
    registry.connect_phone(PHONE)
    registry.configure("board-1", CameraConfig(rules="free"))
    restarted = CameraRegistry()
    assert restarted.rules("board-1") == "free"
    assert restarted.rules("phone") == "standard"  # never configured: standard
    assert "phone" not in restarted._boards  # asking did not create the board


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


CAMERA = f"/cameras/{PHONE}"


def _hello(phone: WebSocketTestSession) -> str:
    """Read what a phone hears on connect; its board id."""
    assigned = json.loads(phone.receive_text())
    assert assigned["type"] == "assigned"
    assert json.loads(phone.receive_text())["type"] == "config"
    assert json.loads(phone.receive_text())["type"] == "previews"
    return assigned["board_id"]


def test_a_config_set_on_control_is_pushed_to_the_phone():
    with _client() as client, client.websocket_connect(CAMERA) as phone:
        assert json.loads(phone.receive_text()) == {
            "type": "assigned", "board_id": "board-1", "name": None,
        }
        assert json.loads(phone.receive_text())["type"] == "config"
        assert json.loads(phone.receive_text()) == {"type": "previews", "on": False}
        phone.send_text(json.dumps({"type": "report", "report": REPORT}))

        with client.websocket_connect("/boards/board-1/control") as control:
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
            assert _hello(phone) == "board-1"
        with client.websocket_connect("/boards/board-1/control") as control:
            control.send_text(json.dumps({"type": "configure", "config": CONFIG}))
            _until(control, "state")
        with client.websocket_connect(CAMERA) as phone:
            assert json.loads(phone.receive_text()) == {
                "type": "assigned", "board_id": "board-1", "name": None,
            }
            assert json.loads(phone.receive_text()) == {"type": "config", "config": SENT_CONFIG}


def test_a_second_page_of_one_phone_takes_over_and_the_first_is_closed():
    with _client() as client, client.websocket_connect(CAMERA) as first:
        _hello(first)
        with client.websocket_connect(CAMERA) as second:
            assert _hello(second) == "board-1"
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
            assert listed == [{"device_id": PHONE, "board_id": "board-1", "name": None,
                               "connected": True, "report": None, "frame_age_s": None}]

            cameras.send_text(json.dumps({"type": "rename", "device_id": PHONE, "name": "left"}))
            assert json.loads(phone.receive_text()) == {
                "type": "assigned", "board_id": "board-1", "name": "left",
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
        _client() as client, client.websocket_connect("/boards/board-1/control") as control,
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
        with client.websocket_connect("/boards/board-1/control") as control:
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
    import board_reader_inference.control.camera as camera_control

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
