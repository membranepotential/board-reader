"""The camera control contract (``board_reader_core.control``): what is accepted, what is refused."""

from __future__ import annotations

import base64
import json

import pytest

from board_reader_core.control import (
    AssignedMessage,
    BoardDirection,
    CameraConfig,
    CameraEntry,
    CameraReport,
    CamerasMessage,
    ConfigMessage,
    CropRect,
    ForgetMessage,
    FrameSize,
    FullFrameMessage,
    HealthMessage,
    PreviewMessage,
    PreviewsMessage,
    RenameMessage,
    ReplacedMessage,
    ReportMessage,
    SendFullFrameMessage,
    StateMessage,
    camera_board_id,
    camera_name,
    parse_cameras_message,
    parse_control_message,
    parse_phone_message,
)


def test_a_config_round_trips_through_its_wire_form():
    config = CameraConfig(
        crop=CropRect(x=0.1, y=0.2, w=0.6, h=0.7),
        device="Back Ultra Wide Camera",
        resolution="2160p",
        streaming=True,
        rules="free",
    )
    assert CameraConfig.from_dict(config.to_dict()) == config


def test_omitted_fields_take_the_defaults_and_the_default_sends_the_whole_frame():
    config = CameraConfig.from_dict({})
    assert config == CameraConfig()
    assert (config.crop, config.device, config.resolution, config.streaming, config.rules) == (
        None,
        None,
        "1080p",
        False,
        "standard",
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"streming": True},  # a typo must not silently do nothing
        {"resolution": "4k"},
        {"streaming": "yes"},
        {"rules": "chess960"},  # not a mode the guard knows
        {"device": ""},
        {"device": 3},
        {"crop": {"x": 0, "y": 0, "w": 1}},
        {"crop": {"x": 0.5, "y": 0, "w": 0.6, "h": 0.5}},  # past the right edge
        {"crop": {"x": -0.1, "y": 0, "w": 0.5, "h": 0.5}},
        {"crop": {"x": 0, "y": 0, "w": 0.01, "h": 0.5}},  # a slip, not a board
        {"crop": {"x": True, "y": 0, "w": 0.5, "h": 0.5}},
        [],
    ],
)
def test_a_malformed_config_is_refused(raw):
    with pytest.raises(ValueError):
        CameraConfig.from_dict(raw)


def test_the_crop_maps_to_pixels_inside_the_frame():
    assert CropRect(x=0.25, y=0.5, w=0.5, h=0.5).pixels(1920, 1080) == (480, 540, 1440, 1080)
    # The full frame is the full frame, at any size.
    assert CropRect(x=0, y=0, w=1, h=1).pixels(641, 359) == (0, 0, 641, 359)


def test_the_cli_spelling_parses_and_refuses_what_is_not_a_rectangle():
    assert CropRect.parse("0.1,0.2,0.5,0.6") == CropRect(x=0.1, y=0.2, w=0.5, h=0.6)
    for bad in ("0.1,0.2,0.5", "a,b,c,d", "0.6,0,0.5,0.5"):
        with pytest.raises(ValueError):
            CropRect.parse(bad)


def test_a_report_round_trips_and_knows_when_the_phone_is_upright():
    raw = {
        "devices": [{"id": "d1", "label": "Back Camera"}, {"id": "d2", "label": "Front Camera"}],
        "device": "Back Camera",
        "frame": {"width": 1080, "height": 1920},
        "ingest": "streaming",
    }
    report = CameraReport.from_dict(raw)
    assert report.to_dict() == raw
    assert report.frame == FrameSize(width=1080, height=1920)
    assert report.frame.portrait


def test_a_report_before_the_camera_opens_has_no_frame():
    report = CameraReport.from_dict({"devices": [], "device": None, "frame": None, "ingest": "idle"})
    assert report.frame is None


@pytest.mark.parametrize(
    "raw",
    [
        {"frame": {"width": 0, "height": 10}},
        {"frame": {"width": 1.5, "height": 10}},
        {"ingest": "flying"},
        {"devices": [{"id": "d1"}]},
        {"devices": "Back Camera"},
        {"zoom": 2},
    ],
)
def test_a_malformed_report_is_refused(raw):
    with pytest.raises(ValueError):
        CameraReport.from_dict(raw)


# ---- the socket messages --------------------------------------------------------

REPORT = {"devices": [{"id": "d1", "label": "Back Camera"}], "device": "Back Camera",
          "frame": {"width": 1920, "height": 1080}, "ingest": "idle"}


def test_the_phone_reports_with_a_typed_message():
    message = parse_phone_message(json.dumps({"type": "report", "report": REPORT}))
    assert message.report == CameraReport.from_dict(REPORT)


def test_a_control_page_configures_with_a_typed_message():
    config = {"crop": None, "device": None, "resolution": "720p", "streaming": True}
    message = parse_control_message(json.dumps({"type": "configure", "config": config}))
    assert message.config == CameraConfig.from_dict(config)


@pytest.mark.parametrize(
    "text",
    [
        "streaming=1",  # not JSON
        "[]",  # not an object
        json.dumps({"report": REPORT}),  # no type
        json.dumps({"type": "configure", "config": {}}),  # the other socket's message
        json.dumps({"type": "report"}),  # its body missing
        json.dumps({"type": "report", "report": REPORT, "extra": 1}),  # unknown key
        json.dumps({"type": "report", "report": {"ingest": "flying"}}),  # bad body
    ],
)
def test_the_phone_socket_refuses_what_is_not_a_report(text):
    with pytest.raises(ValueError):
        parse_phone_message(text)


def test_a_control_page_cannot_configure_a_crop_outside_the_frame():
    bad = {"type": "configure", "config": {"crop": {"x": 0.9, "y": 0, "w": 0.5, "h": 0.5}}}
    with pytest.raises(ValueError, match="crop"):
        parse_control_message(json.dumps(bad))


def test_every_message_down_names_its_type():
    config = CameraConfig(streaming=True)
    assert json.loads(ConfigMessage(config).to_json()) == {"type": "config", "config": config.to_dict()}
    assert json.loads(PreviewsMessage(on=True).to_json()) == {"type": "previews", "on": True}
    state = json.loads(
        StateMessage(config, None, phone_connected=False, autoframe="framed").to_json()
    )
    assert state == {"type": "state", "config": config.to_dict(), "report": None,
                     "phone_connected": False, "autoframe": "framed"}
    assert json.loads(SendFullFrameMessage(request=4).to_json()) == {
        "type": "send_full_frame", "request": 4,
    }


def test_the_phone_answers_a_full_frame_request_with_a_header_and_reports_alike():
    header = {"type": "full_frame", "request": 4, "capture_ts": 1764000000.25}
    assert parse_phone_message(json.dumps(header)) == FullFrameMessage(4, 1764000000.25)
    report = {"type": "report", "report": {"ingest": "idle"}}
    assert parse_phone_message(json.dumps(report)) == ReportMessage(CameraReport())
    with pytest.raises(ValueError):
        parse_phone_message(json.dumps({**header, "request": -1}))


def test_auto_frame_is_on_unless_switched_off():
    assert CameraConfig().auto is True
    assert CameraConfig.from_dict({"crop": None}).auto is True
    health = json.loads(HealthMessage("chessq", None, 3, 0, None, None, None).to_json())
    assert health["type"] == "health"
    assert (health["frame_age_s"], health["fps"], health["direction"]) == (None, None, None)
    turned = BoardDirection(degrees=89.0, white_side="left", camera_moved=False)
    health = json.loads(HealthMessage("chessq", None, 3, 0, 0.5, 4.0, turned).to_json())
    assert health["direction"] == {"degrees": 89.0, "white_side": "left", "camera_moved": False}


def test_a_preview_carries_its_jpeg_as_base64():
    sent = json.loads(PreviewMessage(jpeg=b"\xff\xd8jpeg", width=480, height=270).to_json())
    assert (sent["type"], sent["width"], sent["height"]) == ("preview", 480, 270)
    assert base64.b64decode(sent["jpeg"]) == b"\xff\xd8jpeg"


# ---- camera identity -------------------------------------------------------------

DEVICE = "kitchen-phone-id"


@pytest.mark.parametrize("name", ["board-1", "phone", "left.table_2", "A1"])
def test_a_camera_board_id_is_one_plain_path_segment(name):
    assert camera_board_id(name) == name


@pytest.mark.parametrize("name", ["", "left board", "a/b", "-x", ".hidden", "x" * 33, "brett-ü"])
def test_a_camera_board_id_refuses_what_a_url_path_would_mangle(name):
    with pytest.raises(ValueError, match="is not a board id: use up to 32 letters"):
        camera_board_id(name)


@pytest.mark.parametrize("name", ["kitchen", "Board 1", "left.table_2", "-x", "A"])
def test_a_camera_name_may_hold_spaces_inside(name):
    assert camera_name(name) == name


@pytest.mark.parametrize("name", ["", " kitchen", "kitchen ", "a/b", 'say "hi"', "x" * 33, "Küche"])
def test_a_camera_name_refuses_edges_of_space_and_what_is_not_ascii(name):
    with pytest.raises(ValueError, match="is not a camera name: use up to 32 letters"):
        camera_name(name)


def test_the_control_ui_renames_and_forgets_with_typed_messages():
    message = parse_cameras_message(json.dumps({"type": "rename", "device_id": DEVICE, "name": "left"}))
    assert message == RenameMessage(DEVICE, "left")
    cleared = parse_cameras_message(json.dumps({"type": "rename", "device_id": DEVICE, "name": None}))
    assert cleared == RenameMessage(DEVICE, None)
    assert parse_cameras_message(json.dumps({"type": "forget", "device_id": DEVICE})) == ForgetMessage(
        DEVICE
    )
    with pytest.raises(ValueError):  # not a device id
        parse_cameras_message(json.dumps({"type": "forget", "device_id": "x"}))


def test_the_identity_messages_name_their_type():
    assert json.loads(AssignedMessage("board-2").to_json()) == {
        "type": "assigned", "board_id": "board-2", "name": None,
    }
    assert json.loads(ReplacedMessage().to_json()) == {"type": "replaced"}
    entry = CameraEntry(DEVICE, "board-1", "kitchen", True, None, 0.5)
    assert json.loads(CamerasMessage(cameras=(entry,)).to_json()) == {
        "type": "cameras",
        "cameras": [
            {"device_id": DEVICE, "board_id": "board-1", "name": "kitchen", "connected": True,
             "report": None, "frame_age_s": 0.5},
        ],
    }
