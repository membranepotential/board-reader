"""The camera control contract (``board_reader_core.control``): what is accepted, what is refused."""

from __future__ import annotations

import base64
import json

import pytest

from board_reader_core.control import (
    BoardDirection,
    CameraConfig,
    CameraReport,
    ConfigMessage,
    CropRect,
    FrameSize,
    HealthMessage,
    PreviewMessage,
    PreviewsMessage,
    StateMessage,
    parse_control_message,
    parse_phone_message,
)


def test_a_config_round_trips_through_its_wire_form():
    config = CameraConfig(
        crop=CropRect(x=0.1, y=0.2, w=0.6, h=0.7),
        device="Back Ultra Wide Camera",
        resolution="2160p",
        streaming=True,
    )
    assert CameraConfig.from_dict(config.to_dict()) == config


def test_omitted_fields_take_the_defaults_and_the_default_sends_the_whole_frame():
    config = CameraConfig.from_dict({})
    assert config == CameraConfig()
    assert (config.crop, config.device, config.resolution, config.streaming) == (
        None,
        None,
        "1080p",
        False,
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"streming": True},  # a typo must not silently do nothing
        {"resolution": "4k"},
        {"streaming": "yes"},
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
    state = json.loads(StateMessage(config, None, phone_connected=False).to_json())
    assert state == {"type": "state", "config": config.to_dict(), "report": None,
                     "phone_connected": False}
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
