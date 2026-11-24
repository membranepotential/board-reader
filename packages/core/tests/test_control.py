"""The camera control contract (``board_reader_core.control``): what is accepted, what is refused."""

from __future__ import annotations

import pytest

from board_reader_core.control import (
    CameraConfig,
    CameraReport,
    CropRect,
    FrameSize,
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
