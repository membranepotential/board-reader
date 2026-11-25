"""The registry's Auto-frame wiring: what it asks the phone, pushes and publishes."""

from __future__ import annotations

from dataclasses import replace

from board_reader_core.control import (
    CameraConfig,
    CameraReport,
    ConfigMessage,
    CropRect,
    FrameSize,
    FullFrameMessage,
    SendFullFrameMessage,
    StateMessage,
)

from board_reader_inference.control.autoframe.actions import PublishRead
from board_reader_inference.control.autoframe.events import (
    FullFrameFailed,
    FullFrameLocated,
)
from board_reader_inference.control.autoframe.geometry import FrameProposal, Located
from board_reader_inference.control.camera.records import FullFrame
from board_reader_inference.control.camera.registry import CameraRegistry

REPORT = {
    "devices": [{"id": "d1", "label": "Back Camera"}, {"id": "d2", "label": "Back Ultra Wide Camera"}],
    "device": "Back Camera",
    "frame": {"width": 1920, "height": 1080},
    "ingest": "idle",
}

PHONE = "kitchen-phone-id"

BOARD = Located(corners=((0.3, 0.3), (0.6, 0.3), (0.6, 0.7), (0.3, 0.7)), r2=0.99)
FOUND_CROP = CropRect(0.27, 0.26, 0.36, 0.48)


class Clock:
    """The registry's monotonic clock, moved by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _framing() -> tuple[CameraRegistry, Clock, list[FullFrame]]:
    clock = Clock()
    asked: list[FullFrame] = []
    return CameraRegistry(on_full_frame=asked.append, clock=clock), clock, asked


def _answer(registry: CameraRegistry, link, request: int) -> None:
    registry.full_frame(link, FullFrameMessage(request, 50.0), b"jpeg")


def test_a_new_camera_is_asked_for_a_full_frame_and_framed_from_it():
    registry, _, asked = _framing()
    phone = registry.connect_phone(PHONE)
    page = registry.connect_watcher("board-1")
    assert SendFullFrameMessage(0) in phone.mailbox.take_nowait()
    assert page.take_nowait()[0].autoframe == "locating"

    _answer(registry, phone, 0)
    assert asked == [FullFrame("board-1", 0, 50.0, b"jpeg")]
    read = object()
    published = registry.autoframe_result(
        "board-1", FullFrameLocated(0, FrameProposal(FOUND_CROP, BOARD, read), 50.0)
    )
    assert published == [PublishRead(read, 50.0)]
    # The node's own crop: pushed to the phone, Auto-frame stays on.
    framed = CameraConfig(crop=FOUND_CROP)
    assert phone.mailbox.take_nowait() == [ConfigMessage(framed)]
    assert page.take_nowait() == [
        StateMessage(framed, None, phone_connected=True, autoframe="framed")
    ]


def test_an_answer_to_another_request_costs_no_locate():
    registry, _, asked = _framing()
    phone = registry.connect_phone(PHONE)
    _answer(registry, phone, 7)
    assert asked == []


def test_no_board_waits_and_asks_again():
    registry, clock, _ = _framing()
    phone = registry.connect_phone(PHONE)
    page = registry.connect_watcher("board-1")
    phone.mailbox.take_nowait()
    _answer(registry, phone, 0)
    registry.autoframe_result("board-1", FullFrameFailed(0))
    assert page.take_nowait()[-1].autoframe == "no-board"

    clock.now += 1.0
    registry.tick()
    assert phone.mailbox.take_nowait() == []  # still backing off
    clock.now += 2.5
    registry.tick()
    assert phone.mailbox.take_nowait() == [SendFullFrameMessage(1)]


def test_a_crop_set_on_control_switches_auto_frame_off_and_the_switch_back_on():
    registry, _, _ = _framing()
    phone = registry.connect_phone(PHONE)
    phone.mailbox.take_nowait()
    drawn = CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5))  # auto as /control had it: on
    registry.configure("board-1", drawn)
    manual = CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5), auto=False)
    assert registry.board("board-1").config == manual
    # The answer to the request made before arrives, and changes nothing.
    assert registry.autoframe_result(
        "board-1", FullFrameLocated(0, FrameProposal(FOUND_CROP, BOARD, object()), 50.0)
    ) == []
    assert phone.mailbox.take_nowait() == [ConfigMessage(manual)]

    registry.configure("board-1", CameraConfig(crop=manual.crop, auto=True))
    assert phone.mailbox.take_nowait()[-1] == SendFullFrameMessage(1)
    assert registry.board("board-1").autoframe.auto is True


def test_a_new_lens_or_frame_size_asks_for_a_new_full_frame():
    registry, _, _ = _framing()
    phone = registry.connect_phone(PHONE)
    _answer(registry, phone, 0)
    registry.autoframe_result(
        "board-1", FullFrameLocated(0, FrameProposal(FOUND_CROP, BOARD, None), 50.0)
    )
    report = CameraReport.from_dict(REPORT)
    registry.report(phone, report)
    phone.mailbox.take_nowait()
    upright = FrameSize(width=1080, height=1920)
    registry.report(phone, replace(report, frame=upright))
    assert phone.mailbox.take_nowait() == [SendFullFrameMessage(1)]
