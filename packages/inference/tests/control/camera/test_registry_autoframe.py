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
    PreviewMessage,
    PreviewsMessage,
    SendFullFrameMessage,
    StateMessage,
)

from board_reader_inference.control.autoframe.actions import PublishRead
from board_reader_inference.control.autoframe.events import (
    FullFrameFailed,
    FullFrameLocated,
)
from board_reader_inference.control.autoframe.geometry import FrameProposal, Located
from board_reader_inference.control.camera.records import FullFrame, PhoneLink
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
    page = registry.connect_watcher(PHONE)
    assert SendFullFrameMessage(0) in phone.mailbox.take_nowait()
    assert page.take_nowait()[0].autoframe == "locating"

    _answer(registry, phone, 0)
    assert asked == [FullFrame(PHONE, 0, 50.0, b"jpeg")]
    read = object()
    published = registry.autoframe_result(
        PHONE, FullFrameLocated(0, FrameProposal(FOUND_CROP, BOARD, read), 50.0)
    )
    assert published == [PublishRead(read, 50.0)]
    # The node's own crop: pushed to the phone, Auto-frame stays on.
    framed = CameraConfig(crop=FOUND_CROP)
    assert phone.mailbox.take_nowait() == [ConfigMessage(framed)]
    assert page.take_nowait() == [
        StateMessage(framed, None, phone_connected=True, autoframe="framed", watching=False)
    ]


def test_an_answer_to_another_request_costs_no_locate():
    registry, _, asked = _framing()
    phone = registry.connect_phone(PHONE)
    _answer(registry, phone, 7)
    assert asked == []


def test_no_board_waits_and_asks_again():
    registry, clock, _ = _framing()
    phone = registry.connect_phone(PHONE)
    page = registry.connect_watcher(PHONE)
    phone.mailbox.take_nowait()
    _answer(registry, phone, 0)
    registry.autoframe_result(PHONE, FullFrameFailed(0))
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
    registry.configure(PHONE, drawn)
    manual = CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5), auto=False)
    assert registry.board(PHONE).config == manual
    # The answer to the request made before arrives, and changes nothing.
    assert registry.autoframe_result(
        PHONE, FullFrameLocated(0, FrameProposal(FOUND_CROP, BOARD, object()), 50.0)
    ) == []
    assert phone.mailbox.take_nowait() == [ConfigMessage(manual)]

    registry.configure(PHONE, CameraConfig(crop=manual.crop, auto=True))
    assert phone.mailbox.take_nowait()[-1] == SendFullFrameMessage(1)
    assert registry.board(PHONE).autoframe.auto is True


def test_a_new_lens_or_frame_size_asks_for_a_new_full_frame():
    registry, _, _ = _framing()
    phone = registry.connect_phone(PHONE)
    _answer(registry, phone, 0)
    registry.autoframe_result(
        PHONE, FullFrameLocated(0, FrameProposal(FOUND_CROP, BOARD, None), 50.0)
    )
    report = CameraReport.from_dict(REPORT)
    registry.report(phone, report)
    phone.mailbox.take_nowait()
    upright = FrameSize(width=1080, height=1920)
    registry.report(phone, replace(report, frame=upright))
    assert phone.mailbox.take_nowait() == [SendFullFrameMessage(1)]


# ---- the previews: a moved camera ----------------------------------------------------

#: A board 0.4 of the frame wide: large enough to be watched on the previews.
WIDE = Located(corners=((0.3, 0.3), (0.7, 0.3), (0.7, 0.8), (0.3, 0.8)), r2=0.99)
MOVED = Located(corners=((0.35, 0.3), (0.75, 0.3), (0.75, 0.8), (0.35, 0.8)), r2=0.99)
PREVIEW = PreviewMessage(jpeg=b"preview", width=480, height=270)


def _watched() -> tuple[CameraRegistry, Clock, list[str], PhoneLink]:
    """A camera framed on ``WIDE``, with every preview handed out recorded."""
    clock = Clock()
    handed: list[str] = []
    registry = CameraRegistry(
        on_preview=lambda board_id, preview: handed.append(board_id), clock=clock
    )
    phone = registry.connect_phone(PHONE)
    registry.full_frame(phone, FullFrameMessage(0, 50.0), b"jpeg")
    registry.autoframe_result(
        PHONE, FullFrameLocated(0, FrameProposal(FOUND_CROP, WIDE, None), 50.0)
    )
    return registry, clock, handed, phone


def test_previews_stay_on_while_auto_frame_watches_with_no_control_page_open():
    registry, _, _, phone = _watched()
    assert PreviewsMessage(on=True) in phone.mailbox.take_nowait()
    registry.configure(PHONE, CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5)))
    assert PreviewsMessage(on=False) in phone.mailbox.take_nowait()  # manual: not watched


def test_a_preview_is_located_one_at_a_time_and_paced():
    registry, clock, handed, phone = _watched()
    registry.preview(phone, PREVIEW)
    registry.preview(phone, PREVIEW)  # the first is still being located
    assert handed == [PHONE]
    registry.preview_result(PHONE, WIDE, 16 / 9)
    clock.now += 1.5
    registry.preview(phone, PREVIEW)  # sooner than preview_every
    assert handed == [PHONE]
    clock.now += 1.5
    registry.preview(phone, PREVIEW)
    assert handed == [PHONE, PHONE]
    registry.preview_skipped(PHONE)  # the read queue was busy: the next may go
    registry.preview(phone, PREVIEW)
    assert len(handed) == 3


def test_two_previews_off_the_crop_ask_for_a_new_full_frame():
    registry, clock, _, phone = _watched()
    phone.mailbox.take_nowait()
    clock.now += 10.0  # past the re-frame rate limit
    for _ in range(2):
        registry.preview(phone, PREVIEW)
        registry.preview_result(PHONE, MOVED, 16 / 9)
        clock.now += 3.0
    assert SendFullFrameMessage(1) in phone.mailbox.take_nowait()


def test_re_frame_now_asks_again_and_switches_a_manual_camera_back_on():
    registry, _, _, phone = _watched()
    registry.reframe(PHONE)
    assert SendFullFrameMessage(1) in phone.mailbox.take_nowait()
    registry.full_frame(phone, FullFrameMessage(1, 60.0), b"jpeg")
    registry.autoframe_result(
        PHONE, FullFrameLocated(1, FrameProposal(FOUND_CROP, WIDE, None), 60.0)
    )
    registry.configure(PHONE, CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5)))
    assert registry.board(PHONE).config.auto is False
    registry.reframe(PHONE)
    assert registry.board(PHONE).config.auto is True
    assert SendFullFrameMessage(2) in phone.mailbox.take_nowait()
