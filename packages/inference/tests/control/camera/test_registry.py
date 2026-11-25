"""The camera registry: the fan-out to phones and watchers, names, forgetting and who may stream."""

from __future__ import annotations

import pytest

from board_reader_core.control import (
    AssignedMessage,
    CameraConfig,
    CameraReport,
    ConfigMessage,
    CropRect,
    PreviewMessage,
    PreviewsMessage,
    ReplacedMessage,
    SendFullFrameMessage,
    StateMessage,
)
from board_reader_core.protocol import BoardName, NamesMessage

from board_reader_inference.control.camera.registry import CameraRegistry

REPORT = {
    "devices": [{"id": "d1", "label": "Back Camera"}, {"id": "d2", "label": "Back Ultra Wide Camera"}],
    "device": "Back Camera",
    "frame": {"width": 1920, "height": 1080},
    "ingest": "idle",
}

PHONE = "kitchen-phone-id"
OTHER = "garden-phone-id"


def test_a_phone_hears_its_board_its_config_and_that_nobody_watches_on_connect():
    registry = CameraRegistry()
    phone = registry.connect_phone(PHONE)
    assert phone.mailbox.take_nowait() == [
        AssignedMessage(PHONE, "board-1"),  # its board id is its device id
        ConfigMessage(CameraConfig()),
        PreviewsMessage(on=False),
        SendFullFrameMessage(0),  # Auto-frame wants the whole picture
    ]


def test_a_phone_keeps_its_board_and_name_across_a_reconnect():
    registry = CameraRegistry()
    first = registry.connect_phone(PHONE)
    assert registry.connect_phone(OTHER).mailbox.take_nowait()[0] == AssignedMessage(OTHER, "board-2")
    registry.disconnect_phone(first)
    # The same phone again (a reload, a reconnect) streams as the same board.
    assert registry.connect_phone(PHONE).mailbox.take_nowait()[0] == AssignedMessage(PHONE, "board-1")


def test_a_newer_socket_of_one_phone_replaces_the_older():
    registry = CameraRegistry()
    old = registry.connect_phone(PHONE)
    old.mailbox.take_nowait()
    new = registry.connect_phone(PHONE)
    assert old.mailbox.take_nowait() == [ReplacedMessage()]
    assert new.mailbox.take_nowait()[0] == AssignedMessage(PHONE, "board-1")

    # The replaced socket speaks for nobody: its late report and its close change nothing.
    registry.report(old, CameraReport.from_dict({**REPORT, "ingest": "streaming"}))
    registry.disconnect_phone(old)
    watcher = registry.connect_watcher(PHONE)
    assert watcher.take_nowait() == [
        StateMessage(CameraConfig(), None, phone_connected=True, autoframe="locating", watching=False)
    ]


def test_a_config_change_reaches_the_boards_phone_and_every_watcher():
    registry = CameraRegistry()
    phone = registry.connect_phone(PHONE)
    watchers = [registry.connect_watcher(PHONE), registry.connect_watcher(PHONE)]
    for mailbox in (phone.mailbox, *watchers):
        mailbox.take_nowait()

    config = CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5), streaming=True, auto=False)
    registry.configure(PHONE, config)
    assert phone.mailbox.take_nowait() == [ConfigMessage(config)]
    for watcher in watchers:
        assert watcher.take_nowait() == [
            StateMessage(config, None, phone_connected=True, autoframe="manual", watching=False)
        ]


def test_previews_are_wanted_exactly_while_a_control_page_watches():
    registry = CameraRegistry()
    phone = registry.connect_phone(PHONE)
    phone.mailbox.take_nowait()

    first = registry.connect_watcher(PHONE)
    assert phone.mailbox.take_nowait() == [PreviewsMessage(on=True)]
    second = registry.connect_watcher(PHONE)
    assert phone.mailbox.take_nowait() == []  # already on
    registry.disconnect_watcher(PHONE, first)
    assert phone.mailbox.take_nowait() == []  # one still watches
    registry.disconnect_watcher(PHONE, second)
    assert phone.mailbox.take_nowait() == [PreviewsMessage(on=False)]


def test_a_watcher_hears_the_phone_come_and_go_and_its_reports():
    registry = CameraRegistry()
    watcher = registry.connect_watcher(PHONE)
    assert watcher.take_nowait() == [
        StateMessage(CameraConfig(), None, phone_connected=False, autoframe="idle", watching=False)
    ]

    phone = registry.connect_phone(PHONE)
    report = CameraReport.from_dict(REPORT)
    registry.report(phone, report)
    # Latest state wins: the connect and the report collapse into one message.
    assert watcher.take_nowait() == [
        StateMessage(CameraConfig(), report, phone_connected=True, autoframe="locating", watching=False)
    ]

    registry.disconnect_phone(phone)
    # The report outlives the phone, so the lens list survives a reload.
    assert watcher.take_nowait() == [
        StateMessage(CameraConfig(), report, phone_connected=False, autoframe="idle", watching=False)
    ]


def test_a_new_watcher_gets_the_last_preview_at_once():
    registry = CameraRegistry()
    preview = PreviewMessage(jpeg=b"jpeg", width=480, height=270)
    registry.preview(registry.connect_phone(PHONE), preview)
    watcher = registry.connect_watcher(PHONE)
    assert watcher.take_nowait()[1:] == [preview]


def test_boards_are_independent():
    registry = CameraRegistry()
    left = registry.connect_phone(PHONE)
    left.mailbox.take_nowait()
    registry.connect_phone(OTHER)
    registry.connect_watcher(OTHER)
    registry.configure(OTHER, CameraConfig(streaming=True))
    assert left.mailbox.take_nowait() == []
    assert registry.board(PHONE).config.streaming is False


# ---- naming ---------------------------------------------------------------------------


def test_a_new_camera_gets_the_lowest_default_name_no_other_camera_uses():
    registry = CameraRegistry()
    registry.connect_phone(PHONE)
    registry.connect_phone(OTHER)
    registry.rename(PHONE, "board-3")  # frees board-1, takes board-3
    names = [registry.connect_phone(f"phone-{n}-device").mailbox.take_nowait()[0].name for n in range(3)]
    assert names == ["board-1", "board-4", "board-5"]


def test_a_rename_sets_a_name_and_nothing_else():
    """The board id is the device's for good: a rename is an alias people read."""
    heard: list[NamesMessage] = []
    registry = CameraRegistry(on_names=heard.append)
    phone = registry.connect_phone(PHONE)
    config = CameraConfig(crop=CropRect(0.1, 0.1, 0.5, 0.5), streaming=True, auto=False)
    registry.configure(PHONE, config)
    page = registry.connect_watcher(PHONE)
    for mailbox in (phone.mailbox, page):
        mailbox.take_nowait()

    registry.rename(PHONE, "kitchen table")
    assert registry.name_of(PHONE) == "kitchen table"
    assert registry.board(PHONE).config == config
    assert registry.label(PHONE) == "kitchen table"
    # The phone hears its name beside the same board id; the page has nothing new.
    assert phone.mailbox.take_nowait() == [AssignedMessage(PHONE, "kitchen table")]
    assert page.take_nowait() == []
    assert heard == [
        NamesMessage(names=()),
        NamesMessage(names=(BoardName(PHONE, "board-1"),)),
        NamesMessage(names=(BoardName(PHONE, "kitchen table"),)),
    ]
    assert registry.cameras(lambda board_id: None).cameras[0].name == "kitchen table"

    # The same name again changes nothing; None gives it a default name again.
    registry.rename(PHONE, "kitchen table")
    assert len(heard) == 3
    registry.rename(PHONE, None)
    assert heard[-1] == NamesMessage(names=(BoardName(PHONE, "board-1"),))


def test_a_rename_is_refused_for_a_taken_or_malformed_name_or_an_unknown_camera():
    registry = CameraRegistry()
    registry.connect_phone(PHONE)
    registry.connect_phone(OTHER)
    registry.rename(OTHER, "garden")
    with pytest.raises(ValueError, match="garden-phone-id is taken by camera garden"):
        registry.rename(PHONE, OTHER)  # another camera's board id
    with pytest.raises(ValueError, match="garden is taken by camera garden"):
        registry.rename(PHONE, "garden")  # another camera's name
    with pytest.raises(ValueError, match="is not a camera name"):
        registry.rename(PHONE, " left")
    with pytest.raises(ValueError, match="no camera"):
        registry.rename("unknown-device", "left")
    assert registry.label(PHONE) == "board-1"


# ---- forgetting ---------------------------------------------------------------------------


def test_forgetting_a_disconnected_camera_drops_its_name_and_config():
    forgotten: list[str] = []
    heard: list[NamesMessage] = []
    registry = CameraRegistry(on_names=heard.append, on_forget=forgotten.append)
    phone = registry.connect_phone(PHONE)
    registry.configure(PHONE, CameraConfig(streaming=True))
    registry.rename(PHONE, "kitchen")
    registry.connect_phone(OTHER)  # takes board-1, which the rename freed
    page = registry.connect_watcher(PHONE)
    registry.disconnect_phone(phone)
    page.take_nowait()

    registry.forget(PHONE)
    assert registry.name_of(PHONE) is None
    assert [e.device_id for e in registry.cameras(lambda board_id: None).cameras] == [OTHER]
    assert forgotten == [PHONE]
    assert heard[-1] == NamesMessage(names=(BoardName(OTHER, "board-1"),))
    # The open page stays, on a board with nothing configured.
    assert page.take_nowait() == [
        StateMessage(CameraConfig(), None, phone_connected=False, autoframe="idle", watching=False)
    ]
    # The device that comes back is a new camera: the same board id, a fresh default
    # name, no config.
    assert registry.connect_phone(PHONE).mailbox.take_nowait()[:2] == [
        AssignedMessage(PHONE, "board-2"), ConfigMessage(CameraConfig()),
    ]


def test_a_connected_or_unknown_camera_cannot_be_forgotten():
    forgotten: list[str] = []
    registry = CameraRegistry(on_forget=forgotten.append)
    registry.connect_phone(PHONE)
    with pytest.raises(ValueError, match="camera board-1 is connected: close its /camera page"):
        registry.forget(PHONE)
    with pytest.raises(ValueError, match="no camera"):
        registry.forget(OTHER)
    assert registry.name_of(PHONE) == "board-1" and forgotten == []


# ---- which frames /ingest takes -------------------------------------------------------


def test_a_cameras_board_takes_frames_from_that_camera_only():
    registry = CameraRegistry()
    registry.connect_phone(PHONE)
    registry.connect_phone(OTHER)
    assert registry.ingest_refusal(PHONE, PHONE) is None
    assert registry.ingest_refusal(PHONE, OTHER) == (
        "camera board-2 streams as its device id, not kitchen-phone-id: reload /camera"
    )
    # An edge without a device id: only under a board id no camera holds, a default
    # name included (a name is no board id).
    assert registry.ingest_refusal(PHONE, None) == (
        "board kitchen- belongs to camera board-1; an edge without a device id may "
        "stream only under a board id no camera holds"
    )
    assert registry.ingest_refusal("feed", None) is None
    assert registry.ingest_refusal("board-1", None) is None


def test_a_forgotten_camera_may_not_stream_until_it_registers_again():
    registry = CameraRegistry()
    registry.disconnect_phone(registry.connect_phone(PHONE))
    registry.forget(PHONE)
    refused = "camera kitchen- is not known to the node (forgotten?): reload /camera"
    assert registry.ingest_refusal(PHONE, PHONE) == refused
    assert registry.ingest_refusal(PHONE, None) is None  # no camera holds it now
    registry.connect_phone(PHONE)
    assert registry.ingest_refusal(PHONE, PHONE) is None


def test_the_list_names_every_camera_in_name_order_with_its_link_and_frame_age():
    registry = CameraRegistry()
    devices = [f"phone-{n:02d}-device" for n in range(1, 11)]
    links = [registry.connect_phone(device) for device in reversed(devices)]
    registry.disconnect_phone(links[0])
    listed = registry.cameras(lambda board_id: 2.0 if board_id == devices[0] else None).cameras
    assert [e.name for e in listed] == [f"board-{n}" for n in range(1, 11)]
    assert listed[0].device_id == devices[-1] and listed[-1].device_id == devices[0]
    assert [e.connected for e in listed[:2]] == [False, True]
    assert listed[-1].frame_age_s == 2.0 and listed[0].frame_age_s is None
