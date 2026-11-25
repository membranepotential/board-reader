"""The records the registry holds: one board's camera, one phone's socket, one full frame."""

from __future__ import annotations

from dataclasses import dataclass, field

from board_reader_core.control import (
    CameraConfig,
    CameraReport,
    ControlDown,
    PhoneDown,
    PreviewMessage,
)

from board_reader_inference.control.autoframe.state import AutoFrame
from board_reader_inference.control.camera.direction import DirectionTrack
from board_reader_inference.control.camera.mailbox import Mailbox


@dataclass(slots=True)
class BoardCamera:
    """One board's camera state.

    ``autoframe`` is Auto-frame's controller state; its ``auto`` always equals
    ``config.auto``, which is the persisted copy.
    """

    config: CameraConfig = field(default_factory=CameraConfig)
    report: CameraReport | None = None
    preview: PreviewMessage | None = None
    direction: DirectionTrack = field(default_factory=DirectionTrack)
    watchers: set[Mailbox[ControlDown]] = field(default_factory=set)
    autoframe: AutoFrame = field(default_factory=AutoFrame)


@dataclass(frozen=True, slots=True)
class FullFrame:
    """A full frame a camera sent for Auto-frame's ``request``, still a JPEG."""

    board_id: str
    request: int
    capture_ts: float
    jpeg: bytes


@dataclass(eq=False, slots=True)
class PhoneLink:
    """One open phone socket: the device it speaks for and its outbox."""

    device_id: str
    mailbox: Mailbox[PhoneDown] = field(default_factory=Mailbox)
