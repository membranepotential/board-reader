"""What happens to a camera, as Auto-frame sees it."""

from __future__ import annotations

from dataclasses import dataclass

from board_reader_inference.control.autoframe.geometry import FrameProposal, Located


@dataclass(frozen=True)
class CameraRegistered:
    """The board's camera connected its control link."""


@dataclass(frozen=True)
class StreamingStarted:
    """The camera started streaming frames to ``/ingest``."""


@dataclass(frozen=True)
class FullFrameLocated[R]:
    """``autoframe`` found the board in the full frame of ``request``."""

    request: int
    proposal: FrameProposal[R]
    capture_ts: float


@dataclass(frozen=True)
class FullFrameFailed:
    """``autoframe`` found no board in the full frame of ``request``."""

    request: int


@dataclass(frozen=True)
class PreviewLocated:
    """A whole-frame preview was located: its trusted board, or ``None``.
    ``aspect`` is the preview's width over its height."""

    board: Located | None
    aspect: float


@dataclass(frozen=True)
class ManualCropSet:
    """Someone set the crop by hand on /control (never the node's own push)."""


@dataclass(frozen=True)
class AutoToggled:
    """/control switched auto-frame on or off."""

    on: bool


@dataclass(frozen=True)
class Reframed:
    """The camera's lens or resolution changed: the whole frame is a new picture."""


@dataclass(frozen=True)
class Disconnected:
    """The camera's control link dropped."""


@dataclass(frozen=True)
class Tick:
    """Time passed: lets a deferred request, a retry or a timeout happen."""


type Event[R] = (
    CameraRegistered
    | StreamingStarted
    | FullFrameLocated[R]
    | FullFrameFailed
    | PreviewLocated
    | ManualCropSet
    | AutoToggled
    | Reframed
    | Disconnected
    | Tick
)
