"""What the node must do when Auto-frame says so."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum

from board_reader_core.control import CropRect


class Why(Enum):
    """Why a full frame is wanted."""

    #: Nothing is framed yet, or the frame itself changed (lens, resolution).
    FRESH = "fresh"
    #: The camera moved away from the pushed crop.
    DRIFT = "drift"


class Status(StrEnum):
    """What /control shows about auto-frame for this camera."""

    IDLE = "idle"
    LOCATING = "locating"
    FRAMED = "framed"
    NO_BOARD = "no-board"
    MANUAL = "manual"


@dataclass(frozen=True)
class RequestFullFrame:
    """Ask the phone for one zoomed-out full frame; answer with ``request``."""

    request: int
    why: Why


@dataclass(frozen=True)
class PushCrop:
    """Push this crop to the camera's config."""

    crop: CropRect


@dataclass(frozen=True)
class PublishRead[R]:
    """Put this read into the position stream as a normal read at ``capture_ts``."""

    read: R
    capture_ts: float


@dataclass(frozen=True)
class ReportStatus:
    """Auto-frame's status for this camera changed."""

    status: Status


type Action[R] = RequestFullFrame | PushCrop | PublishRead[R] | ReportStatus
