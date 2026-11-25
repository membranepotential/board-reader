"""Auto-frame's state for one board's camera."""

from __future__ import annotations

from dataclasses import dataclass

from board_reader_inference.control.autoframe.actions import Why
from board_reader_inference.control.autoframe.geometry import Located


@dataclass(frozen=True)
class InFlight:
    """The one full-frame request the phone has not answered yet."""

    request: int
    why: Why
    since: float


@dataclass(frozen=True)
class Backoff:
    """``failures`` locates in a row found no board; retry at ``until``."""

    failures: int
    until: float


@dataclass(frozen=True)
class Drifting:
    """``run`` previews in a row lay off the anchor and settled; ``last`` is the
    latest of them."""

    last: Located
    run: int


@dataclass(frozen=True)
class AutoFrame:
    """Auto-frame's state for one board's camera. Start from ``AutoFrame()`` (or
    ``AutoFrame(auto=False)`` for a camera whose crop was set by hand)."""

    auto: bool = True
    connected: bool = False
    #: A full frame is wanted and not yet requested.
    want: Why | None = None
    inflight: InFlight | None = None
    #: When the latest request was sent; the drift rate limit counts from it.
    last_request: float | None = None
    #: The next request's number; never reset, so a late answer matches nothing.
    next_request: int = 0
    backoff: Backoff | None = None
    #: The board the pushed crop was built from.
    anchor: Located | None = None
    drifting: Drifting | None = None
