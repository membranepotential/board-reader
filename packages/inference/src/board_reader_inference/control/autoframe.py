"""Auto-frame's decisions for one board's camera: when to look for the board, when to
push a crop, when the camera moved. Pure: events in, actions out, no I/O.

The node frames each camera by itself. When a camera registers (or starts streaming
without a crop of ours), it asks the phone for **one zoomed-out full frame**,
ChessQ locates the board in it (``recognition/locate.py``'s ``autoframe``), and the
node pushes the proposed crop. When the locate's last read already saw the board
large enough (``Proposal.dominant``), that read goes into the position stream as an
ordinary read, stamped with the full frame's ``capture_ts``.

**Drift.** After a crop is pushed, the phone's whole-frame previews (480 px, every
1.5 s) are located too (``locate_preview``). Each trusted preview's board is compared
with the board the crop was built from (the **anchor**, the full-frame locate's
corners): the displacement is the symmetric Hausdorff distance between the two
corner sets, in pixels, over the anchor's width in pixels (:func:`displacement`).
It ignores corner labels, because a read that turns the board by a quarter
relabels the same four points. The camera has moved when :attr:`Tuning.drift_previews`
consecutive trusted previews are each farther than :attr:`Tuning.drift` from the
anchor and each within :attr:`Tuning.settle` of the one before, that is, the camera
came to rest somewhere else; then the node asks for a full frame again. A preview
whose R^2 is below :attr:`Tuning.min_r2` (a hand, a person in the way) neither
counts nor breaks a run: it behaves exactly like a :class:`Tick`.

**Robustness.**

* At most one full-frame request is in flight; a request the phone does not answer
  within :attr:`Tuning.full_frame_timeout` counts as failed.
* A failed locate (no board) is retried after a growing, capped delay
  (:attr:`Tuning.backoff`), and the status says ``no-board`` meanwhile.
* A re-frame for drift waits :attr:`Tuning.min_reframe_interval` after the previous
  request; a fresh start (registration, auto switched on, a new lens or resolution)
  does not, and clears the backoff.
* A crop set by hand on /control switches auto-frame off for that camera until it
  is switched on again; results that arrive while it is off are dropped. A camera
  disconnect resets everything except that switch, so a hand-set crop survives a
  reconnect. A new lens or resolution (:class:`Reframed`) or a re-register while a
  request is in flight makes it stale: its answer is dropped and a new one follows.

The locate results are plain dataclasses here (:class:`Located`,
:class:`FrameProposal`), because ``control`` imports nothing from ``recognition``
(.claude/rules/code-organization.md); the package root converts ``locate``'s
``Grid`` and ``Proposal`` into them. The read is generic (``R``): this module only
hands it back in :class:`PublishRead`. Times are the node's monotonic seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum, StrEnum
from itertools import pairwise

from board_reader_core.control import CropRect

# --- tuning ---------------------------------------------------------------------


@dataclass(frozen=True)
class Tuning:
    """Auto-frame's thresholds. The defaults are placeholders from the auto-crop
    probe (docs/reports/2026-11-25-autocrop-probe.md: corners jitter 1-2 % of the
    board's width while the camera is still); retune them from the drift
    measurement."""

    #: The smallest R^2 of a preview's grid that is trusted. Mirrors
    #: ``recognition.locate.MIN_R2``; the root passes that value.
    min_r2: float = 0.90
    #: A preview is off the anchor when it is farther than this share of the
    #: anchor's width (:func:`displacement`).
    drift: float = 0.06
    #: Two off-anchor previews in a row within this share of the board's width of
    #: each other: the camera has stopped moving.
    settle: float = 0.03
    #: How many off-anchor, settled previews in a row mean the camera moved.
    drift_previews: int = 2
    #: A re-frame for drift waits at least this long after the previous request (s).
    min_reframe_interval: float = 10.0
    #: Retry delays after the 1st, 2nd, 3rd ... failed locate in a row (s); the
    #: last one repeats.
    backoff: tuple[float, ...] = (3.0, 10.0, 30.0)
    #: A full-frame request unanswered for this long has failed (s).
    full_frame_timeout: float = 15.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_r2 <= 1:
            raise ValueError("min_r2 must lie in [0, 1]")
        if self.drift <= 0 or self.settle <= 0:
            raise ValueError("drift and settle must be positive")
        if self.drift_previews < 1:
            raise ValueError("drift_previews must be at least 1")
        if self.min_reframe_interval < 0 or self.full_frame_timeout <= 0:
            raise ValueError("min_reframe_interval must be >= 0, full_frame_timeout > 0")
        delays = self.backoff
        if not delays or delays[0] <= 0 or any(a > b for a, b in pairwise(delays)):
            raise ValueError("backoff must be positive, non-empty and non-decreasing")

    def retry_delay(self, failures: int) -> float:
        """The wait after ``failures`` (>= 1) failed locates in a row."""
        return self.backoff[min(failures, len(self.backoff)) - 1]


TUNING = Tuning()

# --- what locate found ----------------------------------------------------------

#: A point in fractions of the whole camera frame: x of its width, y of its height.
Point = tuple[float, float]
#: The board's outer corners a8, h8, h1, a1 (the order does not matter here).
Quad = tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class Located:
    """The board in a whole camera frame: its corners and its grid fit's R^2."""

    corners: Quad
    r2: float


@dataclass(frozen=True)
class FrameProposal[R]:
    """``autoframe`` found the board in a full frame: the crop to push (already
    snapped and valid), the board it was built from, and the last read when it saw
    the board large enough to publish (``Proposal.dominant``), else ``None``."""

    crop: CropRect
    board: Located
    read: R | None


def displacement(anchor: Located, board: Located, aspect: float) -> float:
    """How far ``board`` lies from ``anchor``, in widths of ``anchor``: the symmetric
    Hausdorff distance between the two corner sets in pixels, over the anchor's
    bounding-box width in pixels. ``aspect`` is the frame's width over its height."""

    def scaled(quad: Quad) -> list[Point]:
        return [(x * aspect, y) for x, y in quad]

    a, b = scaled(anchor.corners), scaled(board.corners)

    def farthest(src: list[Point], dst: list[Point]) -> float:
        return max(min(math.dist(p, q) for q in dst) for p in src)

    xs = [x for x, _ in a]
    width = max(max(xs) - min(xs), 1e-9)
    return max(farthest(a, b), farthest(b, a)) / width


# --- events ---------------------------------------------------------------------


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

# --- actions --------------------------------------------------------------------


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

# --- state ----------------------------------------------------------------------


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


def status(state: AutoFrame) -> Status:
    """What /control shows for ``state``."""
    if not state.auto:
        return Status.MANUAL
    if not state.connected:
        return Status.IDLE
    if state.inflight is not None:
        return Status.LOCATING
    if state.backoff is not None:
        return Status.NO_BOARD
    if state.want is not None:
        return Status.LOCATING
    if state.anchor is not None:
        return Status.FRAMED
    return Status.IDLE


# --- the step -------------------------------------------------------------------


def step[R](
    state: AutoFrame, event: Event[R], now: float, tuning: Tuning = TUNING
) -> tuple[AutoFrame, list[Action[R]]]:
    """Apply ``event`` at ``now`` (monotonic seconds): the next state and what the
    node must do. A timed-out request expires first; a due request fires last."""
    before = status(state)
    state = _expire(state, now, tuning)
    state, actions = _apply(state, event, now, tuning)
    state, fired = _fire(state, now, tuning)
    after = status(state)
    report: list[Action[R]] = [ReportStatus(after)] if after != before else []
    return state, [*actions, *fired, *report]


def _apply[R](
    state: AutoFrame, event: Event[R], now: float, tuning: Tuning
) -> tuple[AutoFrame, list[Action[R]]]:
    match event:
        case CameraRegistered():
            fresh = replace(state, connected=True)
            return (_fresh(fresh) if state.auto else fresh), []
        case StreamingStarted():
            idle = state.anchor is None and state.inflight is None
            return (_fresh(state) if state.auto and state.connected and idle else state), []
        case FullFrameLocated(request=request, proposal=proposal, capture_ts=ts):
            answered = _answered(state, request)
            if answered is None:
                return state, []
            if not _accepts(answered):
                return answered, []
            framed = replace(answered, backoff=None, anchor=proposal.board, drifting=None)
            actions: list[Action[R]] = [PushCrop(proposal.crop)]
            if proposal.read is not None:
                actions.append(PublishRead(proposal.read, ts))
            return framed, actions
        case FullFrameFailed(request=request):
            answered = _answered(state, request)
            if answered is None:
                return state, []
            if not _accepts(answered):
                return answered, []
            assert state.inflight is not None  # _answered matched it
            return _failed(answered, state.inflight.why, now, tuning), []
        case PreviewLocated(board=board, aspect=aspect):
            return _preview(state, board, aspect, tuning), []
        case ManualCropSet() | AutoToggled(on=False):
            return _manual(state), []
        case AutoToggled(on=True):
            return (state if state.auto else _fresh(replace(state, auto=True))), []
        case Reframed():
            return (_fresh(state) if state.auto else state), []
        case Disconnected():
            return AutoFrame(auto=state.auto, next_request=state.next_request), []
        case Tick():
            return state, []


def _fresh(state: AutoFrame) -> AutoFrame:
    """Want a full frame from scratch: the frame or the intent is new."""
    return replace(state, want=Why.FRESH, backoff=None, anchor=None, drifting=None)


def _manual(state: AutoFrame) -> AutoFrame:
    """Auto-frame off; an in-flight request stays in flight (the phone still
    answers it) and its answer is dropped."""
    return AutoFrame(
        auto=False,
        connected=state.connected,
        inflight=state.inflight,
        last_request=state.last_request,
        next_request=state.next_request,
    )


def _answered(state: AutoFrame, request: int) -> AutoFrame | None:
    """``state`` with ``request`` answered, or ``None`` when it is not the one in
    flight (late, timed out, or from before a disconnect)."""
    if state.inflight is None or state.inflight.request != request:
        return None
    return replace(state, inflight=None)


def _accepts(answered: AutoFrame) -> bool:
    """Whether the answer just taken off ``answered`` still counts: auto-frame is on
    and nothing new was wanted meanwhile. A newer want (a new lens, a re-register)
    makes the answer stale; the want stays, so a new request follows."""
    return answered.auto and answered.connected and answered.want is None


def _failed(state: AutoFrame, why: Why, now: float, tuning: Tuning) -> AutoFrame:
    failures = 1 if state.backoff is None else state.backoff.failures + 1
    return replace(
        state, want=why, backoff=Backoff(failures, now + tuning.retry_delay(failures))
    )


def _expire(state: AutoFrame, now: float, tuning: Tuning) -> AutoFrame:
    """Give up on a request the phone has not answered in time."""
    inflight = state.inflight
    if inflight is None or now < inflight.since + tuning.full_frame_timeout:
        return state
    cleared = replace(state, inflight=None)
    if not _accepts(cleared):
        return cleared
    return _failed(cleared, inflight.why, now, tuning)


def _preview(
    state: AutoFrame, board: Located | None, aspect: float, tuning: Tuning
) -> AutoFrame:
    anchor = state.anchor
    watching = (
        state.auto
        and state.connected
        and anchor is not None
        and state.inflight is None
        and state.want is None
        and state.backoff is None
    )
    if not watching or board is None or board.r2 < tuning.min_r2:
        return state
    assert anchor is not None  # watching
    if displacement(anchor, board, aspect) <= tuning.drift:
        return replace(state, drifting=None)
    previous = state.drifting
    settled = (
        previous is not None and displacement(previous.last, board, aspect) <= tuning.settle
    )
    run = previous.run + 1 if settled and previous is not None else 1
    if run >= tuning.drift_previews:
        return replace(state, want=Why.DRIFT, drifting=None)
    return replace(state, drifting=Drifting(board, run))


def _fire[R](
    state: AutoFrame, now: float, tuning: Tuning
) -> tuple[AutoFrame, list[Action[R]]]:
    """Send the wanted request once nothing blocks it."""
    why = state.want
    if why is None or not (state.auto and state.connected) or state.inflight is not None:
        return state, []
    if state.backoff is not None and now < state.backoff.until:
        return state, []
    limited = state.last_request is not None and (
        now < state.last_request + tuning.min_reframe_interval
    )
    if why is Why.DRIFT and limited:
        return state, []
    request = state.next_request
    sent = replace(
        state,
        want=None,
        inflight=InFlight(request, why, now),
        last_request=now,
        next_request=request + 1,
    )
    return sent, [RequestFullFrame(request, why)]
