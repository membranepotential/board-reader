"""Auto-frame's decisions for one board's camera: when to look for the board, when to
push a crop, when the camera moved. Pure: events in, actions out, no I/O.

The node frames each camera by itself. When a camera registers (or starts streaming
without a crop of ours), it asks the phone for **one zoomed-out full frame**,
ChessQ locates the board in it (``recognition/locate.py``'s ``autoframe``), and the
node pushes the proposed crop. When the locate's last read already saw the board
large enough (``Proposal.dominant``), that read goes into the position stream as an
ordinary read, stamped with the full frame's ``capture_ts``.

**Drift.** After a crop is pushed, the phone's whole-frame previews (480 px, every
1.5 s) are located too (``locate_preview``), at most one per
:attr:`Tuning.preview_every` (:func:`wants_preview`), and only while the board is
large enough in the frame to be found on a preview (:func:`watching`). A preview
counts when its grid's R^2 is at least :attr:`Tuning.preview_min_r2`; its board is
then compared with the board the crop was built from (the **anchor**, the
full-frame locate's corners): the displacement is the symmetric Hausdorff distance
between the two corner sets, in pixels, over the anchor's width in pixels
(:func:`displacement`). It ignores corner labels, because a read that turns the
board by a quarter relabels the same four points. The camera has moved when
:attr:`Tuning.drift_previews` counted previews in a row are each farther than
:attr:`Tuning.drift` from the anchor; then the node asks for a full frame again. A
preview that does not count (a hand, a person in the way) neither adds to the run
nor breaks it; :attr:`Tuning.unseen_previews` of them in a row say "board not
visible" and re-frame nothing.

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

The read is generic (``R``): this module only hands it back in
:class:`PublishRead`. Times are the node's monotonic seconds. The pieces live in
sibling modules: ``tuning``, ``geometry`` (:class:`Located`, :func:`displacement`),
``events``, ``actions`` and ``state``.
"""

from __future__ import annotations

from dataclasses import replace

from board_reader_inference.control.autoframe.actions import (
    Action,
    PublishRead,
    PushCrop,
    ReportStatus,
    RequestFullFrame,
    Status,
    Why,
)
from board_reader_inference.control.autoframe.events import (
    AutoToggled,
    CameraRegistered,
    Disconnected,
    Event,
    FullFrameFailed,
    FullFrameLocated,
    ManualCropSet,
    PreviewLocated,
    Reframed,
    StreamingStarted,
    Tick,
)
from board_reader_inference.control.autoframe.geometry import Located, displacement
from board_reader_inference.control.autoframe.state import (
    AutoFrame,
    Backoff,
    InFlight,
)
from board_reader_inference.control.autoframe.tuning import TUNING, Tuning


def watching(state: AutoFrame, tuning: Tuning = TUNING) -> bool:
    """Whether the previews are watched for a camera move: framed, nothing asked
    for, and the board wide enough in the frame to be found on a preview."""
    anchor = state.anchor
    return (
        state.auto
        and state.connected
        and anchor is not None
        and state.inflight is None
        and state.want is None
        and state.backoff is None
        and _width(anchor) >= tuning.min_preview_share
    )


def wants_preview(state: AutoFrame, now: float, tuning: Tuning = TUNING) -> bool:
    """Whether a preview arriving at ``now`` is worth locating (one ChessQ read)."""
    last = state.last_preview
    return watching(state, tuning) and (last is None or now - last >= tuning.preview_every)


def _width(board: Located) -> float:
    """The board's width as a share of the frame's."""
    xs = [x for x, _ in board.corners]
    return max(xs) - min(xs)


def status(state: AutoFrame, tuning: Tuning = TUNING) -> Status:
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
        return Status.NOT_VISIBLE if state.unseen >= tuning.unseen_previews else Status.FRAMED
    return Status.IDLE


# --- the step -------------------------------------------------------------------


def step[R](
    state: AutoFrame, event: Event[R], now: float, tuning: Tuning = TUNING
) -> tuple[AutoFrame, list[Action[R]]]:
    """Apply ``event`` at ``now`` (monotonic seconds): the next state and what the
    node must do. A timed-out request expires first; a due request fires last."""
    before = status(state, tuning)
    state = _expire(state, now, tuning)
    state, actions = _apply(state, event, now, tuning)
    state, fired = _fire(state, now, tuning)
    after = status(state, tuning)
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
            framed = replace(
                answered, backoff=None, anchor=proposal.board, drifting=0, unseen=0
            )
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
            return _preview(state, board, aspect, now, tuning), []
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
    return replace(state, want=Why.FRESH, backoff=None, anchor=None, drifting=0, unseen=0)


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
    state: AutoFrame, board: Located | None, aspect: float, now: float, tuning: Tuning
) -> AutoFrame:
    if not watching(state, tuning):
        return state
    anchor = state.anchor
    assert anchor is not None  # watching
    seen = replace(state, last_preview=now)
    if board is None or board.r2 < tuning.preview_min_r2:
        return replace(seen, unseen=min(seen.unseen + 1, tuning.unseen_previews))
    seen = replace(seen, unseen=0)
    if displacement(anchor, board, aspect) <= tuning.drift:
        return replace(seen, drifting=0)
    run = seen.drifting + 1
    if run >= tuning.drift_previews:
        return replace(seen, want=Why.DRIFT, drifting=0)
    return replace(seen, drifting=run)


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
