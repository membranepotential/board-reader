"""Auto-frame's decisions (control/autoframe/machine.py): scripted scenarios, then a seeded
property test over random event streams that also proves it can fail."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from enum import Enum, auto

import pytest

from board_reader_core.control import CropRect

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
from board_reader_inference.control.autoframe.geometry import FrameProposal, Located, Quad
from board_reader_inference.control.autoframe.machine import status, step
from board_reader_inference.control.autoframe.state import AutoFrame
from board_reader_inference.control.autoframe.tuning import TUNING

ASPECT = 16 / 9
#: A board 0.3 of the frame wide; 0.01 of the frame is 3.3 % of its width.
HOME = Located(((0.30, 0.30), (0.60, 0.30), (0.60, 0.80), (0.30, 0.80)), 0.97)
CROP = CropRect(0.27, 0.25, 0.36, 0.6)


def moved(board: Located, dx: float, dy: float = 0.0, r2: float | None = None) -> Located:
    corners = tuple((x + dx, y + dy) for x, y in board.corners)
    assert len(corners) == 4
    quad: Quad = (corners[0], corners[1], corners[2], corners[3])
    return Located(quad, board.r2 if r2 is None else r2)


def proposal(board: Located = HOME, read: str | None = "read") -> FrameProposal[str]:
    return FrameProposal(CROP, board, read)


Step = Callable[[AutoFrame, Event[str], float], tuple[AutoFrame, list[Action[str]]]]


def real_step(
    state: AutoFrame, event: Event[str], now: float
) -> tuple[AutoFrame, list[Action[str]]]:
    return step(state, event, now)


def run(
    script: Sequence[tuple[float, Event[str]]], state: AutoFrame | None = None
) -> tuple[AutoFrame, list[tuple[float, Action[str]]]]:
    state = AutoFrame() if state is None else state
    out: list[tuple[float, Action[str]]] = []
    for now, event in script:
        state, actions = step(state, event, now)
        out.extend((now, a) for a in actions)
    return state, out


def requests(out: list[tuple[float, Action[str]]]) -> list[tuple[float, RequestFullFrame]]:
    return [(t, a) for t, a in out if isinstance(a, RequestFullFrame)]


#: Registered at 0 and framed on HOME at 1 (request 0).
FRAMED: list[tuple[float, Event[str]]] = [
    (0.0, CameraRegistered()),
    (1.0, FullFrameLocated(0, proposal(), capture_ts=0.5)),
]


def previews(start: float, *boards: Located | None) -> list[tuple[float, Event[str]]]:
    return [(start + 1.5 * i, PreviewLocated(b, ASPECT)) for i, b in enumerate(boards)]


# --- framing --------------------------------------------------------------------


def test_registration_asks_for_one_full_frame_and_frames_the_board():
    state, out = run(FRAMED)
    assert [a for _, a in out] == [
        RequestFullFrame(0, Why.FRESH),
        ReportStatus(Status.LOCATING),
        PushCrop(CROP),
        PublishRead("read", 0.5),
        ReportStatus(Status.FRAMED),
    ]
    assert status(state) is Status.FRAMED


def test_a_read_that_is_not_board_dominant_is_not_published():
    _, out = run(
        [(0.0, CameraRegistered()), (1.0, FullFrameLocated(0, proposal(read=None), 0.5))]
    )
    assert not any(isinstance(a, PublishRead) for _, a in out)
    assert PushCrop(CROP) in [a for _, a in out]


def test_streaming_start_and_a_re_register_never_send_a_second_request_in_flight():
    _, out = run(
        [(0.0, CameraRegistered()), (0.5, StreamingStarted()), (1.0, CameraRegistered())]
    )
    assert [r for _, r in requests(out)] == [RequestFullFrame(0, Why.FRESH)]


def test_a_new_lens_drops_the_in_flight_answer_and_asks_again():
    state, out = run(
        [
            (0.0, CameraRegistered()),
            (1.0, Reframed()),
            (2.0, FullFrameLocated(0, proposal(), 0.5)),
        ]
    )
    assert not any(isinstance(a, PushCrop) for _, a in out)
    assert requests(out) == [
        (0.0, RequestFullFrame(0, Why.FRESH)),
        (2.0, RequestFullFrame(1, Why.FRESH)),
    ]
    assert status(state) is Status.LOCATING


# --- drift ----------------------------------------------------------------------

AWAY = moved(HOME, 0.03)  # 10 % of the board's width


@pytest.mark.parametrize(
    ("boards", "reframes"),
    [
        pytest.param([AWAY, moved(AWAY, 0.001)], True, id="two settled previews away"),
        pytest.param([AWAY], False, id="one preview away"),
        pytest.param([moved(HOME, 0.005), moved(HOME, -0.005)], False, id="still jitter"),
        pytest.param(
            [moved(AWAY, 0, r2=0.8), moved(AWAY, 0, r2=0.8), moved(AWAY, 0, r2=0.8)],
            False,
            id="untrusted previews",
        ),
        pytest.param([None, None, None], False, id="no board in the previews"),
        pytest.param(
            [AWAY, moved(AWAY, 0, r2=0.5), None, moved(AWAY, 0.001)],
            True,
            id="a hand between two previews away",
        ),
        pytest.param([AWAY, HOME, AWAY], False, id="back home breaks the run"),
        pytest.param([AWAY, moved(AWAY, 0.03)], False, id="camera still moving"),
    ],
)
def test_drift_needs_two_trusted_settled_previews_away_from_the_crop(boards, reframes):
    _, out = run([*FRAMED, *previews(20.0, *boards), (40.0, Tick())])
    assert [r.why for _, r in requests(out)][1:] == ([Why.DRIFT] if reframes else [])


def test_a_drift_re_frame_waits_for_the_rate_limit():
    # Framed by request 0 at t=0; the camera moves at once.
    _, out = run([*FRAMED, *previews(2.0, AWAY, AWAY), (6.0, Tick()), (10.0, Tick())])
    assert requests(out)[1:] == [(10.0, RequestFullFrame(1, Why.DRIFT))]
    assert ReportStatus(Status.LOCATING) in [a for t, a in out if t == 3.5]


def test_a_drift_re_frame_pushes_the_new_crop_and_anchors_there():
    new_crop = CropRect(0.3, 0.25, 0.36, 0.6)
    script = [
        *FRAMED,
        *previews(20.0, AWAY, AWAY),
        (22.0, FullFrameLocated(1, FrameProposal(new_crop, AWAY, None), 21.6)),
        *previews(23.0, AWAY, moved(AWAY, 0.001)),
    ]
    state, out = run(script)
    assert PushCrop(new_crop) in [a for _, a in out]
    assert state.anchor == AWAY
    assert len(requests(out)) == 2


# --- failures -------------------------------------------------------------------


def test_no_board_retries_with_a_growing_capped_backoff():
    """Each locate fails 1 s after its request; ticks every 0.5 s find the retry."""
    now = 0.0
    state, actions = step(AutoFrame(), CameraRegistered(), now)
    failed_at: list[float] = []
    sent_at: list[float] = []
    for _ in range(5):
        request = next(a for a in actions if isinstance(a, RequestFullFrame))
        sent_at.append(now)
        now += 1.0
        state, actions = step(state, FullFrameFailed(request.request), now)
        assert ReportStatus(Status.NO_BOARD) in actions
        failed_at.append(now)
        while not any(isinstance(a, RequestFullFrame) for a in actions):
            now += 0.5
            state, actions = step(state, Tick(), now)
    gaps = [sent - failed for failed, sent in zip(failed_at, sent_at[1:])]
    assert gaps == [3.0, 10.0, 30.0, 30.0]


def test_an_unanswered_request_times_out_into_the_backoff():
    timeout = TUNING.full_frame_timeout
    state, out = run(
        [
            (0.0, CameraRegistered()),
            (timeout, Tick()),
            (timeout + 3.0, Tick()),
            (timeout + 4.0, FullFrameLocated(0, proposal(), 0.1)),  # too late
        ]
    )
    assert requests(out) == [
        (0.0, RequestFullFrame(0, Why.FRESH)),
        (timeout + 3.0, RequestFullFrame(1, Why.FRESH)),
    ]
    assert not any(isinstance(a, PushCrop) for _, a in out)


# --- manual and disconnect ------------------------------------------------------


def test_a_manual_crop_switches_auto_frame_off_until_it_is_switched_on():
    state, out = run(
        [
            (0.0, CameraRegistered()),
            (0.5, ManualCropSet()),
            (1.0, FullFrameLocated(0, proposal(), 0.5)),
            *previews(2.0, AWAY, AWAY),
            (30.0, Reframed()),
            (31.0, Tick()),
        ]
    )
    assert status(state) is Status.MANUAL
    assert not any(isinstance(a, PushCrop | PublishRead) for _, a in out)
    assert len(requests(out)) == 1
    state, out = run([(40.0, AutoToggled(on=True))], state)
    assert [a for _, a in out] == [
        RequestFullFrame(1, Why.FRESH),
        ReportStatus(Status.LOCATING),
    ]


def test_disconnect_resets_everything_but_the_manual_switch():
    state, _ = run([*FRAMED, *previews(20.0, AWAY), (30.0, Disconnected())])
    assert state == AutoFrame(next_request=1)
    assert status(state) is Status.IDLE
    manual, _ = run(
        [(0.0, CameraRegistered()), (1.0, ManualCropSet()), (2.0, Disconnected())]
    )
    assert manual.auto is False
    _, out = run([(3.0, CameraRegistered())], manual)
    assert requests(out) == []


def test_an_answer_from_before_a_disconnect_matches_nothing_after_it():
    state, out = run(
        [
            (0.0, CameraRegistered()),
            (1.0, Disconnected()),
            (2.0, CameraRegistered()),
            (3.0, FullFrameLocated(0, proposal(), 0.5)),
        ]
    )
    assert [r for _, r in requests(out)] == [
        RequestFullFrame(0, Why.FRESH),
        RequestFullFrame(1, Why.FRESH),
    ]
    assert not any(isinstance(a, PushCrop) for _, a in out)
    assert status(state) is Status.LOCATING


# --- displacement and tuning ----------------------------------------------------


# --- property -------------------------------------------------------------------


class Broken(Enum):
    """An invariant a run of events broke."""

    TWO_IN_FLIGHT = auto()
    CROP_WHILE_MANUAL = auto()
    RATE_LIMIT = auto()
    UNTRUSTED_TRIGGER = auto()
    BACKOFF = auto()


class Path(Enum):
    """A code path a run took (the property test must take all of them)."""

    REQUEST_FRESH = auto()
    REQUEST_DRIFT = auto()
    PUSH = auto()
    PUBLISH = auto()
    UNTRUSTED_PREVIEW = auto()
    LATE_ANSWER = auto()
    TIMEOUT = auto()
    BACKOFF_CAPPED = auto()


#: Where the camera may sit: home, a bit off (jitter), moved, and pointed away
#: from the board (``None``: every locate fails).
POSES = (HOME, moved(HOME, 0.004), moved(HOME, 0.05), moved(HOME, -0.04, 0.03), None)


def _event(
    rng: random.Random, churn: float, pose: Located | None, open_request: int | None
) -> Event[str]:
    """One event; ``churn`` scales how often the rare ones (connects, switches,
    lens changes) happen, so calm streams reach long runs of failures."""
    roll = rng.random() / churn
    if roll < 0.03:
        return CameraRegistered()
    if roll < 0.05:
        return Disconnected()
    if roll < 0.07:
        return StreamingStarted()
    if roll < 0.09:
        return ManualCropSet()
    if roll < 0.12:
        return AutoToggled(on=rng.random() < 0.8)
    if roll < 0.14:
        return Reframed()
    roll = rng.random()
    if roll < (0.30 if open_request is not None else 0.03):
        known = open_request is not None and rng.random() < 0.9
        request = open_request if known and open_request is not None else rng.randrange(40)
        if pose is not None and rng.random() < 0.6:
            read = "read" if rng.random() < 0.5 else None
            return FullFrameLocated(request, proposal(pose, read), 0.0)
        return FullFrameFailed(request)
    if roll < 0.80:
        if pose is None or rng.random() < 0.1:
            return PreviewLocated(None, ASPECT)
        r2 = 0.97 if rng.random() < 0.7 else rng.uniform(0.3, 0.89)
        jitter = rng.uniform(-0.002, 0.002)
        return PreviewLocated(moved(pose, jitter, jitter, r2=r2), ASPECT)
    return Tick()


def _check(step_fn: Step, seed: int, events: int = 400) -> tuple[set[Broken], set[Path]]:
    """Run one seeded stream through ``step_fn``: the invariants it broke (judged
    against the shipped :data:`TUNING`) and the paths it took."""
    rng = random.Random(seed)
    broken: set[Broken] = set()
    paths: set[Path] = set()
    state = AutoFrame()
    now = 0.0
    churn = 1.0 if seed % 2 else 0.03
    pose: Located | None = rng.choice(POSES)
    auto_on = True
    open_request: tuple[int, float] | None = None  # (request, sent at)
    last_request: float | None = None
    last_delay = 0.0
    for i in range(events):
        now += rng.uniform(0.2, 2.0)
        if rng.random() < 0.06 * churn:
            pose = rng.choice(POSES)
        if open_request is not None and now >= open_request[1] + TUNING.full_frame_timeout:
            open_request = None
            paths.add(Path.TIMEOUT)
        event: Event[str] = (
            _event(rng, churn, pose, None if open_request is None else open_request[0])
            if i
            else CameraRegistered()
        )
        match event:
            case ManualCropSet() | AutoToggled(on=False):
                auto_on = False
            case AutoToggled(on=True):
                auto_on = True
            case Disconnected():
                open_request, last_request = None, None
            case FullFrameLocated(request=r) | FullFrameFailed(request=r):
                if open_request is not None and r == open_request[0]:
                    open_request = None
                else:
                    paths.add(Path.LATE_ANSWER)
            case PreviewLocated(board=board) if board is None or board.r2 < TUNING.min_r2:
                paths.add(Path.UNTRUSTED_PREVIEW)
                if step_fn(state, event, now) != step_fn(state, Tick(), now):
                    broken.add(Broken.UNTRUSTED_TRIGGER)
            case _:
                pass
        before = state.backoff
        state, actions = step_fn(state, event, now)
        for action in actions:
            match action:
                case RequestFullFrame(request=r, why=why):
                    if open_request is not None:
                        broken.add(Broken.TWO_IN_FLIGHT)
                    if why is Why.DRIFT:
                        paths.add(Path.REQUEST_DRIFT)
                        if (
                            last_request is not None
                            and now < last_request + TUNING.min_reframe_interval
                        ):
                            broken.add(Broken.RATE_LIMIT)
                    else:
                        paths.add(Path.REQUEST_FRESH)
                    open_request, last_request = (r, now), now
                case PushCrop():
                    paths.add(Path.PUSH)
                    if not auto_on:
                        broken.add(Broken.CROP_WHILE_MANUAL)
                case PublishRead():
                    paths.add(Path.PUBLISH)
                case _:
                    pass
        if state.backoff is None:
            last_delay = 0.0
        elif state.backoff != before:
            delay = round(state.backoff.until - now, 9)
            if delay < last_delay or delay not in TUNING.backoff:
                broken.add(Broken.BACKOFF)
            if delay == TUNING.backoff[-1]:
                paths.add(Path.BACKOFF_CAPPED)
            last_delay = delay
    return broken, paths


SEEDS = range(40)


def test_random_event_streams_keep_every_invariant():
    taken: set[Path] = set()
    for seed in SEEDS:
        broken, paths = _check(real_step, seed)
        assert broken == set(), f"seed {seed}"
        taken |= paths
    assert taken == set(Path)


def _tuned(**changes: object) -> Step:
    tuning = replace(TUNING, **changes)
    return lambda s, e, t: step(s, e, t, tuning)


@pytest.mark.parametrize(
    ("sabotaged", "invariant"),
    [
        pytest.param(
            lambda s, e, t: step(replace(s, inflight=None), e, t),
            Broken.TWO_IN_FLIGHT,
            id="forgets the request in flight",
        ),
        pytest.param(
            lambda s, e, t: step(replace(s, auto=True), e, t),
            Broken.CROP_WHILE_MANUAL,
            id="ignores the manual switch",
        ),
        pytest.param(
            _tuned(min_reframe_interval=0.0), Broken.RATE_LIMIT, id="no rate limit"
        ),
        pytest.param(
            _tuned(min_r2=0.0), Broken.UNTRUSTED_TRIGGER, id="trusts every preview"
        ),
        pytest.param(_tuned(backoff=(3.0, 10.0, 30.0, 60.0)), Broken.BACKOFF, id="no cap"),
    ],
)
def test_the_property_test_catches_a_broken_step(sabotaged: Step, invariant: Broken):
    assert any(invariant in _check(sabotaged, seed)[0] for seed in SEEDS)
