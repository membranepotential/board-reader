"""Motion updates on ``WS /viewer``: one per ingested frame, throttled, never late.

``FrameIngest.submit`` turns each arriving frame's ``stability`` into a
``{"type": "motion"}`` message on the hub. :class:`MotionThrottle` keeps a board
to about five a second but always lets a change of ``moving`` through, so a hand
reaching for a piece shows at once however many frames came before it.
"""

from __future__ import annotations

import asyncio
import json
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from board_reader_core.protocol import MotionUpdate
from board_reader_core.wire import Frame

from board_reader_inference.recognition.recognizer import StubRecognizer, frame_reader
from board_reader_inference.server import create_app
from board_reader_inference.stream import ingest as ingest_mod
from board_reader_inference.stream.hub import PositionHub
from board_reader_inference.stream.ingest import (
    MOTION_INTERVAL_S,
    FrameIngest,
    MotionThrottle,
)
from board_reader_inference.stream.schedule.constants import ENTER_BURST


def test_the_first_frame_and_every_change_of_moving_go_out_at_once():
    throttle = MotionThrottle()
    assert throttle.due(False, now=10.0)
    assert not throttle.due(False, now=10.05)  # same state, inside the interval
    assert throttle.due(True, now=10.06)  # the hand arrives: never held back
    assert throttle.due(False, now=10.07)  # and leaves
    assert not throttle.due(False, now=10.08)


def test_a_steady_state_is_sent_at_most_once_per_interval():
    throttle = MotionThrottle()
    ticks = np.arange(0.0, 1.0, 1 / 30)  # a 30 fps edge, one second, at rest
    sent = [t for t in ticks if throttle.due(False, now=float(t))]
    assert len(sent) == round(1.0 / MOTION_INTERVAL_S)
    assert all(b - a >= MOTION_INTERVAL_S - 1e-6 for a, b in pairwise(sent))


def test_a_change_restarts_the_interval():
    throttle = MotionThrottle()
    throttle.due(False, now=0.0)
    throttle.due(True, now=0.15)
    assert not throttle.due(True, now=0.25)  # 0.25 s after the first, 0.1 after the change
    assert throttle.due(True, now=0.35)


def _frame(seq: int, stability: float | None) -> Frame:
    return Frame(
        board_id="A1",
        seq=seq,
        capture_ts=1000.0 + seq / 10,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        stability=stability,
    )


def test_submit_sends_the_moving_verdict_with_the_schedulers_threshold(monkeypatch):
    """The verdict is ``stability < ENTER_BURST`` (the UI's MOVING_BELOW); a frame
    with no usable stability counts as at rest, as the scheduler counts it."""
    # A frozen node clock, so only changes of `moving` pass the throttle.
    monkeypatch.setattr(
        ingest_mod, "time", SimpleNamespace(monotonic=lambda: 50.0, time=lambda: 50.0)
    )
    stabilities = [1.0, 0.95, ENTER_BURST - 0.01, 0.2, ENTER_BURST, float("nan"), None]

    async def run() -> list[MotionUpdate]:
        hub = PositionHub()
        updates = hub.subscribe(replay_latest=False)
        first = asyncio.ensure_future(anext(updates))
        await asyncio.sleep(0)  # let the subscription register
        ingest = FrameIngest(frame_reader(StubRecognizer()), hub)
        for seq, stability in enumerate(stabilities):
            ingest.submit(_frame(seq, stability))
        await ingest.drain()
        await ingest.close()
        got = [await first]
        while True:
            try:
                got.append(await asyncio.wait_for(anext(updates), timeout=0.05))
            except TimeoutError:
                break
        return [m for m in got if isinstance(m, MotionUpdate)]

    updates = asyncio.run(run())
    assert [(round(m.ts, 6), m.moving) for m in updates] == [
        (1000.0, False),  # first frame
        (1000.2, True),  # just under the threshold: moving
        (1000.4, False),  # exactly at the threshold: at rest again
    ]
    assert updates[1].stability == pytest.approx(ENTER_BURST - 0.01)


def test_viewer_receives_the_motion_message_in_the_published_schema(gateway):
    """Through the real sockets: the exact keys the UI is built against."""
    client = gateway(create_app(StubRecognizer(), PositionHub()))
    frame = Frame(
        board_id="A1",
        seq=4,
        capture_ts=1_700_000_000.5,
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        stability=0.25,
    )
    header, jpeg = frame.encode()
    with (
        client.websocket_connect("/viewer") as viewer,
        client.websocket_connect("/ingest") as ingest,
    ):
        ingest.send_text(json.dumps(header))
        ingest.send_bytes(jpeg)
        ingest.receive_json()  # ack: submit, and so the broadcast, happened first
        motion = json.loads(viewer.receive_text())
        event = json.loads(viewer.receive_text())
    assert motion == {
        "type": "motion",
        "board_id": "A1",
        "moving": True,
        "stability": 0.25,
        "ts": 1_700_000_000.5,
    }
    # The position event keeps its shape: no `type`.
    assert "type" not in event and event["seq"] == 4
