"""The orientation guard: a read whose board points the wrong way is dropped.

ChessQ reports the e1 -> e8 direction with every read, and on every recorded take
it held within 2 degrees. A read more than 45 degrees off the board's running
angle is the model reading the board turned, so the stream drops it before the
repair and the debounce see it, counts it, and logs it.
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
import pytest

from board_reader_core.protocol import START_PLACEMENT, dist_from_placement
from board_reader_core.wire import Frame

from board_reader_inference.hub import PositionHub
from board_reader_inference.ingest import (
    MAX_TURN_DEG,
    REANCHOR_AFTER,
    FrameIngest,
    Misoriented,
    OrientationGuard,
)
from board_reader_inference.recognizer import (
    FrameReading,
    Recognition,
    StubRecognizer,
    event_from_dist,
)
from board_reader_inference.server import create_app

E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"


def _feed(guard: OrientationGuard, directions: list[float | None], first_seq: int = 0) -> list[bool]:
    """Which of ``directions`` (one per consecutive seq) the guard publishes."""
    return [guard.check(first_seq + i, d) is None for i, d in enumerate(directions)]


def test_a_quarter_turn_is_dropped_and_does_not_move_the_running_angle():
    guard = OrientationGuard()
    assert _feed(guard, [89.0, 90.0, 91.0]) == [True, True, True]
    dropped = guard.check(3, 0.0)  # read a quarter turn off: White at the bottom
    assert dropped == Misoriented(seq=3, direction=0.0, running=90.0)
    assert guard.running == 90.0
    assert guard.check(4, 180.0) is not None  # the other quarter turn
    assert guard.check(5, 92.0) is None


def test_the_limit_is_45_degrees_either_way():
    guard = OrientationGuard()
    _feed(guard, [90.0] * 5)
    assert guard.check(5, 90.0 + MAX_TURN_DEG) is None
    assert guard.running == 90.0
    assert guard.check(6, 90.0 - MAX_TURN_DEG - 0.5) is not None


def test_the_running_angle_is_a_median_and_wraps_at_180():
    guard = OrientationGuard()
    _feed(guard, [179.0, -179.0, 178.0])  # one direction across the seam
    assert guard.running == pytest.approx(179.0)
    assert guard.check(3, -170.0) is None  # 11 degrees away, not 349
    assert guard.check(4, 0.0) is not None  # upside down


def test_a_read_without_a_direction_passes_and_leaves_the_angle_alone():
    guard = OrientationGuard()
    assert _feed(guard, [None, 90.0, None]) == [True, True, True]
    assert guard.running == 90.0


def test_a_seq_restart_starts_afresh():
    guard = OrientationGuard()
    _feed(guard, [90.0, 90.0], first_seq=10)
    assert guard.check(12, 0.0) is not None
    # A new streaming session (seq does not advance): the old angle is forgotten.
    assert guard.check(0, 0.0) is None
    assert guard.running == 0.0


def test_a_camera_that_really_turned_is_adopted_after_a_run_of_drops(caplog):
    guard = OrientationGuard()
    _feed(guard, [90.0] * 5)
    turned = _feed(guard, [0.0] * (REANCHOR_AFTER + 1), first_seq=5)
    assert turned == [False] * (REANCHOR_AFTER - 1) + [True, True]
    assert guard.running == 0.0
    assert "re-anchored" in caplog.text


def test_one_good_read_ends_a_run_of_drops():
    guard = OrientationGuard()
    _feed(guard, [90.0] * 3)
    pattern = [0.0] * (REANCHOR_AFTER - 1) + [90.0] + [0.0]
    assert _feed(guard, pattern, first_seq=3) == [False] * (REANCHOR_AFTER - 1) + [True, False]


def _scripted(directions: dict[int, float]):
    """A FrameRead reading E4 at every seq, pointing ``directions[seq]``."""

    def read(frame: Frame) -> FrameReading:
        return FrameReading(
            event_from_dist(frame, dist_from_placement(E4)), directions[frame.seq]
        )

    return read


def _frame(seq: int) -> Frame:
    return Frame(
        board_id="A1", seq=seq, capture_ts=100.0 + seq, image=np.zeros((8, 8, 3), np.uint8)
    )


def test_the_ingest_worker_drops_counts_and_logs_a_turned_read(caplog):
    published: list[FrameReading] = []

    async def run() -> FrameIngest:
        ingest = FrameIngest(
            _scripted({0: 90.0, 1: -90.0, 2: 91.0}),
            PositionHub(),
            on_event=lambda reading, frame: published.append(reading),
        )
        for seq in range(3):
            ingest.submit(_frame(seq))
            await asyncio.wait_for(ingest.drain(), timeout=2.0)
        await ingest.close()
        return ingest

    with caplog.at_level(logging.WARNING, logger="board_reader_inference.ingest"):
        ingest = asyncio.run(run())
    assert [r.event.seq for r in published] == [0, 2]
    assert (ingest.frames_estimated, ingest.frames_misoriented) == (2, 1)
    assert "seq 1: dropped a read pointing -90 degrees" in caplog.text


class _Pointing(StubRecognizer):
    """The stub's board, read pointing ``direction`` degrees."""

    def __init__(self, direction: float) -> None:
        super().__init__(START_PLACEMENT)
        self.direction = direction

    def read(self, image: np.ndarray) -> Recognition:
        return Recognition(super().read(image).square_dist, self.direction)


def _send(ws, seq: int) -> None:
    header, jpeg = _frame(seq).encode()
    ws.send_text(json.dumps(header))
    ws.send_bytes(jpeg)
    ws.receive_json()


def test_a_live_reads_direction_reaches_the_camera_registry(gateway, drain_ingest, monkeypatch):
    recognizer = _Pointing(88.0)
    client = gateway(create_app(recognizer, PositionHub()))
    seen: list[tuple[str, float, float]] = []
    monkeypatch.setattr(
        client.app.state.cameras,
        "observe_direction",
        lambda board_id, degrees, ts: seen.append((board_id, degrees, ts)),
    )
    with client.websocket_connect("/ingest") as ws:
        _send(ws, 0)
        drain_ingest(client)
        recognizer.direction = -2.0  # read a quarter turn off: dropped, never observed
        _send(ws, 1)
        drain_ingest(client)
    assert seen == [("A1", 88.0, 100.0)]
    assert client.get("/healthz").json()["ingest"]["misoriented"] == 1
