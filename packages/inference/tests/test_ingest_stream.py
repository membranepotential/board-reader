"""Tests for the streaming frame ingress (DESIGN.md §8 WP-G1/G2).

The estimator is faked with a controllable ``time.sleep`` so "the model is slower
than the stream" is a property of the test, not of the machine: it runs in the
worker thread, exactly where the real ONNX call runs.
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.wire import Frame

from board_reader_inference.hub import PositionHub
from board_reader_inference.ingest import SHARPNESS_WINDOW_S, FrameIngest, prefer_frame
from board_reader_inference.recognizer import FrameReading, StubRecognizer, frame_reader
from board_reader_inference.server import create_app


class SlowRecognizer(StubRecognizer):
    """The stub recognizer, at a controllable cost per read."""

    def __init__(self, delay: float) -> None:
        super().__init__(START_PLACEMENT)
        self.delay = delay

    def read(self, image: np.ndarray) -> np.ndarray:
        time.sleep(self.delay)
        return super().read(image)


class FakeEstimator:
    """A frame read with a controllable cost, recording what it was handed."""

    def __init__(
        self,
        delay: float = 0.0,
        *,
        delays: dict[str, float] | None = None,
        fail_seqs: set[int] | None = None,
    ) -> None:
        self.delay = delay
        self.delays = delays or {}
        self.fail_seqs = fail_seqs or set()
        self.seen: list[tuple[str, int]] = []
        self._stub = frame_reader(StubRecognizer(START_PLACEMENT))

    def estimate(self, frame: Frame) -> FrameReading:
        time.sleep(self.delays.get(frame.board_id, self.delay))
        self.seen.append((frame.board_id, frame.seq))
        if frame.seq in self.fail_seqs:
            raise RuntimeError(f"model choked on seq {frame.seq}")
        return self._stub(frame)

    def seqs(self, board_id: str) -> list[int]:
        return [seq for board, seq in self.seen if board == board_id]


def _frame(board_id: str = "A1", seq: int = 0) -> Frame:
    return Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=float(seq),
        image=np.zeros((100, 100, 3), dtype=np.uint8),
    )


def test_burst_is_prefiltered_to_the_freshest_frame():
    """A burst outrunning the estimator is dropped from the stale end, not the fresh one."""

    async def run():
        est = FakeEstimator(delay=0.02)
        hub = PositionHub()
        ingest = FrameIngest(est.estimate, hub)
        for seq in range(20):
            ingest.submit(_frame(seq=seq))
            await asyncio.sleep(0.002)  # stream faster than the model
        await asyncio.wait_for(ingest.drain(), timeout=5.0)

        assert ingest.frames_received == 20
        assert 0 < ingest.frames_estimated < 20  # the pre-filter did its job
        assert ingest.frames_estimated + ingest.frames_dropped == 20
        assert ingest.frames_failed == 0
        # The newest frame is never the one dropped, and one worker per board keeps
        # what *is* estimated in submission order.
        assert hub.latest("A1").seq == 19
        assert est.seqs("A1") == sorted(est.seqs("A1"))
        assert est.seqs("A1")[-1] == 19
        await ingest.close()

    asyncio.run(run())


def test_boards_are_independent_and_a_slow_board_does_not_starve_a_fast_one():
    async def run():
        est = FakeEstimator(delays={"SLOW": 0.3, "FAST": 0.0})
        hub = PositionHub()
        ingest = FrameIngest(est.estimate, hub)

        ingest.submit(_frame("SLOW", seq=1))
        for seq in range(1, 4):
            ingest.submit(_frame("FAST", seq=seq))
            await asyncio.sleep(0.01)

        # FAST is done long before SLOW's single frame comes back.
        await asyncio.wait_for(ingest.drain("FAST"), timeout=2.0)
        assert hub.latest("FAST").seq == 3
        assert hub.latest("SLOW") is None

        await asyncio.wait_for(ingest.drain(), timeout=5.0)
        assert hub.latest("SLOW").seq == 1
        assert ingest.frames_received == 4
        await ingest.close()

    asyncio.run(run())


def test_estimator_exception_does_not_end_the_boards_stream():
    async def run():
        est = FakeEstimator(fail_seqs={1})
        hub = PositionHub()
        ingest = FrameIngest(est.estimate, hub)

        ingest.submit(_frame(seq=1))
        await asyncio.wait_for(ingest.drain(), timeout=5.0)
        assert hub.latest("A1") is None
        assert ingest.frames_failed == 1

        ingest.submit(_frame(seq=2))
        await asyncio.wait_for(ingest.drain(), timeout=5.0)
        assert hub.latest("A1").seq == 2  # the worker survived the bad frame
        assert (ingest.frames_received, ingest.frames_estimated) == (2, 1)
        assert ingest.frames_dropped == 0
        await ingest.close()

    asyncio.run(run())


def test_drain_and_close_are_idempotent_and_prompt():
    async def run():
        est = FakeEstimator(delay=1.0)
        hub = PositionHub()
        ingest = FrameIngest(est.estimate, hub)

        await asyncio.wait_for(ingest.drain(), timeout=0.5)  # nothing to drain
        await asyncio.wait_for(ingest.drain("nosuchboard"), timeout=0.5)

        ingest.submit(_frame(seq=0))
        await asyncio.sleep(0.05)  # let the worker enter the slow estimate

        started = time.perf_counter()
        await asyncio.wait_for(ingest.close(), timeout=1.0)
        await asyncio.wait_for(ingest.close(), timeout=1.0)
        # Closing abandons the in-flight estimate instead of waiting the full second.
        assert time.perf_counter() - started < 0.5
        await asyncio.wait_for(ingest.drain(), timeout=0.5)
        assert ingest.frames_estimated == 0

        # A closed ingest is reusable: the next frame starts a fresh worker.
        est.delay = 0.0
        ingest.submit(_frame(seq=1))
        await asyncio.wait_for(ingest.drain(), timeout=5.0)
        assert hub.latest("A1").seq == 1
        await ingest.close()

    asyncio.run(run())


def test_ingest_socket_acks_every_frame_promptly_while_the_estimator_is_busy(
    gateway, drain_ingest
):
    """The /ingest contract: one ack per received frame, in order, never behind the model."""
    client = gateway(create_app(SlowRecognizer(0.05), PositionHub()))
    frames = 10

    with client.websocket_connect("/ingest") as ws:
        started = time.perf_counter()
        for seq in range(frames):
            header, jpeg = _frame(seq=seq).encode()
            ws.send_text(json.dumps(header))
            ws.send_bytes(jpeg)
            assert ws.receive_json() == {"ack": seq, "board_id": "A1"}
        acked = time.perf_counter() - started
        # Serialized estimation would cost >= 0.5s; acks must not wait on it at all.
        assert acked < frames * 0.05 / 2

        ingest = client.app.state.ingest
        assert ingest.frames_received == frames

    # The workers keep going after the acks — wait for them, don't sleep-poll.
    drain_ingest(client)
    hub = client.app.state.hub
    assert hub.latest("A1").seq == frames - 1
    assert ingest.frames_estimated + ingest.frames_dropped == frames
    assert ingest.frames_failed == 0


def test_app_shutdown_stops_the_ingest_workers():
    with TestClient(create_app(StubRecognizer(), PositionHub())) as client:
        ingest = client.app.state.ingest
        with client.websocket_connect("/ingest") as ws:
            header, jpeg = _frame(seq=0).encode()
            ws.send_text(json.dumps(header))
            ws.send_bytes(jpeg)
            ws.receive_json()
        assert ingest.frames_received == 1
    assert ingest._boards == {}  # lifespan shutdown cancelled and forgot the workers


def test_settled_rides_the_stream_and_holds_through_a_one_read_flicker():
    """Every streamed event carries ``settled``, and a single flickered read (a
    hand over the board, a mid-move blur) never changes the settled placement.

    The flicker satisfies every hard constraint on purpose: the stabilizer sees
    the *repaired* read, so a constraint-violating flicker would be (correctly)
    erased by the prior decoder before the debounce is even exercised.
    """
    flicker = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"  # 1.e4 for a read

    async def run():
        placements = [START_PLACEMENT, flicker, START_PLACEMENT]
        events = []
        hub = PositionHub()
        ingest = FrameIngest(
            lambda frame: frame_reader(StubRecognizer(placements[frame.seq]))(frame),
            hub,
            on_event=lambda reading, frame: events.append(reading.event),
        )
        for seq in range(3):
            ingest.submit(_frame(seq=seq))
            await asyncio.wait_for(ingest.drain(), timeout=2.0)
        await ingest.close()
        assert [e.settled.placement for e in events] == [START_PLACEMENT] * 3
        assert events[1].settled.pending == ("e4", "e2")  # board order: rank 8 first
        assert events[2].settled.pending == ()

    asyncio.run(run())


def _entry(seq: int, ts: float, sharpness: float | None) -> tuple[Frame, float]:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    return (
        Frame(
            board_id="T1", seq=seq, capture_ts=ts, image=img, sharpness=sharpness
        ),
        ts,
    )


def test_prefer_frame_keeps_the_sharper_recent_frame():
    """A shaky hand smears half the frames; the ~1 s read should be spent on the
    crisp one, not blindly on the newest."""
    sharp = _entry(1, 10.0, 900.0)
    blurry = _entry(2, 10.4, 120.0)
    assert prefer_frame(sharp, blurry) is sharp


def test_prefer_frame_lets_freshness_win_past_the_window():
    """A sharp frame from a move ago describes the WRONG board — staleness caps
    how long sharpness may hold the inbox."""
    sharp = _entry(1, 10.0, 900.0)
    late = _entry(9, 10.0 + SHARPNESS_WINDOW_S + 0.1, 120.0)
    assert prefer_frame(sharp, late) is late


def test_prefer_frame_without_the_measurement_is_latest_wins():
    """A sharpness-less edge (the synthetic feed, an old page) behaves exactly as
    before: the freshest board state wins."""
    older = _entry(1, 10.0, None)
    newer = _entry(2, 10.1, None)
    assert prefer_frame(older, newer) is newer
    assert prefer_frame(_entry(1, 10.0, 500.0), newer) is newer


def test_prefer_frame_ties_go_to_the_newcomer():
    a = _entry(1, 10.0, 500.0)
    b = _entry(2, 10.1, 500.0)
    assert prefer_frame(a, b) is b


def test_each_board_settles_with_the_ingest_settle_config():
    """``FrameIngest(settle=...)`` builds every board's debounce: with a zero
    window and one read, the first changed read settles at once."""
    from board_reader_inference.recognizer import event_from_dist
    from board_reader_inference.stabilize import SettleConfig

    e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    boards = {0: START_PLACEMENT, 1: e4}

    def read(frame: Frame) -> FrameReading:
        dist = StubRecognizer(boards[frame.seq]).read(frame.image).square_dist
        return FrameReading(event_from_dist(frame, dist))

    async def run(settle: SettleConfig) -> str:
        hub = PositionHub()
        ingest = FrameIngest(read, hub, settle=settle)
        for seq in (0, 1):
            ingest.submit(_frame(seq=seq))
            await asyncio.wait_for(ingest.drain(), timeout=5.0)
        await ingest.close()
        return hub.latest("A1").settled.placement

    assert asyncio.run(run(SettleConfig(seconds=0.0, min_reads=1))) == e4
    assert asyncio.run(run(SettleConfig())) == START_PLACEMENT


def test_healthz_reports_the_settle_config_from_the_environment(monkeypatch):
    monkeypatch.setenv("BR_SETTLE_SECONDS", "1.5")
    monkeypatch.setenv("BR_SETTLE_MIN_READS", "1")
    health = TestClient(create_app(StubRecognizer(), PositionHub())).get("/healthz").json()
    assert health["settle"]["settle_seconds"] == 1.5
    assert health["settle"]["settle_min_reads"] == 1
