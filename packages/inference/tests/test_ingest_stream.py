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

from board_reader_core.protocol import START_PLACEMENT, PositionEvent
from board_reader_core.wire import Frame
from board_reader_inference.estimator import StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.ingest import FrameIngest
from board_reader_inference.server import create_app


class FakeEstimator:
    """A :class:`~board_reader_inference.estimator.Estimator` with a controllable cost."""

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
        self._stub = StubEstimator(START_PLACEMENT)

    def estimate(self, frame: Frame) -> PositionEvent:
        time.sleep(self.delays.get(frame.board_id, self.delay))
        self.seen.append((frame.board_id, frame.seq))
        if frame.seq in self.fail_seqs:
            raise RuntimeError(f"model choked on seq {frame.seq}")
        return self._stub.estimate(frame)

    def seqs(self, board_id: str) -> list[int]:
        return [seq for board, seq in self.seen if board == board_id]


def _frame(board_id: str = "A1", seq: int = 0) -> Frame:
    return Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=float(seq),
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def test_burst_is_prefiltered_to_the_freshest_frame():
    """A burst outrunning the estimator is dropped from the stale end, not the fresh one."""

    async def run():
        est = FakeEstimator(delay=0.02)
        hub = PositionHub()
        ingest = FrameIngest(lambda: est, hub)
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
        ingest = FrameIngest(lambda: est, hub)

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
        ingest = FrameIngest(lambda: est, hub)

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
        ingest = FrameIngest(lambda: est, hub)

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


def test_ingest_socket_acks_every_frame_promptly_while_the_estimator_is_busy():
    """The /ingest contract: one ack per received frame, in order, never behind the model."""
    est = FakeEstimator(delay=0.05)
    client = TestClient(create_app(est, PositionHub()))
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
        hub = client.app.state.hub
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:  # the workers keep going after the acks
            latest = hub.latest("A1")
            if latest is not None and latest.seq == frames - 1:
                break
            time.sleep(0.01)
        assert hub.latest("A1").seq == frames - 1

    assert ingest.frames_estimated + ingest.frames_dropped == frames
    assert ingest.frames_failed == 0


def test_app_shutdown_stops_the_ingest_workers():
    est = FakeEstimator(delay=0.0)
    with TestClient(create_app(est, PositionHub())) as client:
        ingest = client.app.state.ingest
        with client.websocket_connect("/ingest") as ws:
            header, jpeg = _frame(seq=0).encode()
            ws.send_text(json.dumps(header))
            ws.send_bytes(jpeg)
            ws.receive_json()
        assert ingest.frames_received == 1
    assert ingest._boards == {}  # lifespan shutdown cancelled and forgot the workers
