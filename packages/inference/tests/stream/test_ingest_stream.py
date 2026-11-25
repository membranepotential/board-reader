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

from board_reader_inference.recognition.recognizer import (
    FrameReading,
    StubRecognizer,
    frame_reader,
)
from board_reader_inference.server import create_app
from board_reader_inference.stream.hub import PositionHub
from board_reader_inference.stream.ingest import READ_BACKLOG, FrameIngest
from board_reader_inference.stream.schedule import ReadKind


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


def test_a_burst_outrunning_the_estimator_is_read_in_full_and_in_order():
    """A scheduled frame waits for the estimator however long it takes: nothing is
    dropped short of the overload rule, and the reads run in arrival order."""

    async def run():
        est = FakeEstimator(delay=0.02)
        hub = PositionHub()
        ingest = FrameIngest(est.estimate, hub)
        for seq in range(20):
            ingest.submit(_frame(seq=seq))
            await asyncio.sleep(0.002)  # stream faster than the model
        await asyncio.wait_for(ingest.drain(), timeout=5.0)

        assert (ingest.frames_received, ingest.frames_estimated) == (20, 20)
        assert (ingest.frames_dropped, ingest.frames_failed) == (0, 0)
        assert est.seqs("A1") == list(range(20))
        assert hub.latest("A1").seq == 19
        stats = ingest.queue_stats()
        assert stats.served == 20 and stats.max_waiting > 1 and stats.wait_max_s > 0
        await ingest.close()

    asyncio.run(run())


def test_boards_share_one_estimator_oldest_read_first_one_at_a_time():
    """Every board's reads go through one queue: oldest first across boards, and
    never two estimates at once (one GPU session, one arena)."""

    async def run():
        running, most = 0, 0

        def estimate(frame: Frame) -> FrameReading:
            nonlocal running, most
            running += 1
            most = max(most, running)
            try:
                return est.estimate(frame)
            finally:
                running -= 1

        est = FakeEstimator(delays={"SLOW": 0.1, "FAST": 0.0})
        hub = PositionHub()
        ingest = FrameIngest(estimate, hub)
        ingest.submit(_frame("SLOW", seq=1))
        ingest.submit(_frame("FAST", seq=1))
        ingest.submit(_frame("SLOW", seq=2))
        ingest.submit(_frame("FAST", seq=2))
        await asyncio.wait_for(ingest.drain(), timeout=5.0)
        await ingest.close()
        return est.seen, most

    seen, most = asyncio.run(run())
    assert seen == [("SLOW", 1), ("FAST", 1), ("SLOW", 2), ("FAST", 2)]
    assert most == 1


def test_sustained_overload_sheds_keepalive_reads_first_and_reports_each_drop():
    """The overload rule: a board with READ_BACKLOG reads waiting sheds its oldest
    keepalive read, never a burst-end one while it has a keepalive to give up, and
    every drop reaches the drop observer."""
    from board_reader_inference.stream.schedule import EveryFrame

    class _Kinds(EveryFrame):
        """Every frame read; even seqs are burst-end reads, odd ones keepalives."""

        def due(self, capture_ts, stability):
            return ReadKind.BURST_END if int(capture_ts) % 2 == 0 else ReadKind.KEEPALIVE

    async def run():
        gate = asyncio.Event()
        est = FakeEstimator()
        drops: list[tuple[int, ReadKind]] = []

        def estimate(frame: Frame) -> FrameReading:
            if frame.seq == 0:
                asyncio.run_coroutine_threadsafe(gate.wait(), loop).result(timeout=5)
            return est.estimate(frame)

        loop = asyncio.get_running_loop()
        ingest = FrameIngest(
            estimate,
            PositionHub(),
            policy=_Kinds,
            on_drop=lambda frame, kind, rules, waited: drops.append((frame.seq, kind)),
        )
        ingest.submit(_frame(seq=0))  # parks the estimator
        await asyncio.sleep(0.05)
        extra = 10
        for seq in range(1, 1 + READ_BACKLOG + extra):
            ingest.submit(_frame(seq=seq))
        gate.set()
        await asyncio.wait_for(ingest.drain(), timeout=5.0)
        await ingest.close()
        return est.seqs("A1"), drops, ingest

    seen, drops, ingest = asyncio.run(run())
    assert [kind for _, kind in drops] == [ReadKind.KEEPALIVE] * 10
    assert [seq for seq, _ in drops] == list(range(1, 21, 2))  # the oldest keepalives
    assert (ingest.frames_dropped, ingest.frames_dropped_burst_end) == (10, 0)
    assert seen == sorted(seen) and len(seen) == 1 + READ_BACKLOG
    assert all(seq % 2 == 0 for seq in seen[1:11]), "every burst-end read was kept"


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


def test_the_settled_board_rides_the_stream_and_holds_through_a_one_read_flicker():
    """Every streamed event's ``placement`` is the settled board, and a single
    flickered read (a hand over the board, a mid-move blur) never changes it; the
    read itself stays the argmax of ``square_dist``.

    The flicker is a possible board on purpose, so it is the debounce that
    holds it back, not the guard.
    """
    flicker = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"  # 1.e4 for a read

    async def run():
        placements = [START_PLACEMENT, flicker, START_PLACEMENT]
        events = []
        hub = PositionHub()
        ingest = FrameIngest(
            lambda frame: frame_reader(StubRecognizer(placements[frame.seq]))(frame),
            hub,
            on_event=lambda reading, frame, rules: events.append(reading.event),
        )
        for seq in range(3):
            ingest.submit(_frame(seq=seq))
            await asyncio.wait_for(ingest.drain(), timeout=2.0)
        await ingest.close()
        assert [e.placement for e in events] == [START_PLACEMENT] * 3
        assert events[1].raw_placement == flicker
        assert events[1].pending == ("e4", "e2")  # board order: rank 8 first
        assert events[2].pending == ()
        # The board never changed, so neither did the time it last changed.
        assert {e.changed_utc for e in events} == {events[0].utc}
        for event in events:
            event.validate()

    asyncio.run(run())


def test_each_board_settles_under_its_own_rules_read_per_read():
    """The worker asks for the board's setting before every read, so a board
    switched to ``free`` settles a 17th piece that a ``standard`` board refuses,
    and the setting each read used is handed to the observers (the recorder)."""
    phantom = "rnbqkbnr/pppppppp/1r6/8/8/8/PPPPPPPP/RNBQKBNR"  # 17 black pieces

    async def run():
        placements = [START_PLACEMENT, phantom, phantom]
        setting = {"variant": "free", "game": "standard"}
        seen: dict[str, list] = {"variant": [], "game": []}
        ingest = FrameIngest(
            lambda frame: frame_reader(StubRecognizer(placements[frame.seq]))(frame),
            PositionHub(),
            on_event=lambda reading, frame, rules: seen[frame.board_id].append(
                (reading.event.placement, rules)
            ),
            rules=setting.__getitem__,
        )
        for board in setting:
            for seq in range(3):
                ingest.submit(_frame(board_id=board, seq=seq))
                await asyncio.wait_for(ingest.drain(), timeout=2.0)
        await ingest.close()
        assert seen["variant"][-1] == (phantom, "free")
        assert seen["game"][-1] == (START_PLACEMENT, "standard")

    asyncio.run(run())


def test_each_board_settles_with_the_ingest_settle_config():
    """``FrameIngest(settle=...)`` builds every board's debounce: with a zero
    window and one read, the first changed read settles at once."""
    from board_reader_inference.recognition.recognizer import event_from_dist
    from board_reader_inference.stream.stabilize import SettleConfig

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
        return hub.latest("A1").placement

    assert asyncio.run(run(SettleConfig(seconds=0.0, min_reads=1))) == e4
    assert asyncio.run(run(SettleConfig())) == START_PLACEMENT


def test_an_obscured_read_is_published_as_such_and_commits_nothing():
    """The frame energy reaches the debounce through ``_settle_read``: with a
    window that commits any changed read at once, the obscured one still waits."""
    from board_reader_inference.recognition.recognizer import event_from_dist
    from board_reader_inference.stream.stabilize import SettleConfig

    e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    #: seq -> (the read, its frame energy): clear, obscured, clear.
    reads = {0: (START_PLACEMENT, -14.0), 1: (e4, -10.0), 2: (e4, -14.0)}

    def read(frame: Frame) -> FrameReading:
        board, energy = reads[frame.seq]
        dist = StubRecognizer(board).read(frame.image).square_dist
        return FrameReading(event_from_dist(frame, dist), frame_energy=energy)

    async def run() -> list[PositionEvent]:
        hub = PositionHub()
        ingest = FrameIngest(
            read, hub, settle=SettleConfig(seconds=0.0, min_reads=1, obscured_energy=-12.4)
        )
        published = []
        for seq in reads:
            ingest.submit(_frame(seq=seq))
            await asyncio.wait_for(ingest.drain(), timeout=5.0)
            published.append(hub.latest("A1"))
        await ingest.close()
        return published

    events = asyncio.run(run())
    assert [(e.placement, e.obscured) for e in events] == [
        (START_PLACEMENT, False),
        (START_PLACEMENT, True),
        (e4, False),
    ]


def test_healthz_reports_the_settle_config_from_the_environment(monkeypatch):
    monkeypatch.setenv("BR_SETTLE_SECONDS", "1.5")
    monkeypatch.setenv("BR_SETTLE_MIN_READS", "1")
    health = TestClient(create_app(StubRecognizer(), PositionHub())).get("/healthz").json()
    assert health["settle"]["settle_seconds"] == 1.5
    assert health["settle"]["settle_min_reads"] == 1
