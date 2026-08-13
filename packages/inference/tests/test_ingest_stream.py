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

from board_reader_inference.centroid import build_exemplars
from board_reader_inference.embed import LiteEmbedder
from board_reader_inference.estimator import CentroidEstimator, StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.ingest import FrameIngest
from board_reader_inference.routing import Route, Uncalibrated, WholeFrameRead, route_for
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
        self.shapes: list[tuple[int, int]] = []
        self._stub = StubEstimator(START_PLACEMENT)

    def estimate(self, frame: Frame) -> PositionEvent:
        time.sleep(self.delays.get(frame.board_id, self.delay))
        self.seen.append((frame.board_id, frame.seq))
        self.shapes.append(frame.image.shape[:2])
        if frame.seq in self.fail_seqs:
            raise RuntimeError(f"model choked on seq {frame.seq}")
        return self._stub.estimate(frame)

    def seqs(self, board_id: str) -> list[int]:
        return [seq for board, seq in self.seen if board == board_id]


class FakePhotoEstimator:
    """A photo-mode estimator, recording which entry point each frame took."""

    crop_mode = "photo"

    def __init__(self) -> None:
        self.photo_calls: list[tuple[int, np.ndarray]] = []
        self.plain_calls: list[int] = []
        self._stub = StubEstimator(START_PLACEMENT)

    def estimate(self, frame: Frame) -> PositionEvent:
        self.plain_calls.append(frame.seq)
        return self._stub.estimate(frame)

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        self.photo_calls.append((frame.seq, np.asarray(corners)))
        return self._stub.estimate(frame)


QUAD = np.array([[4.0, 6.0], [96.0, 8.0], [98.0, 94.0], [2.0, 92.0]], dtype=np.float32)


def _occupancy_bank() -> CentroidEstimator:
    """The smallest possible occupancy-bank estimator.

    Only its existence matters here: it is what makes the §4.6 uncalibrated route
    constructible. Nothing in this module ever reads a position through it.
    """
    emb = LiteEmbedder()
    crops = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(64)]
    return CentroidEstimator(emb, build_exemplars(emb, crops, START_PLACEMENT))


def _frame(board_id: str = "A1", seq: int = 0, corners=None) -> Frame:
    return Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=float(seq),
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        corners=corners,
    )


def test_burst_is_prefiltered_to_the_freshest_frame():
    """A burst outrunning the estimator is dropped from the stale end, not the fresh one."""

    async def run():
        est = FakeEstimator(delay=0.02)
        hub = PositionHub()
        ingest = FrameIngest(lambda frame: route_for(est, frame), hub)
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
        ingest = FrameIngest(lambda frame: route_for(est, frame), hub)

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
        ingest = FrameIngest(lambda frame: route_for(est, frame), hub)

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
        ingest = FrameIngest(lambda frame: route_for(est, frame), hub)

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
    est = FakeEstimator(delay=0.05)
    client = gateway(create_app(est, PositionHub()))
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


def test_corners_route_to_a_photo_mode_estimators_photo_path():
    est = FakePhotoEstimator()
    event = route_for(est, _frame(seq=7, corners=QUAD)).read()

    assert est.plain_calls == []  # never the rectified path
    assert [seq for seq, _ in est.photo_calls] == [7]
    assert np.allclose(est.photo_calls[0][1], QUAD)  # the quad, verbatim
    assert event.seq == 7


def test_corners_are_rectified_for_an_estimator_without_a_photo_path():
    est = FakeEstimator()
    frame = Frame(
        board_id="B2",
        seq=5,
        capture_ts=12.5,
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        stability=0.25,
        clock="black",
        burst_id="b7",
        corners=QUAD,
    )
    event = route_for(est, frame).read()

    # The estimator saw the warp, not the original photo it cannot read.
    assert est.shapes == [(512, 512)]
    assert est.seen == [("B2", 5)]
    # ...and everything the envelope carries survived the rebuild.
    assert (event.board_id, event.seq, event.stability) == ("B2", 5, 0.25)
    assert event.clock.running == "black"


def test_a_frame_without_corners_takes_the_unchanged_path():
    est = FakeEstimator()
    route = route_for(est, _frame(seq=1))
    assert isinstance(route, WholeFrameRead)
    route.read()
    assert est.shapes == [(100, 100)]  # untouched pixels


def test_a_photo_estimator_without_corners_reads_the_whole_frame_as_the_board():
    """A corner-less body is a rectified crop, and a photo-mode estimator reads it
    through its photo path with the whole frame as the quad — the contract
    ``POST /estimate`` has always had, and now the stream's too (they share one
    router, so this can only ever be one behaviour)."""
    photo = FakePhotoEstimator()
    route_for(photo, _frame(seq=1)).read()
    assert photo.plain_calls == []
    assert [seq for seq, _ in photo.photo_calls] == [1]
    assert np.allclose(
        photo.photo_calls[0][1], [[0, 0], [100, 0], [100, 100], [0, 100]]
    )


def test_streamed_corners_reach_the_photo_path_end_to_end(gateway, drain_ingest):
    est = FakePhotoEstimator()
    client = gateway(create_app(est, PositionHub()))
    with client.websocket_connect("/ingest") as ws:
        header, jpeg = _frame(seq=3, corners=QUAD).encode()
        ws.send_text(json.dumps(header))
        ws.send_bytes(jpeg)
        assert ws.receive_json() == {"ack": 3, "board_id": "A1"}

    drain_ingest(client)
    hub = client.app.state.hub
    assert hub.latest("A1").seq == 3
    assert [seq for seq, _ in est.photo_calls] == [3]
    assert np.allclose(est.photo_calls[0][1], QUAD)


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
            lambda frame: WholeFrameRead(
                StubEstimator(placements[frame.seq]), frame
            ),
            hub,
            on_event=lambda event, frame: events.append(event),
        )
        for seq in range(3):
            ingest.submit(_frame(seq=seq))
            await asyncio.wait_for(ingest.drain(), timeout=2.0)
        await ingest.close()
        assert [e.settled.placement for e in events] == [START_PLACEMENT] * 3
        assert events[1].settled.pending == ("e4", "e2")  # board order: rank 8 first
        assert events[2].settled.pending == ()

    asyncio.run(run())


def test_an_uncalibrated_route_skips_frames_and_the_stream_wakes_on_hot_swap():
    """§4.6 uncalibrated state: the router answers ``Uncalibrated`` and the stream
    publishes NOTHING (the hub carries estimated positions, never a placeholder) —
    then a calibration hot-swap brings the very same stream alive mid-flight."""

    async def run():
        stub = StubEstimator(START_PLACEMENT)
        routes: dict[str, Route] = {
            "current": Uncalibrated(_occupancy_bank(), _frame(), QUAD)
        }
        hub = PositionHub()
        ingest = FrameIngest(lambda frame: routes["current"], hub)

        ingest.submit(_frame(seq=1))
        await asyncio.wait_for(ingest.drain(), timeout=2.0)
        assert hub.latest("A1") is None
        assert ingest.frames_skipped == 1
        assert ingest.frames_estimated == 0

        routes["current"] = WholeFrameRead(stub, _frame(seq=2))
        ingest.submit(_frame(seq=2))
        await asyncio.wait_for(ingest.drain(), timeout=2.0)
        assert hub.latest("A1").seq == 2
        assert ingest.frames_skipped == 1
        await ingest.close()

    asyncio.run(run())
