"""End-to-end: pixels go in one end, a capturable stream of positions comes out.

This is the proof-of-concept claim the project exists to make, asserted through
the real pieces and nothing mocked: a synthetic game is rendered frame by frame,
forwarded by the real edge adapter over a real ``WS /ingest`` socket to a real
uvicorn gateway, estimated by a real (pixel-reading) estimator, published to the
hub, and consumed off ``WS /viewer`` by the real headless consumer
(:mod:`board_reader_inference.positions`) — which writes the stream to disk.

What ``tests/test_m2_stream.py`` asserts is the stream's *mechanics* (adaptive
rate, monotone seq, stability). What this asserts is different and was previously
untestable: that the system's **output** exists as an artefact. Until the headless
consumer landed, the only way to observe a run was to watch a browser, so "we can
take a stream from a board and emit a stream of positions" was a claim no test
could reach.

Recognition here is deliberately easy — a template match on synthetic renders
(``tests/_sim_recognizer.py``), which is exact — and the feed is settled frames
only (no hand, no transitions). Neither recognition accuracy nor burst behaviour is on trial;
those have their own suites. What is on trial is that the positions *travel* and
that the capture reproduces them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time

import httpx
import uvicorn
from _sim_recognizer import SimRecognizer

from board_reader_core.protocol import START_PLACEMENT, PositionEvent
from board_reader_core.sim import SAMPLE_GAME_PLACEMENTS

from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.client import stream_to_gateway
from board_reader_edge.source import GameSource
from board_reader_inference.hub import PositionHub
from board_reader_inference.positions import stream_to
from board_reader_inference.recognizer import RecognizerEstimator
from board_reader_inference.server import create_app

PLACEMENTS = SAMPLE_GAME_PLACEMENTS[:4]
# Two frames per position is the smallest feed that still proves the point: the
# stream has to carry each position and the capture has to show them in order.
# Everything above that is the same assertion, paid for again. Small squares for
# the same reason.
FRAMES_PER_POSITION = 2
SQUARE_PX = 32


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerThread:
    def __init__(self, port: int, app):
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            # Without this, shutdown waits on the WebSocket connections this test
            # opened and the join below burns its full 5s timeout — pure teardown
            # cost, paid after every assertion has already passed.
            timeout_graceful_shutdown=1,
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)


def _health(port: int) -> dict | None:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=0.5)
    except httpx.HTTPError:
        return None
    return r.json() if r.status_code == 200 else None


def _wait_healthy(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _health(port) is not None:
            return
        time.sleep(0.05)
    raise TimeoutError("inference server did not become healthy")


async def _wait_subscribed(port: int, timeout: float = 5.0) -> None:
    """Block until the consumer's ``/viewer`` subscription is registered.

    Without this the feed races the subscriber and the test is flaky-by-design:
    frames estimated before anyone is listening are only recoverable from the
    hub's retained latest, which would collapse a *stream* assertion into a
    single-value one. ``/healthz`` already reports ``subscribers``, so the
    readiness signal costs no new API surface.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health = await asyncio.to_thread(_health, port)
        if health and health.get("subscribers", 0) >= 1:
            return
        await asyncio.sleep(0.02)
    raise TimeoutError("position consumer never subscribed")


async def _wait_drained(port: int, sent: int, timeout: float = 10.0) -> None:
    """Wait until the gateway has disposed of every frame the edge sent.

    "Disposed of" counts estimated + pre-filter drops + failures, which is exactly
    the set that can no longer produce an event. Polling that beats sleeping a
    guessed interval in both directions — it returns as soon as the work is done,
    and it does not go flaky when a loaded machine is slower than the guess.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health = await asyncio.to_thread(_health, port)
        counts = (health or {}).get("ingest")
        if counts and counts["estimated"] + counts["dropped"] + counts["failed"] >= sent:
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"gateway did not finish the {sent} frames it was sent")


def test_a_frame_stream_becomes_a_capturable_position_stream(tmp_path):
    port = _free_port()
    capture = tmp_path / "run.ndjson"
    app = create_app(RecognizerEstimator(SimRecognizer()), PositionHub())

    with _ServerThread(port, app):
        _wait_healthy(port)

        async def run() -> int:
            viewer_url = f"ws://127.0.0.1:{port}/viewer"
            ingest_url = f"ws://127.0.0.1:{port}/ingest"
            with capture.open("w", encoding="utf-8") as handle:
                # The real consumer, following until cancelled — the same code
                # path `board-reader-inference positions --out` runs.
                consumer = asyncio.create_task(
                    stream_to(viewer_url, out=handle, tail=None)
                )
                await _wait_subscribed(port)
                source = GameSource(
                    PLACEMENTS,
                    board_id="A1",
                    frames_per_position=FRAMES_PER_POSITION,
                    transition_frames=0,
                    hand=False,
                    square_px=SQUARE_PX,
                )
                sent = await stream_to_gateway(FileEdgeAdapter(source), ingest_url)
                # Wait for the gateway to finish the frames it accepted, rather
                # than sleeping a guessed interval: a fixed sleep is either slower
                # than it needs to be or too short on a loaded machine, and this
                # is neither.
                await _wait_drained(port, sent)
                consumer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await consumer
            return sent

        sent = asyncio.run(run())

    assert sent == len(PLACEMENTS) * FRAMES_PER_POSITION

    lines = capture.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "the run produced no position events at all"

    # Every captured line is a full §5 event that rebuilds exactly — this is what
    # makes a capture a reproduction of the stream rather than a rendering of it.
    events = [PositionEvent.from_dict(json.loads(line)) for line in lines]
    for event in events:
        event.validate()
    assert all(e.board_id == "A1" for e in events)
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    # The positions track the board: every position the game passed through shows
    # up in the captured stream, in the order the game played them.
    captured = [e.placement for e in events]
    assert captured[0] == START_PLACEMENT
    first_seen = [p for i, p in enumerate(captured) if i == 0 or captured[i - 1] != p]
    assert first_seen == list(PLACEMENTS), (
        f"captured position sequence {first_seen} does not follow the game {list(PLACEMENTS)}"
    )
