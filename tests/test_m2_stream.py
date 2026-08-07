"""M2 end-to-end: a recorded game streams adaptively through the whole pipe.

Boots a real uvicorn gateway (like ``test_end_to_end.py``) and replays the
synthetic recorded game — settled positions, hand-occluded transitions — through
the edge's real M2 stages: MotionMeter → AdaptiveForwarder → FileEdgeAdapter →
``WS /ingest`` → per-board ingest worker → hub → ``WS /viewer``.

What M2 promises (DESIGN §7): the edge forwards a keepalive trickle at rest and
bursts through motion (D4); the gateway streams position events continuously with
monotone per-board seq; stability travels on every event (D3) and visibly drops
during transitions. Recognition accuracy is M1's business — the estimator here is
the stub, and the assertions are about the *stream mechanics*.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import uvicorn
import websockets

from board_reader_core.sim import SAMPLE_GAME_PLACEMENTS

from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.client import stream_to_gateway
from board_reader_edge.forwarder import AdaptiveForwarder, ForwarderConfig
from board_reader_edge.source import GameSource
from board_reader_inference.server import create_app

PLACEMENTS = SAMPLE_GAME_PLACEMENTS[:4]
FRAMES_PER_POSITION = 15
TRANSITION_FRAMES = 6
CAPTURE_FPS = 30.0
TOTAL_FRAMES = len(PLACEMENTS) * FRAMES_PER_POSITION + (len(PLACEMENTS) - 1) * TRANSITION_FRAMES


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerThread:
    def __init__(self, port: int):
        config = uvicorn.Config(
            create_app(),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            # Shutdown otherwise waits on the WebSocket connections this test
            # opened, and the join below burns its full 5s timeout — teardown
            # cost paid after every assertion has already passed.
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


def _wait_healthy(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise TimeoutError("inference server did not become healthy")


def test_m2_adaptive_stream_over_a_recorded_game():
    port = _free_port()
    with _ServerThread(port):
        _wait_healthy(port)

        source = GameSource(
            PLACEMENTS,
            board_id="A1",
            fps=CAPTURE_FPS,
            frames_per_position=FRAMES_PER_POSITION,
            transition_frames=TRANSITION_FRAMES,
            square_px=24,
        )
        adapter = FileEdgeAdapter(
            source,
            forwarder=AdaptiveForwarder(ForwarderConfig(keepalive_fps=2.0)),
        )

        async def run():
            viewer_url = f"ws://127.0.0.1:{port}/viewer"
            ingest_url = f"ws://127.0.0.1:{port}/ingest"
            async with websockets.connect(viewer_url) as viewer:
                sent = await stream_to_gateway(adapter, ingest_url)
                # The disconnect handler drains the board's worker, so shortly
                # after the stream ends the event flow dries up: collect until
                # a quiet second.
                events = []
                while True:
                    try:
                        msg = await asyncio.wait_for(viewer.recv(), timeout=1.0)
                    except TimeoutError:
                        break
                    events.append(json.loads(msg))
                return sent, events

        sent, events = asyncio.run(run())

    # D4 — adaptive forwarding: the ~30 fps capture stream leaves the edge as a
    # keepalive trickle plus bursts, far below the raw frame count.
    assert sent < TOTAL_FRAMES / 2, f"forwarded {sent}/{TOTAL_FRAMES}: not adaptive"
    assert sent >= 5  # but the stream never goes silent

    # The gateway streamed positions continuously and in order.
    assert len(events) >= 3
    assert all(e["board_id"] == "A1" for e in events)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)  # monotone per board
    assert all(e["status"] == "ok" for e in events)

    # D3 — stability travels on every event, drops through the hand-occluded
    # transitions (bursts) and reads at-rest on settled keepalives.
    stabilities = [e["stability"] for e in events]
    assert all(0.0 <= s <= 1.0 for s in stabilities)
    assert min(stabilities) < 0.7, "no burst-time event made it through"
    assert max(stabilities) > 0.85, "no settled keepalive event made it through"
