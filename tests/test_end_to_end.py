"""Real-network walking-skeleton test (M0): edge -> inference -> viewer over WS.

Unlike ``test_server.py`` (in-process TestClient), this boots a real uvicorn server,
drives frames through the actual edge WebSocket client, and reads them back through a
real WebSocket viewer connection. It proves the end-to-end pipe the user runs.
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

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_frames

from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.client import stream_to_gateway
from board_reader_edge.source import SyntheticSource
from board_reader_inference.server import create_app
from board_reader_inference.stream.schedule import DEFAULT_POLICY_SPEC


async def _next_position(viewer, timeout: float = 5.0) -> str:
    """The next position event off ``/viewer``, past any motion updates."""
    async with asyncio.timeout(timeout):
        while True:
            msg = await viewer.recv()
            if "type" not in json.loads(msg):
                return msg


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerThread:
    def __init__(self, port: int, app=None):
        config = uvicorn.Config(
            create_app() if app is None else app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            # Shutdown otherwise waits on the WebSocket connections this test
            # opened, and the join below burns its full 5s timeout — teardown
            # cost paid after every assertion has already passed (measured: it
            # was the entire runtime of this test).
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


def test_walking_skeleton_edge_to_viewer():
    port = _free_port()
    with _ServerThread(port):
        _wait_healthy(port)

        source = SyntheticSource(render_frames(5 * [START_PLACEMENT]), board_id="A1")
        adapter = FileEdgeAdapter(source)

        async def run():
            viewer_url = f"ws://127.0.0.1:{port}/viewer"
            ingest_url = f"ws://127.0.0.1:{port}/ingest"
            async with websockets.connect(viewer_url) as viewer:
                sent = await stream_to_gateway(adapter, ingest_url)
                msg = await _next_position(viewer)
                return sent, msg

        sent, msg = asyncio.run(run())

    assert sent == 5
    event = json.loads(msg)
    assert event["board_id"] == "A1"
    assert event["placement"] == START_PLACEMENT
    assert event["status"] == "ok"


def test_the_pipe_still_publishes_under_the_shipped_read_schedule(monkeypatch):
    """The same pipe, with the scheduler `serve` turns on (``BR_SCHEDULE``).

    Scheduling is the one change that can make an end-to-end run look *broken by
    working*: a policy that declines every frame publishes nothing, the viewer
    stays empty, and every layer below is green. So the walking skeleton is run
    once more under the shipped policy, and the claim is the minimal one that
    matters — frames still become a position.

    The synthetic feed is the adversarial case for it, not the friendly one. Its
    frames are stamped ``stability=1.0`` (``SyntheticSource`` has no motion meter),
    so there is never a burst and never a burst *end* — every read on this stream
    has to come from the D3 keepalive floor. Four frames at the source's 2 fps
    span 1.5 s of capture time, inside one 2 s keepalive, so exactly one of them is
    worth an estimate.
    """
    monkeypatch.setenv("BR_SCHEDULE", DEFAULT_POLICY_SPEC)
    port = _free_port()
    with _ServerThread(port, app=create_app()):
        _wait_healthy(port)

        source = SyntheticSource(render_frames(4 * [START_PLACEMENT]), board_id="A1")
        adapter = FileEdgeAdapter(source)

        async def run():
            async with websockets.connect(f"ws://127.0.0.1:{port}/viewer") as viewer:
                sent = await stream_to_gateway(adapter, f"ws://127.0.0.1:{port}/ingest")
                msg = await _next_position(viewer)
                return sent, msg

        sent, msg = asyncio.run(run())
        stats = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=5.0).json()

    assert sent == 4, "every frame is still forwarded and acked; only reads are sparse"
    assert json.loads(msg)["placement"] == START_PLACEMENT
    ingest = stats["ingest"]
    assert ingest["received"] == 4
    assert ingest["estimated"] == 1
    assert ingest["estimated"] + ingest["dropped"] + ingest["unscheduled"] == 4
    assert ingest["backstopped"] == 0, "the edge's capture clock is sound"
