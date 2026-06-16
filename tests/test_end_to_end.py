"""Real-network walking-skeleton test (M0): edge -> inference -> viewer over WS.

Unlike ``test_server.py`` (in-process TestClient), this boots a real uvicorn server,
drives frames through the actual edge WebSocket client, and reads them back through a
real WebSocket viewer connection. It proves the end-to-end pipe the user runs.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import httpx
import uvicorn
import websockets

from board_reader_core.sim import render_frames
from board_reader_core.protocol import START_PLACEMENT
from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.client import stream_to_gateway
from board_reader_edge.source import SyntheticSource
from board_reader_inference.server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerThread:
    def __init__(self, port: int):
        config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
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
                msg = await asyncio.wait_for(viewer.recv(), timeout=5.0)
                return sent, msg

        sent, msg = asyncio.run(run())

    assert sent == 5
    import json

    event = json.loads(msg)
    assert event["board_id"] == "A1"
    assert event["placement"] == START_PLACEMENT
    assert event["status"] == "ok"
