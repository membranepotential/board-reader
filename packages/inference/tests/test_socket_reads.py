"""The bounded socket reads fail fast on a silent node."""

from __future__ import annotations

import contextlib
import time

import pytest
from _socket_reads import RECEIVE_S, receive_text
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.testclient import TestClient


def test_a_read_of_a_silent_socket_times_out_within_the_bound():
    app = FastAPI()

    @app.websocket("/silent")
    async def silent(ws: WebSocket) -> None:
        await ws.accept()
        with contextlib.suppress(WebSocketDisconnect):
            await ws.receive_text()  # sends nothing until the test closes

    with TestClient(app) as client, client.websocket_connect("/silent") as ws:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            receive_text(ws)
        assert time.monotonic() - started < RECEIVE_S + 2
