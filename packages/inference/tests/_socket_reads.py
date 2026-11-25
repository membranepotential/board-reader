"""Bounded reads of a Starlette ``WebSocketTestSession``.

``WebSocketTestSession.receive_text()`` and its siblings wait forever, so a test
that awaits a message the node never sends hangs until pytest-timeout fails it at
10 s, with the stack pointing at nothing useful. These helpers fail after
:data:`RECEIVE_S` with a ``TimeoutError`` raised at the read itself. They raise
``WebSocketDisconnect`` on a close exactly like Starlette's own methods.

Import them by module name, like ``tests/_sim_recognizer.py``.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
from starlette.testclient import WebSocketTestSession

#: How long one message may take; the node answers within milliseconds in a test.
RECEIVE_S = 5


def _receive(ws: WebSocketTestSession) -> dict[str, Any]:
    async def bounded() -> dict[str, Any]:
        with anyio.fail_after(RECEIVE_S):
            return await ws._send_rx.receive()

    message = ws.portal.call(bounded)
    ws._raise_on_close(message)
    return message


def receive_text(ws: WebSocketTestSession) -> str:
    return _receive(ws)["text"]


def receive_bytes(ws: WebSocketTestSession) -> bytes:
    return _receive(ws)["bytes"]


def receive_json(ws: WebSocketTestSession) -> Any:
    return json.loads(receive_text(ws))
