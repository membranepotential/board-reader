"""Edge -> inference WebSocket client (DESIGN.md §4.5 ingress).

Streams wire envelopes from any :class:`EnvelopeSource` to the gateway's
``/ingest`` socket: a JSON header text message followed by the binary JPEG, per
frame. This is the real network path the browser "virtual Pi" will mirror later
(WP-C4); for now it lets the file-based adapter — or a replayed recording — drive
a running inference node.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

import websockets

from board_reader_core.control import IngestState

from board_reader_edge.adapter import EnvelopeSource

log = logging.getLogger(__name__)

#: A wire envelope: the JSON header and its JPEG (``Frame.encode``).
Envelope = tuple[dict, bytes]


async def stream_to_gateway(
    adapter: EnvelopeSource,
    url: str = "ws://127.0.0.1:8000/ingest",
    *,
    drain_acks: bool = True,
) -> int:
    """Send every envelope from ``adapter`` to ``url``. Returns frames sent."""
    sent = 0
    async with websockets.connect(url, max_size=None) as ws:
        for header, jpeg in adapter.envelopes():
            await ws.send(json.dumps(header))
            await ws.send(jpeg)
            if drain_acks:
                await ws.recv()  # one ack per frame, keeps backpressure honest
            sent += 1
    return sent


class IngestLink:
    """A live camera's ``/ingest`` socket: the latest frame wins, nothing queues.

    :func:`stream_to_gateway` sends a finite stream frame by frame and waits for each
    ack. A live camera must never wait: while one frame is being written, a newer
    one (:meth:`offer`) replaces the frame still pending, as the phone drops a frame
    when its link is behind (DESIGN D3/D4). Acks are read and counted, never waited
    for; a refused frame is logged once per distinct reason.

    :meth:`run` holds the socket, reconnecting until it is cancelled, and says how
    the stream is in the report's terms (:data:`IngestState`).
    """

    def __init__(
        self,
        url: str,
        *,
        on_state: Callable[[IngestState], None],
        retry_s: float = 1.0,
        max_retry_s: float = 10.0,
    ) -> None:
        self.url = url
        self._on_state = on_state
        self._retry_s = retry_s
        self._max_retry_s = max_retry_s
        self._pending: Envelope | None = None
        self._wake = asyncio.Event()
        self._refusals: set[str] = set()
        self.sent = 0
        self.dropped = 0
        self.acked = 0

    def offer(self, envelope: Envelope) -> None:
        """Send this frame next; a frame still pending is dropped for it."""
        if self._pending is not None:
            self.dropped += 1
        self._pending = envelope
        self._wake.set()

    async def run(self) -> None:
        delay = self._retry_s
        self._on_state("connecting")
        try:
            while True:
                try:
                    async with websockets.connect(self.url, max_size=None) as ws:
                        self._on_state("streaming")
                        delay = self._retry_s
                        await self._serve(ws)
                    reason = "closed by the node"
                except (OSError, websockets.WebSocketException) as e:
                    reason = str(e) or type(e).__name__
                self._on_state("reconnecting")
                log.warning("ingest link lost (%s); retrying in %.0f s", reason, delay)
                self._pending = None  # stale by the time the link is back
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_retry_s)
        finally:
            self._pending = None
            self._on_state("idle")

    async def _serve(self, ws: websockets.ClientConnection) -> None:
        reader = asyncio.create_task(self._read(ws))
        try:
            while not reader.done():
                await self._wake.wait()
                self._wake.clear()
                envelope, self._pending = self._pending, None
                if envelope is None:
                    continue
                header, jpeg = envelope
                # Header then JPEG, adjacent: the node reads the two in lockstep.
                await ws.send(json.dumps(header))
                await ws.send(jpeg)
                self.sent += 1
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def _read(self, ws: websockets.ClientConnection) -> None:
        async for message in ws:
            try:
                reply = json.loads(message)
            except ValueError:
                continue
            if not isinstance(reply, dict):
                continue
            if "ack" in reply:
                self.acked += 1
            elif isinstance(error := reply.get("error"), str) and error not in self._refusals:
                self._refusals.add(error)
                log.warning("the node refused a frame: %s", error)
        self._wake.set()  # the socket closed: let the writer see it
