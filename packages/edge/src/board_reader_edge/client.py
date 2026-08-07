"""Edge -> inference WebSocket client (DESIGN.md §4.5 ingress).

Streams wire envelopes from any :class:`EnvelopeSource` to the gateway's
``/ingest`` socket: a JSON header text message followed by the binary JPEG, per
frame. This is the real network path the browser "virtual Pi" will mirror later
(WP-C4); for now it lets the file-based adapter — or a replayed recording — drive
a running inference node.
"""

from __future__ import annotations

import json

import websockets

from board_reader_edge.adapter import EnvelopeSource


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
