"""``WS /streams``: the control UI's videos and live streams (DESIGN D22).

Up: ``add_stream`` and ``remove_stream``. Down: the whole list on connect, on every
change and about once a second (a stream's camera connecting changes its state from
``starting`` to ``running``, which the registry, not the supervisor, sees), and a
``refused`` with the reason for a message the node refused. No polling, as on the
other control sockets.
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket

from board_reader_core.control import (
    AddStreamMessage,
    RemoveStreamMessage,
    StreamsDown,
    parse_streams_message,
)

from board_reader_inference.control.camera.mailbox import Mailbox
from board_reader_inference.control.camera.sockets import (
    Refusals,
    deliver,
    until_first_ends,
)
from board_reader_inference.control.streams.supervisor import StreamSupervisor

#: How often an open list hears the streams even when nothing it was told of changed.
LIST_EVERY_S = 1.0


def router(supervisor: StreamSupervisor) -> APIRouter:
    api = APIRouter()

    @api.websocket("/streams")
    async def streams(ws: WebSocket) -> None:
        await ws.accept()
        changed = supervisor.connect_list()
        mailbox: Mailbox[StreamsDown] = Mailbox()
        refuse = Refusals("/streams", mailbox)

        async def receive() -> None:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                try:
                    match parse_streams_message(message.get("text") or ""):
                        case AddStreamMessage(url=url, start_s=start_s):
                            supervisor.add(url, start_s)
                        case RemoveStreamMessage(stream_id=stream_id):
                            supervisor.remove(stream_id)
                except ValueError as e:
                    refuse(str(e))

        async def push() -> None:
            while True:
                changed.clear()
                mailbox.post(supervisor.streams())
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(LIST_EVERY_S):
                        await changed.wait()

        try:
            await until_first_ends(receive(), deliver(ws, mailbox), push())
        finally:
            supervisor.disconnect_list(changed)

    return api
