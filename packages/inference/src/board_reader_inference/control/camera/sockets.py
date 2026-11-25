"""What every control socket handler shares: sending, ending together, refusing, previews."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Coroutine
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from board_reader_core.control import PreviewMessage, RefusedMessage, ReplacedMessage
from board_reader_core.wire import decode_jpeg

from board_reader_inference.control.camera.mailbox import Mailbox, Message

log = logging.getLogger(__name__)

#: A preview is a small thumbnail; anything bigger is not one.
MAX_PREVIEW_BYTES = 1_000_000


async def deliver[M: Message](ws: WebSocket, mailbox: Mailbox[M]) -> None:
    while True:
        for message in await mailbox.take():
            await ws.send_text(message.to_json())
            if isinstance(message, ReplacedMessage):
                # The device's newer socket took over: this one closes, said aloud,
                # because a handler that only returns leaves the socket open.
                await ws.close(code=1000, reason="replaced")
                return


class _HungUp(Exception):
    """One of a socket's loops ended: the client hung up (or a send failed)."""


async def _ending(loop: Coroutine[Any, Any, None]) -> None:
    with contextlib.suppress(WebSocketDisconnect):
        await loop
    raise _HungUp


async def until_first_ends(*loops: Coroutine[Any, Any, None]) -> None:
    """Run a socket's loops together; the first to end (a hang-up) ends them all.

    A ``TaskGroup``, not ``asyncio.wait`` plus a hand-rolled cancel: the server (or
    Starlette's test client) may cancel the handler from outside at any moment, and
    only the task group keeps that cancellation intact while it stops the loops.
    """
    try:
        async with asyncio.TaskGroup() as group:
            for loop in loops:
                group.create_task(_ending(loop))
    except* _HungUp:
        pass


class Refusals:
    """Refuse one message on one socket: tell the client, log each reason once."""

    def __init__(self, socket: str, mailbox: Mailbox[Any]) -> None:
        self._socket = socket
        self._mailbox = mailbox
        self._logged: set[str] = set()

    def __call__(self, reason: str) -> None:
        self._mailbox.post(RefusedMessage(reason))
        if reason not in self._logged:
            self._logged.add(reason)
            log.warning("refused a message on %s: %s", self._socket, reason)


def parse_preview(jpeg: bytes) -> PreviewMessage:
    if len(jpeg) > MAX_PREVIEW_BYTES:
        raise ValueError(f"preview larger than {MAX_PREVIEW_BYTES} bytes")
    try:
        image = decode_jpeg(jpeg)
    except ValueError as e:
        raise ValueError("preview is not a JPEG") from e
    height, width = image.shape[:2]
    return PreviewMessage(jpeg=jpeg, width=width, height=height)
