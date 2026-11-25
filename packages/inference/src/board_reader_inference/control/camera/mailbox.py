"""The messages one socket has yet to send: the latest of each type.

Every message on the control sockets is a snapshot (a config, a state, a picture, a
health reading), so a slow client that misses an intermediate one loses nothing it
needs, and no client can make the node buffer without bound.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar, Protocol


class Message(Protocol):
    TYPE: ClassVar[str]

    def to_json(self) -> str: ...


class Mailbox[M: Message]:
    """The messages one socket has yet to send: the latest of each type.

    Posting a type again replaces the pending one and moves it to the back, so
    messages leave in the order they were last posted.
    """

    def __init__(self) -> None:
        self._pending: dict[str, M] = {}
        self._ready = asyncio.Event()

    def post(self, message: M) -> None:
        self._pending.pop(message.TYPE, None)
        self._pending[message.TYPE] = message
        self._ready.set()

    def take_nowait(self) -> list[M]:
        taken = list(self._pending.values())
        self._pending.clear()
        self._ready.clear()
        return taken

    async def take(self) -> list[M]:
        await self._ready.wait()
        return self.take_nowait()
