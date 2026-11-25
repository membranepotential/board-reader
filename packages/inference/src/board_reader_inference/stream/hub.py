"""Position-stream hub (DESIGN.md §4.5 — pub/sub, retains latest per board).

The hub is the L3 fan-out point: the estimator publishes a
:class:`~board_reader_core.protocol.PositionEvent` per board, the hub stores the
latest per board (so late subscribers get an immediate snapshot) and broadcasts to
every subscriber. It is stateless beyond "last value per board" — no move/game logic.

It also fans out :class:`~board_reader_core.protocol.MotionUpdate` messages
(:meth:`PositionHub.broadcast`), which it never retains: they are display state for
whoever is watching now, and a retained "moving" would outlive the hand.

And it holds the cameras' names (:class:`~board_reader_core.protocol.NamesMessage`,
:meth:`PositionHub.set_names`), one snapshot for all boards: sent to every
subscriber on a change, and first to each new one once any camera has a name.

asyncio-based so it drops straight into the FastAPI gateway, but the core is small
and testable with ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from board_reader_core.protocol import (
    MotionUpdate,
    NamesMessage,
    PositionEvent,
    ViewerMessage,
)


class PositionHub:
    """In-process pub/sub with retained latest-per-board."""

    def __init__(self, queue_maxsize: int = 256) -> None:
        self._latest: dict[str, PositionEvent] = {}
        self._names = NamesMessage(names=())
        self._subscribers: set[asyncio.Queue[ViewerMessage]] = set()
        self._queue_maxsize = queue_maxsize
        self._lock = asyncio.Lock()

    async def publish(self, event: PositionEvent) -> None:
        """Store as latest for its board and fan out to all subscribers."""
        async with self._lock:
            self._latest[event.board_id] = event
        self._fan_out(event)

    def broadcast(self, update: MotionUpdate) -> None:
        """Fan ``update`` out to every subscriber now; nothing is retained.

        Synchronous, so the ingest path can call it per frame without awaiting.
        Safe without the lock: it runs on the event loop and never yields.
        """
        self._fan_out(update)

    def set_names(self, names: NamesMessage) -> None:
        """Hold the cameras' names and send them to every subscriber now.

        Synchronous, like :meth:`broadcast`; the camera registry calls it.
        """
        self._names = names
        self._fan_out(names)

    def _fan_out(self, message: ViewerMessage) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # Slow consumer: drop the oldest, keep the newest (latest-wins).
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(message)
                except asyncio.QueueFull:
                    pass

    def forget(self, board_id: str) -> None:
        """Drop ``board_id``'s retained event, so no new subscriber is shown it.

        For a camera /control forgot: a forgotten camera leaves the live page, and
        should the device come back, it is a new camera that must not open under its
        last position. Synchronous, like :meth:`broadcast`.
        """
        self._latest.pop(board_id, None)

    def latest(self, board_id: str) -> PositionEvent | None:
        return self._latest.get(board_id)

    def latest_all(self) -> dict[str, PositionEvent]:
        return dict(self._latest)

    async def subscribe(
        self, *, replay_latest: bool = True
    ) -> AsyncIterator[ViewerMessage]:
        """Async-iterate messages. By default replays the cameras' names (when any
        camera has one) and then the retained latest position event per board first,
        so a viewer connecting mid-stream renders at once, under the right names."""
        q: asyncio.Queue[ViewerMessage] = asyncio.Queue(maxsize=self._queue_maxsize)
        async with self._lock:
            self._subscribers.add(q)
            snapshot: list[ViewerMessage] = []
            if replay_latest:
                if self._names.names:
                    snapshot.append(self._names)
                snapshot.extend(self._latest.values())
        try:
            for event in snapshot:
                yield event
            while True:
                yield await q.get()
        finally:
            async with self._lock:
                self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
