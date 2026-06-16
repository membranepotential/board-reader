"""Position-stream hub (DESIGN.md §4.5 — pub/sub, retains latest per board).

The hub is the L3 fan-out point: the estimator publishes a
:class:`~board_reader_core.protocol.PositionEvent` per board, the hub stores the
latest per board (so late subscribers get an immediate snapshot) and broadcasts to
every subscriber. It is stateless beyond "last value per board" — no move/game logic.

asyncio-based so it drops straight into the FastAPI gateway, but the core is small
and testable with ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from board_reader_core.protocol import PositionEvent


class PositionHub:
    """In-process pub/sub with retained latest-per-board."""

    def __init__(self, queue_maxsize: int = 256) -> None:
        self._latest: dict[str, PositionEvent] = {}
        self._subscribers: set[asyncio.Queue[PositionEvent]] = set()
        self._queue_maxsize = queue_maxsize
        self._lock = asyncio.Lock()

    async def publish(self, event: PositionEvent) -> None:
        """Store as latest for its board and fan out to all subscribers."""
        async with self._lock:
            self._latest[event.board_id] = event
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer: drop the oldest, keep the newest (latest-wins).
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    def latest(self, board_id: str) -> PositionEvent | None:
        return self._latest.get(board_id)

    def latest_all(self) -> dict[str, PositionEvent]:
        return dict(self._latest)

    async def subscribe(
        self, *, replay_latest: bool = True
    ) -> AsyncIterator[PositionEvent]:
        """Async-iterate events. By default replays the retained latest per board
        first, so a viewer connecting mid-stream renders immediately."""
        q: asyncio.Queue[PositionEvent] = asyncio.Queue(maxsize=self._queue_maxsize)
        async with self._lock:
            self._subscribers.add(q)
            snapshot = list(self._latest.values()) if replay_latest else []
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
