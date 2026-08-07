"""Streaming frame ingress for the gateway (DESIGN.md §8 WP-G1/G2, D4).

The edge forwards frames at an *adaptive* rate (D4): a slow keepalive at rest, a
10–15 fps burst while the position changes. Estimation is synchronous CPU work
(ONNX/numpy), so running it inline in the ``/ingest`` WebSocket handler blocks the
whole event loop for the duration of every frame — viewers stall, HTTP endpoints
stall, and the ingest socket backs up behind the model.

:class:`FrameIngest` is the seam that decouples the two:

* **G1 — per-board inbox + pre-filter.** Each board has a latest-wins inbox holding
  a single frame: a newer frame displaces an unconsumed older one. Bursts are
  therefore streamed at whatever rate the estimator sustains, and what it skips is
  always the *stale* end of the burst — never the freshest view of the board.
* **G2 — shared estimator worker.** One asyncio task per board pulls from the inbox
  and runs the estimator in a worker thread (:func:`asyncio.to_thread`), leaving the
  event loop free to ack, serve viewers and answer HTTP. One worker per board also
  keeps published events in submission order per board (monotone ``seq``) without
  any extra bookkeeping.

Acks are decoupled from estimation: the gateway acks every frame it *receives*,
including frames the pre-filter later drops. An ack means "received", not
"estimated" — it is the edge's flow-control signal, not a result.

The workers belong to the ingest object, not to a connection: a reconnecting edge
never interrupts an estimate in flight, and shutdown is the only thing that stops
them (:meth:`FrameIngest.lifespan`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from board_reader_core.wire import Frame

from board_reader_inference.estimator import Estimator
from board_reader_inference.hub import PositionHub

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _BoardWorker:
    """Per-board ingress state: the latest-wins inbox and the task draining it."""

    inbox: asyncio.Queue[Frame]  # maxsize=1 — the pre-filter (G1)
    idle: asyncio.Event  # set while the inbox is empty and nothing is in flight
    task: asyncio.Task[None]


class FrameIngest:
    """Latest-wins frame ingress: :meth:`submit` never blocks on the model.

    The estimator is resolved *per frame* through a callable rather than captured,
    because ``POST /calibrate`` hot-swaps the gateway's estimator (§4.6) and a live
    stream must pick the new one up mid-flight.
    """

    def __init__(self, estimator: Callable[[], Estimator], hub: PositionHub) -> None:
        self._estimator = estimator
        self._hub = hub
        self._boards: dict[str, _BoardWorker] = {}
        self.frames_received = 0
        self.frames_estimated = 0
        self.frames_dropped = 0
        self.frames_failed = 0

    def submit(self, frame: Frame) -> None:
        """Hand ``frame`` to its board's worker and return immediately.

        Called from the event-loop thread (the ``/ingest`` handler). If the board's
        inbox already holds an un-estimated frame, that older frame is dropped —
        the estimator is behind, and the freshest board state is the useful one.
        """
        self.frames_received += 1
        worker = self._boards.get(frame.board_id)
        if worker is None or worker.task.done():
            # No worker yet, or a stopped one (post-shutdown): start a fresh board.
            worker = self._spawn(frame.board_id)
            self._boards[frame.board_id] = worker
        worker.idle.clear()
        try:
            worker.inbox.put_nowait(frame)
        except asyncio.QueueFull:
            # No await since the put, so the queue is still full and this cannot miss.
            worker.inbox.get_nowait()
            self.frames_dropped += 1
            worker.inbox.put_nowait(frame)

    async def drain(self, *board_ids: str) -> None:
        """Wait until the named boards (default: all) have nothing left to estimate.

        A stopped worker counts as idle, so this never outlives :meth:`close`.
        """
        if board_ids:
            workers = [self._boards[b] for b in board_ids if b in self._boards]
        else:
            workers = list(self._boards.values())
        for worker in workers:
            await worker.idle.wait()

    async def close(self) -> None:
        """Stop every worker and forget the boards. Idempotent.

        Prompt by design: an estimate already running in its worker thread is
        abandoned rather than awaited (the thread finishes on its own; its result
        is simply never published).
        """
        workers = list(self._boards.values())
        self._boards.clear()
        for worker in workers:
            worker.task.cancel()
        if workers:
            await asyncio.gather(*(w.task for w in workers), return_exceptions=True)

    @contextlib.asynccontextmanager
    async def lifespan(self, _app: object) -> AsyncIterator[None]:
        """ASGI lifespan hook — stops the workers when the gateway shuts down."""
        yield
        await self.close()

    def _spawn(self, board_id: str) -> _BoardWorker:
        inbox: asyncio.Queue[Frame] = asyncio.Queue(maxsize=1)
        idle = asyncio.Event()
        idle.set()
        task = asyncio.create_task(
            self._run(board_id, inbox, idle), name=f"ingest:{board_id}"
        )
        return _BoardWorker(inbox=inbox, idle=idle, task=task)

    async def _run(
        self, board_id: str, inbox: asyncio.Queue[Frame], idle: asyncio.Event
    ) -> None:
        """Estimate whatever is in the inbox, forever, one frame at a time."""
        try:
            while True:
                frame = await inbox.get()
                try:
                    event = await asyncio.to_thread(self._estimator().estimate, frame)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A frame the model chokes on must not end the board's stream.
                    self.frames_failed += 1
                    log.exception(
                        "estimation failed (board %s, seq %d)", board_id, frame.seq
                    )
                else:
                    self.frames_estimated += 1
                    await self._hub.publish(event)
                finally:
                    if inbox.empty():
                        idle.set()
        finally:
            idle.set()  # a stopped worker is never busy — never strand a drain()
