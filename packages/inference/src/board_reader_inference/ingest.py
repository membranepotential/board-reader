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

What a worker *does* with a frame is not this module's decision: it asks
:mod:`board_reader_inference.routing` — the same function ``POST /estimate`` asks —
and runs the :class:`~board_reader_inference.routing.Route` it gets back.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from board_reader_core.protocol import PositionEvent
from board_reader_core.wire import Frame

from board_reader_inference.hub import PositionHub
from board_reader_inference.prior import decode
from board_reader_inference.routing import Route, Uncalibrated

log = logging.getLogger(__name__)


def _read_with_prior(route: Route) -> PositionEvent:
    """One estimate plus its hard-constraint repair, both off the event loop.

    The repair rides the event as :attr:`PositionEvent.prior` (raw ``placement``
    and ``square_dist`` untouched), so every consumer of the stream — the viewer,
    ``GET /boards/{id}``, a recording — sees the same repaired board the HTTP
    endpoints already serve beside their reads. ``quarter_turns`` is deliberately
    not passed: the stream's orientation is operator-claimed, not established,
    and a guessed frame makes the back-rank constraint circular (see
    :mod:`.prior`).
    """
    event = route.read()
    event.prior = decode(event.square_dist).to_repair()
    return event


@dataclass(frozen=True, slots=True)
class _BoardWorker:
    """Per-board ingress state: the latest-wins inbox and the task draining it."""

    #: maxsize=1 — the pre-filter (G1). Each entry is (frame, received_ts): the
    #: gateway's own clock at submit, because the frame's ``capture_ts`` is the
    #: *edge's* clock (a phone seconds off NTP is normal) and subtracting one
    #: from the other says nothing about how far behind this node runs.
    inbox: asyncio.Queue[tuple[Frame, float]]
    idle: asyncio.Event  # set while the inbox is empty and nothing is in flight
    task: asyncio.Task[None]


class FrameIngest:
    """Latest-wins frame ingress: :meth:`submit` never blocks on the model.

    The route is resolved *per frame* through a callable rather than captured,
    because ``POST /calibrate`` hot-swaps the gateway's estimator (§4.6) and a live
    stream must pick the new one up mid-flight. The route may come back
    :class:`~board_reader_inference.routing.Uncalibrated` — "nothing here can read a
    position yet" (the §4.6 state where ``POST /estimate`` answers occupancy-only
    and publishes nothing). A stream has no HTTP response to answer with, so its
    frames are counted and *skipped*: the hub carries positions someone actually
    estimated, never a placeholder. Calibrating (the camera page's button, or a
    snapshot of the start position) hot-swaps a real estimator in and the same
    stream comes alive mid-flight.
    """

    def __init__(
        self,
        route: Callable[[Frame], Route],
        hub: PositionHub,
        *,
        on_event: Callable[[PositionEvent, Frame], None] | None = None,
    ) -> None:
        self._route = route
        self._hub = hub
        #: Called (on the event loop) for every published event, with the frame it
        #: was estimated from — the gateway hangs its trial log off this.
        self._on_event = on_event
        self._boards: dict[str, _BoardWorker] = {}
        self._warned_unready = False
        self.frames_received = 0
        self.frames_estimated = 0
        self.frames_dropped = 0
        self.frames_failed = 0
        self.frames_skipped = 0  # arrived while no estimator was ready
        #: Incremented by the gateway's ``/ingest`` handler for frames refused at
        #: the wire (malformed header, truncated JPEG) — they never reach
        #: :meth:`submit`, but a stream that is *all* refusals must not be
        #: indistinguishable from one that never sent.
        self.frames_refused = 0

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
        entry = (frame, time.time())
        try:
            worker.inbox.put_nowait(entry)
        except asyncio.QueueFull:
            # No await since the put, so the queue is still full and this cannot miss.
            worker.inbox.get_nowait()
            self.frames_dropped += 1
            worker.inbox.put_nowait(entry)

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
        inbox: asyncio.Queue[tuple[Frame, float]] = asyncio.Queue(maxsize=1)
        idle = asyncio.Event()
        idle.set()
        task = asyncio.create_task(
            self._run(board_id, inbox, idle), name=f"ingest:{board_id}"
        )
        return _BoardWorker(inbox=inbox, idle=idle, task=task)

    async def _run(
        self,
        board_id: str,
        inbox: asyncio.Queue[tuple[Frame, float]],
        idle: asyncio.Event,
    ) -> None:
        """Estimate whatever is in the inbox, forever, one frame at a time."""
        try:
            while True:
                frame, received = await inbox.get()
                try:
                    route = self._route(frame)
                    if isinstance(route, Uncalibrated):
                        self.frames_skipped += 1
                        if not self._warned_unready:
                            self._warned_unready = True
                            log.info(
                                "frames arriving with no estimator ready (board %s):"
                                " skipping until a calibration lands", board_id,
                            )
                        continue
                    event = await asyncio.to_thread(_read_with_prior, route)
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
                    previous = self._hub.latest(board_id)
                    await self._hub.publish(event)
                    # Observers only, and guarded as such: a logging or recording
                    # failure must not take the board's worker down with it.
                    try:
                        self._log_read(event, frame, previous, received)
                        if self._on_event is not None:
                            self._on_event(event, frame)
                    except Exception:
                        log.exception(
                            "post-publish observer failed (board %s, seq %d)",
                            board_id,
                            frame.seq,
                        )
                finally:
                    if inbox.empty():
                        idle.set()
        finally:
            idle.set()  # a stopped worker is never busy — never strand a drain()

    def _log_read(
        self,
        event: PositionEvent,
        frame: Frame,
        previous: PositionEvent | None,
        received: float,
    ) -> None:
        """One line per published read — INFO when the board changed, DEBUG otherwise.

        This is the trial log an operator tails: which placement the stream
        settled on, how long this node took, and what the prior decoder had to
        repair. ``proc`` is receive→publish on this node's own clock — never
        capture→publish, which subtracts the *edge's* clock (a phone off NTP)
        from ours and would print nonsense in the one log meant to diagnose lag.
        "Changed" is judged on the *repaired* placement, because that is what a
        viewer displays; a raw argmax flickering between impossible boards that
        repair to the same position is a DEBUG detail, not news.
        """
        repaired = event.prior.placement if event.prior else event.placement
        before = (
            (previous.prior.placement if previous.prior else previous.placement)
            if previous
            else None
        )
        changed = repaired != before
        flips = event.prior.flips if event.prior else ()
        residual = event.prior.residual if event.prior else {}
        log.log(
            logging.INFO if changed else logging.DEBUG,
            "read %s seq=%d conf=%.3f proc=%.2fs %s%s%s",
            event.board_id,
            frame.seq,
            event.confidence,
            time.time() - received,
            repaired,
            "".join(
                f" [{f.square} {f.from_class}->{f.to_class}]" for f in flips
            ),
            f" residual={residual}" if residual else "",
        )
