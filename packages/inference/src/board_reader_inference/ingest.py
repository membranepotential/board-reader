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

What a worker *does* with a frame is :func:`estimate_frame`, which honours the
optional per-frame board corners the wire may carry — the same routing the
snapshot ``POST /estimate`` path applies.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace

import numpy as np
from board_reader_core.protocol import PositionEvent
from board_reader_core.wire import Frame

from board_reader_inference.estimator import Estimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.squares import rectify_photo

log = logging.getLogger(__name__)

PhotoEstimate = Callable[[Frame, np.ndarray], PositionEvent]


def photo_estimator(est: Estimator) -> PhotoEstimate | None:
    """``est``'s original-photo entry point, if it is calibrated/trained for one.

    Photo-mode is declared either by the classifier (kNN/centroid banks) or by the
    estimator itself (the trained crop classifier). Shared with the gateway's
    ``POST /estimate`` so the streamed and the posted photo can never be routed
    differently.
    """
    photo_mode = (
        getattr(getattr(est, "classifier", None), "crop_mode", None) == "photo"
        or getattr(est, "crop_mode", None) == "photo"
    )
    if not photo_mode:
        return None
    fn = getattr(est, "estimate_photo", None)
    return fn if callable(fn) else None


def estimate_frame(est: Estimator, frame: Frame) -> PositionEvent:
    """Run ``est`` on ``frame``, honouring the frame's optional corners (§4.5).

    Three cases, mirroring ``POST /estimate``:

    * corners + a photo-mode estimator — hand it the original pixels, so it cuts
      tall full-piece boxes instead of decapitating pieces in a warped square;
    * corners + anything else — rectify server-side and estimate the top-down crop,
      which is what a rectified-mode bank was calibrated on (and what the M0 stub's
      centroids assume);
    * no corners — the JPEG *is* the board, exactly as before.
    """
    if frame.corners is None:
        return est.estimate(frame)
    photo_fn = photo_estimator(est)
    if photo_fn is not None:
        return photo_fn(frame, frame.corners)
    rectified = rectify_photo(frame.image, frame.corners)
    return est.estimate(replace(frame, image=rectified, corners=None))


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
    stream must pick the new one up mid-flight. The callable may return ``None`` —
    "no estimator is ready for real positions yet" (the §4.6 uncalibrated state,
    where ``POST /estimate`` answers occupancy-only and publishes nothing). A stream
    has no HTTP response to answer with, so its frames are counted and *skipped*:
    the hub carries positions someone actually estimated, never a placeholder.
    Calibrating (the camera page's button, or a snapshot of the start position)
    hot-swaps a real estimator in and the same stream comes alive mid-flight.
    """

    def __init__(
        self, estimator: Callable[[], Estimator | None], hub: PositionHub
    ) -> None:
        self._estimator = estimator
        self._hub = hub
        self._boards: dict[str, _BoardWorker] = {}
        self._warned_unready = False
        self.frames_received = 0
        self.frames_estimated = 0
        self.frames_dropped = 0
        self.frames_failed = 0
        self.frames_skipped = 0  # arrived while no estimator was ready

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
                    estimator = self._estimator()
                    if estimator is None:
                        self.frames_skipped += 1
                        if not self._warned_unready:
                            self._warned_unready = True
                            log.info(
                                "frames arriving with no estimator ready (board %s):"
                                " skipping until a calibration lands", board_id,
                            )
                        continue
                    event = await asyncio.to_thread(estimate_frame, estimator, frame)
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
