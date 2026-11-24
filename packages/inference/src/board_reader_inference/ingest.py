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
* **The read scheduler (D3/D4).** Not every frame the worker picks is worth ~0.2
  GPU-seconds (ChessQ fp32 on CUDA). A :class:`~board_reader_inference.schedule.FramePolicy` — by default
  "read at the end of a move, plus a keepalive floor at rest" — decides, and the
  declined frames cost nothing beyond the decode that already happened. It is
  *observed* on every arriving frame in :meth:`FrameIngest.submit` and *asked* in
  the worker, so a burst-end read lands on whichever contender the inbox kept
  rather than on whichever frame happened to arrive at that instant. The event
  stream is therefore SPARSE by design: consumers already tolerate that (nothing
  downstream assumes a per-frame cadence), but tests that count published events
  must not.

Acks are decoupled from estimation: the gateway acks every frame it *receives*,
including frames the pre-filter later drops. An ack means "received", not
"estimated" — it is the edge's flow-control signal, not a result.

The workers belong to the ingest object, not to a connection: a reconnecting edge
never interrupts an estimate in flight, and shutdown is the only thing that stops
them (:meth:`FrameIngest.lifespan`).

What a worker *does* with a frame is not this module's decision: it runs the
:data:`~board_reader_inference.recognizer.FrameRead` it was built with — the same
read ``POST /estimate`` runs (:func:`~board_reader_inference.recognizer.frame_reader`).
What it does with the *read* is: the :class:`OrientationGuard` drops a read whose
board points the wrong way, and the rest are repaired, debounced and published.

**Motion updates.** :meth:`FrameIngest.submit` also turns every arriving frame's
``stability`` into a :class:`~board_reader_core.protocol.MotionUpdate` on the hub
(:class:`MotionThrottle` keeps it to about five a second per board), so a viewer
can show a hand on the board between the sparse reads. Display state only: it is
not a read, so it is neither recorded nor replayed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import statistics
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from board_reader_core.protocol import MotionUpdate, PositionEvent
from board_reader_core.wire import Frame

from board_reader_inference.hub import PositionHub
from board_reader_inference.prior import decode
from board_reader_inference.recognizer import FrameRead, FrameReading
from board_reader_inference.schedule import (
    DUE_EPS,
    ENTER_BURST,
    EveryFrame,
    FramePolicy,
    stability_of,
)
from board_reader_inference.stabilize import DEFAULT_SETTLE, BoardStabilizer, SettleConfig

log = logging.getLogger(__name__)

#: How much older than the incoming frame a sharper queued frame may be and still
#: win the inbox (seconds of edge capture clock). One read period, roughly: any
#: longer and the estimator would be reading a board state a move behind.
SHARPNESS_WINDOW_S = 1.0

#: The wall-clock backstop under the scheduler's own keepalive, as a multiple of
#: it (4 s under the shipped policy's 2 s keepalive, :data:`~.schedule.DEFAULT_POLICY_SPEC`).
#:
#: A policy's floor is measured in the EDGE's ``capture_ts``, on purpose: that is
#: what makes an offline replay deterministic (a scheduler that consulted
#: ``time.time()`` would pick different frames on every run). Live, that clock
#: belongs to a phone, and a floor built on a stopped clock is not a floor — a
#: stalled camera pipeline restamping its last timestamp, or a clock stepping
#: backwards, would leave the board unread forever. So the worker keeps the same
#: promise one layer down, on the node's own monotonic clock.
#:
#: The 2x is what keeps this a *diagnostic*. Wall time and capture time run at the
#: same rate on a healthy stream, so a backstop at exactly ``keepalive_s`` would
#: also fire whenever ingest latency grew a little between two reads — and
#: ``frames_backstopped`` would stop meaning "this edge's clock is broken". At 2x
#: only a genuinely stalled clock trips it, and the worst case is still a bounded
#: gap of two keepalives rather than silence.
BACKSTOP_SLACK = 2.0

#: The most a read's direction may differ from its board's running angle and
#: still be published, in degrees. ChessQ's direction held within 2 degrees over
#: every frame of each recorded take, so a read 45 degrees off is not the camera
#: moving: it is the model reading the board a quarter turn or a half turn wrong.
MAX_TURN_DEG = 45.0

#: The running angle is the circular median of this many recent published reads.
DIRECTION_WINDOW = 9

#: This many dropped reads in a row mean the camera really turned (a phone
#: knocked in its holder), not a misread: the guard re-anchors on them, so a
#: turned camera costs a few reads instead of blinding the board until the
#: stream restarts.
REANCHOR_AFTER = 5

#: Motion updates for one board go out at least this far apart (the node's own
#: clock): at most five a second, whatever the edge's frame rate. A change of
#: ``moving`` always goes out at once.
MOTION_INTERVAL_S = 0.2


def _wrap_deg(deg: float) -> float:
    """``deg`` as the same direction in ``[-180, 180)``."""
    return (deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True, slots=True)
class Misoriented:
    """A read the :class:`OrientationGuard` dropped: its board pointed elsewhere."""

    seq: int
    direction: float
    running: float


class OrientationGuard:
    """Drops a read whose board direction contradicts the board's recent reads.

    One per board stream, like the stabilizer. The running angle is the circular
    median of the last :data:`DIRECTION_WINDOW` published reads; a read more than
    :data:`MAX_TURN_DEG` from it is dropped (:class:`Misoriented`) and does not
    move it. A ``seq`` that does not advance starts the guard afresh, because that
    is a new streaming session (the recorder rolls on the same rule). A read with
    no direction is published unchecked. No clock: replay stays deterministic.
    """

    def __init__(self) -> None:
        self._recent: deque[float] = deque(maxlen=DIRECTION_WINDOW)
        self._dropped: list[float] = []
        self._last_seq: int | None = None

    @property
    def running(self) -> float | None:
        """The board's running angle in degrees, or ``None`` before any read."""
        if not self._recent:
            return None
        ref = self._recent[-1]
        offset = statistics.median(_wrap_deg(a - ref) for a in self._recent)
        return _wrap_deg(ref + offset)

    def check(self, seq: int, direction: float | None) -> Misoriented | None:
        """``None`` publishes the read; a :class:`Misoriented` drops it."""
        if self._last_seq is not None and seq <= self._last_seq:
            self._recent.clear()
            self._dropped.clear()
        self._last_seq = seq
        if direction is None:
            return None
        running = self.running
        if running is None or abs(_wrap_deg(direction - running)) <= MAX_TURN_DEG:
            self._recent.append(direction)
            self._dropped.clear()
            return None
        self._dropped.append(direction)
        if len(self._dropped) < REANCHOR_AFTER:
            return Misoriented(seq=seq, direction=direction, running=running)
        log.warning(
            "board direction re-anchored from %.0f to %.0f degrees after %d "
            "dropped reads in a row: the camera turned",
            running, direction, len(self._dropped),
        )
        self._recent = deque(self._dropped, maxlen=DIRECTION_WINDOW)
        self._dropped = []
        return None


class MotionThrottle:
    """Whether a board's next motion update goes out (module docstring)."""

    def __init__(self) -> None:
        self._moving: bool | None = None
        self._sent_at = -math.inf

    def due(self, moving: bool, now: float) -> bool:
        """``now`` is the node's monotonic clock. True also records the send."""
        if moving == self._moving and now - self._sent_at < MOTION_INTERVAL_S - DUE_EPS:
            return False
        self._moving, self._sent_at = moving, now
        return True


def _schedule_signal(frame: Frame) -> tuple[float, float]:
    """The two numbers a :class:`~.schedule.FramePolicy` reads, off a live frame.

    An explicit adapter, not a duck-type. The policies were written against a
    recorded :class:`~board_reader_core.recording.FrameEntry`, whose ``header`` is
    a *dict*; a live :class:`~board_reader_core.wire.Frame`'s ``header`` is a
    *method*, and its ``stability`` is an ordinary attribute that a hand-rolled
    edge can leave as ``None`` or a NaN. Converting here — through the same
    :func:`~.schedule.stability_of` coercion the recorded path uses — is what
    makes "one policy, two callers" true rather than approximately true.
    """
    return frame.capture_ts, stability_of(frame.stability)


def prefer_frame(
    queued: tuple[Frame, float], incoming: tuple[Frame, float]
) -> tuple[Frame, float]:
    """The inbox entry worth keeping when the estimator is behind.

    Latest-wins, EXCEPT that a measurably sharper queued frame within
    :data:`SHARPNESS_WINDOW_S` beats a blurrier newcomer — motion blur is the one
    thing tracking cannot compensate (the corners follow the board; the pixels
    inside are still smeared), and the edge's per-frame ``sharpness`` says which
    frames are worth a ~1 s read. Frames without the measurement fall back to
    latest-wins, which is what a sharpness-less edge always did.
    """
    fq, fi = queued[0], incoming[0]
    if fq.sharpness is None or fi.sharpness is None:
        return incoming
    if fi.capture_ts - fq.capture_ts > SHARPNESS_WINDOW_S:
        return incoming  # the sharp frame has gone stale: freshness wins again
    return queued if fq.sharpness > fi.sharpness else incoming


def _read_with_prior(
    read: FrameRead, frame: Frame, stabilizer: BoardStabilizer, guard: OrientationGuard
) -> FrameReading | Misoriented:
    """One estimate plus its hard-constraint repair, both off the event loop.

    The :class:`OrientationGuard` sees the read first: a dropped read never
    reaches the repair or the debounce, so a misread board cannot start settling.

    The repair rides the event as :attr:`PositionEvent.prior` (raw ``placement``
    and ``square_dist`` untouched), so every consumer of the stream — the viewer,
    ``GET /boards/{id}``, a recording — sees the same repaired board the HTTP
    endpoints already serve beside their reads. ``quarter_turns`` is deliberately
    not passed: the chess frame is the recognizer's inference, not established,
    and an inferred frame makes the back-rank constraint circular (see
    :mod:`.prior`).

    The repaired read then feeds the board's :class:`.stabilize.BoardStabilizer`,
    and the debounced board rides as :attr:`PositionEvent.settled` — safe here
    despite the worker thread because each board's worker estimates strictly one
    frame at a time, in order, which is exactly the stream a debounce needs. The
    guard relies on the same one-at-a-time order.
    """
    reading = read(frame)
    dropped = guard.check(frame.seq, reading.direction)
    if dropped is not None:
        return dropped
    event = reading.event
    repair = decode(event.square_dist)
    event.prior = repair.to_repair()
    event.settled = stabilizer.observe(repair.placement, frame.capture_ts)
    return reading


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
    #: The board's debounce state (:attr:`PositionEvent.settled`). Lives with the
    #: worker so a restarted board starts settling afresh rather than holding a
    #: board from before the gap.
    stabilizer: BoardStabilizer
    #: The board's read scheduler. Per board and per worker for the same reason as
    #: the stabilizer: burst state describes *this* camera's stream, and a stream
    #: that went away and came back is a new camera, not a continuation.
    policy: FramePolicy
    #: The board's orientation guard, per worker like the stabilizer.
    guard: OrientationGuard
    #: The board's motion-update rate limit (display only).
    motion: MotionThrottle


class FrameIngest:
    """Latest-wins frame ingress: :meth:`submit` never blocks on the model.

    ``read`` is the gateway's one read (module docstring).
    """

    def __init__(
        self,
        read: FrameRead,
        hub: PositionHub,
        *,
        on_event: Callable[[FrameReading, Frame], None] | None = None,
        policy: Callable[[], FramePolicy] = EveryFrame,
        settle: SettleConfig = DEFAULT_SETTLE,
    ) -> None:
        self._read = read
        self._hub = hub
        #: Builds this board's read scheduler — a *factory*, because a policy is
        #: stateful and each board owns its own (and gets a fresh one when its
        #: worker restarts). Defaults to reading everything, which is what the
        #: gateway did before scheduling existed; ``serve`` supplies the shipped
        #: burst-end policy through ``BR_SCHEDULE`` (see
        #: :data:`~.schedule.DEFAULT_POLICY_SPEC`).
        self._policy = policy
        #: Each board's debounce is built with these knobs (``BR_SETTLE_*``).
        self._settle = settle
        #: Called (on the event loop) for every published read, with the frame it
        #: was estimated from — the gateway hangs its trial log and the camera's
        #: direction off this.
        self._on_event = on_event
        self._boards: dict[str, _BoardWorker] = {}
        self._warned_backstop: set[str] = set()
        self.frames_received = 0
        self.frames_estimated = 0
        self.frames_dropped = 0
        self.frames_failed = 0
        #: Reached a worker but were not worth an estimate (the scheduler declined
        #: them). At the shipped policy this is the *majority* of a live stream and
        #: is the whole point: ~12x fewer reads than every-frame on the swept takes.
        self.frames_unscheduled = 0
        #: Read because the wall-clock backstop fired, not because the policy asked
        #: — i.e. the edge's ``capture_ts`` stopped telling the truth. Nonzero here
        #: is a clock problem on the edge, and it is the only place it shows.
        self.frames_backstopped = 0
        #: Incremented by the gateway's ``/ingest`` handler for frames refused at
        #: the wire (malformed header, truncated JPEG) — they never reach
        #: :meth:`submit`, but a stream that is *all* refusals must not be
        #: indistinguishable from one that never sent.
        self.frames_refused = 0
        #: Read, then dropped by the board's :class:`OrientationGuard`: the board
        #: pointed more than :data:`MAX_TURN_DEG` away from its recent reads.
        self.frames_misoriented = 0

    def submit(self, frame: Frame) -> None:
        """Hand ``frame`` to its board's worker and return immediately.

        Called from the event-loop thread (the ``/ingest`` handler). If the board's
        inbox already holds an un-estimated frame, one of the two is dropped —
        the estimator is behind, and (:func:`prefer_frame`) the SHARPEST recent
        board state is the useful one, not blindly the freshest: a handheld phone
        smears half its frames, and at ~1 read/s the read might as well spend
        itself on the crisp one.

        The read scheduler is *observed* here and *asked* in the worker. Every
        frame that lands must reach :meth:`~.schedule.FramePolicy.observe` — burst
        end is only visible from the whole run of frames, and the latest-wins inbox
        would otherwise hide most of them — while the decision belongs where the
        frame is picked, so that a burst-end read spends itself on whichever
        contender :func:`prefer_frame` kept. Header-only and synchronous: it costs
        two float comparisons, nothing that could hold up an ack.
        """
        self.frames_received += 1
        worker = self._boards.get(frame.board_id)
        if worker is None or worker.task.done():
            # No worker yet, or a stopped one (post-shutdown): start a fresh board.
            worker = self._spawn(frame.board_id)
            self._boards[frame.board_id] = worker
        capture_ts, stability = _schedule_signal(frame)
        worker.policy.observe(capture_ts, stability)
        # The scheduler's own burst threshold, which the UI mirrors as MOVING_BELOW.
        moving = stability < ENTER_BURST
        if worker.motion.due(moving, time.monotonic()):
            self._hub.broadcast(
                MotionUpdate(
                    board_id=frame.board_id, moving=moving, stability=stability, ts=capture_ts
                )
            )
        worker.idle.clear()
        entry = (frame, time.time())
        try:
            worker.inbox.put_nowait(entry)
        except asyncio.QueueFull:
            # No await since the put, so the queue is still full and this cannot miss.
            queued = worker.inbox.get_nowait()
            self.frames_dropped += 1
            worker.inbox.put_nowait(prefer_frame(queued, entry))

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
        stabilizer = BoardStabilizer(self._settle)
        policy = self._policy()
        guard = OrientationGuard()
        task = asyncio.create_task(
            self._run(board_id, inbox, idle, stabilizer, policy, guard),
            name=f"ingest:{board_id}",
        )
        return _BoardWorker(
            inbox=inbox,
            idle=idle,
            task=task,
            stabilizer=stabilizer,
            policy=policy,
            guard=guard,
            motion=MotionThrottle(),
        )

    async def _run(
        self,
        board_id: str,
        inbox: asyncio.Queue[tuple[Frame, float]],
        idle: asyncio.Event,
        stabilizer: BoardStabilizer,
        policy: FramePolicy,
        guard: OrientationGuard,
    ) -> None:
        """Estimate whatever is in the inbox, forever, one frame at a time."""
        backstop_s = BACKSTOP_SLACK * policy.keepalive_s
        last_read = time.monotonic()
        try:
            while True:
                frame, received = await inbox.get()
                try:
                    now = time.monotonic()
                    if not policy.due(*_schedule_signal(frame)):
                        if now - last_read < backstop_s - DUE_EPS:
                            self.frames_unscheduled += 1
                            continue
                        self.frames_backstopped += 1
                        if board_id not in self._warned_backstop:
                            self._warned_backstop.add(board_id)
                            log.warning(
                                "board %s: %.1fs with no read while frames keep "
                                "arriving — this edge's capture_ts is stalled or "
                                "stepping backwards, so the scheduler's own floor "
                                "cannot fire. Reading on the node's clock instead;"
                                " see /healthz ingest.backstopped for the count.",
                                board_id, now - last_read,
                            )
                    last_read = now
                    outcome = await asyncio.to_thread(
                        _read_with_prior, self._read, frame, stabilizer, guard
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A frame the model chokes on must not end the board's stream.
                    self.frames_failed += 1
                    log.exception(
                        "estimation failed (board %s, seq %d)", board_id, frame.seq
                    )
                else:
                    if isinstance(outcome, Misoriented):
                        self.frames_misoriented += 1
                        log.warning(
                            "board %s seq %d: dropped a read pointing %.0f degrees, "
                            "%.0f from the board's running %.0f — almost certainly "
                            "a rotated misread; see /healthz ingest.misoriented",
                            board_id, outcome.seq, outcome.direction,
                            abs(_wrap_deg(outcome.direction - outcome.running)),
                            outcome.running,
                        )
                        continue
                    self.frames_estimated += 1
                    event = outcome.event
                    previous = self._hub.latest(board_id)
                    await self._hub.publish(event)
                    # Observers only, and guarded as such: a logging or recording
                    # failure must not take the board's worker down with it.
                    try:
                        self._log_read(event, frame, previous, received)
                        if self._on_event is not None:
                            self._on_event(outcome, frame)
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
        "Changed" is judged on the *settled* placement, because that is what a
        viewer displays; a read flickering between boards that debounce to the
        same position is a DEBUG detail, not news.
        """

        def displayed(ev: PositionEvent) -> str:
            if ev.settled is not None:
                return ev.settled.placement
            return ev.prior.placement if ev.prior else ev.placement

        shown = displayed(event)
        before = displayed(previous) if previous else None
        changed = shown != before
        flips = event.prior.flips if event.prior else ()
        residual = event.prior.residual if event.prior else {}
        log.log(
            logging.INFO if changed else logging.DEBUG,
            "read %s seq=%d conf=%.3f proc=%.2fs %s%s%s",
            event.board_id,
            frame.seq,
            event.confidence,
            time.time() - received,
            shown,
            "".join(
                f" [{f.square} {f.from_class}->{f.to_class}]" for f in flips
            ),
            f" residual={residual}" if residual else "",
        )
