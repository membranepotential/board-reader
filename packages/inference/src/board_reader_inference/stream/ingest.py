"""Streaming frame ingress for the gateway (DESIGN.md §8 WP-G1/G2, D4).

The edge forwards frames at an *adaptive* rate (D4): a slow keepalive at rest, a
10–15 fps burst while the position changes. Estimation is synchronous CPU work
(ONNX/numpy), so running it inline in the ``/ingest`` WebSocket handler blocks the
whole event loop for the duration of every frame — viewers stall, HTTP endpoints
stall, and the ingest socket backs up behind the model.

:class:`FrameIngest` is the seam that decouples the two:

* **The read scheduler (D3/D4).** Not every frame is worth ~0.2 GPU-seconds
  (ChessQ fp32 on CUDA). A :class:`~.schedule.seam.FramePolicy`
  — by default "read at the end of a move, plus a keepalive floor at rest" —
  decides in :meth:`FrameIngest.submit`, on every arriving frame, in arrival order:
  exactly the ``observe`` then ``due`` that replay asks per recorded frame. The
  declined frames cost nothing beyond the decode that already happened. The event
  stream is therefore SPARSE by design: consumers already tolerate that (nothing
  downstream assumes a per-frame cadence), but tests that count published events
  must not.
* **G1 — the read queue.** A scheduled frame joins one queue shared by every
  board, oldest first, together with everything its read depends on besides the
  pixels: the board's held frames (:meth:`~.backdate.FrameRing.snapshot`) and its
  ``rules``, both taken at arrival. So which frames a board reads, and what each
  read settles and back-dates to, depend on the board's own frames alone, never
  on how far behind the recognizer runs or how many boards share it: load and
  input spikes add latency, never a different result. Oldest first keeps each
  board's reads in its arrival order for free and makes the worst wait the
  shortest one; no board can starve, because the scheduler bounds each board's
  demand (three burst-end reads per move, one keepalive per 2 s).
* **The overload rule.** Only a sustained overload drops reads, and it says so:
  a keepalive read that waited :data:`MAX_KEEPALIVE_WAIT_S` is dropped instead of
  read, and a board with :data:`READ_BACKLOG` reads waiting sheds its oldest
  keepalive read (a burst-end read only when it has nothing else waiting). Every
  drop is counted on ``/healthz`` (``ingest.dropped``,
  ``ingest.dropped_burst_end``), logged, and recorded (:func:`drop_line`).
* **G2 — one read worker.** One asyncio task takes the next read off the queue
  and runs the recognizer and the board's debounce in a worker thread
  (:func:`asyncio.to_thread`), one read at a time across every board: the event
  loop stays free to ack, serve viewers and answer HTTP, the GPU session runs one
  call at a time (each concurrent call would hold its own activations in the
  session's capped arena), and each board's events publish in its arrival order
  (monotone ``seq``).

**Work between reads.** :meth:`FrameIngest.call` runs other work that needs the
recognizer (Auto-frame locating the board in a full frame, 2 to 8 reads) on the same
worker, in the same queue, oldest first, so the GPU still runs one call at a time;
:meth:`FrameIngest.call_spare` runs skippable work (Auto-frame's look at a preview)
only when no read waits, behind everything else.
:meth:`FrameIngest.publish_read` puts a read made that way into its board's stream:
it joins the queue like a scheduled read and is settled and published in turn, but
not recorded, because its frame never crossed ``/ingest``.

Acks are decoupled from estimation: the gateway acks every frame it *receives*,
including frames the scheduler declines. An ack means "received", not
"estimated" — it is the edge's flow-control signal, not a result.

The worker belongs to the ingest object, not to a connection: a reconnecting edge
never interrupts an estimate in flight, and shutdown is the only thing that stops
it (:meth:`FrameIngest.lifespan`).

What the worker *does* with a frame is not this module's decision: it runs the
:data:`~board_reader_inference.recognition.recognizer.FrameRead` it was built with — the same
read ``POST /estimate`` runs (:func:`~board_reader_inference.recognition.recognizer.frame_reader`).
What it does with the *read* is: the :class:`OrientationGuard` drops a read whose
board points the wrong way, and the rest are debounced and published.

**Motion updates.** :meth:`FrameIngest.submit` also turns every arriving frame's
``stability`` into a :class:`~board_reader_core.protocol.MotionUpdate` on the hub
(:class:`MotionThrottle` keeps it to about five a second per board), so a viewer
can show a hand on the board between the sparse reads. Display state only: it is
not a read, so it is neither recorded nor replayed.

**The recorded line.** :func:`event_line` turns a published event into its
``events.ndjson`` line. It lives on the read path, not in the gateway, because
the live recorder and the replay harness (:mod:`~board_reader_inference.measurement`)
must write that line with the same code.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import statistics
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict, dataclass, replace

from board_reader_core.control import DEFAULT_RULES, Rules
from board_reader_core.protocol import MotionUpdate, PositionEvent, iso_utc, square_evidence
from board_reader_core.wire import Frame

from board_reader_inference.recognition.recognizer import FrameRead, FrameReading
from board_reader_inference.stream.backdate import ChangeClock, FrameRing, HeldFrame
from board_reader_inference.stream.hub import PositionHub
from board_reader_inference.stream.schedule.constants import DUE_EPS, ENTER_BURST
from board_reader_inference.stream.schedule.policies import EveryFrame
from board_reader_inference.stream.schedule.seam import FramePolicy, ReadKind, stability_of
from board_reader_inference.stream.stabilize import (
    DEFAULT_SETTLE,
    BoardStabilizer,
    SettleConfig,
)

log = logging.getLogger(__name__)


def event_line(
    event: PositionEvent, frame: Frame, *, published_ts: float | None, rules: Rules
) -> dict:
    """One published event as its ``events.ndjson`` line.

    ``square_dist`` is stripped — replaying ``frames.ndjson`` through the same
    estimator reproduces it, and at burst rate it is ~25 kB per line. What replay
    cannot cheaply give back is *why* a square read the way it did, so the top-2
    digest (:func:`square_evidence`, ~1.8 kB) rides along instead: it is what
    turns "d8 read k" into "d8 read k at 0.51 over q at 0.49", and it is the
    whole evidence base for the trial report's margin section. Its first class
    per square is the argmax, so the line holds both boards: ``placement`` is the
    settled one, and the read's own board is the digest's top row
    (:func:`~board_reader_inference.measurement.trial.events.read_events` derives it).

    The frame's own ``seq`` and ``capture_ts`` ride along too, so an event can be
    joined back to the exact recorded frame it was estimated from, and so does
    ``rules``, the board's setting when this read settled, so a replay can tell
    which mode a live run used (``replay --rules``). A line without the key
    predates the setting and settled under ``standard``.

    ``published_ts`` is this node's wall clock at publish, and ``None`` omits it:
    an offline replay (:mod:`board_reader_inference.measurement.replay`) never published
    anything, and stamping it with either the replay's wall clock or the capture
    time would put a fabricated capture->publish latency in the trial report. A
    line without the key scores identically and reports "not stamped".

    Shared with the replay harness on purpose: a replayed events.ndjson is only
    worth comparing to a recorded one if the same code wrote both lines.
    """
    line = event.to_dict()
    line.pop("square_dist", None)
    line["evidence"] = square_evidence(event.square_dist)
    line["frame_seq"] = frame.seq
    line["capture_ts"] = frame.capture_ts
    line["rules"] = rules
    if published_ts is not None:
        line["published_ts"] = published_ts
    return line


def drop_line(
    frame: Frame, kind: ReadKind, *, rules: Rules, waited_s: float, published_ts: float
) -> dict:
    """One scheduled read the queue dropped, as its ``events.ndjson`` line.

    Into the events file so that a recorded run shows where load changed its
    result. ``type`` marks it: it has no ``seq`` or ``placement``, so the trial
    scorer (:func:`~board_reader_inference.measurement.trial.events.read_events`) skips it,
    and it carries ``rules`` so :func:`~board_reader_inference.measurement.replay.recorded_rules`
    still reads the run's setting. Replay never drops, so only the live node
    writes it.
    """
    return {
        "type": "read_dropped",
        "board_id": frame.board_id,
        "frame_seq": frame.seq,
        "capture_ts": frame.capture_ts,
        "kind": kind.value,
        "waited_s": round(waited_s, 3),
        "rules": rules,
        "published_ts": published_ts,
    }

#: The most scheduled reads one board may have waiting. Under the shipped policy a
#: board asks for one read per keepalive (2 s) at rest and three in 0.6 s after a
#: move, so a board never comes near this unless the recognizer is overloaded for
#: many seconds. Each waiting read holds its decoded frame (~1.5 MB for the phone's
#: 960 x 520 crop), so this is also the memory bound: ~50 MB per board. A board at
#: the bound sheds its oldest keepalive read, and a burst-end read only when every
#: read it has waiting is one (``frames_dropped_burst_end``).
READ_BACKLOG = 32

#: A keepalive read that waited this long (node clock) is dropped instead of read:
#: the first half of the overload rule. Keepalives say the same thing again, so
#: under a sustained overload they are what the node gives up first, and a node
#: this far behind catches up instead of falling further back. A spike that clears
#: (every board moving at once) waits far less: three boards' burst-end runs are
#: nine reads, ~2 s on the GTX 1650.
MAX_KEEPALIVE_WAIT_S = 5.0

#: The wall-clock backstop under the scheduler's own keepalive, as a multiple of
#: it (4 s under the shipped policy's 2 s keepalive, :data:`~.schedule.default.DEFAULT_POLICY_SPEC`).
#:
#: A policy's floor is measured in the EDGE's ``capture_ts``, on purpose: that is
#: what makes an offline replay deterministic (a scheduler that consulted
#: ``time.time()`` would pick different frames on every run). Live, that clock
#: belongs to a phone, and a floor built on a stopped clock is not a floor — a
#: stalled camera pipeline restamping its last timestamp, or a clock stepping
#: backwards, would leave the board unread forever. So :meth:`FrameIngest.submit`
#: keeps the same promise one layer down, on the node's own monotonic clock at
#: arrival (never at read time: a slow recognizer is not a stalled edge clock).
#:
#: The 2x is what keeps this a *diagnostic*. Wall time and capture time run at the
#: same rate on a healthy stream, so a backstop at exactly ``keepalive_s`` would
#: also fire whenever the network delayed one frame a little — and
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
    """The two numbers a :class:`~.schedule.seam.FramePolicy` reads, off a live frame.

    An explicit adapter, not a duck-type. The policies were written against a
    recorded :class:`~board_reader_core.recording.FrameEntry`, whose ``header`` is
    a *dict*; a live :class:`~board_reader_core.wire.Frame`'s ``header`` is a
    *method*, and its ``stability`` is an ordinary attribute that a hand-rolled
    edge can leave as ``None`` or a NaN. Converting here — through the same
    :func:`~.schedule.seam.stability_of` coercion the recorded path uses — is what
    makes "one policy, two callers" true rather than approximately true.
    """
    return frame.capture_ts, stability_of(frame.stability)


def _settle_read(
    reading: FrameReading,
    frame: Frame,
    stabilizer: BoardStabilizer,
    guard: OrientationGuard,
    rules: Rules,
    clock: ChangeClock,
    held: Sequence[HeldFrame],
) -> FrameReading | Misoriented:
    """One frame's read through its board's guard, debounce and change clock.

    The live worker and replay both call this on every read, in order. The
    :class:`OrientationGuard` sees the read first: a dropped read never
    reaches the debounce, so a misread board cannot start settling.

    The raw read then feeds the board's :class:`.stabilize.BoardStabilizer`, and
    the event is published with the debounced board as its ``placement``
    (:meth:`PositionEvent.with_placement`; ``square_dist`` untouched, so the raw
    board stays its argmax) and the time it last changed as ``changed_utc``: the
    stabilizer's :attr:`~.stabilize.BoardStabilizer.changed_ts`, back-dated by the
    board's :class:`~.backdate.ChangeClock` over ``held``, the board's held frames
    (:meth:`~.backdate.FrameRing.snapshot`, taken when the frame arrived) — safe here
    despite the worker thread because the read worker estimates strictly one
    frame at a time, each board's in arrival order, which is exactly the stream a
    debounce needs. The
    guard and the clock rely on the same one-at-a-time order. ``rules`` is the
    board's setting for this read. A read whose frame energy the stabilizer's
    config calls obscured (:meth:`~.stabilize.SettleConfig.obscures`) commits
    only balanced piece groups and is published with ``obscured`` set.
    """
    dropped = guard.check(frame.seq, reading.direction)
    if dropped is not None:
        return dropped
    event = reading.event
    obscured = stabilizer.config.obscures(reading.frame_energy)
    settled = stabilizer.observe(event.placement, frame.capture_ts, rules, obscured=obscured)
    changed_ts = stabilizer.changed_ts
    assert changed_ts is not None  # observe() has just seen a read
    changed = clock.observe(
        frame.seq, frame.capture_ts, event.placement, settled, changed_ts, held
    )
    published = event.with_placement(
        settled, changed_utc=iso_utc(changed), obscured=obscured
    )
    return replace(reading, event=published)


@dataclass(eq=False, slots=True)
class _Board:
    """One board's stream state. Only the read worker touches the read-side state
    (stabilizer, guard, clock), one read at a time, in the board's arrival order."""

    #: The board's read scheduler. Per board for the same reason as the stabilizer:
    #: burst state describes *this* camera's stream.
    policy: FramePolicy
    #: The board's recent frames as small grey copies, pushed by
    #: :meth:`FrameIngest.submit` for every arriving frame, read or not.
    ring: FrameRing
    #: The board's debounce state (the published ``placement``). A board that went
    #: away and came back (after :meth:`FrameIngest.close`) starts settling afresh.
    stabilizer: BoardStabilizer
    #: The board's orientation guard, per board like the stabilizer.
    guard: OrientationGuard
    #: Back-dates the board's ``changed_utc`` over the ring, per board like the stabilizer.
    clock: ChangeClock
    #: The board's motion-update rate limit (display only).
    motion: MotionThrottle
    #: Set while none of the board's reads is waiting or running (:meth:`FrameIngest.drain`).
    idle: asyncio.Event
    #: The node's monotonic clock at the board's last scheduled read, for the
    #: wall-clock backstop (:data:`BACKSTOP_SLACK`).
    last_scheduled: float
    #: The capture time of the board's latest queued read: the stabilizer sees each
    #: board's reads in capture order, so a read made elsewhere may not go behind it.
    last_read_ts: float = -math.inf
    #: The board's reads in the queue, not counting one that is running.
    waiting: int = 0
    #: Whether one of the board's reads is running.
    busy: bool = False

    def settle_if_idle(self) -> None:
        if not self.waiting and not self.busy:
            self.idle.set()


@dataclass(frozen=True, slots=True)
class _ScheduledRead:
    """A frame the policy chose to read, with what its read needs besides pixels.

    ``held`` and ``rules`` are taken when the frame arrives, as replay takes them
    per recorded frame, so a read that waits behind other reads still back-dates
    over the frames held at its own arrival and settles under the setting in
    force then.
    """

    board: _Board
    frame: Frame
    kind: ReadKind
    held: tuple[HeldFrame, ...]
    rules: Rules
    #: The node's monotonic clock when the read joined the queue.
    queued: float
    #: A read made elsewhere (:meth:`FrameIngest.publish_read`): settled and
    #: published, but not read again and not recorded. ``None`` reads ``frame``.
    made: FrameReading | None = None


@dataclass(frozen=True, slots=True)
class Busy:
    """Reads were waiting, so spare work was skipped (:meth:`FrameIngest.call_spare`)."""


@dataclass(frozen=True, slots=True)
class _Work:
    """Other recognizer work (:meth:`FrameIngest.call`) waiting its turn."""

    #: Runs on the worker thread; returns what hands its result over on the loop.
    run: Callable[[], Callable[[], None]]
    #: Gives up on it: the worker stopped first.
    cancel: Callable[[], None]
    #: Spare work (:meth:`FrameIngest.call_spare`): it waits behind everything else.
    spare: bool = False


@dataclass(frozen=True, slots=True)
class ReadQueueStats:
    """The read queue as ``/healthz`` shows it (``ingest.queue``)."""

    #: Reads waiting now, not counting the one running.
    waiting: int
    #: The most reads ever waiting at once.
    max_waiting: int
    #: Reads the recognizer has started.
    served: int
    #: Seconds a read waited for the recognizer, mean and worst, over ``served``.
    wait_mean_s: float
    wait_max_s: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class FrameIngest:
    """Scheduled, queued frame ingress: :meth:`submit` never blocks on the model.

    ``read`` is the gateway's one read (module docstring).
    """

    def __init__(
        self,
        read: FrameRead,
        hub: PositionHub,
        *,
        on_event: Callable[[FrameReading, Frame, Rules], None] | None = None,
        on_drop: Callable[[Frame, ReadKind, Rules, float], None] | None = None,
        policy: Callable[[], FramePolicy] = EveryFrame,
        settle: SettleConfig = DEFAULT_SETTLE,
        rules: Callable[[str], Rules] = lambda _board_id: DEFAULT_RULES,
    ) -> None:
        self._read = read
        self._hub = hub
        #: Builds this board's read scheduler — a *factory*, because a policy is
        #: stateful and each board owns its own (and gets a fresh one after
        #: :meth:`close`). Defaults to reading everything, which is what the
        #: gateway did before scheduling existed; ``serve`` supplies the shipped
        #: burst-end policy through ``BR_SCHEDULE`` (see
        #: :data:`~.schedule.default.DEFAULT_POLICY_SPEC`).
        self._policy = policy
        #: Each board's debounce is built with these knobs (``BR_SETTLE_*``).
        self._settle = settle
        #: Which chess a board plays, asked on the event loop for each scheduled
        #: frame as it arrives — ``serve`` answers from the camera registry
        #: (``/control``), so a change there applies from the board's next frame.
        self._rules = rules
        #: Called (on the event loop) for every published read, with the frame it
        #: was estimated from and the setting it settled under — the gateway hangs
        #: its trial log and the camera's direction off this.
        self._on_event = on_event
        #: Called (on the event loop) for every scheduled read the queue dropped,
        #: with its kind, rules and seconds waited — the gateway records it.
        self._on_drop = on_drop
        self._boards: dict[str, _Board] = {}
        #: Every board's scheduled reads and the other work, oldest first (module
        #: docstring, G1).
        self._queue: deque[_ScheduledRead | _Work] = deque()
        #: The one read worker and its wake-up, made on the first scheduled read
        #: after construction or :meth:`close`: the gateway is built before its
        #: event loop runs.
        self._worker: asyncio.Task[None] | None = None
        self._ready: asyncio.Event | None = None
        self._warned_backstop: set[str] = set()
        self._max_waiting = 0
        self._served = 0
        self._wait_total_s = 0.0
        self._wait_max_s = 0.0
        self.frames_received = 0
        self.frames_estimated = 0
        #: Scheduled, then dropped by the overload rule (:data:`READ_BACKLOG`,
        #: :data:`MAX_KEEPALIVE_WAIT_S`): the recognizer fell that far behind.
        self.frames_dropped = 0
        #: Of those, burst-end reads: only when a board has :data:`READ_BACKLOG`
        #: of them waiting. Should be 0 on any real deployment.
        self.frames_dropped_burst_end = 0
        self.frames_failed = 0
        #: Not worth an estimate (the scheduler declined them). At the shipped
        #: policy this is the *majority* of a live stream and is the whole point:
        #: ~12x fewer reads than every-frame on the swept takes.
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
        #: Scheduled, then dropped because their board was forgotten
        #: (:meth:`forget`) before the recognizer got to them.
        self.frames_forgotten = 0

    def submit(self, frame: Frame) -> None:
        """Schedule ``frame`` for its board and return immediately.

        Called from the event-loop thread (the ``/ingest`` handler), for every
        arriving frame, in arrival order. The frame goes to the board's ring and
        its policy, which decides here, on this frame: replay asks the same
        ``observe`` then ``due`` per recorded frame, so the live node reads the
        frames replay reads. A scheduled frame joins the read queue with its ring
        snapshot and rules (:class:`_ScheduledRead`); a declined one is done.
        Header-only apart from the ring's grey copy: nothing here waits on a read.
        """
        self.frames_received += 1
        board = self._boards.get(frame.board_id)
        if board is None:
            board = self._boards[frame.board_id] = self._new_board()
        capture_ts, stability = _schedule_signal(frame)
        board.policy.observe(capture_ts, stability)
        board.ring.push(frame.seq, capture_ts, frame.image)
        # The scheduler's own burst threshold, which the UI mirrors as MOVING_BELOW.
        moving = stability < ENTER_BURST
        now = time.monotonic()
        if board.motion.due(moving, now):
            self._hub.broadcast(
                MotionUpdate(
                    board_id=frame.board_id, moving=moving, stability=stability, ts=capture_ts
                )
            )
        kind = board.policy.due(capture_ts, stability)
        if kind is None and self._backstop(frame.board_id, board, now):
            kind = ReadKind.KEEPALIVE
        if kind is None:
            self.frames_unscheduled += 1
            return
        board.last_scheduled = now
        self._enqueue(board, frame, kind, now, made=None)

    def publish_read(self, frame: Frame, reading: FrameReading) -> bool:
        """Settle and publish ``reading``, made elsewhere, as ``frame``'s board's next
        read (Auto-frame's last read of a full frame); ``False`` drops it.

        It joins the queue like a scheduled read (its board's ring as it stands, its
        rules now), so the board's stabilizer still sees one read at a time in
        capture order. A read no newer than the board's latest queued one is
        dropped: the stream already says more. Never recorded and never shown to
        ``on_event``: its frame did not come through ``/ingest``, so no recording
        holds it. ``frame.seq`` should be the board's latest seen, so the published
        seqs never go back.
        """
        board = self._boards.get(frame.board_id)
        if board is None:
            board = self._boards[frame.board_id] = self._new_board()
        if frame.capture_ts <= board.last_read_ts:
            return False
        self._enqueue(board, frame, ReadKind.BURST_END, time.monotonic(), made=reading)
        return True

    def _enqueue(
        self, board: _Board, frame: Frame, kind: ReadKind, now: float, *, made: FrameReading | None
    ) -> None:
        if board.waiting >= READ_BACKLOG:
            self._shed(board)
        self._queue.append(
            _ScheduledRead(
                board=board,
                frame=frame,
                kind=kind,
                held=board.ring.snapshot(),
                rules=self._rules(frame.board_id),
                queued=now,
                made=made,
            )
        )
        board.waiting += 1
        board.last_read_ts = max(board.last_read_ts, frame.capture_ts)
        board.idle.clear()
        self._max_waiting = max(self._max_waiting, len(self._queue))
        self._wake()

    async def call[T](self, work: Callable[[], T]) -> T:
        """Run ``work`` on the read worker thread, in its turn in the read queue, and
        return its result (or raise its exception).

        For other work that needs the recognizer, so that the GPU still runs one
        call at a time: it waits behind every read queued before it, and the reads
        queued after it wait for it. ``CancelledError`` when the worker stops first.
        """
        return await self._call(work, spare=False)

    async def call_spare[T](self, work: Callable[[], T]) -> T | Busy:
        """Run ``work`` on the read worker only when it has nothing else to do;
        :class:`Busy` (and ``work`` never runs) when a read is waiting right now.

        For work that may be skipped, Auto-frame's look at a preview: it never
        delays a read, because every read and every :meth:`call` queued after it
        still goes first. Raises what ``work`` raises, like :meth:`call`.
        """
        if any(isinstance(item, _ScheduledRead) for item in self._queue):
            return Busy()
        return await self._call(work, spare=True)

    async def _call[T](self, work: Callable[[], T], *, spare: bool) -> T:
        done: asyncio.Future[T] = asyncio.get_running_loop().create_future()

        def run() -> Callable[[], None]:
            try:
                result = work()
            except Exception as e:  # handed to the caller
                failed = e
                return lambda: None if done.done() else done.set_exception(failed)
            return lambda: None if done.done() else done.set_result(result)

        self._queue.append(_Work(run=run, cancel=done.cancel, spare=spare))
        self._max_waiting = max(self._max_waiting, len(self._queue))
        self._wake()
        return await done

    def _backstop(self, board_id: str, board: _Board, now: float) -> bool:
        """Whether to read a frame the policy declined, on the node's own clock.

        Arrival time, not read time: the backstop asks whether the EDGE's clock
        stalled, and a slow recognizer must not answer that question for it.
        """
        silent = now - board.last_scheduled
        if silent < BACKSTOP_SLACK * board.policy.keepalive_s - DUE_EPS:
            return False
        self.frames_backstopped += 1
        if board_id not in self._warned_backstop:
            self._warned_backstop.add(board_id)
            log.warning(
                "board %s: %.1fs with no read while frames keep arriving — this "
                "edge's capture_ts is stalled or stepping backwards, so the "
                "scheduler's own floor cannot fire. Reading on the node's clock "
                "instead; see /healthz ingest.backstopped for the count.",
                board_id, silent,
            )
        return True

    def _shed(self, board: _Board) -> None:
        """Make room in ``board``'s share of the queue: drop its oldest keepalive
        read, or, when every read it has waiting is a burst end, its oldest."""
        mine = [s for s in self._queue if isinstance(s, _ScheduledRead) and s.board is board]
        victim = next((s for s in mine if s.kind is ReadKind.KEEPALIVE), mine[0])
        self._queue.remove(victim)
        self._drop(victim, time.monotonic() - victim.queued)

    def _drop(self, scheduled: _ScheduledRead, waited: float) -> None:
        """Count, log and report one dropped read; its board's result may now differ."""
        board, frame = scheduled.board, scheduled.frame
        board.waiting -= 1
        board.settle_if_idle()
        self.frames_dropped += 1
        if scheduled.kind is ReadKind.BURST_END:
            self.frames_dropped_burst_end += 1
        log.warning(
            "board %s: dropped the %s read of seq %d after %.1fs in the queue — the "
            "recognizer cannot keep up with this many boards; see /healthz "
            "ingest.dropped",
            frame.board_id, scheduled.kind.value, frame.seq, waited,
        )
        if self._on_drop is not None:
            try:
                self._on_drop(frame, scheduled.kind, scheduled.rules, waited)
            except Exception:
                log.exception("drop observer failed (board %s, seq %d)", frame.board_id, frame.seq)

    def queue_stats(self) -> ReadQueueStats:
        """The read queue's depth and waits, for ``/healthz``."""
        return ReadQueueStats(
            waiting=len(self._queue),
            max_waiting=self._max_waiting,
            served=self._served,
            wait_mean_s=self._wait_total_s / self._served if self._served else 0.0,
            wait_max_s=self._wait_max_s,
        )

    async def drain(self, *board_ids: str) -> None:
        """Wait until the named boards (default: all) have nothing left to estimate.

        :meth:`close` releases every waiter, so this never outlives it.
        """
        if board_ids:
            boards = [self._boards[b] for b in board_ids if b in self._boards]
        else:
            boards = list(self._boards.values())
        for board in boards:
            await board.idle.wait()

    def forget(self, board_id: str) -> None:
        """Release one board's stream state: its scheduler, ring, stabilizer, guard
        and change clock, and its queued reads (counted in ``frames_forgotten``).

        For a camera /control forgot: the node stops holding ~2 MB per camera ever
        seen, and the device, should it come back, is a new camera that starts
        settling afresh. A read of the board already running finishes but is not
        published (:meth:`_serve`). A frame that arrives later starts a new board.
        """
        board = self._boards.pop(board_id, None)
        if board is None:
            return
        kept = deque(
            item
            for item in self._queue
            if not (isinstance(item, _ScheduledRead) and item.board is board)
        )
        self.frames_forgotten += len(self._queue) - len(kept)
        self._queue = kept
        board.waiting = 0
        self._warned_backstop.discard(board_id)
        board.settle_if_idle()

    async def close(self) -> None:
        """Stop the read worker and forget the boards and their queued reads.
        Idempotent; the next frame starts afresh.

        Prompt by design: a read already running in its thread is abandoned rather
        than awaited (the thread finishes on its own; its result is simply never
        published).
        """
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        for item in self._queue:
            if isinstance(item, _Work):
                item.cancel()
        self._queue.clear()
        boards = list(self._boards.values())
        self._boards.clear()
        for board in boards:
            board.idle.set()  # never strand a drain()

    @contextlib.asynccontextmanager
    async def lifespan(self, _app: object) -> AsyncIterator[None]:
        """ASGI lifespan hook — stops the worker when the gateway shuts down."""
        yield
        await self.close()

    def _new_board(self) -> _Board:
        idle = asyncio.Event()
        idle.set()
        return _Board(
            policy=self._policy(),
            ring=FrameRing(),
            stabilizer=BoardStabilizer(self._settle),
            guard=OrientationGuard(),
            clock=ChangeClock(),
            motion=MotionThrottle(),
            idle=idle,
            last_scheduled=time.monotonic(),
        )

    def _wake(self) -> None:
        if self._worker is None or self._worker.done():
            self._ready = asyncio.Event()
            self._worker = asyncio.create_task(self._serve(self._ready), name="ingest-reads")
        assert self._ready is not None
        self._ready.set()

    def _take(self) -> _ScheduledRead | _Work:
        """The oldest item waiting that is not spare work, else the oldest spare."""
        item = next(
            (i for i in self._queue if not (isinstance(i, _Work) and i.spare)), self._queue[0]
        )
        self._queue.remove(item)
        return item

    async def _next(self, ready: asyncio.Event) -> _ScheduledRead | _Work:
        """The read (or other work) the recognizer runs next: the oldest one waiting.

        The one place that takes reads off the queue, so batching would go here
        (take every waiting read, pad to a fixed batch for the GPU arena rule). A
        keepalive read that has waited :data:`MAX_KEEPALIVE_WAIT_S` is dropped
        instead: the overload rule's second half.
        """
        while True:
            while self._queue:
                scheduled = self._take()
                if isinstance(scheduled, _Work):
                    return scheduled
                waited = time.monotonic() - scheduled.queued
                if scheduled.kind is ReadKind.KEEPALIVE and waited > MAX_KEEPALIVE_WAIT_S:
                    self._drop(scheduled, waited)
                    continue
                scheduled.board.waiting -= 1
                scheduled.board.busy = True
                self._served += 1
                self._wait_total_s += waited
                self._wait_max_s = max(self._wait_max_s, waited)
                return scheduled
            ready.clear()
            await ready.wait()

    async def _serve(self, ready: asyncio.Event) -> None:
        """Estimate every scheduled read, forever, one at a time, oldest first."""
        while True:
            scheduled = await self._next(ready)
            if isinstance(scheduled, _Work):
                try:
                    deliver = await asyncio.to_thread(scheduled.run)
                except asyncio.CancelledError:
                    scheduled.cancel()
                    raise
                deliver()
                continue
            board, frame = scheduled.board, scheduled.frame
            try:
                outcome = await asyncio.to_thread(self._read_and_settle, scheduled)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A frame the model chokes on must not end the board's stream.
                self.frames_failed += 1
                log.exception("estimation failed (board %s, seq %d)", frame.board_id, frame.seq)
            else:
                # A board forgotten while its read ran publishes nothing: the event
                # would put the forgotten camera back on the live page.
                if self._boards.get(frame.board_id) is board:
                    await self._publish(scheduled, outcome)
            finally:
                board.busy = False
                board.settle_if_idle()

    def _read_and_settle(self, scheduled: _ScheduledRead) -> FrameReading | Misoriented:
        """The read and its debounce, on the worker thread."""
        board = scheduled.board
        reading = self._read(scheduled.frame) if scheduled.made is None else scheduled.made
        return _settle_read(
            reading, scheduled.frame, board.stabilizer, board.guard,
            scheduled.rules, board.clock, scheduled.held,
        )

    async def _publish(
        self, scheduled: _ScheduledRead, outcome: FrameReading | Misoriented
    ) -> None:
        frame = scheduled.frame
        if isinstance(outcome, Misoriented):
            self.frames_misoriented += 1
            log.warning(
                "board %s seq %d: dropped a read pointing %.0f degrees, "
                "%.0f from the board's running %.0f — almost certainly "
                "a rotated misread; see /healthz ingest.misoriented",
                frame.board_id, outcome.seq, outcome.direction,
                abs(_wrap_deg(outcome.direction - outcome.running)),
                outcome.running,
            )
            return
        self.frames_estimated += 1
        event = outcome.event
        previous = self._hub.latest(frame.board_id)
        await self._hub.publish(event)
        # Observers only, and guarded as such: a logging or recording failure must
        # not take the read worker down with it.
        try:
            self._log_read(event, frame, previous, scheduled.queued)
            if self._on_event is not None and scheduled.made is None:
                self._on_event(outcome, frame, scheduled.rules)
        except Exception:
            log.exception(
                "post-publish observer failed (board %s, seq %d)", frame.board_id, frame.seq
            )

    def _log_read(
        self,
        event: PositionEvent,
        frame: Frame,
        previous: PositionEvent | None,
        queued: float,
    ) -> None:
        """One line per published read — INFO when the board changed, DEBUG otherwise.

        This is the trial log an operator tails: which placement the stream
        settled on and how long this node took. ``proc`` is queue→publish on
        this node's own clock — never capture→publish, which subtracts the
        *edge's* clock (a phone off NTP) from ours and would print nonsense in the
        one log meant to diagnose lag. "Changed" is judged on the *settled*
        placement, because that is what a viewer displays; a read flickering
        between boards that debounce to the same position is a DEBUG detail, not
        news.
        """

        shown = event.placement
        before = previous.placement if previous else None
        log.log(
            logging.INFO if shown != before else logging.DEBUG,
            "read %s seq=%d conf=%.3f proc=%.2fs %s",
            event.board_id,
            frame.seq,
            event.confidence,
            time.monotonic() - queued,
            shown,
        )
