"""The read policies: which arriving frames are worth an estimate (DESIGN D3/D4).

The node used to read every frame that reached it, which is two kinds of waste at
once. A resting board is re-read at stream rate and every read says the same
thing; a board mid-move is read *during* the hand, and those frames are the
project's worst input — a fist over the d-file, a piece in the air over the wrong
square, a shadow across four ranks. The edge already knows the difference: it
meters motion and stamps every frame with a ``stability`` in [0,1]
(:mod:`board_reader_edge.motion`, mirrored in the browser edge's
``lib/edge/motion.ts``). Nothing downstream had ever used it.

A **policy** here is the thing that does: fed the header of each frame in order, it
answers "spend an estimate on this one?". :class:`BurstTriggered` is the shape this
milestone is about — estimate at the **end of a burst**, the move-complete moment,
skip the burst itself, and read a slow **keepalive** at rest. It is what the live
gateway schedules with (``DEFAULT_POLICY_SPEC``, ``BR_SCHEDULE`` to override),
and what :mod:`board_reader_inference.measurement.replay` sweeps offline — one implementation,
because a scheduler measured on recordings and a different one in production is a
sweep that proved nothing.

Live, both halves of the question run on the ``/ingest`` handler, on every
arriving frame in order (:meth:`FramePolicy.observe`, then :meth:`FramePolicy.due`),
exactly as replay asks them per recorded frame; a scheduled frame then waits in
its board's read queue however long the recognizer takes. That is what makes a
board's reads independent of load. See :class:`FramePolicy`.

**The keepalive is a floor, not a nicety** (DESIGN D3: stability is a *hint*, never
a gate). The meter that shipped before 2026-08-27 measured an absolute pixel
difference and, on the deployment phone, read under 0.35 all day — permanently
"moving", never once resting. Everything gated on it was silently dead. A
scheduler that only reads at burst ends would, on that phone, have read the board
exactly never. So every policy in this package reads on a wall of its own: if
``keepalive_s`` of *capture* time has passed without a read, the next frame is
read whatever the meter claims. A stuck meter costs read rate, never reads.

**The interplay with the debounce is the whole design.**
:class:`~board_reader_inference.stream.settle.stabilizer.BoardStabilizer` adopts a changed square
only after :data:`~board_reader_inference.stream.settle.config.SETTLE_MIN_READS` agreeing
reads spanning :data:`~board_reader_inference.stream.settle.config.SETTLE_SECONDS` — a
*time* window, so that it survives the order-of-magnitude swing in read rate. A
scheduler that reads once per move therefore never settles anything: one read is
not a quorum however long you wait. Hence ``burst_end_reads``/``spacing_s``: a
burst end can be worth two or three reads a beat apart, which hands the debounce
its quorum 0.3 s after the burst end instead of never. Sparsity and settling pull
against each other and the pair of knobs is where the trade is made.

The client-side burst thresholds are mirrored here value for value (see
``ENTER_BURST``), because a policy that segments bursts differently from the
edge that forwarded them would be reasoning about a stream nobody sent. The one
deliberate exception is the calm hold: the shipped node ends a burst after 0.3 s of
calm, the phone after its 0.7 s (``EXIT_HOLD_S``), so the node's burst-end run
reads frames the phone is still sending at burst rate. See ``DEFAULT_POLICY_SPEC``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from board_reader_core.recording import FrameEntry

from board_reader_inference.stream.schedule.constants import (
    DEFAULT_KEEPALIVE_S,
    DUE_EPS,
    ENTER_BURST,
    EXIT_BURST,
    EXIT_HOLD_S,
)
from board_reader_inference.stream.schedule.limits import (
    FLOOR_REASON,
    SPACING_REASON,
    finite_positive,
    under_the_floor,
)
from board_reader_inference.stream.schedule.render import canonical_spec
from board_reader_inference.stream.schedule.seam import ReadKind, header_signal


@dataclass(frozen=True, slots=True)
class EveryFrame:
    """Estimate every recorded frame — the densest benchmark, and the control.

    Not what any scheduler should do (live, it fills the read queue as soon as
    the estimator is slower than the stream, and the queue sheds reads); it is the
    row every sparse policy is measured against, since it is the only reading of
    a take that leaves no frame unread.
    """

    PARAMS: ClassVar[Mapping[str, type]] = {}

    @property
    def name(self) -> str:
        return "every-frame"

    @property
    def keepalive_s(self) -> float:
        """Nominal — this policy reads everything, so its floor never binds.

        It exists because the live worker's wall-clock backstop
        (:class:`~board_reader_inference.stream.ingest.FrameIngest`) reads one number off
        whatever policy it was given, and a seam with a hole in it is a seam that
        gets an ``isinstance`` check bolted onto it later.
        """
        return DEFAULT_KEEPALIVE_S

    def observe(self, capture_ts: float, stability: float) -> None:
        return None

    def due(self, capture_ts: float, stability: float) -> ReadKind | None:
        return ReadKind.KEEPALIVE

    def should_estimate(self, entry: FrameEntry) -> bool:
        return True


@dataclass
class BurstTriggered:
    """Estimate at burst END plus a keepalive floor; skip the burst itself.

    The move-complete policy. Motion is tracked with the edge's own hysteresis —
    ``stability < enter`` starts a burst, and only ``exit_hold_s`` of sustained
    ``stability > exit`` ends one — so the frames it picks are the first calm
    frames after a hand left the board, which is precisely the moment the board
    holds a new position and nothing is in front of it.

    Three knobs, in the order they matter:

    ``keepalive_s``
        The floor (DESIGN D3). At most this much recorded time passes without a
        read, *whatever* the meter says — including "0.2 forever", which is what
        the old absolute-scale meter said on the deployment phone all day. This is
        why the policy cannot go blind: a dead meter never leaves a burst, so
        every read on such a stream comes from here.

    ``burst_end_reads`` / ``spacing_s``
        How much evidence a burst end is worth. One read is enough for the raw
        stream but can never move the settled board:
        :class:`~board_reader_inference.stream.settle.stabilizer.BoardStabilizer` wants
        ``SETTLE_MIN_READS`` reads spanning ``SETTLE_SECONDS``, so a single read
        per move leaves the debounce holding the previous position until the
        keepalive slowly catches up. The shipped three reads 0.3 s apart hand
        the 0.3 s window its quorum at the second, 0.3 s after the burst end, and the
        round-3 sweep kept the third (:data:`DEFAULT_POLICY_SPEC`). That needs the edge to keep sending densely
        after its burst ends (the forwarder's tail), or the run waits on keepalive
        frames instead.

    ``enter`` / ``exit`` / ``exit_hold_s``
        The burst hysteresis, defaulted to the client's own constants. Tune
        ``enter``/``exit`` only together with the forwarder — they describe the
        same bursts. ``exit_hold_s`` may be shorter than the edge's hold (the
        shipped spec uses 0.3 s against the phone's 0.7 s), because the edge keeps
        sending densely through its own hold and tail, so a shorter node hold
        still finds the frames its run reads.

    The state is capture-time only and the policy is otherwise pure, so replaying
    a recording twice schedules the identical frames.
    """

    #: Never longer than this without a read. The floor; not negotiable, D3.
    keepalive_s: float = DEFAULT_KEEPALIVE_S
    #: Reads owed by a burst end, ``spacing_s`` apart — the debounce's quorum.
    burst_end_reads: int = 1
    #: Recorded seconds between those reads.
    spacing_s: float = 1.3
    enter: float = ENTER_BURST
    exit: float = EXIT_BURST
    exit_hold_s: float = EXIT_HOLD_S

    PARAMS: ClassVar[Mapping[str, type]] = {
        "keepalive_s": float,
        "burst_end_reads": int,
        "spacing_s": float,
        "enter": float,
        "exit": float,
        "exit_hold_s": float,
    }

    _bursting: bool = field(default=False, init=False, repr=False, compare=False)
    _calm_since: float | None = field(default=None, init=False, repr=False, compare=False)
    _owed: int = field(default=0, init=False, repr=False, compare=False)
    _last_read_ts: float | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        finite_positive("keepalive_s", self.keepalive_s, FLOOR_REASON)
        if self.burst_end_reads < 1:
            raise ValueError(f"burst_end_reads must be >= 1, got {self.burst_end_reads}")
        finite_positive("spacing_s", self.spacing_s, SPACING_REASON)
        under_the_floor("spacing_s", self.spacing_s, self.keepalive_s)
        if not math.isfinite(self.exit_hold_s) or self.exit_hold_s < 0:
            raise ValueError(
                f"exit_hold_s must be >= 0 and finite, got {self.exit_hold_s!r}"
            )
        if not 0.0 <= self.enter <= self.exit <= 1.0:
            raise ValueError(
                f"need 0 <= enter <= exit <= 1, got enter={self.enter} exit={self.exit}; "
                "the gap between them is the hysteresis that keeps one move from "
                "reading as several bursts"
            )

    @property
    def name(self) -> str:
        return canonical_spec(self, "burst-end", self.PARAMS)

    @property
    def bursting(self) -> bool:
        """Whether the policy currently believes a move is in progress."""
        return self._bursting

    def observe(self, capture_ts: float, stability: float) -> None:
        self._track_burst(stability, capture_ts)

    def due(self, capture_ts: float, stability: float) -> ReadKind | None:
        ts = capture_ts
        if self._last_read_ts is not None:
            if ts < self._last_read_ts:
                # The clock stepped backwards — an NTP correction, a spliced
                # session, or two devices sharing one board_id. Re-anchor HERE
                # rather than clearing the anchor: clearing it would make this
                # frame due, and a monotonically *decreasing* clock would then be
                # read every frame — a rate limit that a broken clock switches
                # off is the dead-meter incident again. Anchoring means elapsed
                # clamps to zero and the stream degrades to the floor rate.
                self._last_read_ts = ts
            # While a burst end is still owed reads the interval is the tight one;
            # at rest it is the floor. Either way the read that fires satisfies one
            # owed read, so a keepalive that happens to land first is not wasted.
            interval = self.spacing_s if self._owed else self.keepalive_s
            # Phrased as "is the interval satisfied?" rather than "is it too soon?"
            # so a non-finite capture_ts answers NO and falls back to the floor,
            # instead of losing every comparison and reading every frame.
            if not ts - self._last_read_ts >= interval - DUE_EPS:
                return None
        kind = ReadKind.BURST_END if self._owed else ReadKind.KEEPALIVE
        self._owed = max(self._owed - 1, 0)
        self._last_read_ts = ts
        return kind

    def should_estimate(self, entry: FrameEntry) -> bool:
        ts, stability = header_signal(entry)
        self.observe(ts, stability)
        return self.due(ts, stability) is not None

    def _track_burst(self, stability: float, ts: float) -> None:
        """The forwarder's own ``trackBurst``, on the receiving side of the wire."""
        if not self._bursting:
            if stability < self.enter:
                self._bursting = True
                self._calm_since = None
                # A new move CANCELS whatever the last burst end still owed. Those
                # reads were evidence for a position that no longer exists, and
                # while owed the interval is the tight one — so on two moves closer
                # together than the run is long (the dim take over-segments to 26
                # bursts across 10 moves) the leftovers fire at spacing_s *inside*
                # the next move, feeding the debounce a quorum of hand-over-board
                # frames that disagree by construction. The new burst end will owe
                # a fresh set, timed from its own move-complete moment.
                self._owed = 0
            return
        if stability <= self.exit:
            self._calm_since = None
        elif self._calm_since is None:
            self._calm_since = ts
        elif ts - self._calm_since >= self.exit_hold_s - DUE_EPS:
            self._bursting = False
            self._calm_since = None
            self._owed = self.burst_end_reads
            # Read NOW, not one interval from the last keepalive: this frame is the
            # move-complete moment the whole policy exists to catch.
            self._last_read_ts = None


@dataclass
class WhenStill:
    """Estimate calm frames at a fixed rate — the naive alternative to burst end.

    No edge detection at all: any frame the meter scores at or above ``theta`` is
    eligible, subject to ``spacing_s`` between reads, and the same keepalive floor
    catches everything else. It exists to answer whether the burst *edge* is worth
    detecting or whether "read the still frames, slowly" gets there too — a
    cheaper story to explain, and one fewer piece of state to get wrong.

    Its weakness is structural and worth stating: with no notion of a transition
    it cannot spend reads where they are worth most, so it pays the same rate for
    a board nobody has touched in a minute as for the second after a move.
    """

    keepalive_s: float = DEFAULT_KEEPALIVE_S
    #: Stability at or above which a frame counts as calm enough to read.
    theta: float = EXIT_BURST
    #: Minimum recorded seconds between two calm-frame reads.
    spacing_s: float = 1.0

    PARAMS: ClassVar[Mapping[str, type]] = {
        "keepalive_s": float,
        "theta": float,
        "spacing_s": float,
    }

    _last_read_ts: float | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        finite_positive("keepalive_s", self.keepalive_s, FLOOR_REASON)
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {self.theta}")
        finite_positive("spacing_s", self.spacing_s, SPACING_REASON)
        under_the_floor("spacing_s", self.spacing_s, self.keepalive_s)

    @property
    def name(self) -> str:
        return canonical_spec(self, "when-still", self.PARAMS)

    def observe(self, capture_ts: float, stability: float) -> None:
        """Nothing to track: this policy has no notion of a transition, which is
        exactly the weakness it exists to measure."""
        return None

    def due(self, capture_ts: float, stability: float) -> ReadKind | None:
        ts = capture_ts
        if self._last_read_ts is not None:
            if ts < self._last_read_ts:
                self._last_read_ts = ts  # see BurstTriggered.due
            interval = self.spacing_s if stability >= self.theta else self.keepalive_s
            if not ts - self._last_read_ts >= interval - DUE_EPS:
                return None
        self._last_read_ts = ts
        return ReadKind.KEEPALIVE

    def should_estimate(self, entry: FrameEntry) -> bool:
        ts, stability = header_signal(entry)
        self.observe(ts, stability)
        return self.due(ts, stability) is not None
