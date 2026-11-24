"""Which arriving frames are worth an estimate — the scheduling layer (DESIGN D3/D4).

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
gateway schedules with (:data:`DEFAULT_POLICY_SPEC`, ``BR_SCHEDULE`` to override),
and what :mod:`board_reader_inference.measurement.replay` sweeps offline — one implementation,
because a scheduler measured on recordings and a different one in production is a
sweep that proved nothing.

Live, the two halves of the question come apart: the policy must see **every**
arriving frame (:meth:`FramePolicy.observe`, on the ``/ingest`` handler) or it
segments bursts nobody sent, while the decision (:meth:`FramePolicy.due`) belongs
at the moment the worker picks a frame out of the latest-wins inbox — which is how
a burst-end read lands on the *sharpest* frame of the moment rather than on
whichever one arrived at the instant the burst ended. See :class:`FramePolicy`.

**The keepalive is a floor, not a nicety** (DESIGN D3: stability is a *hint*, never
a gate). The meter that shipped before 2026-08-27 measured an absolute pixel
difference and, on the deployment phone, read under 0.35 all day — permanently
"moving", never once resting. Everything gated on it was silently dead. A
scheduler that only reads at burst ends would, on that phone, have read the board
exactly never. So every policy in this module reads on a wall of its own: if
``keepalive_s`` of *capture* time has passed without a read, the next frame is
read whatever the meter claims. A stuck meter costs read rate, never reads.

**What a policy may look at.** The header only — ``seq``, ``capture_ts``,
``stability``, ``sharpness``, ``burst_id`` — never the pixels: that is what lets a
declined frame cost nothing at all (its JPEG is never read off disk), and it is
all a scheduler needs. And **only ``capture_ts``**, never the wall clock: a policy
is replayed offline at whatever speed the GPU manages, so a decision that consults
``time.time()`` would answer differently on every run and take the replay
harness's determinism down with it.

That purity has a live cost, and the live path pays it rather than this module:
a camera whose ``capture_ts`` freezes or steps backwards stalls a capture-time
floor, so the ingest worker carries a **wall-clock backstop** on top of the
policy's own. In here, both clock pathologies are made to converge on the floor
rather than on either extreme — a backwards step re-anchors without becoming due,
and an unreadable interval reads as "not yet".

**The interplay with the debounce is the whole design.**
:class:`~board_reader_inference.stream.stabilize.BoardStabilizer` adopts a changed square
only after :data:`~board_reader_inference.stream.stabilize.SETTLE_MIN_READS` agreeing
reads spanning :data:`~board_reader_inference.stream.stabilize.SETTLE_SECONDS` — a
*time* window, so that it survives the order-of-magnitude swing in read rate. A
scheduler that reads once per move therefore never settles anything: one read is
not a quorum however long you wait. Hence ``burst_end_reads``/``spacing_s``: a
burst end can be worth two or three reads a beat apart, which hands the debounce
its quorum 0.6 s after the burst end instead of never. Sparsity and settling pull
against each other and the pair of knobs is where the trade is made.

The client-side burst thresholds are mirrored here value for value (see
:data:`ENTER_BURST`), because a policy that segments bursts differently from the
edge that forwarded them would be reasoning about a stream nobody sent.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import ClassVar, Protocol, runtime_checkable

import cv2
import numpy as np

from board_reader_core.recording import FrameEntry

#: Stability under this starts a burst; it must climb back over
#: :data:`EXIT_BURST` and stay there for :data:`EXIT_HOLD_S` to end one.
#: Mirrored value for value from the browser edge's ``AdaptiveForwarder``
#: (``packages/viewer/src/lib/edge/motion.ts``) — those are the thresholds that
#: decided which frames were *sent*, so segmenting the arriving stream by any
#: other pair would describe bursts that never happened.
ENTER_BURST = 0.7
EXIT_BURST = 0.85
#: Sustained calm before a burst is declared over. The pause between two strides
#: of one move is shorter than this; splitting there would call the middle of a
#: move a move-complete moment.
EXIT_HOLD_S = 0.7

#: Float slack, so an interval due "exactly now" is not missed by a rounding bit
#: (the same guard the forwarder's rate check uses).
DUE_EPS = 1e-6

#: The keepalive floor every policy carries unless told otherwise. Three seconds
#: is two settle windows: slow enough that a resting board is nearly free, fast
#: enough that a scheduler which has gone blind still assembles a settled board
#: within a handful of seconds.
DEFAULT_KEEPALIVE_S = 3.0


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #


@runtime_checkable
class FramePolicy(Protocol):
    """Which frames are worth an estimate.

    Stateful by design: burst end is only recognisable from the *run* of frames —
    a burst's last frame looks exactly like its first until the ones after it
    stay calm — so the policy must see every frame's header, in order.

    That is why the seam is **two** calls rather than one, and the split is what
    lets the same policy drive the offline replay and the live gateway:

    :meth:`observe`
        Fold one arriving frame's header into the burst state. Called for
        **every** frame, in capture order, exactly once each — including frames
        that are never estimated and frames the ingest inbox goes on to drop. A
        policy blinded to part of the stream segments bursts that nobody sent.

    :meth:`due`
        Decide whether *this* frame is the one to spend an estimate on, and
        record the read if so. Called at most once per frame, and only for a
        frame the caller is actually able to estimate right now.

    Offline (:mod:`board_reader_inference.measurement.replay`) the two collapse into
    :meth:`should_estimate`: frames are read one at a time, in order, so the frame
    just observed is always the frame being decided. Live
    (:class:`~board_reader_inference.stream.ingest.FrameIngest`) they come apart —
    :meth:`observe` runs on the ``/ingest`` handler for every frame that lands,
    while :meth:`due` runs on the board's worker for whichever frame survived the
    latest-wins inbox. Deciding at the *drain* rather than at arrival is the
    point: :func:`~board_reader_inference.stream.ingest.prefer_frame` keeps the sharper
    of two contenders within its window, so a burst-end read spends itself on the
    crispest view of the moment instead of on whichever frame happened to arrive
    at the instant the burst ended.

    :attr:`name` is the policy's canonical spec string: it round-trips through
    :func:`parse_policy`, so the ``policy`` field of a replay manifest is literally
    the ``--policy`` argument that reproduces that run, and ``BR_SCHEDULE`` is the
    same string again.
    """

    #: Canonical, re-parseable spec — ``"burst-end:keepalive_s=5"`` and friends.
    name: str
    #: The floor, in seconds of capture time (D3). Part of the seam because the
    #: live worker backstops it against the *wall* clock, and needs the number
    #: from whatever policy it was handed.
    keepalive_s: float

    def observe(self, capture_ts: float, stability: float) -> None:
        """Fold one arriving frame's header in. Every frame, in capture order."""

    def due(self, capture_ts: float, stability: float) -> bool:
        """Whether to spend an estimate on this frame — and, if so, record it.

        Not a pure query: answering ``True`` consumes the read (the keepalive
        clock restarts, a burst end's owed read is struck off), so it must be
        asked exactly once per frame the caller can estimate, and never for a
        frame the caller then declines to read for its own reasons.
        """

    def should_estimate(self, entry: FrameEntry) -> bool:
        """:meth:`observe` then :meth:`due` for a recorded frame — the replay path.

        Called once per frame, in seq order.
        """


def stability_of(raw: object) -> float:
    """The stability a policy reads, from whatever a header actually carried.

    The one coercion boundary, shared by the recorded path (:func:`_signal`) and
    the live one (``ingest._schedule_signal``), because a scheduler that reasons
    about ``None`` or a NaN is a scheduler with two behaviours depending on which
    edge connected. :meth:`FramePolicy.observe` and :meth:`FramePolicy.due` are
    entitled to a finite number.

    Anything not a finite number reads as **1.0, fully at rest** — the D3 default.
    A JSON ``null`` from a hand-rolled edge, a missing field, and a NaN from a
    meter that divided by zero all mean "this stream does not tell me about
    motion", and the honest answer to that is the keepalive floor, not an
    exception in the ``/ingest`` handler and not a burst that ends spuriously
    (NaN loses every comparison, so it would slip through the hysteresis' calm
    branch and declare a move complete in the middle of a hand).
    """
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    return value if math.isfinite(value) else 1.0


def _signal(entry: FrameEntry) -> tuple[float, float]:
    """The two header fields a policy reads: capture time and stability."""
    return entry.capture_ts, stability_of(entry.header.get("stability"))


def _number(value: float, kind: type) -> str:
    """A parameter value rendered so that :func:`parse_policy` reads it back exactly.

    ``%g`` — what this used to use — is a *display* format and loses on both
    sides: it renders ``burst_end_reads=1000000`` as ``1e+06``, which the int
    coercion in :func:`parse_policy` then refuses outright, and it rounds floats
    to six significant digits, so two policies differing past that share one name
    and one replay directory. Ints print as ints; floats print at ``repr``'s
    shortest round-tripping precision, with a bare ``.0`` trimmed because
    ``keepalive_s=5`` is how anyone would write it and ``float("5")`` is the same
    number.
    """
    if kind is int:
        return str(int(value))
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _spec(policy: object, name: str, params: Mapping[str, type]) -> str:
    """``name:key=value,…`` for the parameters that differ from the defaults.

    Only the differences, so the common case reads as the bare policy name and a
    manifest is not a wall of restated defaults — and the defaults themselves are
    free to be re-tuned without every old spec string silently meaning something
    new, because an old spec that *pinned* a value still says so.
    """
    defaults = {f.name: f.default for f in fields(policy) if f.init}  # type: ignore[arg-type]
    parts = [
        f"{key}={_number(getattr(policy, key), kind)}"
        for key, kind in params.items()
        if getattr(policy, key) != defaults[key]
    ]
    return f"{name}:{','.join(parts)}" if parts else name


def _finite_positive(name: str, value: float, because: str) -> None:
    """Reject the two spellings of "no limit" that would switch a floor off.

    ``inf`` passes a bare ``> 0`` check and turns a keepalive into "never" — one
    read in the 60 recorded seconds of the dead-meter incident. NaN passes it too
    and is worse: every comparison against NaN is ``False``, so the rate limit
    inverts and the policy reads *every* frame. Both are the incident this module
    exists to prevent, arriving through a config string instead of a broken
    sensor, so an interval is a finite positive number of seconds and nothing else.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be > 0 and finite, got {value!r}: {because}")


#: Why ``keepalive_s`` has no "off" spelling.
_FLOOR_REASON = (
    "the keepalive is a floor under the read rate (D3), and a policy without one "
    "goes blind the moment the motion meter does"
)
#: Why a spacing of zero is not a thing either.
_SPACING_REASON = (
    "it is the interval between two reads, and an interval of zero (or of none at "
    "all) is a burst of estimates at stream rate, which is what the floor exists "
    "to bound"
)


def _under_the_floor(name: str, spacing_s: float, keepalive_s: float) -> None:
    """A tighter interval may govern over the floor; a looser one may not.

    ``spacing_s`` is the *tight* interval — the beat of a burst-end run, or the
    calm-frame rate — and the whole promise of D3 is that no configuration makes
    the node read more slowly than ``keepalive_s``. A spacing above the floor
    quietly breaks that promise (``when-still`` with ``keepalive_s=3,
    spacing_s=10`` leaves 10 s gaps under a claimed 3 s floor), and it does so in
    the one direction nobody watches: fewer reads, which looks like a quiet board.
    """
    if spacing_s > keepalive_s:
        raise ValueError(
            f"{name}={spacing_s!r} is looser than keepalive_s={keepalive_s!r}: the "
            "spacing is the tight interval and may only ever read MORE often than "
            "the floor, never less (D3). Raise keepalive_s if longer gaps are "
            "really what you want."
        )


@dataclass(frozen=True, slots=True)
class EveryFrame:
    """Estimate every recorded frame — the densest benchmark, and the control.

    Not what the live gateway does (its latest-wins inbox drops frames whenever
    the estimator falls behind), and not what any scheduler should do; it is the
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

    def due(self, capture_ts: float, stability: float) -> bool:
        return True

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
        :class:`~board_reader_inference.stream.stabilize.BoardStabilizer` wants
        ``SETTLE_MIN_READS`` reads spanning ``SETTLE_SECONDS``, so a single read
        per move leaves the debounce holding the previous position until the
        keepalive slowly catches up. The shipped three reads 0.3 s apart span
        0.6 s and hand it the quorum at the third, 0.6 s after the burst end
        (:data:`DEFAULT_POLICY_SPEC`). That needs the edge to keep sending densely
        after its burst ends (the forwarder's tail), or the run waits on keepalive
        frames instead.

    ``enter`` / ``exit`` / ``exit_hold_s``
        The burst hysteresis, defaulted to the client's own constants. Tune only
        together with the forwarder — they describe the same bursts.

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
        _finite_positive("keepalive_s", self.keepalive_s, _FLOOR_REASON)
        if self.burst_end_reads < 1:
            raise ValueError(f"burst_end_reads must be >= 1, got {self.burst_end_reads}")
        _finite_positive("spacing_s", self.spacing_s, _SPACING_REASON)
        _under_the_floor("spacing_s", self.spacing_s, self.keepalive_s)
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
        return _spec(self, "burst-end", self.PARAMS)

    @property
    def bursting(self) -> bool:
        """Whether the policy currently believes a move is in progress."""
        return self._bursting

    def observe(self, capture_ts: float, stability: float) -> None:
        self._track_burst(stability, capture_ts)

    def due(self, capture_ts: float, stability: float) -> bool:
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
                return False
        self._owed = max(self._owed - 1, 0)
        self._last_read_ts = ts
        return True

    def should_estimate(self, entry: FrameEntry) -> bool:
        ts, stability = _signal(entry)
        self.observe(ts, stability)
        return self.due(ts, stability)

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
        _finite_positive("keepalive_s", self.keepalive_s, _FLOOR_REASON)
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {self.theta}")
        _finite_positive("spacing_s", self.spacing_s, _SPACING_REASON)
        _under_the_floor("spacing_s", self.spacing_s, self.keepalive_s)

    @property
    def name(self) -> str:
        return _spec(self, "when-still", self.PARAMS)

    def observe(self, capture_ts: float, stability: float) -> None:
        """Nothing to track: this policy has no notion of a transition, which is
        exactly the weakness it exists to measure."""
        return None

    def due(self, capture_ts: float, stability: float) -> bool:
        ts = capture_ts
        if self._last_read_ts is not None:
            if ts < self._last_read_ts:
                self._last_read_ts = ts  # see BurstTriggered.due
            interval = self.spacing_s if stability >= self.theta else self.keepalive_s
            if not ts - self._last_read_ts >= interval - DUE_EPS:
                return False
        self._last_read_ts = ts
        return True

    def should_estimate(self, entry: FrameEntry) -> bool:
        ts, stability = _signal(entry)
        self.observe(ts, stability)
        return self.due(ts, stability)


# --------------------------------------------------------------------------- #
# Spec strings
# --------------------------------------------------------------------------- #

#: Every policy the ``--policy`` flag accepts, by spec name.
POLICIES: Mapping[str, type] = {
    "every-frame": EveryFrame,
    "burst-end": BurstTriggered,
    "when-still": WhenStill,
}

#: What the live gateway schedules with unless ``BR_SCHEDULE`` says otherwise —
#: ``BurstTriggered(keepalive_s=2.0, burst_end_reads=3, spacing_s=0.3)``. Only the
#: non-default parameters appear, because a spec states what was *asked for* (see
#: :func:`_spec`); the class defaults (:data:`DEFAULT_KEEPALIVE_S` among them) stay
#: put so every recorded spec string still rebuilds the run it names.
#:
#: Three reads 0.3 s apart span 0.6 s, just over ``SETTLE_SECONDS=0.5``, so a
#: completed move settles on the burst end's own evidence 0.6 s after the burst
#: ends — provided the frames are there: the edge forwarder keeps its burst rate
#: for ``tail_s`` (1 s) after a burst, because at the keepalive's 1 fps each of
#: these reads waited ~1 s for a frame (2026-11-24 live take 2).
#: Swept 2026-11-24 (round 2: spacing 0.2-0.6 s x 2-4 reads x settle 0.3-1.0 s,
#: on cached reads with the tail simulated): with the tail, the gate settles 11 of
#: 27 reached boundaries within 1 s of the hand leaving instead of 8, the real
#: gate (no tail) holds 140/156 settled perfect against 138/154, and no hand
#: settles on the gate. Before that: ``spacing_s=0.6`` against a 1.0 s window
#: (docs/reports/2026-11-24-settle-tuning.md), and ``burst_end_reads=3`` at
#: 1.3 s spacing against 2.5 s (docs/reports/2026-08-27-move-trigger-sweep.md §8).
#: ``BR_SCHEDULE=every-frame`` is the way back to reading everything.
#:
#: The keepalive is 2 s, not 3 s, since the 2026-11-24 live take: its phone's
#: meter never opened a burst, so every move waited on the keepalive and took
#: 3-4 s to settle. Replayed on cached reads, 2 s cut the gate's median
#: move-to-settle from 3.0 s to 2.0 s with no hand settling (104/119 -> 138/154
#: settled perfect, +29 % reads) and the live take's from 3.1 s to 2.1 s. 1.5 s
#: had the same medians at +48 % reads; 1.0 s let a hand settle on two gate takes
#: (a lifted b8 knight, a c8 bishop under the hand). The keepalive is the floor
#: when the meter is blind, so it is sized for that case.
DEFAULT_POLICY_SPEC = "burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.3"


def parse_policy(spec: str) -> FramePolicy:
    """Build a policy from ``name[:key=value,…]`` — the ``--policy`` surface.

    Deliberately tiny: this is an experiment knob, and every value it accepts is a
    number naming a field of the policy it configures, so there is nothing to
    invent. Unknown names and unknown keys raise with the list of what exists,
    because the alternative — silently ignoring a typo'd parameter — is a whole
    GPU sweep measuring the defaults under another policy's name.

    Round-trips: ``parse_policy(p.name) == p`` for every policy here.
    """
    head, _, rest = spec.strip().partition(":")
    head = head.strip()
    cls = POLICIES.get(head)
    if cls is None:
        raise ValueError(
            f"unknown policy {head!r}; known policies: {', '.join(POLICIES)}"
        )
    params: dict[str, float | int] = {}
    for item in (p.strip() for p in rest.split(",")):
        if not item:
            continue
        key, sep, raw = item.partition("=")
        key = key.strip()
        coerce = cls.PARAMS.get(key)
        if not sep or coerce is None:
            known = ", ".join(cls.PARAMS) or "(none — it takes no parameters)"
            raise ValueError(
                f"policy {head!r} has no parameter {key!r}; it takes: {known}"
            )
        try:
            params[key] = coerce(raw.strip())
        except ValueError as e:
            raise ValueError(
                f"policy {head!r}: {key}={raw.strip()!r} is not a {coerce.__name__}"
            ) from e
    return cls(**params)  # type: ignore[return-value]


def policy_slug(policy: FramePolicy) -> str:
    """A filesystem-safe rendering of a policy's spec, for a replay directory name.

    Lossless enough to read back at a glance (``burst-end-keepalive_s5``), which is
    the point: a sweep leaves a dozen sibling directories and their names are the
    only index anyone reads.
    """
    return re.sub(r"[^0-9A-Za-z._]+", "-", policy.name.replace("=", "")).strip("-")


# --------------------------------------------------------------------------- #
# Recomputing the signal the policies read
# --------------------------------------------------------------------------- #

#: Meter constants, mirrored from :mod:`board_reader_edge.motion` (itself mirrored
#: in ``packages/viewer/src/lib/edge/motion.ts``). The three meters are one
#: algorithm in three places on purpose: the edge/inference split (DESIGN §4,
#: pinned by ``tests/test_layer_separation.py``) forbids this package from
#: importing the edge's, and the browser's is in another language entirely. Any
#: change here is a change in all three — they are compared by their numbers.
MOTION_DOWNSCALE = 32
MOTION_GAIN = 1.5
MOTION_ATTACK = 0.7
MOTION_RELEASE = 0.4
MOTION_TOLERANCE = 2.0
MOTION_FLOOR_UP = 0.02
MOTION_FLOOR_DOWN = 0.08
MOTION_NOISE_FLOOR = 0.0015
MOTION_FLOOR_NEAR = 4.0
MOTION_FLOOR_FAR_DIVISOR = 5.0


class MotionMeter:
    """Mean absolute inter-frame difference of a tiny thumbnail -> stability [0,1].

    A port, not a variant: same thumbnail, same adaptive floor, same asymmetric
    smoother, same constants as :class:`board_reader_edge.motion.MotionMeter`. See
    :data:`MOTION_DOWNSCALE` for why it is copied rather than imported.

    The first frame scores 1.0 — there is nothing to difference against, and an
    unknown board reports at rest (D3: a hint, never a gate).
    """

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._floor: float | None = None
        self._stability = 1.0

    @property
    def stability(self) -> float:
        """The current smoothed stability, without consuming a frame."""
        return self._stability

    def update(self, image: np.ndarray) -> float:
        """Fold one decoded frame in and return the stability to stamp on it."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        thumb = (
            cv2.resize(
                gray,
                (MOTION_DOWNSCALE, MOTION_DOWNSCALE),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            / 255.0
        )
        prev, self._prev = self._prev, thumb
        if prev is None:
            self._stability = 1.0
            return self._stability
        diff = float(np.abs(thumb - prev).mean())
        if self._floor is None:
            self._floor = MOTION_NOISE_FLOOR
        if diff < self._floor:
            step = MOTION_FLOOR_DOWN
        elif diff / self._floor <= MOTION_FLOOR_NEAR:
            step = MOTION_FLOOR_UP
        else:
            step = MOTION_FLOOR_UP / MOTION_FLOOR_FAR_DIVISOR
        self._floor = max(MOTION_NOISE_FLOOR, self._floor + step * (diff - self._floor))
        instant = float(
            np.exp(-MOTION_GAIN * max(diff / self._floor - MOTION_TOLERANCE, 0.0))
        )
        weight = MOTION_ATTACK if instant < self._stability else MOTION_RELEASE
        self._stability = float(
            np.clip((1.0 - weight) * self._stability + weight * instant, 0.0, 1.0)
        )
        return self._stability


#: Header flag marking a stability the meter recomputed rather than the camera
#: stamped. Written into the replay's ``frames.ndjson``, because a benchmark whose
#: index silently claims a number no camera ever sent is a trap for whoever reads
#: it next.
RESTAMPED_KEY = "stability_restamped"


def restamped(entry: FrameEntry, meter: MotionMeter, image: np.ndarray) -> FrameEntry:
    """``entry`` with the stability today's meter scores for ``image``.

    **Why this exists.** Policies read the header's ``stability``, and on every
    take recorded before the 2026-08-27 evening meter fix that field is a *broken
    sensor's* output: the old meter scored an absolute pixel difference, the
    deployment phone's resting difference measured about six times the constant it
    assumed, and the field sat under 0.35 from the first frame of a session to the
    last. Scheduled against that, every policy would be measuring the bug rather
    than itself. Re-metering the recorded JPEGs in seq order produces the number a
    phone streaming that same scene *today* would send.

    The caller owns the meter and must feed it **every** frame of the session, in
    order, whether or not the policy goes on to read that one: the floor is
    learned from the run of frames, and a floor learned from a subsample is a
    different meter. It must also use a fresh meter per streaming session — a
    reconnect is a new camera.

    The header is marked (:data:`RESTAMPED_KEY`) rather than quietly rewritten: a
    benchmark index that claims a number no camera ever sent is a trap for
    whoever reads it next.

    **Expect a burst at the top of every restamped session.** The floor is seeded
    at :data:`MOTION_NOISE_FLOOR` — sensor noise, deliberately below any real rig's
    resting difference, so a camera opened onto motion cannot learn motion as its
    rest — and chases upward slowly. A recorded stream's first second or two of
    frames therefore difference far above the floor the meter has learned so far
    and score as movement. It costs one spurious burst-end read at the start of a
    take and nothing after it. Live this does not arise the same way: the browser
    edge meters from the moment the camera opens, long before anyone presses
    start.
    """
    return FrameEntry(
        header={**entry.header, "stability": meter.update(image), RESTAMPED_KEY: True},
        jpeg_rel=entry.jpeg_rel,
        received_ts=entry.received_ts,
    )
