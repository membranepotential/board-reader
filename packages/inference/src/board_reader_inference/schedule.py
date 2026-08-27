"""Which arriving frames are worth an estimate — the scheduling layer (DESIGN D3/D4).

The node used to read every frame that reached it, which is two kinds of waste at
once. A resting board is re-read at stream rate and every read says the same
thing; a board mid-move is read *during* the hand, and those frames are the
project's worst input — a fist over the d-file, a piece in the air over the wrong
square, a shadow across four ranks. The edge already knows the difference: it
meters motion and stamps every frame with a ``stability`` in [0,1]
(:mod:`board_reader_edge.motion`, mirrored in the browser edge's
``lib/edge/motion.ts``). Nothing downstream had ever used it.

A **policy** here is the thing that does: fed the recorded header of each frame in
order, it answers "spend an estimate on this one?". :class:`BurstTriggered` is the
shape this milestone is about — estimate at the **end of a burst**, the
move-complete moment, skip the burst itself, and read a slow **keepalive** at rest.

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

**The interplay with the debounce is the whole design.**
:class:`~board_reader_inference.stabilize.BoardStabilizer` adopts a changed square
only after :data:`~board_reader_inference.stabilize.SETTLE_MIN_READS` agreeing
reads spanning :data:`~board_reader_inference.stabilize.SETTLE_SECONDS` — a
*time* window, so that it survives the order-of-magnitude swing in read rate. A
scheduler that reads once per move therefore never settles anything: one read is
not a quorum however long you wait. Hence ``burst_end_reads``/``spacing_s``: a
burst end can be worth two or three reads a beat apart, which hands the debounce
its quorum in ~2.6 s instead of never. Sparsity and settling pull against each
other and the pair of knobs is where the trade is made.

The client-side burst thresholds are mirrored here value for value (see
:data:`ENTER_BURST`), because a policy that segments bursts differently from the
edge that forwarded them would be reasoning about a stream nobody sent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import ClassVar, Protocol, runtime_checkable

import cv2
import numpy as np

from board_reader_core.recording import FrameEntry

#: Stability under this starts a burst; it must climb back over
#: :data:`EXIT_BURST` and stay there for :data:`EXIT_HOLD_S` to end one.
#: Mirrored value for value from the browser edge's ``AdaptiveForwarder``
#: (``packages/viewer/ui/src/lib/edge/motion.ts``) — those are the thresholds that
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
    """Which recorded frames are worth an estimate.

    Stateful by design: burst end is only recognisable from the *run* of frames —
    a burst's last frame looks exactly like its first until the ones after it
    stay calm — so a policy is asked frame by frame, in order, exactly once each.

    :attr:`name` is the policy's canonical spec string: it round-trips through
    :func:`parse_policy`, so the ``policy`` field of a replay manifest is literally
    the ``--policy`` argument that reproduces that run.
    """

    #: Canonical, re-parseable spec — ``"burst-end:keepalive_s=5"`` and friends.
    name: str

    def should_estimate(self, entry: FrameEntry) -> bool:
        """Whether ``entry`` gets estimated. Called once per frame, in seq order."""


def _spec(policy: object, name: str, keys: Sequence[str]) -> str:
    """``name:key=value,…`` for the parameters that differ from the defaults.

    Only the differences, so the common case reads as the bare policy name and a
    manifest is not a wall of restated defaults — and the defaults themselves are
    free to be re-tuned without every old spec string silently meaning something
    new, because an old spec that *pinned* a value still says so.
    """
    defaults = {f.name: f.default for f in fields(policy) if f.init}  # type: ignore[arg-type]
    parts = [
        f"{key}={getattr(policy, key):g}"
        for key in keys
        if getattr(policy, key) != defaults[key]
    ]
    return f"{name}:{','.join(parts)}" if parts else name


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
        :class:`~board_reader_inference.stabilize.BoardStabilizer` wants
        ``SETTLE_MIN_READS`` reads spanning ``SETTLE_SECONDS``, so a single read
        per move leaves the debounce holding the previous position until the
        keepalive slowly catches up. Three reads ~1.3 s apart span 2.6 s and hand
        it the quorum at the third, roughly a beat after the move.

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
        if self.keepalive_s <= 0:
            raise ValueError(
                f"keepalive_s must be > 0, got {self.keepalive_s}: the keepalive is a "
                "floor under the read rate (D3), and a policy without one goes blind "
                "the moment the motion meter does"
            )
        if self.burst_end_reads < 1:
            raise ValueError(f"burst_end_reads must be >= 1, got {self.burst_end_reads}")
        if self.spacing_s < 0:
            raise ValueError(f"spacing_s must be >= 0, got {self.spacing_s}")
        if self.exit_hold_s < 0:
            raise ValueError(f"exit_hold_s must be >= 0, got {self.exit_hold_s}")
        if not 0.0 <= self.enter <= self.exit <= 1.0:
            raise ValueError(
                f"need 0 <= enter <= exit <= 1, got enter={self.enter} exit={self.exit}; "
                "the gap between them is the hysteresis that keeps one move from "
                "reading as several bursts"
            )

    @property
    def name(self) -> str:
        return _spec(self, "burst-end", list(self.PARAMS))

    @property
    def bursting(self) -> bool:
        """Whether the policy currently believes a move is in progress."""
        return self._bursting

    def should_estimate(self, entry: FrameEntry) -> bool:
        ts = entry.capture_ts
        self._track_burst(float(entry.header.get("stability", 1.0)), ts)
        if self._last_read_ts is not None and ts < self._last_read_ts:
            # The clock stepped backwards — a spliced session, or a recorder wart.
            # Re-anchor rather than stall: a policy must never stop reading because
            # a timestamp surprised it.
            self._last_read_ts = None
        # While a burst end is still owed reads the interval is the tight one; at
        # rest it is the floor. Either way the read that fires satisfies one owed
        # read, so a keepalive that happens to land first is not wasted.
        interval = self.spacing_s if self._owed else self.keepalive_s
        if self._last_read_ts is not None and ts - self._last_read_ts < interval - DUE_EPS:
            return False
        self._owed = max(self._owed - 1, 0)
        self._last_read_ts = ts
        return True

    def _track_burst(self, stability: float, ts: float) -> None:
        """The forwarder's own ``trackBurst``, on the receiving side of the wire."""
        if not self._bursting:
            if stability < self.enter:
                self._bursting = True
                self._calm_since = None
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
        if self.keepalive_s <= 0:
            raise ValueError(f"keepalive_s must be > 0, got {self.keepalive_s} (D3 floor)")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {self.theta}")
        if self.spacing_s < 0:
            raise ValueError(f"spacing_s must be >= 0, got {self.spacing_s}")

    @property
    def name(self) -> str:
        return _spec(self, "when-still", list(self.PARAMS))

    def should_estimate(self, entry: FrameEntry) -> bool:
        ts = entry.capture_ts
        if self._last_read_ts is not None and ts < self._last_read_ts:
            self._last_read_ts = None
        calm = float(entry.header.get("stability", 1.0)) >= self.theta
        interval = self.spacing_s if calm else self.keepalive_s
        if self._last_read_ts is not None and ts - self._last_read_ts < interval - DUE_EPS:
            return False
        self._last_read_ts = ts
        return True


# --------------------------------------------------------------------------- #
# Spec strings
# --------------------------------------------------------------------------- #

#: Every policy the ``--policy`` flag accepts, by spec name.
POLICIES: Mapping[str, type] = {
    "every-frame": EveryFrame,
    "burst-end": BurstTriggered,
    "when-still": WhenStill,
}


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
#: in ``packages/viewer/ui/src/lib/edge/motion.ts``). The three meters are one
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
