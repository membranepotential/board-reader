"""L3 position-stream protocol — the shared wire contract (DESIGN.md §5).

This module is the spine of the system and is imported by BOTH the edge (Pi) and
the inference node. It carries NO node-specific logic: only the canonical
position-event shape, the 13-class vocabulary, and pure derivations over the
per-square distribution. Kept dependency-light (numpy only) so the same contract
runs on the Pi, the inference node, and the eventual browser port.

Canonical output is the full 8x8x13 ``square_dist`` (decision #14); ``placement``
(FEN field 1) and ``confidence`` are *derived* from it, never authored alongside.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# 13-class vocabulary. Index order is canonical and travels in every event so a
# consumer can map every per-square vector. Uppercase = white, lowercase = black,
# ``empty`` is index 0 (DESIGN.md §5).
CLASSES: tuple[str, ...] = (
    "empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k",
)
NUM_CLASSES = len(CLASSES)
EMPTY = 0
CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

GRID = 8  # board is 8x8

# Placement is FEN field 1 only. Rows are rank 8 -> rank 1; files a -> h.
START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
EMPTY_PLACEMENT = "8/8/8/8/8/8/8/8"

FILES = "abcdefgh"


def square_name(row: int, col: int) -> str:
    """Algebraic name of grid cell ``(row, col)`` — row 0 is rank 8, col 0 is file a.

    The grid frame is the one every ``square_dist`` is expressed in, so this is
    contract material: it lives here rather than being re-derived in each consumer
    that wants to name a square (it was independently spelled out in four modules
    before it moved).
    """
    if not (0 <= row < GRID and 0 <= col < GRID):
        raise ValueError(f"({row}, {col}) is off an {GRID}x{GRID} board")
    return f"{FILES[col]}{GRID - row}"


def square_name_at(index: int) -> str:
    """Algebraic name of the ``index``-th square in row-major (rank 8 first) order."""
    if not 0 <= index < GRID * GRID:
        raise ValueError(f"square index {index} is off the board")
    return square_name(index // GRID, index % GRID)

# ``status`` deliberately has no ``illegal`` value — legality is not ours to judge
# (DESIGN.md §5). An illegal-looking placement is reported as ``ok``.
VALID_STATUS: tuple[str, ...] = ("ok", "recalibrating", "needs_setup", "stale")
VALID_RUNNING: tuple[str, ...] = ("white", "black", "none")
DEFAULT_ORIENTATION = "a1_bottom_left"

# Distributions are floating point; allow a little slack when checking they sum to 1.
_SUM_TOL = 1e-3


def iso_utc(ts: float) -> str:
    """Epoch seconds -> RFC3339 UTC string with millisecond precision and a ``Z``.

    ``utc`` in an event is the edge *capture* (sensor) time, not server-receive
    time (DESIGN.md D5/§5).
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_placement(placement: str) -> list[list[str]]:
    """FEN field 1 -> 8x8 grid of class symbols ("empty" for empties).

    Row 0 is rank 8, column 0 is file a.
    """
    rows = placement.split("/")
    if len(rows) != GRID:
        raise ValueError(f"placement must have {GRID} ranks, got {len(rows)}")
    grid: list[list[str]] = []
    for rank in rows:
        cells: list[str] = []
        for ch in rank:
            if ch.isdigit():
                cells.extend(["empty"] * int(ch))
            elif ch in CLASS_INDEX and ch != "empty":
                cells.append(ch)
            else:
                raise ValueError(f"invalid placement symbol {ch!r}")
        if len(cells) != GRID:
            raise ValueError(f"rank {rank!r} expands to {len(cells)} squares, need {GRID}")
        grid.append(cells)
    return grid


def encode_placement(grid: list[list[str]]) -> str:
    """8x8 grid of class symbols -> FEN field 1 (run-length empties as digits)."""
    rows: list[str] = []
    for row in grid:
        out = ""
        run = 0
        for cell in row:
            if cell == "empty":
                run += 1
                continue
            if run:
                out += str(run)
                run = 0
            out += cell
        if run:
            out += str(run)
        rows.append(out)
    return "/".join(rows)


def placement_from_dist(square_dist: np.ndarray) -> str:
    """Derive the FEN-field-1 placement as the per-square argmax of ``square_dist``."""
    arr = np.asarray(square_dist)
    idx = arr.argmax(axis=-1)
    grid = [[CLASSES[int(idx[r, c])] for c in range(GRID)] for r in range(GRID)]
    return encode_placement(grid)


def dist_from_placement(placement: str, confidence: float = 0.95) -> np.ndarray:
    """Build a peaked 8x8x13 distribution from a known placement.

    The true class of each square gets probability ``confidence``; the remaining
    mass is spread uniformly over the other 12 classes. Used by the stub estimator
    and by tests to synthesise a valid, argmax-consistent distribution.
    """
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    grid = parse_placement(placement)
    other = (1.0 - confidence) / (NUM_CLASSES - 1)
    arr = np.full((GRID, GRID, NUM_CLASSES), other, dtype=np.float32)
    for r in range(GRID):
        for c in range(GRID):
            arr[r, c, CLASS_INDEX[grid[r][c]]] = confidence
    return arr


def square_confidences(square_dist: np.ndarray) -> np.ndarray:
    """Per-square max probability — an 8x8 scalar grid derived from the distribution."""
    return np.asarray(square_dist).max(axis=-1)


def position_confidence(square_dist: np.ndarray, reduce: str = "mean") -> float:
    """Overall position confidence: mean (default) or min of per-square max prob."""
    sc = square_confidences(square_dist)
    if reduce == "mean":
        return float(sc.mean())
    if reduce == "min":
        return float(sc.min())
    raise ValueError("reduce must be 'mean' or 'min'")


#: How many classes per square a recording keeps as evidence. Two is the whole
#: point of the digest: the winner and its nearest rival are what say whether a
#: misread was a coin-flip or a conviction.
EVIDENCE_TOP_K = 2


def square_evidence(
    square_dist: np.ndarray, *, top_k: int = EVIDENCE_TOP_K, ndigits: int = 4
) -> list[list[tuple[str, float]]]:
    """The ``top_k`` likeliest classes per square, board order, as (class, prob).

    A deliberately lossy digest of ``square_dist``, for the recorder: it keeps the
    margin the argmax throws away, so a wrong square can be read afterwards as
    "0.51 k against 0.49 q" rather than only "k". 64 pairs cost ~1.8 kB a line
    against ~25 kB for the full distribution, which is why the recording carries
    this and not the array (see ``_record_published``).

    Squares run rank 8 -> rank 1, files a -> h — the same order as ``placement``,
    so entry ``i`` belongs to ``square_name_at(i)``.
    """
    arr = np.asarray(square_dist, dtype=np.float64)
    if arr.shape != (GRID, GRID, NUM_CLASSES):
        raise ValueError(f"square_dist must be {GRID}x{GRID}x{NUM_CLASSES}, got {arr.shape}")
    if not 1 <= top_k <= NUM_CLASSES:
        raise ValueError(f"top_k must be in 1..{NUM_CLASSES}, got {top_k}")
    # Descending by probability; ties break on class index so the digest is
    # reproducible across runs and platforms.
    order = np.argsort(-arr, axis=-1, kind="stable")[..., :top_k]
    return [
        [
            (CLASSES[int(order[r, c, k])], round(float(arr[r, c, order[r, c, k]]), ndigits))
            for k in range(top_k)
        ]
        for r in range(GRID)
        for c in range(GRID)
    ]


@dataclass
class ClockState:
    """Which side's clock is running (DESIGN.md D7). ``none`` = neither/unknown."""

    running: str = "none"

    def __post_init__(self) -> None:
        if self.running not in VALID_RUNNING:
            raise ValueError(f"clock.running must be one of {VALID_RUNNING}")

    def to_dict(self) -> dict:
        return {"running": self.running}

    @classmethod
    def from_dict(cls, d: dict | None) -> ClockState:
        if not d:
            return cls()
        return cls(running=d.get("running", "none"))


@dataclass(frozen=True)
class PriorFlip:
    """One square the hard-constraint decoder relabelled, as it travels on the wire.

    The producer is the inference node's prior decoder; this is only its wire
    shape. ``cost`` is the log-probability margin given up, in nats.
    """

    square: str  # algebraic, e.g. "d8"
    from_class: str
    to_class: str
    cost: float
    constraint: str

    def to_dict(self) -> dict:
        return {
            "square": self.square,
            "from": self.from_class,
            "to": self.to_class,
            "cost": round(self.cost, 4),
            "constraint": self.constraint,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PriorFlip:
        return cls(
            square=str(d["square"]),
            from_class=str(d["from"]),
            to_class=str(d["to"]),
            cost=float(d.get("cost", 0.0)),
            constraint=str(d.get("constraint", "")),
        )


@dataclass(frozen=True)
class PriorRepair:
    """The nearest *possible* board to a read, served beside it (never instead).

    ``square_dist`` and ``placement`` stay exactly what the estimator produced —
    a consumer that wants the repaired board takes :attr:`placement` from here,
    and one that runs its own smoothing over the raw distribution can ignore this
    block entirely, so the prior is never double-counted. ``residual`` holds the
    per-constraint excess the decoder could not repair (empty = fully possible).
    """

    placement: str
    flips: tuple[PriorFlip, ...] = ()
    residual: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "placement": self.placement,
            "flips": [f.to_dict() for f in self.flips],
            "residual": dict(self.residual),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PriorRepair:
        return cls(
            placement=str(d["placement"]),
            flips=tuple(PriorFlip.from_dict(f) for f in d.get("flips") or ()),
            residual={str(k): int(v) for k, v in (d.get("residual") or {}).items()},
        )


@dataclass(frozen=True)
class SettledBoard:
    """A temporally debounced board, served beside the raw read (never instead).

    The producer is the inference node's ingest stabilizer: a square must hold the
    same class for K consecutive *estimated* reads before the settled board adopts
    it, so one-frame flicker (a hand crossing the board, a coin-flip repair) never
    reaches a display that draws this block. ``pending`` names the squares
    currently reading differently from the settled board — a move in flight looks
    like its from/to squares going pending, then being adopted.

    Advisory like :class:`PriorRepair`: ``placement``/``square_dist`` on the event
    stay exactly what the estimator produced, and a consumer running its own
    smoothing ignores this block entirely.
    """

    placement: str
    pending: tuple[str, ...] = ()  # algebraic squares, board order

    def to_dict(self) -> dict:
        return {"placement": self.placement, "pending": list(self.pending)}

    @classmethod
    def from_dict(cls, d: dict) -> SettledBoard:
        return cls(
            placement=str(d["placement"]),
            pending=tuple(str(s) for s in d.get("pending") or ()),
        )


@dataclass
class PositionEvent:
    """One L3 position event (DESIGN.md §5).

    Construct via :meth:`from_square_dist` so ``placement`` and ``confidence`` stay
    derived from the canonical ``square_dist`` and can never drift out of sync.
    """

    board_id: str
    seq: int
    utc: str
    square_dist: np.ndarray  # (8, 8, 13) float, rank 8 -> rank 1, file a -> h
    placement: str
    confidence: float
    stability: float = 1.0
    clock: ClockState = field(default_factory=ClockState)
    status: str = "ok"
    orientation: str = DEFAULT_ORIENTATION
    #: Optional hard-constraint repair of this read, produced by the inference
    #: node's prior decoder. ``placement`` above stays the raw argmax; a display
    #: that must never show an impossible board draws ``prior.placement`` instead.
    prior: PriorRepair | None = None
    #: Optional debounced board, produced by the ingest stabilizer on streamed
    #: frames only (a one-shot ``/estimate`` has no stream to settle over). A
    #: display that must not flicker draws ``settled.placement``.
    settled: SettledBoard | None = None

    @classmethod
    def from_square_dist(
        cls,
        *,
        board_id: str,
        seq: int,
        utc: str,
        square_dist: np.ndarray,
        stability: float = 1.0,
        clock: ClockState | None = None,
        status: str = "ok",
        orientation: str = DEFAULT_ORIENTATION,
        confidence_reduce: str = "mean",
    ) -> PositionEvent:
        arr = np.asarray(square_dist, dtype=np.float32)
        if arr.shape != (GRID, GRID, NUM_CLASSES):
            raise ValueError(
                f"square_dist must be {(GRID, GRID, NUM_CLASSES)}, got {arr.shape}"
            )
        return cls(
            board_id=board_id,
            seq=seq,
            utc=utc,
            square_dist=arr,
            placement=placement_from_dist(arr),
            confidence=position_confidence(arr, confidence_reduce),
            stability=stability,
            clock=clock or ClockState(),
            status=status,
            orientation=orientation,
        )

    def validate(self) -> PositionEvent:
        """Structural self-check; raises ``ValueError`` on any violation. Returns self."""
        arr = np.asarray(self.square_dist)
        if arr.shape != (GRID, GRID, NUM_CLASSES):
            raise ValueError(f"square_dist shape {arr.shape} != {(GRID, GRID, NUM_CLASSES)}")
        if arr.min() < -_SUM_TOL or arr.max() > 1.0 + _SUM_TOL:
            raise ValueError("square_dist entries must be probabilities in [0, 1]")
        sums = arr.sum(axis=-1)
        if np.any(np.abs(sums - 1.0) > _SUM_TOL):
            raise ValueError("each square distribution must sum to 1")
        if self.placement != placement_from_dist(arr):
            raise ValueError("placement is not the argmax of square_dist")
        for name, v in (("confidence", self.confidence), ("stability", self.stability)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.status not in VALID_STATUS:
            raise ValueError(f"status must be one of {VALID_STATUS}")
        if self.clock.running not in VALID_RUNNING:
            raise ValueError(f"clock.running must be one of {VALID_RUNNING}")
        if self.seq < 0:
            raise ValueError("seq must be non-negative")
        if self.prior is not None:
            parse_placement(self.prior.placement)  # raises on a malformed repair
        if self.settled is not None:
            parse_placement(self.settled.placement)
        return self

    def to_dict(self, *, round_dist: int | None = 4) -> dict:
        """Serialise to the §5 JSON shape. ``round_dist`` trims distribution noise."""
        arr = np.asarray(self.square_dist, dtype=float)
        if round_dist is not None:
            arr = np.round(arr, round_dist)
        d = {
            "board_id": self.board_id,
            "utc": self.utc,
            "seq": self.seq,
            "orientation": self.orientation,
            "classes": list(CLASSES),
            "placement": self.placement,
            "square_dist": arr.tolist(),
            "confidence": round(self.confidence, 6),
            "stability": round(self.stability, 6),
            "clock": self.clock.to_dict(),
            "status": self.status,
        }
        if self.prior is not None:
            d["prior"] = self.prior.to_dict()
        if self.settled is not None:
            d["settled"] = self.settled.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> PositionEvent:
        classes = d.get("classes")
        if classes is not None and tuple(classes) != CLASSES:
            raise ValueError("event 'classes' does not match this protocol's CLASSES")
        arr = np.asarray(d["square_dist"], dtype=np.float32)
        return cls(
            board_id=d["board_id"],
            seq=int(d["seq"]),
            utc=d["utc"],
            square_dist=arr,
            placement=d.get("placement") or placement_from_dist(arr),
            confidence=float(d.get("confidence", position_confidence(arr))),
            stability=float(d.get("stability", 1.0)),
            clock=ClockState.from_dict(d.get("clock")),
            status=d.get("status", "ok"),
            orientation=d.get("orientation", DEFAULT_ORIENTATION),
            prior=PriorRepair.from_dict(d["prior"]) if d.get("prior") else None,
            settled=SettledBoard.from_dict(d["settled"]) if d.get("settled") else None,
        )

    @classmethod
    def from_json(cls, s: str) -> PositionEvent:
        return cls.from_dict(json.loads(s))


#: The ``type`` of a :class:`MotionUpdate` on ``WS /viewer``. A position event
#: carries no ``type`` key, so every message before this one stays valid as-is.
MOTION_TYPE = "motion"


@dataclass(frozen=True, slots=True)
class MotionUpdate:
    """Whether a hand is over the board right now: display state, not a read.

    The inference node sends one per ingested frame on ``WS /viewer`` (rate-limited
    per board, but always on a change of ``moving``), so a viewer can show "hand on
    board" between the sparse position events. It is never recorded and never
    retained: a stale ``moving`` would show a hand on a board nobody touches.

    ``stability`` is the frame's edge motion meter (1 = at rest), ``moving`` the
    node's verdict on it, and ``ts`` the frame's ``capture_ts`` (edge clock).
    """

    board_id: str
    moving: bool
    stability: float
    ts: float

    def to_dict(self) -> dict:
        return {
            "type": MOTION_TYPE,
            "board_id": self.board_id,
            "moving": self.moving,
            "stability": round(self.stability, 6),
            "ts": self.ts,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> MotionUpdate:
        return cls(
            board_id=str(d["board_id"]),
            moving=bool(d["moving"]),
            stability=float(d["stability"]),
            ts=float(d["ts"]),
        )


#: Everything ``WS /viewer`` sends.
ViewerMessage = PositionEvent | MotionUpdate


def parse_viewer_message(raw: str) -> ViewerMessage:
    """One ``WS /viewer`` text message as what it is.

    No ``type`` is a position event (§5); ``"motion"`` a :class:`MotionUpdate`.
    Any other ``type`` raises, because it would be our own egress sending
    something this protocol does not define.
    """
    d = json.loads(raw)
    kind = d.get("type")
    if kind is None:
        return PositionEvent.from_dict(d)
    if kind == MOTION_TYPE:
        return MotionUpdate.from_dict(d)
    raise ValueError(f"unknown /viewer message type {kind!r}")
