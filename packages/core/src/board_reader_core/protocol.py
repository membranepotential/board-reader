"""L3 position-stream protocol — the shared wire contract (DESIGN.md §5).

This module is the spine of the system and is imported by BOTH the edge (Pi) and
the inference node. It carries NO node-specific logic: only the canonical
position-event shape, the 13-class vocabulary, and pure derivations over the
per-square distribution. Kept dependency-light (numpy and pydantic) so the same
contract runs on the Pi and the inference node; the browser gets it generated
(:mod:`.schema`).

Canonical output is the full 8x8x13 ``square_dist`` (decision #14); ``placement``
(FEN field 1) and ``confidence`` are *derived* from it, never authored alongside.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, get_args

import numpy as np
from pydantic import (
    AfterValidator,
    Discriminator,
    Field,
    PlainSerializer,
    PlainValidator,
    SerializationInfo,
    StrictBool,
    StrictFloat,
    StrictStr,
    Tag,
    WithJsonSchema,
)

from board_reader_core.wiremodel import (
    BoardId,
    Message,
    Natural,
    Unit,
    Wire,
    absent,
    adapter,
    checked,
    message,
    parse,
    rounded,
    tag,
    wire,
)

# 13-class vocabulary. Index order is canonical and travels in every event so a
# consumer can map every per-square vector. Uppercase = white, lowercase = black,
# ``empty`` is index 0 (DESIGN.md §5).
ClassName = Literal["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
CLASSES: tuple[ClassName, ...] = get_args(ClassName)
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
Status = Literal["ok", "recalibrating", "needs_setup", "stale"]
VALID_STATUS: tuple[Status, ...] = get_args(Status)
#: Which side's clock is running (DESIGN.md D7). ``none`` = neither/unknown.
Running = Literal["white", "black", "none"]
VALID_RUNNING: tuple[Running, ...] = get_args(Running)
#: The only frame an event names (reserved): the placement is in the chess frame.
Orientation = Literal["a1_bottom_left"]
DEFAULT_ORIENTATION: Orientation = "a1_bottom_left"

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

    Row 0 is rank 8, column 0 is file a. A digit is a run of 1 to 8 empty squares.
    """
    rows = placement.split("/")
    if len(rows) != GRID:
        raise ValueError(f"placement must have {GRID} ranks, got {len(rows)}")
    grid: list[list[str]] = []
    for rank in rows:
        cells: list[str] = []
        for ch in rank:
            if ch in "12345678":
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


# ---- the wire models (DESIGN.md §5) -------------------------------------------


def _placement(value: str) -> str:
    parse_placement(value)  # raises on a malformed board
    return value


#: FEN field 1, eight ranks of eight squares each (the ``isPlacement`` check).
Placement = Annotated[
    StrictStr, AfterValidator(_placement), Field(json_schema_extra=checked("isPlacement"))
]
#: An algebraic square name, ``a1`` .. ``h8``.
Square = Annotated[StrictStr, Field(pattern=r"^[a-h][1-8]$")]
#: RFC 3339 UTC with a ``Z``; :func:`iso_utc` writes milliseconds.
Utc = Annotated[
    StrictStr,
    Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$"),
]
#: A unit-interval score, rounded on the way in (see :func:`.schema.rounded`).
Score = Annotated[Unit, AfterValidator(rounded(6))]


def _classes(value: tuple[str, ...]) -> tuple[ClassName, ...]:
    if value != CLASSES:
        raise ValueError("event 'classes' does not match this protocol's CLASSES")
    return CLASSES


#: The class order every per-square vector is in; always exactly :data:`CLASSES`.
Classes = Annotated[
    tuple[StrictStr, ...],
    AfterValidator(_classes),
    WithJsonSchema({"type": "array", "const": list(CLASSES)}),
]

#: How many decimals ``to_dict``/``to_json`` keep of each probability by default.
ROUND_DIST = 4

_DIST_ROWS = adapter(list[list[list[StrictFloat]]])


def _square_dist(value: object) -> np.ndarray:
    """``value`` as the (8, 8, 13) float32 distribution, every entry in [0, 1].

    An array (the recognizers') is taken as it is; anything else must be nested
    JSON numbers, checked strictly so a ``true`` is not a probability.
    """
    if isinstance(value, np.ndarray):
        exact = value
    else:
        exact = np.asarray(_DIST_ROWS.validate_python(value), dtype=np.float64)
    if exact.shape != (GRID, GRID, NUM_CLASSES):
        raise ValueError(
            f"square_dist must be {GRID}x{GRID}x{NUM_CLASSES}, got {exact.shape}"
        )
    if not np.isfinite(exact).all() or exact.min() < 0 or exact.max() > 1:
        raise ValueError("square_dist entries must be probabilities in [0, 1]")
    return exact.astype(np.float32, copy=False)


def _square_dist_json(arr: np.ndarray, info: SerializationInfo) -> list:
    ndigits = (info.context or {}).get("round_dist", ROUND_DIST)
    out = np.asarray(arr, dtype=float)
    return (out if ndigits is None else np.round(out, ndigits)).tolist()


def _fixed(items: dict, n: int) -> dict:
    return {"type": "array", "items": items, "minItems": n, "maxItems": n}


#: The canonical 8x8x13 distribution: rank 8 -> rank 1, file a -> h, :data:`CLASSES`.
SquareDist = Annotated[
    np.ndarray,
    PlainValidator(_square_dist),
    PlainSerializer(_square_dist_json),
    WithJsonSchema(
        _fixed(_fixed(_fixed({"type": "number", "minimum": 0, "maximum": 1}, NUM_CLASSES), GRID), GRID)
    ),
]


@wire(frozen=False)
class ClockState(Wire):
    """Which side's clock is running (DESIGN.md D7). ``none`` = neither/unknown."""

    running: Running = "none"

    @classmethod
    def from_dict(cls, raw: object) -> ClockState:
        return cls() if not raw else parse(cls, raw)


@wire
class SettledBoard(Wire):
    """A temporally debounced board, served beside the raw read (never instead).

    The producer is the inference node's ingest stabilizer: a square must hold the
    same class for K consecutive *estimated* reads before the settled board adopts
    it, so one-frame flicker (a hand crossing the board, a coin-flip repair) never
    reaches a display that draws this block. ``pending`` names the squares
    currently reading differently from the settled board — a move in flight looks
    like its from/to squares going pending, then being adopted.

    Advisory: ``placement``/``square_dist`` on the event stay exactly what the
    estimator produced, and a consumer running its own smoothing ignores this
    block entirely.
    """

    placement: Placement
    pending: tuple[Square, ...] = ()  # board order


@wire(frozen=False, kw_only=True)
class PositionEvent(Wire):
    """One L3 position event (DESIGN.md §5).

    Construct via :meth:`from_square_dist` so ``placement`` and ``confidence`` stay
    derived from the canonical ``square_dist`` and can never drift out of sync.
    Every field is validated on construction and on assignment; :meth:`validate`
    adds the checks that need the whole distribution.
    """

    board_id: BoardId
    utc: Utc
    seq: Natural
    orientation: Orientation = DEFAULT_ORIENTATION
    classes: Classes = CLASSES
    placement: Placement
    square_dist: SquareDist
    confidence: Score
    stability: Score = 1.0
    clock: ClockState = Field(default_factory=ClockState)
    status: Status = "ok"
    #: Optional debounced board, produced by the ingest stabilizer on streamed
    #: frames only (a one-shot ``/estimate`` has no stream to settle over). A
    #: display that must not flicker draws ``settled.placement``.
    settled: SettledBoard | None = Field(default=None, exclude_if=absent)

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
        status: Status = "ok",
        orientation: Orientation = DEFAULT_ORIENTATION,
        confidence_reduce: str = "mean",
    ) -> PositionEvent:
        arr = _square_dist(np.asarray(square_dist, dtype=np.float32))
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
        """The whole-distribution checks; raises ``ValueError``. Returns self.

        Each square's distribution sums to 1 and ``placement`` is its argmax. Not
        part of parsing, because a distribution rounded for the wire can tie two
        classes the unrounded one did not.
        """
        sums = self.square_dist.sum(axis=-1)
        if np.any(np.abs(sums - 1.0) > _SUM_TOL):
            raise ValueError("each square distribution must sum to 1")
        if self.placement != placement_from_dist(self.square_dist):
            raise ValueError("placement is not the argmax of square_dist")
        return self

    def to_dict(self, *, round_dist: int | None = ROUND_DIST) -> dict:
        """Serialise to the §5 JSON shape. ``round_dist`` trims distribution noise."""
        return adapter(PositionEvent).dump_python(
            self, mode="json", context={"round_dist": round_dist}
        )


#: The ``type`` of a :class:`MotionUpdate` on ``WS /viewer``. A position event
#: carries no ``type`` key, so every message before this one stays valid as-is.
MOTION_TYPE = "motion"


@message
class MotionUpdate(Message):
    """Whether a hand is over the board right now: display state, not a read.

    The inference node sends one per ingested frame on ``WS /viewer`` (rate-limited
    per board, but always on a change of ``moving``), so a viewer can show "hand on
    board" between the sparse position events. It is never recorded and never
    retained: a stale ``moving`` would show a hand on a board nobody touches.

    ``stability`` is the frame's edge motion meter (1 = at rest), ``moving`` the
    node's verdict on it, and ``ts`` the frame's ``capture_ts`` (edge clock).
    """

    type: Literal["motion"] = tag(MOTION_TYPE)
    board_id: BoardId
    moving: StrictBool
    stability: Score
    ts: StrictFloat


def _viewer_kind(raw: object) -> str | None:
    if isinstance(raw, PositionEvent | MotionUpdate):
        return "position" if isinstance(raw, PositionEvent) else MOTION_TYPE
    if isinstance(raw, dict):
        return raw.get("type", "position")
    return None


#: Everything ``WS /viewer`` sends: a position event (no ``type``) or a motion update.
ViewerMessage = Annotated[
    Annotated[PositionEvent, Tag("position")] | Annotated[MotionUpdate, Tag(MOTION_TYPE)],
    Discriminator(
        _viewer_kind,
        custom_error_type="viewer_message_type",
        custom_error_message="unknown /viewer message type",
    ),
]


def parse_viewer_message(raw: str) -> PositionEvent | MotionUpdate:
    """One ``WS /viewer`` text message as what it is.

    No ``type`` is a position event (§5); ``"motion"`` a :class:`MotionUpdate`.
    Any other ``type`` raises, because it would be our own egress sending
    something this protocol does not define.
    """
    return parse(ViewerMessage, raw)
