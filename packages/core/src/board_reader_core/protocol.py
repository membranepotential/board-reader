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
    def from_dict(cls, d: dict | None) -> "ClockState":
        if not d:
            return cls()
        return cls(running=d.get("running", "none"))


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
    ) -> "PositionEvent":
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

    def validate(self) -> "PositionEvent":
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
        return self

    def to_dict(self, *, round_dist: int | None = 4) -> dict:
        """Serialise to the §5 JSON shape. ``round_dist`` trims distribution noise."""
        arr = np.asarray(self.square_dist, dtype=float)
        if round_dist is not None:
            arr = np.round(arr, round_dist)
        return {
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

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "PositionEvent":
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
        )

    @classmethod
    def from_json(cls, s: str) -> "PositionEvent":
        return cls.from_dict(json.loads(s))
