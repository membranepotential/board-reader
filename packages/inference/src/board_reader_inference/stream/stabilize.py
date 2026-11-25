"""Per-square temporal debounce over one board's estimated reads (DESIGN §5).

The estimator is stateless per frame by decision (D1), and the 08-13 phone trial
showed exactly what that costs a display: a hand crossing the board reads as a
phantom piece for a frame or two, and a near-coin-flip square (a k/q swap on
rank 8) flips back and forth at burst rate. Both are
short-lived events on a stream that publishes several reads per second.

:class:`BoardStabilizer` is the gateway-side answer, deliberately tiny: a square
changes only after reading the same new class on every read across
:data:`SETTLE_SECONDS` (and at least :data:`SETTLE_MIN_READS` reads — one read
alone is never evidence, however much wall clock passes around it). The window is
*time*, not a read count, because the read rate varies an order of magnitude with
estimator load (the trial ran ~0.9 reads/s; the same stream replayed unloaded is
7.5/s) and a count tuned at one rate is wrong at the other — measured on the
trial replay, the count a quiet board needs is ~3 at 0.9 reads/s but ~12 at
7.5/s. Time passes through both.

**Squares settle independently, but the board they compose must still be a board.**
That is the one thing a naive per-square debounce gets wrong, and it is not a
theoretical worry: on the 08-27 control take replayed at every frame, d8 settled
on ``k`` from one reading of a k/q swap while e8 settled on ``k`` from the
*other* reading, and the published settled board carried **two black kings on
69 % of its events** — a board no read ever claimed and no truth segment can ever
equal. So a displacement is committed only if it does not make the settled board
*less possible* than it already is (:func:`~board_reader_inference.stream.rules.violation`,
the excess over the conditions no standard game can break); when it would, the
squares the same read disagrees on are dragged along with it — see
:meth:`BoardStabilizer.observe`. The invariant is *monotone*, not *possible*: the
first read is adopted wholesale and sets the ceiling, and a read that is itself
impossible is never adopted over a more possible settled board. Under ChessQ the
raw reads are possible almost always (2 of 446 gate reads were not), so the
ceiling is zero in practice; the guard is what keeps the 3 impossible boards per
gate run that settle without it off the settled stream
(docs/reports/2026-11-24-prior-backrank.md).

The result is the event's ``placement`` (:meth:`~board_reader_core.protocol.PositionEvent.with_placement`):
estimation itself stays memoryless, the read's ``square_dist`` stays byte-identical
on the wire (its argmax is the raw board), and ``pending`` names the squares where
the two differ, so a consumer running its own smoothing still has every read. This
is a *display* debounce, not the downstream legality smoother
(DESIGN D3): it consults no move model and holds whatever the reads converge to,
right or wrong. The constraint check above is not legality either — it is the
necessary condition on the pieces, applied to the board the debounce composes out
of several reads.

**An obscured read is evidence, never a commit.** A hand resting still on the
board makes no motion, so the schedule reads it, and ChessQ reads the covered
squares as empty at p >= 0.96: confidence cannot tell. The frame energy can
(:func:`~board_reader_inference.recognition.chessq.frame_energy`), and above
:attr:`SettleConfig.obscured_energy` the read is *obscured*. It still starts and
extends challengers, because a mid-move read with the hand on the piece also says
something true (the lifted piece's square is empty), but no square ripens on it, so
nothing commits until a clear read confirms or resets each challenger. Dropping
obscured reads outright cost 15 gate boards in the probe, this 3
(docs/reports/2026-11-24-hand-signal-probe.md).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from board_reader_core.control import DEFAULT_RULES, Rules
from board_reader_core.protocol import (
    GRID,
    encode_placement,
    parse_placement,
)

from board_reader_inference.stream.rules import excess

#: How long a square must read as the same new class before the settled board
#: adopts it. With the shipped schedule's burst-end run (three reads 0.3 s apart,
#: after a 0.3 s calm hold) two reads span it, so a move settles 0.3 s after its
#: burst ends. Swept three times: over 0-2.5 s (docs/reports/2026-11-24-settle-tuning.md;
#: 2.5 s, sized for the old crop stack's noisy reads, mostly bought lag), then
#: 0.3-1.0 s against the 0.2-0.6 s spacings the forwarder's tail makes possible
#: (round 2 chose 0.5 s: with the energy gate not yet built, a window at or below
#: the spacing let hands settle), then 0-0.5 s with the energy gate on
#: (docs/reports/2026-11-25-latency-round3.md): 0.3 s settled no hand in any cell
#: and cut live take 2's median move-to-settle from 1.66 s to 0.94 s together with
#: the schedule's 0.3 s hold.
#:
#: **This window relies on the energy gate** (:data:`OBSCURED_ENERGY`,
#: ``BR_OBSCURED_ENERGY``, DESIGN D16), not on its own length, to keep hands out:
#: with the gate off, the round-3 settings let hands settle (6 on the gate takes,
#: 12 wrong boards in take 2's resting-hand stretch). Nothing falls back by
#: itself: with ``BR_OBSCURED_ENERGY=off``, return to round 2's settings with
#: ``BR_SETTLE_SECONDS=0.5`` and the schedule's ``exit_hold_s=0.7``.
#:
#: Shorter than one hand crossing (~1 s) on purpose: a hand the motion meter sees
#: is never read (the schedule skips bursts), and a hand resting still on the
#: board is what the energy gate catches.
SETTLE_SECONDS = 0.3

#: And never on fewer than this many consistent reads, however slow the stream —
#: at one read per window the debounce would otherwise be a pass-through. It is
#: also the quorum a square must have before it can be *dragged* into another
#: square's displacement (see :meth:`BoardStabilizer.observe`).
SETTLE_MIN_READS = 2

#: How many settle windows a ripe-but-blocked square may wait before the
#: debounce stops protecting the board it is blocked by and falls through to the
#: current read across the whole contested set. Without a deadline the coherence
#: rule can starve a square forever: a square that reads its challenger on
#: *every* read still never lands if committing it alone would raise the excess
#: and its partner flickers below quorum. 2 windows puts the worst-case
#: staleness at ~3x SETTLE_SECONDS (one window to ripen, two to time out).
STALE_WINDOWS = 2

#: A read whose frame energy is above this is obscured and commits nothing (module
#: docstring). The median of the probe's leave-one-take-out thresholds (each the
#: 99th percentile over the other gate takes' clear frames, -12.13 to -12.93) for
#: ChessQ V6. Energy lives on the model's logit scale, like a temperature: **re-fit
#: it for every model release** (docs/reports/2026-11-24-hand-signal-probe.md).
OBSCURED_ENERGY = -12.4

#: The environment ``serve`` reads the knobs from (:meth:`SettleConfig.from_env`).
SETTLE_ENV = (
    "BR_SETTLE_SECONDS",
    "BR_SETTLE_MIN_READS",
    "BR_SETTLE_STALE_WINDOWS",
    "BR_OBSCURED_ENERGY",
)

#: How ``BR_OBSCURED_ENERGY`` and ``replay --obscured-energy`` spell "no gate".
GATE_OFF = "off"

@dataclass(frozen=True)
class SettleConfig:
    """The debounce's knobs, as one value a node or a replay is built with.

    ``seconds`` and ``min_reads`` are the window a challenger must span
    (:data:`SETTLE_SECONDS`, :data:`SETTLE_MIN_READS`); ``stale_windows`` is how
    many windows a ripe-but-blocked square waits before the coherence rule gives
    way (:data:`STALE_WINDOWS`); ``obscured_energy`` is the frame energy above
    which a read commits nothing (:data:`OBSCURED_ENERGY`, ``None`` switches that
    gate off). ``serve`` reads them from :data:`SETTLE_ENV`, ``replay`` from
    ``--settle-*`` and ``--obscured-energy`` (defaulting to the same environment),
    and a replay records them in its manifest, name and config digest.
    """

    seconds: float = SETTLE_SECONDS
    min_reads: int = SETTLE_MIN_READS
    stale_windows: int = STALE_WINDOWS
    obscured_energy: float | None = OBSCURED_ENERGY

    def __post_init__(self) -> None:
        if not (math.isfinite(self.seconds) and self.seconds >= 0):
            raise ValueError(f"settle seconds must be finite and >= 0, got {self.seconds}")
        if self.min_reads < 1:
            raise ValueError(f"settle min_reads must be >= 1, got {self.min_reads}")
        if self.stale_windows < 1:
            raise ValueError(f"settle stale_windows must be >= 1, got {self.stale_windows}")
        if self.obscured_energy is not None and not math.isfinite(self.obscured_energy):
            raise ValueError(
                f"obscured energy must be finite (or {GATE_OFF!r}), got {self.obscured_energy}"
            )

    @property
    def is_default(self) -> bool:
        return self == DEFAULT_SETTLE

    @property
    def slug(self) -> str:
        """A path-safe name for a non-default config: ``settle1.5s-2r-2w``, with
        ``-e-12.0`` (or ``-eoff``) appended when the energy gate is not the default."""
        slug = f"settle{self.seconds:g}s-{self.min_reads}r-{self.stale_windows}w"
        if self.obscured_energy != OBSCURED_ENERGY:
            slug += f"-e{_energy_text(self.obscured_energy)}"
        return slug

    def obscures(self, frame_energy: float | None) -> bool:
        """Whether a read with this frame energy is obscured (module docstring).

        A read without one (the stub) never is, and neither is any read with the
        gate off.
        """
        return (
            self.obscured_energy is not None
            and frame_energy is not None
            and frame_energy > self.obscured_energy
        )

    def to_dict(self) -> dict[str, float | int | None]:
        """The manifest's form (its first two keys predate this class)."""
        return {
            "settle_seconds": self.seconds,
            "settle_min_reads": self.min_reads,
            "stale_windows": self.stale_windows,
            "obscured_energy": self.obscured_energy,
        }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> SettleConfig:
        """``BR_SETTLE_SECONDS`` / ``_MIN_READS`` / ``_STALE_WINDOWS``; unset keeps
        the default. A malformed value raises, so a typo fails at startup."""
        default = cls()
        seconds, min_reads, stale, energy = SETTLE_ENV
        return cls(
            seconds=_from_env(environ, seconds, float, default.seconds),
            min_reads=_from_env(environ, min_reads, int, default.min_reads),
            stale_windows=_from_env(environ, stale, int, default.stale_windows),
            obscured_energy=_from_env(
                environ,
                energy,
                parse_obscured_energy,
                default.obscured_energy,
                f"float or {GATE_OFF!r}",
            ),
        )


#: The shipped knobs, as the default argument everything that builds a debounce takes.
DEFAULT_SETTLE = SettleConfig()


def parse_obscured_energy(raw: str) -> float | None:
    """``"off"`` -> ``None`` (no gate), anything else a float; raises ``ValueError``.

    Finiteness is :class:`SettleConfig`'s check, so ``"nan"`` fails there.
    """
    return None if raw.strip().lower() == GATE_OFF else float(raw)


def _energy_text(energy: float | None) -> str:
    """``-12.0`` -> ``"-12"``, ``None`` -> ``"off"``: repr, a bare ``.0`` trimmed."""
    if energy is None:
        return GATE_OFF
    text = repr(energy)
    return text.removesuffix(".0")


def _from_env[T](
    environ: Mapping[str, str],
    name: str,
    parse: Callable[[str], T],
    default: T,
    what: str | None = None,
) -> T:
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return parse(raw)
    except ValueError as e:
        raise ValueError(f"{name}={raw!r} is not a valid {what or parse.__name__}") from e


def _excess(grid: list[list[str]], rules: Rules) -> int:
    """How impossible a board is, as the integer the guard keeps from rising."""
    return excess(encode_placement(grid), rules)


@dataclass
class _Candidate:
    """A square's current challenger to the settled class, and its evidence."""

    cls: str
    first_ts: float
    reads: int


class BoardStabilizer:
    """Time-window debounce for one board. Feed it every estimated read, in order.

    ``ts`` is the frame's capture timestamp on the *edge's* clock — only deltas
    between reads of the same stream are ever taken, so its NTP offset is
    irrelevant. The first read is adopted wholesale (a stream's first board is
    news, not flicker; and a square it gets wrong is displaceable like any
    other). After that a square changes only via an unbroken run of the same
    challenger class spanning ``settle_seconds`` and ``settle_min_reads``; any
    read that disagrees with both the settled class and the current challenger
    restarts the run, as does a timestamp going backwards.

    The one rule that is not per-square: **a commit may not make the board less
    possible than it already is.** The ripe displacements of one read are applied
    as a set; if that set would raise the hard-constraint excess, the squares
    this same read disagrees on — those already carrying ``settle_min_reads``
    consistent reads — are dragged into the commit, most evidence first, until
    the excess is no worse. The loop is bounded because it walks a finite set of
    squares once, not because it is guaranteed to arrive: its terminal board is
    the read itself only when *every* contested square carries the quorum. When
    it does not arrive, the ripe squares that can be committed one at a time
    without raising the excess are — greedily, in evidence order, which is not
    the same as the largest such subset: a pair that only works together is
    passed over and lands once both halves ripen on the same read. The rest keep
    their evidence and try again. In practice these partial and empty commits are
    the common case and a completed drag is rare: over the 40 seeded streams the
    tests sweep, 487 partial and 1770 empty commits against 2 completed drags.

    That is what stops one square's settlement being welded to another square's:
    a swap is read as a swap, and the settled board only ever shows halves of two
    incompatible readings for as long as it takes one of them to win.

    **Staleness is bounded.** Coherence alone can starve a square outright — one
    that reads its challenger on every single read still never lands if
    committing it alone would raise the excess and the partner that would relieve
    it flickers below the quorum, which is a real stream (settled ``k`` on d8
    against 200 straight reads of ``q``, held for 260 s). So a ripe square that
    has been blocked for :data:`STALE_WINDOWS` windows gives up on its partners
    and the contested set falls through to this read. Worst-case staleness is one window to ripen plus ``STALE_WINDOWS``
    to time out, ~3x ``settle_seconds``. The excess rule still gates that commit:
    a read that is *itself* less possible than the settled board is never adopted,
    however long it persists, so an impossible read cannot become the one thing
    a debounced display holds on to.
    """

    def __init__(self, config: SettleConfig = DEFAULT_SETTLE) -> None:
        self.config = config
        self._window = config.seconds
        self._min_reads = config.min_reads
        self._deadline = config.seconds * config.stale_windows
        self._settled: list[list[str]] | None = None
        self._candidates: dict[tuple[int, int], _Candidate] = {}
        #: When each ripe-but-uncommitted square first went ripe, so a blocked
        #: one can time out rather than wait on a partner that never arrives.
        self._ripe_since: dict[tuple[int, int], float] = {}
        self._changed_ts: float | None = None

    @property
    def changed_ts(self) -> float | None:
        """When the settled board last changed: the capture time of the earliest
        read that showed it, ``None`` before the first read.

        A commit adopts squares whose unbroken runs began at different reads (a
        move's two squares, a dragged partner); the board as it now stands was
        first read at the latest of those starts. The first read sets it to its
        own time. A read that commits nothing leaves it alone.
        """
        return self._changed_ts

    def observe(
        self,
        placement: str,
        ts: float,
        rules: Rules = DEFAULT_RULES,
        *,
        obscured: bool = False,
    ) -> str:
        """Fold one read into the settled board and return the settled placement.

        The squares where the read still differs from it are the event's
        ``pending`` (:func:`~board_reader_core.protocol.pending_squares`).

        ``rules`` is the board's setting at this read (:data:`~board_reader_core.control.Rules`):
        under ``free`` the guard counts nothing, so every ripe square commits. It
        may change from one read to the next; the guard then judges this read's
        commit by the new setting.

        ``obscured`` (:meth:`SettleConfig.obscures`) makes the read evidence only:
        it starts and extends challengers as usual, but no square ripens on it, so
        it commits nothing, not even through the stale fall-through. The first read
        is still adopted wholesale: there is no settled board to hold yet, and a
        wrong square is displaced by the clear reads that follow.

        Raises ``ValueError`` on an unparseable placement, like
        :func:`~board_reader_core.protocol.parse_placement`.
        """
        grid = parse_placement(placement)
        if self._settled is None:
            self._settled = [list(row) for row in grid]
            self._changed_ts = ts
            return placement

        contested: list[tuple[int, int]] = []
        ripe: list[tuple[int, int]] = []
        for r in range(GRID):
            for c in range(GRID):
                read = grid[r][c]
                if read == self._settled[r][c]:
                    self._forget((r, c))
                    continue
                challenger = self._candidates.get((r, c))
                if (
                    challenger is None
                    or challenger.cls != read
                    or ts < challenger.first_ts
                ):
                    self._forget((r, c))  # a new challenger waits its own turn
                    challenger = _Candidate(cls=read, first_ts=ts, reads=1)
                    self._candidates[(r, c)] = challenger
                else:
                    challenger.reads += 1
                contested.append((r, c))
                if (
                    not obscured
                    and challenger.reads >= self._min_reads
                    and ts - challenger.first_ts >= self._window
                ):
                    ripe.append((r, c))
                    self._ripe_since.setdefault((r, c), ts)

        taken = self._commit(grid, ripe, contested, ts, rules)
        if taken:
            self._changed_ts = max(self._candidates[square].first_ts for square in taken)
        for r, c in taken:
            self._settled[r][c] = grid[r][c]
            self._forget((r, c))
        return encode_placement(self._settled)

    # ----------------------------------------------------------------- commit

    def _commit(
        self,
        grid: list[list[str]],
        ripe: list[tuple[int, int]],
        contested: list[tuple[int, int]],
        ts: float,
        rules: Rules,
    ) -> list[tuple[int, int]]:
        """Which of ``ripe`` (plus any dragged square) to adopt from ``grid``.

        Four paths, tried in order, all of them holding the same guarantee — the
        excess never rises:

        1. the ripe set alone, when it costs nothing;
        2. the ripe set plus quorum-carrying partners of the same read, added in
           evidence order until it costs nothing (the *drag*);
        3. once a ripe square has been blocked for ``STALE_WINDOWS`` windows, and
           only if this read is itself no less possible than the settled board,
           the contested set, dragged in quorum-free until the excess clears;
        4. failing all that, whichever ripe squares can be committed one at a
           time without raising the excess, in evidence order — greedy, not
           maximum: a two-square commit that only works as a pair is passed over
           and lands on a later read once the pair ripens together.

        Paths 1 and 2 are the debounce; 3 is what stops it starving; 4 is what
        keeps one blocked square from freezing the rest of the board.
        """
        assert self._settled is not None
        if not ripe:
            return []
        held_excess = _excess(self._settled, rules)
        trial = [row[:] for row in self._settled]
        for r, c in ripe:
            trial[r][c] = grid[r][c]
        if _excess(trial, rules) <= held_excess:
            return ripe

        taken = list(ripe)
        for r, c in self._by_evidence(ripe, contested):
            trial[r][c] = grid[r][c]
            taken.append((r, c))
            if _excess(trial, rules) <= held_excess:
                return taken

        timed_out = any(
            ts - self._ripe_since[square] >= self._deadline
            for square in ripe
            if square in self._ripe_since
        )
        if timed_out and _excess(grid, rules) <= held_excess:
            # The same drag with the quorum waived. Its last step is the whole
            # contested set — this read, which the excess check above has already
            # cleared — so it cannot fall through, and it usually stops one or
            # two squares in rather than adopting every square in flux.
            trial = [row[:] for row in self._settled]
            taken = list(ripe)
            for r, c in taken:
                trial[r][c] = grid[r][c]
            for r, c in self._by_evidence(ripe, contested, quorum=False):
                trial[r][c] = grid[r][c]
                taken.append((r, c))
                if _excess(trial, rules) <= held_excess:
                    break
            return taken

        trial = [row[:] for row in self._settled]
        taken = []
        for r, c in self._by_evidence([], ripe):
            trial[r][c] = grid[r][c]
            if _excess(trial, rules) <= held_excess:
                taken.append((r, c))
            else:
                trial[r][c] = self._settled[r][c]
        return taken

    def _forget(self, square: tuple[int, int]) -> None:
        """Drop a square's challenger and its ripeness clock together."""
        self._candidates.pop(square, None)
        self._ripe_since.pop(square, None)

    def _by_evidence(
        self,
        exclude: list[tuple[int, int]],
        squares: list[tuple[int, int]],
        *,
        quorum: bool = True,
    ) -> list[tuple[int, int]]:
        """Quorum-carrying squares, best evidence first — the drag's total order.

        Most consistent reads wins; the older run breaks a tie; board order
        breaks what is left, so the same stream always drags the same squares.
        The order is observable in the settled board (which of two partners is
        adopted with the blocked square), not merely an implementation detail.
        """
        skip = set(exclude)
        return sorted(
            (
                square
                for square in squares
                if square not in skip
                and (not quorum or self._candidates[square].reads >= self._min_reads)
            ),
            key=lambda square: (
                -self._candidates[square].reads,
                self._candidates[square].first_ts,
                square,
            ),
        )
