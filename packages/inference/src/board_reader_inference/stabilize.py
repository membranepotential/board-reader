"""Per-square temporal debounce over one board's estimated reads (DESIGN §5).

The estimator is stateless per frame by decision (D1), and the 08-13 phone trial
showed exactly what that costs a display: a hand crossing the board reads as a
phantom piece for a frame or two, and a near-coin-flip constraint repair (two raw
kings, alternating demotion) swaps k/q back and forth at burst rate. Both are
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
69 % of its events** — a board no repaired read ever claimed and no truth segment
can ever equal. That is what "settled per-square 96.1 % below its own prior input
97.2 %, settled-perfect 0/253, never settles a segment" was made of. So a
displacement is committed only if it does not make the settled board *less
possible* than it already is (:func:`~board_reader_inference.prior.violation`,
the same hard-constraint excess the prior decoder descends); when it would, the
squares the same read disagrees on are dragged along with it — see
:meth:`BoardStabilizer.observe`. The invariant is *monotone*, not *possible*: the
first read is adopted wholesale and sets the ceiling, so feeding the debounce the
repaired placement (as the live path does) is what makes that ceiling zero.

The result travels as the advisory
:class:`~board_reader_core.protocol.SettledBoard` block beside the raw read —
estimation itself stays memoryless, the raw ``placement``/``square_dist`` stay
byte-identical on the wire, and a consumer running its own smoothing ignores the
block. This is a *display* debounce, not the downstream legality smoother
(DESIGN D3): it consults no move model and holds whatever the reads converge to,
right or wrong. The constraint check above is not legality either — it is the
same necessary condition on the piece multiset the prior already enforces per
read, applied to the board the debounce composes out of several of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from board_reader_core.protocol import (
    CLASSES,
    GRID,
    SettledBoard,
    encode_placement,
    parse_placement,
    square_name,
)

from .prior import tally, violation

#: How long a square must read as the same new class before the settled board
#: adopts it. Long enough that a hand crossing the board (~1 s) and the trial's
#: alternating repair runs mostly fall inside one window; short enough that a
#: real move appears within a beat of being played.
SETTLE_SECONDS = 2.5

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

#: Class symbol -> index, for handing a placement to the prior's tally.
_CLASS_INDEX = {symbol: index for index, symbol in enumerate(CLASSES)}


def _excess(grid: list[list[str]]) -> int:
    """The hard-constraint excess of a board, as an integer to be minimised.

    ``oriented=False``: the debounce is never told which way round the board is,
    so the back-rank-pawn constraint is left unclaimed rather than counted in an
    arbitrary frame. Kings, piece count and promotion budget need no frame.
    """
    labels = np.array([[_CLASS_INDEX[cls] for cls in row] for row in grid])
    return violation(tally(labels, oriented=False)).total


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
    and the whole contested set falls through to this read — which is the prior's
    own repair, and therefore the most possible board the debounce could compose
    out of it. Worst-case staleness is one window to ripen plus ``STALE_WINDOWS``
    to time out, ~3x ``settle_seconds``. The excess rule still gates that commit:
    a read that is *itself* less possible than the settled board is never adopted,
    however long it persists.

    Feed it the *repaired* placement (``prior.placement``): the raw argmax can be
    persistently impossible (the trial's two-kings rank 8), and an impossible
    board must not become the one thing a debounced display holds on to.
    """

    def __init__(
        self,
        settle_seconds: float = SETTLE_SECONDS,
        settle_min_reads: int = SETTLE_MIN_READS,
    ) -> None:
        if settle_seconds < 0:
            raise ValueError(f"settle_seconds must be >= 0, got {settle_seconds}")
        if settle_min_reads < 1:
            raise ValueError(f"settle_min_reads must be >= 1, got {settle_min_reads}")
        self._window = settle_seconds
        self._min_reads = settle_min_reads
        self._deadline = settle_seconds * STALE_WINDOWS
        self._settled: list[list[str]] | None = None
        self._candidates: dict[tuple[int, int], _Candidate] = {}
        #: When each ripe-but-uncommitted square first went ripe, so a blocked
        #: one can time out rather than wait on a partner that never arrives.
        self._ripe_since: dict[tuple[int, int], float] = {}

    def observe(self, placement: str, ts: float) -> SettledBoard:
        """Fold one read into the settled board and return the current settlement.

        Raises ``ValueError`` on an unparseable placement, like
        :func:`~board_reader_core.protocol.parse_placement`.
        """
        grid = parse_placement(placement)
        if self._settled is None:
            self._settled = [list(row) for row in grid]
            return SettledBoard(placement=placement)

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
                    challenger.reads >= self._min_reads
                    and ts - challenger.first_ts >= self._window
                ):
                    ripe.append((r, c))
                    self._ripe_since.setdefault((r, c), ts)

        for r, c in self._commit(grid, ripe, contested, ts):
            self._settled[r][c] = grid[r][c]
            self._forget((r, c))
        return SettledBoard(
            placement=encode_placement(self._settled),
            pending=tuple(
                square_name(r, c)
                for r, c in contested
                if (r, c) in self._candidates  # survivors: not committed above
            ),
        )

    # ----------------------------------------------------------------- commit

    def _commit(
        self,
        grid: list[list[str]],
        ripe: list[tuple[int, int]],
        contested: list[tuple[int, int]],
        ts: float,
    ) -> list[tuple[int, int]]:
        """Which of ``ripe`` (plus any dragged square) to adopt from ``grid``.

        Four paths, tried in order, all of them holding the same guarantee — the
        excess never rises:

        1. the ripe set alone, when it costs nothing;
        2. the ripe set plus quorum-carrying partners of the same read, added in
           evidence order until it costs nothing (the *drag*);
        3. once a ripe square has been blocked for ``STALE_WINDOWS`` windows, the
           whole contested set — which is exactly this read, and this read is the
           prior's own repair, so it is the one board guaranteed to be as
           possible as anything the debounce could compose out of it;
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
        held_excess = _excess(self._settled)
        trial = [row[:] for row in self._settled]
        for r, c in ripe:
            trial[r][c] = grid[r][c]
        if _excess(trial) <= held_excess:
            return ripe

        taken = list(ripe)
        for r, c in self._by_evidence(ripe, contested):
            trial[r][c] = grid[r][c]
            taken.append((r, c))
            if _excess(trial) <= held_excess:
                return taken

        timed_out = any(
            ts - self._ripe_since[square] >= self._deadline
            for square in ripe
            if square in self._ripe_since
        )
        if timed_out and _excess(grid) <= held_excess:
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
                if _excess(trial) <= held_excess:
                    break
            return taken

        trial = [row[:] for row in self._settled]
        taken = []
        for r, c in self._by_evidence([], ripe):
            trial[r][c] = grid[r][c]
            if _excess(trial) <= held_excess:
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
