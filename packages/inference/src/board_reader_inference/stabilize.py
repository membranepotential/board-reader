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

The result travels as the advisory
:class:`~board_reader_core.protocol.SettledBoard` block beside the raw read —
estimation itself stays memoryless, the raw ``placement``/``square_dist`` stay
byte-identical on the wire, and a consumer running its own smoothing ignores the
block. This is a *display* debounce, not the downstream legality smoother
(DESIGN D3): it consults no move model and holds whatever the reads converge to,
right or wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from board_reader_core.protocol import (
    GRID,
    SettledBoard,
    encode_placement,
    parse_placement,
    square_name,
)

#: How long a square must read as the same new class before the settled board
#: adopts it. Long enough that a hand crossing the board (~1 s) and the trial's
#: alternating repair runs mostly fall inside one window; short enough that a
#: real move appears within a beat of being played.
SETTLE_SECONDS = 2.5

#: And never on fewer than this many consistent reads, however slow the stream —
#: at one read per window the debounce would otherwise be a pass-through.
SETTLE_MIN_READS = 2


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
    news, not flicker). After that a square changes only via an unbroken run of
    the same challenger class spanning ``settle_seconds`` and
    ``settle_min_reads``; any read that disagrees with both the settled class and
    the current challenger restarts the run, as does a timestamp going backwards.

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
        self._settled: list[list[str]] | None = None
        self._candidates: dict[tuple[int, int], _Candidate] = {}

    def observe(self, placement: str, ts: float) -> SettledBoard:
        """Fold one read into the settled board and return the current settlement.

        Raises ``ValueError`` on an unparseable placement, like
        :func:`~board_reader_core.protocol.parse_placement`.
        """
        grid = parse_placement(placement)
        if self._settled is None:
            self._settled = [list(row) for row in grid]
            return SettledBoard(placement=placement)
        pending: list[str] = []
        for r in range(GRID):
            for c in range(GRID):
                read = grid[r][c]
                if read == self._settled[r][c]:
                    self._candidates.pop((r, c), None)
                    continue
                challenger = self._candidates.get((r, c))
                if challenger is None or challenger.cls != read or ts < challenger.first_ts:
                    challenger = _Candidate(cls=read, first_ts=ts, reads=1)
                    self._candidates[(r, c)] = challenger
                else:
                    challenger.reads += 1
                if (
                    challenger.reads >= self._min_reads
                    and ts - challenger.first_ts >= self._window
                ):
                    self._settled[r][c] = read
                    self._candidates.pop((r, c))
                else:
                    pending.append(square_name(r, c))
        return SettledBoard(
            placement=encode_placement(self._settled), pending=tuple(pending)
        )
