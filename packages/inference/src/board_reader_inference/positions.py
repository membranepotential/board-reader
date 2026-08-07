"""Headless consumer of the L3 position stream (DESIGN.md §5).

This is the other end of the thing the project exists to build. The gateway
publishes position events on ``WS /viewer``; until now the only consumer of that
socket was a browser page, which meant the system's *output* could be watched but
never captured, diffed, or asserted. A run's result lived on a screen.

:func:`follow` subscribes to the socket and yields validated
:class:`~board_reader_core.protocol.PositionEvent` objects; :func:`run` is the
``board-reader-inference positions`` command around it — a readable tail on
stdout, and optionally the events themselves as NDJSON:

    board-reader-inference positions                       # tail every board
    board-reader-inference positions --changes-only        # ... only when it moves
    board-reader-inference positions --out run.ndjson      # capture the stream

``--out`` writes each event as one line of the exact §5 JSON the socket carried,
so a captured run is the stream, not a rendering of it. That makes two runs
comparable and makes "a stream of positions came out" a claim a test can check.

.. rubric:: This is where temporal consistency will live, and does not yet

The estimator is stateless per frame by decision (DESIGN D1): it reads pixels and
reports what it sees, with no memory, no move tracking and no legality. Everything
that turns a *sequence* of noisy per-frame reads into a stable position history —
smoothing across frames, segmenting moves on the advisory ``stability`` signal,
rejecting a one-frame flicker as an occluded hand rather than a capture — belongs
to the **consumer** (DESIGN D3, §5). That consumer does not exist yet, in this
repo or outside it.

So the gap is not a missing feature in the CV layer; it is a missing layer. What
lands here today is deliberately memoryless too: :func:`transitions` diffs
consecutive placements so a stream is legible to a human, and that is all. It
smooths nothing, and a piece flickering between two classes on alternate frames
will show up as a transition every single frame — which is the honest picture of
what a stateless estimator emits, and the reason the layer above it is needed.
Anything cleverer than a diff is that layer's job and wants its own design.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from board_reader_core.protocol import (
    GRID,
    PositionEvent,
    parse_placement,
    square_name,
)

#: How an empty square is drawn in the human-readable tail. The NDJSON capture is
#: unaffected — it carries the protocol's own ``"empty"``.
EMPTY_GLYPH = "·"

DEFAULT_URL = "ws://127.0.0.1:8000/viewer"


@dataclass(frozen=True, slots=True)
class SquareTransition:
    """One square whose occupant differs between two consecutive placements.

    Purely a difference between two strings — it carries no claim that a *move*
    happened. A hand crossing the board, a misread, and a real capture all produce
    transitions, and telling them apart is the consumer layer's job (see the module
    docstring).
    """

    square: str
    before: str
    after: str

    def __str__(self) -> str:
        before = EMPTY_GLYPH if self.before == "empty" else self.before
        after = EMPTY_GLYPH if self.after == "empty" else self.after
        return f"{self.square} {before}→{after}"


def transitions(before: str, after: str) -> tuple[SquareTransition, ...]:
    """Squares that differ between two FEN-field-1 placements, in board order.

    Raises ``ValueError`` on an unparseable placement — the same contract
    :func:`~board_reader_core.protocol.parse_placement` has.
    """
    prev, cur = parse_placement(before), parse_placement(after)
    return tuple(
        SquareTransition(square_name(r, c), prev[r][c], cur[r][c])
        for r in range(GRID)
        for c in range(GRID)
        if prev[r][c] != cur[r][c]
    )


def format_event(
    event: PositionEvent, changed: tuple[SquareTransition, ...] | None = None
) -> str:
    """One line of tail: identity, the derived numbers, and what moved.

    ``changed`` is ``None`` for the first event of a board (nothing to diff
    against), which reads differently from an empty tuple (a board that held
    still) — and should.
    """
    head = (
        f"{event.board_id} #{event.seq} {event.utc}"
        f"  conf {event.confidence:.2f}  stab {event.stability:.2f}"
    )
    if event.status != "ok":
        head += f"  [{event.status}]"
    if changed is None:
        return f"{head}  {event.placement}"
    if not changed:
        return f"{head}  (unchanged)"
    return f"{head}  {len(changed)}Δ  " + ", ".join(str(c) for c in changed)


class StreamTail:
    """Per-board memory of the last placement, so a stream can be diffed.

    The *only* state this module keeps, and it exists for legibility rather than
    for estimation — nothing here feeds back into a position (D1).
    """

    def __init__(self) -> None:
        self._last: dict[str, str] = {}

    def observe(self, event: PositionEvent) -> tuple[SquareTransition, ...] | None:
        """Record ``event`` and return what changed since this board's last one.

        ``None`` the first time a board is seen: there is no previous placement,
        which is not the same as nothing having changed.
        """
        previous = self._last.get(event.board_id)
        self._last[event.board_id] = event.placement
        if previous is None:
            return None
        return transitions(previous, event.placement)


async def follow(
    url: str = DEFAULT_URL,
    *,
    board_id: str | None = None,
    limit: int | None = None,
) -> AsyncIterator[PositionEvent]:
    """Yield position events from the gateway's ``/viewer`` socket.

    The gateway replays its retained latest-per-board on connect, so a tail
    started mid-run reports the current position immediately rather than waiting
    for the next frame.

    ``limit`` stops after that many *yielded* events (events filtered out by
    ``board_id`` do not count) — it is what makes this testable and scriptable.
    A malformed event raises: it would be our own egress violating our own
    protocol, and swallowing that would hide a real bug.
    """
    import websockets

    yielded = 0
    async with websockets.connect(url, max_size=None) as ws:
        async for raw in ws:
            event = PositionEvent.from_json(
                raw if isinstance(raw, str) else raw.decode()
            )
            if board_id is not None and event.board_id != board_id:
                continue
            yield event
            yielded += 1
            if limit is not None and yielded >= limit:
                return


@dataclass(slots=True)
class StreamStats:
    """What a follow actually did.

    ``consumed`` and ``reported`` are equal unless ``changes_only`` held some
    events back, and they answer different questions: whether the stream was
    *alive* (consumed) and how much of it was worth writing down (reported).
    Collapsing them into one number makes "wrote N events" a lie the moment
    anything is filtered.

    Deliberately **mutable and caller-owned**: Ctrl-C is the ordinary way to end a
    tail, and it unwinds out of ``asyncio.run`` before any return value can be
    handed back. Counting into an object the caller already holds is what lets an
    interrupted run still report what it did, instead of claiming it saw nothing.
    """

    consumed: int = 0
    reported: int = 0


async def stream_to(
    url: str = DEFAULT_URL,
    *,
    board_id: str | None = None,
    limit: int | None = None,
    changes_only: bool = False,
    out: TextIO | None = None,
    tail: TextIO | None = None,
    stats: StreamStats | None = None,
) -> StreamStats:
    """Follow the stream, writing NDJSON to ``out`` and a readable tail to ``tail``.

    ``changes_only`` suppresses *both* outputs for an event whose placement
    matches the board's previous one — an at-rest board republishes the same
    position at the keepalive rate, and a log of that is noise. A board's first
    event is always reported: "never seen before" and "held still" are different
    facts. Suppressed events are still consumed, and ``limit`` counts them.

    Pass ``stats`` to keep the tally after an interrupt (see :class:`StreamStats`);
    otherwise a fresh one is made and returned.
    """
    tally = stats if stats is not None else StreamStats()
    seen = StreamTail()
    async for event in follow(url, board_id=board_id, limit=limit):
        tally.consumed += 1
        changed = seen.observe(event)
        if changes_only and changed is not None and not changed:
            continue
        tally.reported += 1
        if out is not None:
            out.write(json.dumps(event.to_dict()) + "\n")
            out.flush()  # a tail that buffers is not a tail
        if tail is not None:
            print(format_event(event, changed), file=tail, flush=True)
    return tally


def run(
    url: str = DEFAULT_URL,
    *,
    board_id: str | None = None,
    limit: int | None = None,
    changes_only: bool = False,
    out_path: str | None = None,
    tail: TextIO | None = None,
) -> StreamStats:
    """``board-reader-inference positions`` — blocking shell around :func:`stream_to`.

    Ctrl-C is the normal way to stop a tail, so it ends the run rather than
    raising — and the tally is kept, because a run the operator chose to stop
    still saw everything it reports.
    """
    import asyncio

    stats = StreamStats()
    handle = None
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8")
    try:
        asyncio.run(
            stream_to(
                url,
                board_id=board_id,
                limit=limit,
                changes_only=changes_only,
                out=handle,
                tail=tail,
                stats=stats,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        if handle is not None:
            handle.close()
    return stats
