"""The headless L3 consumer (:mod:`board_reader_inference.positions`).

Two layers, tested apart: the pure diff/format helpers, and :func:`follow` against
a real WebSocket server emitting real §5 JSON. The latter uses a bare
``websockets.serve`` rather than the gateway, so a failure here is the consumer's
and not the estimator's — the whole-pipe proof lives in ``tests/test_end_to_end``.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from websockets.asyncio.server import serve

from board_reader_core.protocol import (
    EMPTY_PLACEMENT,
    START_PLACEMENT,
    MotionUpdate,
    PositionEvent,
    dist_from_placement,
    iso_utc,
)

from board_reader_inference.positions import (
    StreamTail,
    follow,
    format_event,
    stream_to,
    transitions,
)

AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"


def _event(placement: str, *, seq: int = 0, board: str = "A1", **kw) -> PositionEvent:
    return PositionEvent.from_square_dist(
        board_id=board,
        seq=seq,
        utc=iso_utc(1_770_000_000.0 + seq),
        square_dist=dist_from_placement(placement, kw.pop("confidence", 0.9)),
        **kw,
    )


# --- the pure part -----------------------------------------------------------


def test_transitions_of_a_pawn_push_names_both_squares():
    changed = transitions(START_PLACEMENT, AFTER_E4)
    assert [str(c) for c in changed] == ["e4 ·→P", "e2 P→·"]


def test_transitions_are_empty_for_an_unchanged_board():
    assert transitions(START_PLACEMENT, START_PLACEMENT) == ()


def test_transitions_report_the_before_and_after_occupant():
    (change,) = transitions(
        "8/8/8/8/8/8/8/7R", "8/8/8/8/8/8/8/7q"
    )
    assert (change.square, change.before, change.after) == ("h1", "R", "q")


def test_transitions_cover_a_full_clear():
    assert len(transitions(START_PLACEMENT, EMPTY_PLACEMENT)) == 32


def test_transitions_reject_an_unparseable_placement():
    with pytest.raises(ValueError):
        transitions(START_PLACEMENT, "not-a-placement")


def test_first_sighting_of_a_board_is_none_not_empty():
    """``None`` (no previous position) and ``()`` (held still) must not collapse."""
    tail = StreamTail()
    assert tail.observe(_event(START_PLACEMENT)) is None
    assert tail.observe(_event(START_PLACEMENT, seq=1)) == ()


def test_stream_tail_diffs_each_board_independently():
    tail = StreamTail()
    tail.observe(_event(START_PLACEMENT, board="A1"))
    tail.observe(_event(START_PLACEMENT, board="B2"))
    # B2 moving must not be attributed to A1, nor reset A1's memory.
    assert tail.observe(_event(AFTER_E4, seq=1, board="B2")) != ()
    assert tail.observe(_event(START_PLACEMENT, seq=1, board="A1")) == ()


def test_format_event_shows_the_placement_for_a_first_sighting():
    line = format_event(_event(START_PLACEMENT), None)
    assert START_PLACEMENT in line
    assert "A1 #0" in line


def test_format_event_marks_an_unchanged_board():
    assert "(unchanged)" in format_event(_event(START_PLACEMENT), ())


def test_format_event_lists_the_transitions():
    line = format_event(_event(AFTER_E4, seq=3), transitions(START_PLACEMENT, AFTER_E4))
    assert "2Δ" in line
    assert "e4 ·→P" in line and "e2 P→·" in line


def test_format_event_surfaces_a_non_ok_status():
    line = format_event(_event(START_PLACEMENT, status="needs_setup"), ())
    assert "[needs_setup]" in line


# --- the socket part ---------------------------------------------------------


async def _serve(events: list[PositionEvent]):
    """A stand-in for the gateway's ``/viewer``: send the events, then hang up.

    The handler *returns* rather than parking on a never-resolving future. Parking
    is the obvious way to write "the client decides when it is done", and it
    deadlocks teardown: ``Server.__aexit__`` waits for its handlers, so the test
    hangs for as long as the runner allows and reports nothing. Closing cleanly
    also exercises the path where the stream ends on its own.
    """

    async def handler(ws):
        for event in events:
            await ws.send(event.to_json())

    return await serve(handler, "127.0.0.1", 0)


def _url(server) -> str:
    return f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"


def _run(coro, *, seconds: float = 10.0):
    """Run ``coro`` under a hard deadline.

    "Never terminates" is the failure mode a stream consumer regresses into most
    easily, and a hanging test costs a whole suite run while asserting nothing.
    Everything here is in-process, so ten seconds is orders of magnitude of slack.
    """

    async def guarded():
        async with asyncio.timeout(seconds):
            return await coro

    return asyncio.run(guarded())


def test_follow_yields_validated_events_in_order():
    async def go():
        server = await _serve([_event(START_PLACEMENT, seq=n) for n in range(3)])
        async with server:
            return [e async for e in follow(_url(server), limit=3)]

    events = _run(go())
    assert [e.seq for e in events] == [0, 1, 2]
    # The consumer must hand back the canonical event, not a dict: placement is
    # derived from square_dist, so a broken round trip shows up here.
    assert all(e.placement == START_PLACEMENT for e in events)
    assert events[0].square_dist.shape == (8, 8, 13)


def test_follow_skips_motion_updates():
    """The /viewer socket interleaves motion updates; they are not positions."""
    motion = MotionUpdate(board_id="A1", moving=True, stability=0.2, ts=1.0)

    async def go():
        server = await _serve([motion, _event(START_PLACEMENT, seq=4), motion])
        async with server:
            return [e async for e in follow(_url(server))]

    assert [e.seq for e in _run(go())] == [4]


def test_follow_filters_by_board_and_limit_counts_only_yielded_events():
    async def go():
        server = await _serve(
            [
                _event(START_PLACEMENT, seq=0, board="A1"),
                _event(START_PLACEMENT, seq=1, board="B2"),
                _event(AFTER_E4, seq=2, board="A1"),
            ]
        )
        async with server:
            return [e async for e in follow(_url(server), board_id="A1", limit=2)]

    events = _run(go())
    assert [(e.board_id, e.seq) for e in events] == [("A1", 0), ("A1", 2)]


def test_follow_raises_on_an_event_that_violates_the_protocol():
    """Our own egress emitting a malformed event is a bug, not a frame to skip."""

    async def go():
        async def handler(ws):
            await ws.send(json.dumps({"board_id": "A1", "seq": 0}))

        server = await serve(handler, "127.0.0.1", 0)
        async with server:
            return [e async for e in follow(_url(server), limit=1)]

    with pytest.raises((KeyError, ValueError)):
        _run(go())


def test_stream_to_writes_the_stream_verbatim_as_ndjson():
    out, tail = io.StringIO(), io.StringIO()
    sent = [_event(START_PLACEMENT, seq=0), _event(AFTER_E4, seq=1)]

    async def go():
        server = await _serve(sent)
        async with server:
            return await stream_to(_url(server), limit=2, out=out, tail=tail)

    stats = _run(go())
    assert (stats.consumed, stats.reported) == (2, 2)
    lines = out.getvalue().strip().split("\n")
    assert len(lines) == 2
    # "Verbatim" is the claim that makes a capture a reproduction: the written
    # line must rebuild the identical event, square_dist included.
    rebuilt = [PositionEvent.from_dict(json.loads(line)) for line in lines]
    assert [e.placement for e in rebuilt] == [START_PLACEMENT, AFTER_E4]
    assert rebuilt[0].to_dict() == sent[0].to_dict()
    assert "e4 ·→P" in tail.getvalue()


def test_changes_only_suppresses_a_board_that_held_still():
    out, tail = io.StringIO(), io.StringIO()

    async def go():
        server = await _serve(
            [
                _event(START_PLACEMENT, seq=0),
                _event(START_PLACEMENT, seq=1),  # keepalive: same position
                _event(START_PLACEMENT, seq=2),  # keepalive: same position
                _event(AFTER_E4, seq=3),
            ]
        )
        async with server:
            return await stream_to(
                _url(server), limit=4, changes_only=True, out=out, tail=tail
            )

    # All four are consumed; only the first sighting and the move are reported.
    # The two counts must not collapse — "wrote N" is a claim about the file, and
    # `wc -l` has to agree with it.
    stats = _run(go())
    assert (stats.consumed, stats.reported) == (4, 2)
    written = [json.loads(line) for line in out.getvalue().strip().split("\n")]
    assert [e["seq"] for e in written] == [0, 3]
    assert len(written) == stats.reported
    assert "(unchanged)" not in tail.getvalue()
