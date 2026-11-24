"""Tests for the position-stream hub (DESIGN.md §4.5)."""

from __future__ import annotations

import asyncio

from board_reader_core.protocol import (
    START_PLACEMENT,
    MotionUpdate,
    PositionEvent,
    dist_from_placement,
    iso_utc,
)

from board_reader_inference.hub import PositionHub


def _event(board_id="A1", seq=0, placement=START_PLACEMENT):
    return PositionEvent.from_square_dist(
        board_id=board_id, seq=seq, utc=iso_utc(seq),
        square_dist=dist_from_placement(placement),
    )


def test_latest_is_retained_per_board():
    async def run():
        hub = PositionHub()
        await hub.publish(_event("A1", 0))
        await hub.publish(_event("A1", 1))
        await hub.publish(_event("B2", 0))
        assert hub.latest("A1").seq == 1
        assert hub.latest("B2").seq == 0
        assert hub.latest("Z9") is None
        assert set(hub.latest_all()) == {"A1", "B2"}

    asyncio.run(run())


def test_subscriber_receives_live_events():
    async def run():
        hub = PositionHub()
        gen = hub.subscribe(replay_latest=False)
        # Prime the subscription (registers the queue) without blocking forever.
        task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)  # let the generator register its queue
        await hub.publish(_event("A1", 7))
        ev = await asyncio.wait_for(task, timeout=1.0)
        assert ev.seq == 7
        await gen.aclose()

    asyncio.run(run())


def test_late_subscriber_gets_retained_snapshot_first():
    async def run():
        hub = PositionHub()
        await hub.publish(_event("A1", 3))
        await hub.publish(_event("B2", 4))
        gen = hub.subscribe(replay_latest=True)
        seen = {(await gen.__anext__()).board_id, (await gen.__anext__()).board_id}
        assert seen == {"A1", "B2"}  # both retained boards replayed on connect
        await gen.aclose()

    asyncio.run(run())


def test_subscriber_count_tracks_lifecycle():
    async def run():
        hub = PositionHub()
        assert hub.subscriber_count == 0
        gen = hub.subscribe(replay_latest=False)
        task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)
        assert hub.subscriber_count == 1
        await hub.publish(_event("A1", 1))
        await task
        await gen.aclose()
        assert hub.subscriber_count == 0

    asyncio.run(run())


def test_slow_subscriber_drops_oldest_not_newest():
    async def run():
        hub = PositionHub(queue_maxsize=2)
        gen = hub.subscribe(replay_latest=False)
        task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)
        # Publish more than the queue can hold without the consumer draining.
        for seq in range(5):
            await hub.publish(_event("A1", seq))
        first = await asyncio.wait_for(task, timeout=1.0)
        rest = []
        try:
            while True:
                rest.append(await asyncio.wait_for(gen.__anext__(), timeout=0.05))
        except asyncio.TimeoutError:
            pass
        seqs = [first.seq] + [e.seq for e in rest]
        # latest-wins: the newest event survives even under overflow.
        assert 4 in seqs
        await gen.aclose()

    asyncio.run(run())


def test_a_motion_update_reaches_live_subscribers_and_is_never_retained():
    async def run():
        hub = PositionHub()
        live = hub.subscribe(replay_latest=False)
        pending = asyncio.ensure_future(anext(live))
        await asyncio.sleep(0)
        update = MotionUpdate(board_id="A1", moving=True, stability=0.1, ts=5.0)
        hub.broadcast(update)
        assert await asyncio.wait_for(pending, 1.0) is update
        await live.aclose()
        assert hub.latest("A1") is None
        # A late subscriber gets the retained positions only, and here there are none.
        late = hub.subscribe()
        await hub.publish(_event("A1", 3))
        first = await asyncio.wait_for(anext(late), 1.0)
        assert isinstance(first, PositionEvent) and first.seq == 3
        await late.aclose()

    asyncio.run(run())
