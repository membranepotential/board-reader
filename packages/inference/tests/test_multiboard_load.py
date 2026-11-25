"""Several boards and input spikes change a node's latency, never its results.

The user's requirement (2026-11-25): "real-time latency can increase, but the
ultimate results and timestamps should not." A board's published stream (which
frames were read, the settled board after each, its ``changed_utc``) must
therefore be a function of that board's arriving frames alone, never of how fast
the recognizer is, how many other boards share it, or how the frames bunched up on
the way in.

The test drives one :class:`FrameIngest` with three boards at once, a recognizer
far slower than the arrival rate, and every frame sent faster than real time, and
compares each board's lines with the same board streamed alone through an
unloaded node. The reads are scripted per frame (a :data:`FrameRead` fake), with
occasional misreads, so a read landing on a different frame shows up in the lines;
the pixels are :mod:`board_reader_core.sim` renders with a hand over each move, so
``changed_utc``'s back-dating runs on real frames.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cache

import pytest

from board_reader_core.protocol import (
    dist_from_placement,
    encode_placement,
    parse_placement,
)
from board_reader_core.recording import Recording
from board_reader_core.sim import SAMPLE_GAME_PLACEMENTS, render_game_frames
from board_reader_core.wire import Frame

from board_reader_inference.measurement.replay import recorded_rules
from board_reader_inference.measurement.trial import read_events
from board_reader_inference.recognition.recognizer import (
    FrameReading,
    StubRecognizer,
    event_from_dist,
)
from board_reader_inference.server import create_app
from board_reader_inference.stream import ingest as ingest_module
from board_reader_inference.stream.hub import PositionHub
from board_reader_inference.stream.ingest import FrameIngest, drop_line, event_line
from board_reader_inference.stream.schedule.default import DEFAULT_POLICY_SPEC
from board_reader_inference.stream.schedule.parse import parse_policy
from board_reader_inference.stream.schedule.seam import ReadKind

#: Capture seconds between two frames (the edge's 10 fps burst rate).
DT = 0.1
#: Frames of rest per position: 2.5 s, longer than the 2 s keepalive.
REST_FRAMES = 25
#: Frames with a hand over the board per move (stability under ENTER_BURST).
HAND_FRAMES = 6
#: Positions each board plays through.
POSITIONS = 5
#: Seconds the slow recognizer takes per read. The arrival is a flood or clumps,
#: so any read time keeps the node behind; this one keeps the test fast.
SLOW_READ_S = 0.01


@dataclass(frozen=True, slots=True)
class _Board:
    """One board's arriving frames and what the fake recognizer reads in each."""

    board_id: str
    frames: tuple[Frame, ...]
    reads: dict[int, str]


def _misread(placement: str, seq: int) -> str:
    """``placement`` with one square read wrong, chosen by ``seq``."""
    grid = parse_placement(placement)
    row, col = 3 + seq % 2, seq % 8
    grid[row][col] = "empty" if grid[row][col] != "empty" else "P"
    return encode_placement(grid)


@cache
def _board(index: int) -> _Board:
    """A board playing ``POSITIONS`` sample-game positions from ply ``index``.

    Rest frames read their position, except every seventh (offset per board), which
    reads one square wrong; hand frames read the board under the hand with a
    misread too. Each board's clock and phase differ, so no two boards line up.
    """
    placements = SAMPLE_GAME_PLACEMENTS[index : index + POSITIONS]
    images: Iterator = render_game_frames(
        placements,
        frames_per_position=REST_FRAMES,
        transition_frames=HAND_FRAMES,
        seed=index,
        square_px=16,
    )
    board_id = f"B{index}"
    t0 = 1_764_000_000.0 + 37.0 * index
    frames: list[Frame] = []
    reads: dict[int, str] = {}
    seq = 0

    def add(placement: str, *, hand: bool) -> None:
        nonlocal seq
        frames.append(
            Frame(
                board_id=board_id,
                seq=seq,
                capture_ts=t0 + seq * DT,
                image=next(images),
                stability=0.3 if hand else 1.0,
            )
        )
        wrong = hand or (3 * seq + index) % 7 == 0
        reads[seq] = _misread(placement, seq) if wrong else placement
        seq += 1

    for i, placement in enumerate(placements):
        for _ in range(REST_FRAMES):
            add(placement, hand=False)
        if i + 1 < len(placements):
            for t in range(HAND_FRAMES):
                add(placement if 2 * t < HAND_FRAMES else placements[i + 1], hand=True)
    return _Board(board_id, tuple(frames), reads)


class _Scripted:
    """The fake recognizer: each frame reads what its board's script says, after
    ``delay_s``. It also keeps the order it served the reads in."""

    def __init__(self, boards: Sequence[_Board], delay_s: float) -> None:
        self._reads = {board.board_id: board.reads for board in boards}
        self._delay_s = delay_s
        self.served: list[tuple[str, int]] = []

    def __call__(self, frame: Frame) -> FrameReading:
        if self._delay_s:
            time.sleep(self._delay_s)
        self.served.append((frame.board_id, frame.seq))
        placement = self._reads[frame.board_id][frame.seq]
        return FrameReading(event_from_dist(frame, dist_from_placement(placement)))


def _node(read: _Scripted, lines: dict[str, list[dict]], drops: list | None = None) -> FrameIngest:
    """A node with the shipped schedule that records each board's published lines."""

    def on_event(reading: FrameReading, frame: Frame, rules) -> None:
        lines[frame.board_id].append(
            event_line(reading.event, frame, published_ts=None, rules=rules)
        )

    def on_drop(frame: Frame, kind: ReadKind, rules, waited_s: float) -> None:
        assert drops is not None, "a read was dropped"
        drops.append((frame.board_id, frame.seq, kind))

    return FrameIngest(
        read,
        PositionHub(),
        on_event=on_event,
        on_drop=on_drop,
        policy=lambda: parse_policy(DEFAULT_POLICY_SPEC),
    )


def _kinds(board: _Board) -> dict[int, ReadKind]:
    """Which of ``board``'s frames the shipped policy schedules, and as what."""
    policy = parse_policy(DEFAULT_POLICY_SPEC)
    kinds: dict[int, ReadKind] = {}
    for frame in board.frames:
        policy.observe(frame.capture_ts, frame.stability)
        kind = policy.due(frame.capture_ts, frame.stability)
        if kind is not None:
            kinds[frame.seq] = kind
    return kinds


async def _solo(board: _Board) -> list[dict]:
    """``board`` alone through an unloaded node: every read finishes before the
    next frame arrives, so nothing waits and nothing can be dropped."""
    lines: dict[str, list[dict]] = defaultdict(list)
    ingest = _node(_Scripted([board], 0.0), lines)
    for frame in board.frames:
        ingest.submit(frame)
        await ingest.drain()
    await ingest.close()
    return lines[board.board_id]


async def _flood(ingest: FrameIngest, boards: Sequence[_Board]) -> None:
    """Every board's every frame at once, interleaved frame by frame."""
    for nth in range(max(len(b.frames) for b in boards)):
        for board in boards:
            if nth < len(board.frames):
                ingest.submit(board.frames[nth])
        await asyncio.sleep(0)


async def _clumps(ingest: FrameIngest, boards: Sequence[_Board]) -> None:
    """Each board on its own task, its frames in clumps of ten (a network spike)."""

    async def one(board: _Board) -> None:
        for start in range(0, len(board.frames), 10):
            for frame in board.frames[start : start + 10]:
                ingest.submit(frame)
            await asyncio.sleep(0.005)

    await asyncio.gather(*(one(board) for board in boards))


@pytest.mark.parametrize("arrive", [_flood, _clumps], ids=["flood", "clumps"])
def test_three_loaded_boards_publish_exactly_what_each_publishes_alone(arrive):
    """Per board, the loaded node's lines equal the unloaded node's, line for line:
    the same frames read, the same settled boards, the same ``utc`` and
    ``changed_utc``, in the same order. Across boards the one recognizer serves the
    oldest scheduled read first, so no board waits behind another's backlog."""
    boards = [_board(index) for index in (0, 3, 6)]
    kinds = {board.board_id: _kinds(board) for board in boards}

    async def run():
        async with asyncio.timeout(30):
            solo = {board.board_id: await _solo(board) for board in boards}
            lines: dict[str, list[dict]] = defaultdict(list)
            read = _Scripted(boards, SLOW_READ_S)
            ingest = _node(read, lines)
            arrived: list[tuple[str, int]] = []
            submit = ingest.submit

            def watch(frame: Frame) -> None:
                if frame.seq in kinds[frame.board_id]:
                    arrived.append((frame.board_id, frame.seq))
                submit(frame)

            ingest.submit = watch  # type: ignore[method-assign]
            await arrive(ingest, boards)
            read_when_all_arrived = ingest.frames_estimated
            await ingest.drain()
            await ingest.close()
            return solo, lines, read_when_all_arrived, ingest, read.served, arrived

    solo, loaded, read_when_all_arrived, ingest, served, arrived = asyncio.run(run())

    total_reads = sum(len(lines) for lines in solo.values())
    assert read_when_all_arrived < total_reads, "the node was never behind: no load"
    for board in boards:
        expected = solo[board.board_id]
        settled = {line["placement"] for line in expected}
        assert len(settled) >= POSITIONS, "the script must settle every move"
        assert json.dumps(loaded[board.board_id]) == json.dumps(expected), board.board_id
    assert served == arrived, "the recognizer served the oldest scheduled read first"
    assert ingest.frames_received == sum(len(b.frames) for b in boards)
    assert (ingest.frames_dropped, ingest.frames_failed) == (0, 0)
    stats = ingest.queue_stats()
    assert stats.served == total_reads and stats.wait_max_s >= stats.wait_mean_s > 0


def test_a_sustained_overload_drops_only_keepalive_reads_and_counts_each(monkeypatch):
    """Past the overload rule's wait cap, the node gives up keepalive reads, never
    burst-end ones, and every drop reaches the drop observer and the counters."""
    monkeypatch.setattr(ingest_module, "MAX_KEEPALIVE_WAIT_S", 0.15)
    boards = [_board(index) for index in (0, 3, 6)]

    async def run():
        async with asyncio.timeout(30):
            lines: dict[str, list[dict]] = defaultdict(list)
            drops: list[tuple[str, int, ReadKind]] = []
            ingest = _node(_Scripted(boards, SLOW_READ_S), lines, drops)
            await _flood(ingest, boards)
            await ingest.drain()
            await ingest.close()
            return lines, drops, ingest

    lines, drops, ingest = asyncio.run(run())

    assert drops, "the overload never came"
    assert {kind for *_, kind in drops} == {ReadKind.KEEPALIVE}
    assert (ingest.frames_dropped, ingest.frames_dropped_burst_end) == (len(drops), 0)
    for board in boards:
        read = {line["frame_seq"] for line in lines[board.board_id]}
        burst_end = {s for s, k in _kinds(board).items() if k is ReadKind.BURST_END}
        assert burst_end <= read, board.board_id


def test_a_dropped_read_line_is_skipped_by_scoring_and_keeps_the_rules(tmp_path):
    """The drop line lands in ``events.ndjson`` beside the published reads; the
    trial scorer must skip it and ``replay --rules`` must still see the setting."""
    board = _board(0)
    frame = board.frames[0]
    (tmp_path / "frames.ndjson").write_text("")
    (tmp_path / "events.ndjson").write_text(
        json.dumps(
            drop_line(frame, ReadKind.KEEPALIVE, rules="free", waited_s=5.2, published_ts=1.0)
        )
        + "\n"
    )
    recording = Recording(tmp_path)
    assert read_events(recording) == []
    assert recorded_rules(recording) == {"free"}


def test_two_boards_on_one_socket_record_every_frame_under_a_slow_recognizer(
    gateway, drain_ingest, tmp_path, monkeypatch
):
    """Recording sits before the scheduler and the read queue, for every board: a
    recognizer that falls behind costs the recording nothing."""

    class _Slow(StubRecognizer):
        def read(self, image):
            time.sleep(SLOW_READ_S)
            return super().read(image)

    monkeypatch.setenv("BR_SCHEDULE", DEFAULT_POLICY_SPEC)
    monkeypatch.setenv("BR_RECORD_DIR", str(tmp_path / "rec"))
    client = gateway(create_app(_Slow(), PositionHub()))
    boards = [_board(index) for index in (0, 3)]
    frames = [f for pair in zip(*(b.frames[:40] for b in boards)) for f in pair]
    with client.websocket_connect("/ingest") as ws:
        for frame in frames:
            header, jpeg = frame.encode()
            ws.send_text(json.dumps(header))
            ws.send_bytes(jpeg)
            ws.receive_json()
    drain_ingest(client)

    recording = Recording.open(tmp_path / "rec")
    for board in boards:
        seqs = [entry.seq for entry in recording.frame_index(board_id=board.board_id)]
        assert seqs == list(range(40)), board.board_id
    stats = client.get("/healthz").json()["ingest"]
    assert stats["received"] == len(frames)
    assert stats["dropped"] == stats["dropped_burst_end"] == 0
    assert stats["queue"]["served"] == stats["estimated"] > 0
    assert stats["estimated"] + stats["dropped"] + stats["unscheduled"] == len(frames)
