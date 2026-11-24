"""The read scheduler on the LIVE ingest path (DESIGN D3/D4).

``test_schedule.py`` proves the policies decide correctly from a header sequence.
This module proves the gateway *asks them at the right moment*, which is a
different claim and the one that took a design to get right:

* the policy is **observed at submit** — every frame that lands, in arrival order,
  because burst end is only visible from the whole run and the latest-wins inbox
  hides most of it;
* it is **asked at drain** — for whichever frame the worker actually picked, so a
  burst-end read spends itself on the sharper contender
  :func:`~board_reader_inference.stream.ingest.prefer_frame` kept rather than on whichever
  frame happened to arrive at that instant. That interplay is the sweep's own
  recommendation (report §8, "apply the policy to the ingest inbox");
* the floor holds through every way the signal can fail — a stuck meter, an absent
  one, and a ``capture_ts`` that stops advancing (which the policy alone cannot
  see, so the worker backstops it on the node's own clock);
* ``BR_SCHEDULE=every-frame`` is byte-for-byte the gateway that shipped before this.

Frames carry a synthetic capture clock (:data:`DT` seconds apart) so every
expectation is arithmetic rather than a sleep. The one test that must measure real
time says so.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time

import numpy as np
import pytest

from board_reader_core.protocol import START_PLACEMENT, PositionEvent
from board_reader_core.recording import Recording
from board_reader_core.wire import Frame

from board_reader_inference.recognition.recognizer import (
    FrameReading,
    StubRecognizer,
    frame_reader,
)
from board_reader_inference.server import create_app
from board_reader_inference.stream.hub import PositionHub
from board_reader_inference.stream.ingest import BACKSTOP_SLACK, FrameIngest
from board_reader_inference.stream.schedule import (
    DEFAULT_POLICY_SPEC,
    BurstTriggered,
    EveryFrame,
    parse_policy,
)
from board_reader_inference.stream.stabilize import SETTLE_SECONDS

#: Capture seconds between two frames — 10 fps, a plausible in-burst rate.
DT = 0.1

#: The shipped policy, built fresh per board (the factory shape ``FrameIngest``
#: wants). Every expectation below is against these exact constants.
SHIPPED = lambda: parse_policy(DEFAULT_POLICY_SPEC)  # noqa: E731


def _frame(
    seq: int,
    *,
    stability: float = 1.0,
    board_id: str = "A1",
    sharpness: float | None = None,
    capture_ts: float | None = None,
) -> Frame:
    return Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=seq * DT if capture_ts is None else capture_ts,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        stability=stability,
        sharpness=sharpness,
    )


class _Recorder:
    """An estimator that records which frames were spent on it, per board."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []
        self._stub = frame_reader(StubRecognizer(START_PLACEMENT))

    def estimate(self, frame: Frame) -> FrameReading:
        self.seen.append((frame.board_id, frame.seq))
        return self._stub(frame)

    def seqs(self, board_id: str = "A1") -> list[int]:
        return [seq for board, seq in self.seen if board == board_id]


def _one_move(rest: int = 5, move: int = 6, after: int = 49) -> list[float]:
    """Calm, one hand over the board, calm again — as a stability sequence."""
    return [1.0] * rest + [0.4] * move + [0.95] * after


async def _drive(
    policy, stabilities: list[float], *, board_id: str = "A1"
) -> tuple[list[int], FrameIngest]:
    """Stream a stability sequence, draining after each frame.

    Draining between submits keeps the inbox from ever holding two frames, so what
    this measures is the *schedule* and nothing else — the inbox interplay gets its
    own test below, where contention is arranged on purpose.
    """
    est = _Recorder()
    ingest = FrameIngest(
        est.estimate, PositionHub(), policy=policy
    )
    for seq, stability in enumerate(stabilities):
        ingest.submit(_frame(seq, stability=stability, board_id=board_id))
        await asyncio.wait_for(ingest.drain(), timeout=5.0)
    await ingest.close()
    return est.seqs(board_id), ingest


# --------------------------------------------------------------------------- #
# What the shipped policy does to a live stream
# --------------------------------------------------------------------------- #


def test_a_completed_move_buys_three_spaced_reads_and_nothing_during_the_hand():
    """The whole point of the shipped policy, end to end through the worker.

    The move runs from frame 5 to 10; the burst ends ``EXIT_HOLD_S`` of calm later
    at frame 18, which owes three reads ``spacing_s=0.3`` apart — 0.6 s of span,
    which is what :data:`~board_reader_inference.stream.stabilize.SETTLE_SECONDS` needs
    before a completed move can settle on its own evidence.
    """

    async def run():
        seen, ingest = await _drive(SHIPPED, _one_move())
        # frame 0 (the stream's first), the move-complete frame, its two follow-ups
        # a beat apart, then the 2 s keepalive. 6.0 s of capture time, five reads.
        assert seen == [0, 18, 21, 24, 44]
        assert not [s for s in seen if 5 <= s <= 18 - 1], "nothing read inside the hand"
        gaps = np.diff([s * DT for s in seen[1:]])
        assert gaps.min() >= 0.3 - DT
        assert seen[-1] * DT - seen[1] * DT >= SETTLE_SECONDS, "the run spans a settle window"
        # 60 frames arrived; 55 of them cost nothing beyond the decode.
        assert ingest.frames_received == 60
        assert ingest.frames_estimated == 5
        assert ingest.frames_unscheduled == 55
        assert ingest.frames_backstopped == 0

    asyncio.run(run())


def test_a_resting_board_is_read_once_per_keepalive():
    """Nobody has touched the board for ten seconds. The node reads it five times."""

    async def run():
        seen, ingest = await _drive(SHIPPED, [1.0] * 100)
        assert seen == [0, 20, 40, 60, 80]  # 2.0 s apart, the floor exactly
        assert ingest.frames_unscheduled == 95

    asyncio.run(run())


@pytest.mark.parametrize("stability", [0.2, 1.0, None])
def test_a_dead_or_absent_meter_still_reads_at_the_floor(stability):
    """D3, on the live path: stability is a hint and never a gate.

    Three ways the signal fails, all of which have happened. ``0.2`` is the meter
    that shipped before 2026-08-27 on the deployment phone — permanently "moving",
    so a burst never *ends* and every read must come from the floor. ``1.0`` is a
    meter that never dips, so there is no burst to trigger on. ``None`` is an edge
    that does not measure motion at all. None of them may read zero.
    """

    async def run():
        raw = 1.0 if stability is None else stability
        seen, ingest = await _drive(SHIPPED, [raw] * 100)
        assert seen == [0, 20, 40, 60, 80]
        assert ingest.frames_estimated == 5
        assert ingest.frames_backstopped == 0  # the capture clock is fine

    asyncio.run(run())


def test_boards_schedule_independently():
    """One policy per board, like one stabilizer per board.

    Two tables on one node: a move on one must neither spend the read budget of
    the other nor silence it. The burst state is the camera's, so it cannot be
    shared any more than the debounce can.
    """

    async def run():
        est = _Recorder()
        ingest = FrameIngest(
            est.estimate, PositionHub(), policy=SHIPPED
        )
        busy, quiet = _one_move(), [1.0] * 60
        for seq, (moving, still) in enumerate(zip(busy, quiet)):
            ingest.submit(_frame(seq, stability=moving, board_id="BUSY"))
            ingest.submit(_frame(seq, stability=still, board_id="QUIET"))
            await asyncio.wait_for(ingest.drain(), timeout=5.0)
        await ingest.close()
        return est

    est = asyncio.run(run())
    assert est.seqs("BUSY") == [0, 18, 21, 24, 44]  # its own move, its own quorum
    assert est.seqs("QUIET") == [0, 20, 40]  # 6.0 s of rest: the floor, three times



# --------------------------------------------------------------------------- #
# The inbox interplay — the sweep's own recommendation
# --------------------------------------------------------------------------- #


class _GatedEstimator:
    """Blocks the worker inside its FIRST estimate, so the inbox can fill.

    The real thing takes ~1.3 GPU-seconds; this takes exactly as long as the test
    needs it to, in the same place — the worker thread.
    """

    def __init__(self) -> None:
        self.seen: list[int] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self._stub = frame_reader(StubRecognizer(START_PLACEMENT))

    def estimate(self, frame: Frame) -> FrameReading:
        self.seen.append(frame.seq)
        if not self.entered.is_set():
            self.entered.set()
            assert self.release.wait(timeout=10.0), "the test never released the worker"
        return self._stub(frame)


def test_a_burst_end_read_gets_the_sharper_contender_the_inbox_kept():
    """Why the decision happens at the drain and not at arrival.

    Frame 18 is the move-complete moment and it is *sharp*; frame 19 arrives a
    tenth of a second later and is smeared. Latest-wins would hand the burst-end
    read the blurry one — exactly the weakness take 5 exposed in the sweep. Because
    the policy is only *observed* on arrival and *asked* when the worker picks a
    frame, ``prefer_frame`` gets to keep the sharp contender and that is what the
    single expensive read is spent on.
    """

    async def run():
        est = _GatedEstimator()
        ingest = FrameIngest(
            est.estimate, PositionHub(), policy=SHIPPED
        )
        ingest.submit(_frame(0))  # the stream's first read — parks the worker
        while not est.entered.is_set():
            await asyncio.sleep(0.005)

        # The move, then the calm that ends it. All of these pile into the inbox
        # behind the parked worker; every one of them is still OBSERVED.
        for seq in range(5, 18):
            ingest.submit(_frame(seq, stability=0.4 if seq <= 10 else 0.95))
        sharp = _frame(18, stability=0.95, sharpness=900.0)  # ends the burst
        blurry = _frame(19, stability=0.95, sharpness=120.0)  # newer, smeared
        ingest.submit(sharp)
        ingest.submit(blurry)

        est.release.set()
        await asyncio.wait_for(ingest.drain(), timeout=5.0)
        await ingest.close()
        return est.seen, ingest

    seen, ingest = asyncio.run(run())
    assert seen == [0, 18], "the burst-end read landed on the sharp frame"
    assert 19 not in seen, "…and not on the newest frame, which was the blurry one"
    assert ingest.frames_received == 16
    assert ingest.frames_dropped == 14  # everything the inbox displaced


def test_the_policy_sees_every_arriving_frame_not_just_the_ones_it_reads():
    """The reason ``observe`` lives in ``submit``.

    Under a slow estimator the inbox discards most of a burst. If the policy only
    saw the survivors it would see a board that went from calm to calm and never
    notice the move at all — no burst, no burst end, and a completed move that only
    the keepalive eventually catches, seconds late.
    """

    async def run():
        est = _GatedEstimator()
        ingest = FrameIngest(
            est.estimate, PositionHub(), policy=SHIPPED
        )
        ingest.submit(_frame(0))
        while not est.entered.is_set():
            await asyncio.sleep(0.005)
        # The whole move arrives while the worker is busy: the inbox keeps ONE of
        # these, and the burst is only in the run of headers.
        for seq in range(5, 19):
            ingest.submit(_frame(seq, stability=0.4 if seq <= 10 else 0.95))
        est.release.set()
        await asyncio.wait_for(ingest.drain(), timeout=5.0)
        await ingest.close()
        return est.seen

    # Frame 18 is the move-complete frame; the burst end made it due IMMEDIATELY,
    # 1.8 s in, rather than at the 2.0 s keepalive.
    assert asyncio.run(run()) == [0, 18]


# --------------------------------------------------------------------------- #
# The floors under the floor
# --------------------------------------------------------------------------- #


def test_a_frozen_capture_clock_falls_back_to_the_nodes_own_keepalive():
    """The one pathology a capture-time floor cannot see.

    A policy decides from ``capture_ts`` only, which is what makes an offline
    replay deterministic — and it means a camera whose timestamp stops advancing
    (a stalled pipeline restamping its last frame) leaves the policy believing no
    time has passed, forever. That is the dead-meter incident again with a
    different broken sensor, so the worker keeps the same promise on the node's own
    monotonic clock. Real time is the point of this test, hence the tiny keepalive.
    """

    async def run():
        est = _Recorder()
        policy = BurstTriggered(keepalive_s=0.06, burst_end_reads=1, spacing_s=0.02)
        ingest = FrameIngest(
            est.estimate, PositionHub(), policy=lambda: policy
        )
        deadline = time.monotonic() + 5.0
        seq = 0
        while ingest.frames_backstopped < 2 and time.monotonic() < deadline:
            ingest.submit(_frame(seq, capture_ts=5.0))  # the clock never moves
            await asyncio.wait_for(ingest.drain(), timeout=5.0)
            await asyncio.sleep(BACKSTOP_SLACK * policy.keepalive_s / 2)
            seq += 1
        await ingest.close()
        return est.seen, ingest

    seen, ingest = asyncio.run(run())
    assert ingest.frames_backstopped >= 2, "the node stopped reading a live stream"
    assert len(seen) == 1 + ingest.frames_backstopped
    assert ingest.frames_unscheduled > 0, "the policy really did decline them"


def test_a_declined_frame_publishes_nothing_at_all():
    """A skipped frame is not a quiet event, an empty event, or a repeat of the
    last one — the hub simply does not move. Consumers see a sparse stream, which
    is what they already tolerate (there was never a per-frame cadence guarantee)."""

    async def run():
        est = _Recorder()
        hub = PositionHub()
        events: list[PositionEvent] = []
        ingest = FrameIngest(
            est.estimate,
            hub,
            on_event=lambda reading, frame, rules: events.append(reading.event),
            policy=SHIPPED,
        )
        for seq in range(10):  # one capture second: the floor is 3
            ingest.submit(_frame(seq))
            await asyncio.wait_for(ingest.drain(), timeout=5.0)
        await ingest.close()
        return hub, events, ingest

    hub, events, ingest = asyncio.run(run())
    assert [e.seq for e in events] == [0]
    assert hub.latest("A1").seq == 0
    assert (ingest.frames_estimated, ingest.frames_unscheduled) == (1, 9)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_every_frame_restores_the_gateway_that_shipped_before_scheduling():
    """The escape hatch has to be exact, or "turn it off and see" is not a
    diagnosis anyone can trust. Same sequence, same worker, every frame read."""

    async def run():
        seen, ingest = await _drive(EveryFrame, _one_move())
        assert seen == list(range(60))
        assert ingest.frames_estimated == 60
        assert ingest.frames_unscheduled == 0
        assert ingest.frames_backstopped == 0

    asyncio.run(run())


def test_frame_ingest_reads_everything_unless_it_is_given_a_policy():
    """The class default, which every existing gateway test relies on: a
    ``FrameIngest`` nobody configured behaves exactly as it did before this
    milestone. The shipped policy arrives through ``serve``, not through here."""

    async def run():
        est = _Recorder()
        ingest = FrameIngest(est.estimate, PositionHub())
        for seq in range(5):
            ingest.submit(_frame(seq))
            await asyncio.wait_for(ingest.drain(), timeout=5.0)
        await ingest.close()
        return est.seqs()

    assert asyncio.run(run()) == [0, 1, 2, 3, 4]


def _stream(client, frames: list[Frame]) -> None:
    with client.websocket_connect("/ingest") as ws:
        for frame in frames:
            header, jpeg = frame.encode()
            ws.send_text(json.dumps(header))
            ws.send_bytes(jpeg)
            ws.receive_json()


@pytest.mark.parametrize(
    ("spec", "reads"),
    [
        # 0.9 s of capture time under a 2 s floor: one read, and it does not matter
        # which frames the inbox dropped on the way — that is the point of a
        # capture-time floor.
        (DEFAULT_POLICY_SPEC, (1, 1)),
        # Whatever reaches a worker is read. The range is the INBOX, not the
        # scheduler: a fast socket outruns even a stub estimator, and the frames it
        # displaces were never offered to the policy at all.
        ("every-frame", (1, 10)),
        # Its own 0.3 s spacing: at most every arriving frame from 0, 3, 6, 9; at
        # least the first and the last, which drain() guarantees.
        ("when-still:spacing_s=0.3", (2, 4)),
    ],
)
def test_br_schedule_chooses_the_gateways_policy(
    spec, reads, gateway, drain_ingest, monkeypatch
):
    """``BR_SCHEDULE`` takes a ``replay --policy`` spec, so the string a sweep
    measured is literally the string the node runs — and it reaches the live
    socket, not just a constructor."""
    monkeypatch.setenv("BR_SCHEDULE", spec)
    client = gateway(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))
    _stream(client, [_frame(seq) for seq in range(10)])
    drain_ingest(client)

    low, high = reads
    ingest = client.app.state.ingest
    assert ingest.frames_received == 10
    assert low <= ingest.frames_estimated <= high
    # Every arriving frame is accounted for exactly once, and the scheduler's share
    # of that is visible to an operator without shelling into the container.
    stats = client.get("/healthz").json()["ingest"]
    assert stats["estimated"] + stats["dropped"] + stats["unscheduled"] == 10
    assert stats["backstopped"] == 0
    assert (spec == "every-frame") == (stats["unscheduled"] == 0)


def test_an_unparseable_schedule_fails_at_startup_not_on_the_first_frame(monkeypatch):
    """A typo'd policy must not become "the defaults under another name" — that is
    a whole session read by a scheduler nobody asked for."""
    monkeypatch.setenv("BR_SCHEDULE", "burst-end:keepalve_s=3")
    with pytest.raises(ValueError, match="no parameter 'keepalve_s'"):
        create_app(StubRecognizer(START_PLACEMENT), PositionHub())


def test_serve_defaults_the_schedule_to_the_shipped_policy(monkeypatch):
    """Where the product default lives. ``create_app`` reads every frame unless
    told otherwise (so an injected test app is not silently sparse); ``serve`` —
    the only thing ``docker compose up`` and ``make dev`` run — sets the swept
    policy, the same asymmetry ``BR_RECORD_DIR`` has."""
    import uvicorn

    from board_reader_inference import cli

    # A private copy of the environment: `cmd_serve` writes into `os.environ`, and
    # a key the code under test sets is a key monkeypatch cannot restore.
    env = dict(os.environ)
    env.pop("BR_SCHEDULE", None)
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

    cli.cmd_serve(cli.build_parser().parse_args(["serve", "--no-record"]))
    assert env["BR_SCHEDULE"] == DEFAULT_POLICY_SPEC

    env["BR_SCHEDULE"] = "every-frame"
    cli.cmd_serve(cli.build_parser().parse_args(["serve", "--no-record"]))
    assert env["BR_SCHEDULE"] == "every-frame", "an operator's setting outranks it"

    cli.cmd_serve(cli.build_parser().parse_args(["serve", "--no-record", "--schedule",
                                                 "when-still"]))
    assert env["BR_SCHEDULE"] == "when-still", "…and the flag outranks both"


def test_the_ingress_recording_keeps_the_frames_the_scheduler_declined(
    tmp_path, monkeypatch
):
    """A recording is the INPUT to a run, not a log of what the estimator got round
    to — recording happens in the ``/ingest`` handler, before the inbox and long
    before the policy. It has to stay that way, or a sparse live run replays as a
    sparse recording and the whole sweep harness is measuring itself."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("BR_SCHEDULE", DEFAULT_POLICY_SPEC)
    monkeypatch.setenv("BR_RECORD_DIR", str(tmp_path / "rec"))
    with TestClient(create_app(StubRecognizer(START_PLACEMENT), PositionHub())) as client:
        _stream(client, [_frame(seq) for seq in range(10)])
        assert client.app.state.ingest.frames_estimated <= 1

    assert [rec.seq for rec in Recording.open(tmp_path / "rec")] == list(range(10))
