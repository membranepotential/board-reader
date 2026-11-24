"""The recorder's per-session roll, driven through the real gateway (WP-H2).

:mod:`board_reader_core.recording` unit-tests the roll at the ``record()`` /
``record_event()`` level, with a hand-driven recorder and an injected clock. What
that cannot pin is the *wiring*: the roll only ever happens because the
``/ingest`` handler records the normalised header inline, and the late-event
routing only ever matters because ``FrameIngest`` publishes an estimate on the
event loop some time after the frame that produced it was recorded. Both of those
are gateway facts, not recorder facts.

So this file drives the shape a reconnecting phone actually produces — session A's
``seq`` 0..N, then session B restarting at 0 with a later ``capture_ts``, the same
board id throughout (the shape found in ``recordings/20260827-175539``, which
predates the roll and holds two colliding sessions in one directory) — and asserts
the layout on disk, through the app that wrote it.

The race the design exists for is drivable end to end: the estimator runs in a
worker thread (``asyncio.to_thread``), so a fake that parks there holds session
A's last read open while session B's first frame is recorded — and the roll is
inline in the handler, so the ack for that frame *proves* it already happened.
Releasing the fake then publishes an event whose ``capture_ts`` belongs to a
directory that is no longer the current one. See
:func:`test_an_estimate_that_outlives_the_roll_is_recorded_in_its_own_session`.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from starlette.testclient import TestClient, WebSocketTestSession

from board_reader_core.recording import EVENTS_NAME, Recording
from board_reader_core.wire import Frame, decode_jpeg

from board_reader_inference.recognizer import StubRecognizer
from board_reader_inference.replay import find_sessions
from board_reader_inference.server import create_app

BOARD = "A1"
#: Two takes ten minutes apart, as a phone that dropped and reconnected produces.
#: Session ids are second-resolution stamps and both rolls land inside one second
#: of test time, so the *directory* names are the stamp and the stamp plus a
#: suffix — the recording's own time axis is these constants, not the wall clock.
SESSION_A_T0 = 1_000_000.0
SESSION_B_T0 = 1_000_600.0


def _frame(seq: int, *, t0: float, board_id: str = BOARD) -> Frame:
    rng = np.random.default_rng(int(t0) + seq)
    return Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=t0 + seq,
        image=(rng.random((64, 64, 3)) * 255).astype(np.uint8),
        stability=0.25,
    )


def _send(ws: WebSocketTestSession, frame: Frame) -> dict:
    """One frame over an open /ingest socket, returning its ack.

    The ack is the synchronisation point this file leans on: the handler records
    the frame (and therefore rolls) *before* it acks, so a returned ack means the
    recording side effect has already landed on disk.
    """
    header, jpeg = frame.encode()
    ws.send_text(json.dumps(header))
    ws.send_bytes(jpeg)
    return ws.receive_json()


def _stream(client: TestClient, frames: list[Frame]) -> None:
    """One streaming connection's worth of frames, as the edge client sends them."""
    with client.websocket_connect("/ingest") as ws:
        for frame in frames:
            _send(ws, frame)


def _session_dirs(root) -> list:
    return sorted(d for d in root.iterdir() if d.is_dir())


def _events(session_dir) -> list[dict]:
    path = session_dir / EVENTS_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def record_root(tmp_path, monkeypatch):
    """The gateway's ``BR_RECORD_DIR`` for this test, as ``serve --record-dir`` sets it."""
    root = tmp_path / "rec"
    monkeypatch.setenv("BR_RECORD_DIR", str(root))
    return root


class _GatedRecognizer(StubRecognizer):
    """A stub that parks the read of one frame in its worker thread.

    The point is to make the publish order — not the arrival order — the thing
    under test: exactly one frame's read is held open across a roll, and the test
    thread decides when it completes. A recognizer sees pixels only, so the gated
    frame is told apart by its image (each ``_frame`` is distinct noise).
    """

    def __init__(self, gate: Frame) -> None:
        super().__init__()
        # What the gateway will decode off the wire for this frame.
        self._gate_image = decode_jpeg(gate.encode()[1])
        self.entered = threading.Event()  # set once the gated read is in flight
        self.release = threading.Event()  # set by the test to let it finish

    def read(self, image: np.ndarray) -> np.ndarray:
        if np.array_equal(image, self._gate_image):
            self.entered.set()
            self.release.wait(timeout=10.0)
        return super().read(image)


def test_a_reconnecting_edge_rolls_the_gateway_onto_a_second_directory(record_root):
    """The layout: one directory per streaming session, chained by the manifests.

    A plain ``TestClient`` is enough here — every fact asserted is one the
    ``/ingest`` handler wrote before acking (see the conftest docstring) — and
    leaving its ``with`` block runs the lifespan, which closes the recorder and
    finalises the second manifest.
    """
    with TestClient(create_app(recognizer=StubRecognizer())) as client:
        _stream(client, [_frame(s, t0=SESSION_A_T0) for s in range(5)])
        _stream(client, [_frame(s, t0=SESSION_B_T0) for s in range(3)])
        assert client.get("/healthz").json()["recording"]["sessions"] == 2

    first, second = _session_dirs(record_root)
    a, b = Recording(first), Recording(second)

    # Each directory holds one monotone stream, and the split is by capture time.
    assert [e.seq for e in a.frame_index()] == [0, 1, 2, 3, 4]
    assert [e.seq for e in b.frame_index()] == [0, 1, 2]
    assert all(e.capture_ts < SESSION_B_T0 for e in a.frame_index())
    assert all(e.capture_ts >= SESSION_B_T0 for e in b.frame_index())
    assert a.board_ids() == b.board_ids() == [BOARD]

    # The manifests say why the first ended and where the second came from.
    assert a.manifest["closed_reason"] == "rolled: new streaming session"
    assert b.manifest["rolled_from"] == first.name
    assert b.manifest["closed_reason"] == "closed"  # the lifespan's own stop()
    assert "rolled_from" not in a.manifest

    # `--recording recordings/` means "the run that just happened".
    assert Recording.open(record_root).path == second


def test_each_directorys_events_belong_to_the_session_beside_them(
    record_root, gateway, drain_ingest
):
    """Outputs land with the inputs that produced them, on both sides of the roll."""
    client = gateway(create_app(recognizer=StubRecognizer()))
    _stream(client, [_frame(s, t0=SESSION_A_T0) for s in range(3)])
    drain_ingest(client)
    _stream(client, [_frame(s, t0=SESSION_B_T0) for s in range(3)])
    drain_ingest(client)

    first, second = _session_dirs(record_root)
    events_a, events_b = _events(first), _events(second)
    assert events_a and events_b, "both sessions must have published something"

    windows = ((events_a, SESSION_A_T0, SESSION_B_T0), (events_b, SESSION_B_T0, None))
    for lines, lo, hi in windows:
        for line in lines:
            assert line["board_id"] == BOARD
            assert line["capture_ts"] >= lo
            if hi is not None:
                assert line["capture_ts"] < hi

    # Every event names a frame that is actually in ITS directory: the join from
    # events.ndjson back to frames.ndjson is what the trial scorer walks, and
    # before the roll it was ambiguous (both sessions had a frame_seq 0).
    for session_dir in (first, second):
        recorded = {(e.seq, e.capture_ts) for e in Recording(session_dir).frame_index()}
        for line in _events(session_dir):
            assert (line["frame_seq"], line["capture_ts"]) in recorded


def test_an_estimate_that_outlives_the_roll_is_recorded_in_its_own_session(
    record_root, gateway, drain_ingest
):
    """The race the routing exists for, driven deterministically.

    ``frame_seq`` cannot say which session an event came from — both sessions
    start at 0 — so the writer routes on ``capture_ts``. Here session A's last
    read is parked in its worker thread; session B's first frame is then ingested
    (its ack proves the roll happened, since recording is inline in the handler);
    only then is the read released. The event it publishes is written into a
    directory the recorder has already closed.

    One socket carries both sessions on purpose: the roll is decided by ``seq``,
    not by the connection, and reconnecting here would drain the parked read at
    disconnect and dissolve the race being tested.
    """
    est = _GatedRecognizer(_frame(2, t0=SESSION_A_T0))
    client = gateway(create_app(recognizer=est))

    with client.websocket_connect("/ingest") as ws:
        _send(ws, _frame(0, t0=SESSION_A_T0))
        _send(ws, _frame(1, t0=SESSION_A_T0))
        drain_ingest(client)  # session A's ungated reads are published and recorded

        _send(ws, _frame(2, t0=SESSION_A_T0))
        assert est.entered.wait(5.0), "the gated read never reached the worker thread"

        _send(ws, _frame(0, t0=SESSION_B_T0))
        assert len(_session_dirs(record_root)) == 2, "the ack did not imply the roll"
        assert not est.release.is_set()

        est.release.set()  # session A's read completes into session B's present
        drain_ingest(client)

    first, second = _session_dirs(record_root)
    late = [line for line in _events(first) if line["frame_seq"] == 2]
    assert late, "the late event was lost from the session it was estimated from"
    assert late[0]["capture_ts"] == SESSION_A_T0 + 2
    assert all(line["capture_ts"] < SESSION_B_T0 for line in _events(first))
    assert all(line["capture_ts"] >= SESSION_B_T0 for line in _events(second))


def test_replay_finds_exactly_one_stream_inside_each_rolled_directory(record_root):
    """End to end, the wart is gone: no consumer has to split a directory any more.

    ``replay.find_sessions`` is the heuristic that existed because a directory
    could hold several takes. Run over each rolled directory it now finds one —
    and over the two concatenated it still finds two, so the check has teeth.
    """
    with TestClient(create_app(recognizer=StubRecognizer())) as client:
        _stream(client, [_frame(s, t0=SESSION_A_T0) for s in range(5)])
        _stream(client, [_frame(s, t0=SESSION_B_T0) for s in range(4)])

    dirs = _session_dirs(record_root)
    per_dir = [list(Recording(d).frame_index(board_id=BOARD)) for d in dirs]
    assert [len(find_sessions(entries)) for entries in per_dir] == [1, 1]
    assert [len(entries) for entries in per_dir] == [5, 4]
    assert len(find_sessions(per_dir[0] + per_dir[1])) == 2
