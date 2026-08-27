"""The gateway records its ingress (DESIGN.md WP-H2).

``/ingest`` is the chokepoint every edge crosses — the CLI feed, the Pi, and the
phone ``/camera`` page (which arrives here through the viewer's proxy). Recording
there is what makes a run reproducible, and what means a *new* edge is recorded
without anyone remembering to wire it up.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from board_reader_core.recording import INDEX_NAME, Recording
from board_reader_core.wire import Frame

from board_reader_inference.estimator import StubEstimator
from board_reader_inference.server import create_app


def _frame(seq: int, *, board_id: str = "A1", corners=None) -> Frame:
    rng = np.random.default_rng(seq)
    return Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=1000.0 + seq,
        image=(rng.random((64, 64, 3)) * 255).astype(np.uint8),
        stability=0.25,
        corners=corners,
    )


def _await_position(client: TestClient, board_id: str = "A1", timeout: float = 5.0):
    """Wait for a board's first published position, then return the response.

    An ack means "received", never "estimated" (WP-G1/G2): the ingest worker runs
    the model off the event loop, so a frame acked on the socket has not yet
    become an event on the hub. Recording, by contrast, happens inline in the
    ``/ingest`` handler — so a test that checks a recorder fact and then asserts a
    position exists is racing two different clocks, and loses about one run in
    three under `-n auto`.
    """
    deadline = time.monotonic() + timeout
    while True:
        response = client.get(f"/boards/{board_id}")
        if response.status_code == 200 or time.monotonic() > deadline:
            return response
        time.sleep(0.01)


def _send(client: TestClient, *frames: Frame) -> list[tuple[dict, bytes]]:
    """Stream frames over /ingest exactly as the edge client does."""
    envelopes = [f.encode() for f in frames]
    with client.websocket_connect("/ingest") as ws:
        for header, jpeg in envelopes:
            ws.send_text(json.dumps(header))
            ws.send_bytes(jpeg)
            ws.receive_json()
    return envelopes


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A gateway recording into ``tmp_path/rec``. Yields (client, record_root)."""
    root = tmp_path / "rec"
    monkeypatch.setenv("BR_RECORD_DIR", str(root))
    with TestClient(create_app(estimator=StubEstimator())) as client:
        yield client, root


def test_ingested_frames_land_on_disk_byte_for_byte(recorded):
    client, root = recorded
    sent = _send(client, _frame(0), _frame(1), _frame(2))

    replayed = [(r.header, r.jpeg) for r in Recording.open(root)]
    assert replayed == sent


def test_recording_survives_a_reconnecting_edge(recorded):
    """One session spans connections: a dropped socket must not split the run."""
    client, root = recorded
    _send(client, _frame(0))
    _send(client, _frame(1))

    assert [r.seq for r in Recording.open(root)] == [0, 1]


def test_every_board_is_recorded_in_one_session(recorded):
    client, root = recorded
    _send(client, _frame(0, board_id="A1"), _frame(0, board_id="B2"))

    assert Recording.open(root).board_ids() == ["A1", "B2"]


def test_the_photo_path_records_its_corners(recorded):
    """The /camera page streams original frames with corners — replay needs them."""
    client, root = recorded
    quad = np.array([[10, 10], [50, 12], [52, 54], [8, 52]], dtype=np.float32)
    _send(client, _frame(0, corners=quad))

    (rec,) = list(Recording.open(root))
    assert np.allclose(rec.frame().corners, quad)


def test_the_recording_holds_the_normalised_header_not_the_raw_json(recorded):
    """Whatever an edge puts on the wire, the recording holds validated values."""
    client, root = recorded
    frame = _frame(4)
    header, jpeg = frame.encode()
    with client.websocket_connect("/ingest") as ws:
        ws.send_text(json.dumps({**header, "seq": "4", "stability": "0.25"}))
        ws.send_bytes(jpeg)
        ws.receive_json()

    (rec,) = list(Recording.open(root))
    assert rec.header["seq"] == 4  # an int, as the wire defines it
    assert rec.header["stability"] == 0.25


def test_frames_the_prefilter_drops_are_still_recorded(recorded):
    """A recording is the *input* to a run, not a log of what the estimator reached.

    The latest-wins inbox drops stale frames under load; replaying only the
    survivors would reproduce a different, easier run than the one that happened.
    """
    client, root = recorded
    _send(client, *(_frame(i) for i in range(8)))

    assert [r.seq for r in Recording.open(root)] == list(range(8))


def test_healthz_reports_the_recording(recorded):
    client, root = recorded
    _send(client, _frame(0))

    recording = client.get("/healthz").json()["recording"]
    assert recording["frames"] == 1
    assert recording["active"] is True
    assert recording["path"] == str(Recording.open(root).path)


def test_recording_is_off_unless_asked_for(tmp_path, monkeypatch):
    """Opt-in by default: a test app must never write to the filesystem.

    The counterpart, ``serve --record-dir``, opts a real run in — so every served
    run is reproducible while the suite stays clean (see the BR_SESSION_BANK
    clobber incidents for why the default is this way round).
    """
    monkeypatch.delenv("BR_RECORD_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app(estimator=StubEstimator())) as client:
        _send(client, _frame(0))
        assert "recording" not in client.get("/healthz").json()

    assert list(tmp_path.iterdir()) == []


def test_a_recording_failure_does_not_break_the_stream(tmp_path, monkeypatch):
    """Recording is best-effort: the position stream outlives a full disk."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    monkeypatch.setenv("BR_RECORD_DIR", str(blocker))

    with TestClient(create_app(estimator=StubEstimator())) as client:
        _send(client, _frame(0), _frame(1))
        health = client.get("/healthz").json()
        assert health["status"] == "ok"
        assert health["recording"]["active"] is False
        assert _await_position(client).status_code == 200


def test_max_mb_caps_the_session(tmp_path, monkeypatch):
    monkeypatch.setenv("BR_RECORD_DIR", str(tmp_path / "rec"))
    monkeypatch.setenv("BR_RECORD_MAX_MB", "0.001")  # 1000 bytes

    with TestClient(create_app(estimator=StubEstimator())) as client:
        _send(client, *(_frame(i) for i in range(5)))
        assert client.get("/healthz").json()["recording"]["active"] is False

    lines = (Recording.open(tmp_path / "rec").path / INDEX_NAME).read_text().splitlines()
    assert 0 <= len(lines) < 5


def test_published_events_land_in_the_session_beside_their_frames(recorded):
    """The outputs of a run, joined to its inputs: each events.ndjson line names
    the frame it was estimated from and drops square_dist (replay reproduces it)."""
    from board_reader_core.recording import EVENTS_NAME

    client, root = recorded
    _send(client, _frame(0), _frame(1))
    _await_position(client)

    session = next(p for p in root.iterdir() if p.is_dir())
    deadline = time.monotonic() + 5.0
    lines = []
    while time.monotonic() < deadline:
        events = session / EVENTS_NAME
        if events.exists():
            lines = [json.loads(line) for line in events.read_text().splitlines()]
            if lines:
                break
        time.sleep(0.01)

    assert lines, "no published event reached events.ndjson"
    first = lines[0]
    assert first["board_id"] == "A1"
    assert "square_dist" not in first
    assert first["frame_seq"] in (0, 1)
    assert first["capture_ts"] >= 1000.0
    assert first["published_ts"] > 0
    assert "prior" in first  # the repair is part of the run's record


def test_a_refused_frame_is_counted_and_the_socket_survives():
    """A stream that is ALL refusals must not look like one that never sent —
    the refusal used to reach only the phone's banner."""
    with TestClient(create_app(estimator=StubEstimator())) as client:
        with client.websocket_connect("/ingest") as ws:
            ws.send_text(json.dumps({"type": "frame", "board_id": "A1", "seq": 0}))
            ws.send_bytes(b"not a jpeg")
            reply = ws.receive_json()
            assert "error" in reply
            # The socket is still usable for a good frame afterwards.
            header, jpeg = _frame(1).encode()
            ws.send_text(json.dumps(header))
            ws.send_bytes(jpeg)
            assert ws.receive_json()["ack"] == 1

        health = client.get("/healthz").json()
        assert health["ingest"]["refused"] == 1
        assert health["ingest"]["received"] == 1


def test_a_published_event_records_why_each_square_read_that_way(recorded):
    """``square_dist`` is too big to keep and replaying it needs the model back.
    The top-2 digest is the compromise: 64 squares of (class, probability), which
    is what turns "d8 read k" into "d8 read k at 0.51 over q at 0.49"."""
    from board_reader_core.protocol import CLASSES, EVIDENCE_TOP_K, GRID, parse_placement
    from board_reader_core.recording import EVENTS_NAME

    client, root = recorded
    _send(client, _frame(0))
    _await_position(client)

    session = next(p for p in root.iterdir() if p.is_dir())
    deadline = time.monotonic() + 5.0
    lines: list[dict] = []
    while time.monotonic() < deadline and not lines:
        events = session / EVENTS_NAME
        if events.exists():
            lines = [json.loads(line) for line in events.read_text().splitlines()]
        time.sleep(0.01)

    assert lines, "no published event reached events.ndjson"
    evidence = lines[0]["evidence"]
    assert len(evidence) == GRID * GRID
    assert all(len(square) == EVIDENCE_TOP_K for square in evidence)
    assert all(
        cls in CLASSES and 0.0 <= p <= 1.0 for square in evidence for cls, p in square
    )
    # The digest is of the same read the line's placement came from: the top class
    # per square IS that placement, square for square.
    grid = parse_placement(lines[0]["placement"])
    assert [square[0][0] for square in evidence] == [
        grid[r][c] for r in range(GRID) for c in range(GRID)
    ]
