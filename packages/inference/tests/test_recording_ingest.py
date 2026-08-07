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
