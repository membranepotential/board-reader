"""Tests for the inference gateway (DESIGN.md §4.5) via Starlette TestClient."""

from __future__ import annotations

import json

from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.estimator import StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.server import create_app


def _client(**kw):
    return TestClient(create_app(StubEstimator(START_PLACEMENT), PositionHub()))


def _send_frame(ws, seq=0):
    frame = Frame(board_id="A1", seq=seq, capture_ts=float(seq), image=render_board())
    header, jpeg = frame.encode()
    ws.send_text(json.dumps(header))
    ws.send_bytes(jpeg)


def _app_estimator(client: TestClient):
    return client.app.state.estimator


def test_healthz_ok():
    r = _client().get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_boards_404_before_any_frame():
    assert _client().get("/boards/A1").status_code == 404


def test_ingest_frame_then_board_snapshot_available(gateway, drain_ingest):
    # The ack means "received", not "estimated" — hence gateway + drain_ingest
    # rather than a bare client (see conftest.py).
    client = gateway(create_app(StubEstimator(START_PLACEMENT), PositionHub()))
    with client.websocket_connect("/ingest") as ingest:
        _send_frame(ingest, seq=5)
        ack = ingest.receive_json()
        assert ack == {"ack": 5, "board_id": "A1"}
    drain_ingest(client)
    snap = client.get("/boards/A1")
    assert snap.status_code == 200
    assert snap.json()["placement"] == START_PLACEMENT


def test_end_to_end_ingest_to_viewer_stream(gateway):
    """A frame pushed on /ingest reaches a /viewer subscriber as a position event."""
    # Both sockets must share ONE event loop. On a bare TestClient each
    # `websocket_connect` gets its own portal, so the hub's subscriber queue would
    # live on the viewer's loop while `publish` runs on the ingest loop — waking a
    # future across loops, which strands `receive_text()` below forever.
    client = gateway(create_app(StubEstimator(START_PLACEMENT), PositionHub()))
    with (
        client.websocket_connect("/viewer") as viewer,
        client.websocket_connect("/ingest") as ingest,
    ):
        _send_frame(ingest, seq=1)
        ingest.receive_json()  # ack
        event = json.loads(viewer.receive_text())
    assert event["board_id"] == "A1"
    assert event["placement"] == START_PLACEMENT
    assert event["status"] == "ok"
    assert "square_dist" in event


def test_estimate_snapshot_roundtrip():
    """POST /estimate (rectified crop JPEG body) -> position event, retained in hub."""
    from board_reader_core.wire import encode_jpeg

    client = _client()
    r = client.post(
        "/estimate",
        content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    event = r.json()
    assert event["board_id"] == "snapshot"
    assert event["placement"] == START_PLACEMENT
    assert "square_dist" in event
    # The event is retained like any streamed frame's.
    assert client.get("/boards/snapshot").json()["seq"] == event["seq"]


def test_estimate_rejects_garbage_bytes():
    r = _client().post(
        "/estimate", content=b"not a jpeg", headers={"Content-Type": "image/jpeg"}
    )
    assert r.status_code == 400
    assert "error" in r.json()


def test_estimate_with_corners_falls_back_to_server_side_warp():
    """A photo + corners against a rectified-mode estimator still estimates.

    The stub has no photo path, so the gateway warps the photo to the canonical
    crop server-side and runs the normal path.
    """
    from board_reader_core.sim import simulate_angled_view
    from board_reader_core.wire import encode_jpeg

    photo, quad = simulate_angled_view(render_board())
    corners = ",".join(f"{x:.1f},{y:.1f}" for x, y in quad)
    r = _client().post(
        f"/estimate?corners={corners}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    assert r.json()["placement"] == START_PLACEMENT


def test_estimate_rejects_malformed_corners():
    from board_reader_core.wire import encode_jpeg

    r = _client().post(
        "/estimate?corners=1,2,3",
        content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 400


def test_detect_endpoint_success_and_no_board(monkeypatch):
    import sys
    import types

    import numpy as np

    from board_reader_core.wire import encode_jpeg

    quad = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    fake = types.ModuleType("board_reader_core.boarddetect")
    fake.detect_board_quad = lambda image: quad
    monkeypatch.setitem(sys.modules, "board_reader_core.boarddetect", fake)
    client = _client()
    body = encode_jpeg(render_board())
    r = client.post("/detect", content=body, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200
    assert r.json()["corners"] == quad.tolist()

    fake.detect_board_quad = lambda image: None
    r = client.post("/detect", content=body, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 422


def test_detect_endpoint_rejects_garbage():
    r = _client().post(
        "/detect", content=b"junk", headers={"Content-Type": "image/jpeg"}
    )
    assert r.status_code == 400


def test_detect_does_not_block_the_event_loop(monkeypatch):
    """A slow /detect (real misses run 2-4s) must not stall the gateway.

    Regression for the bug where detect_board_quad ran synchronously inside
    the async handler: while it churned, WS /ingest could not ack and the
    /viewer hub could not publish. Proxy that here with /healthz -- if
    /detect blocks the loop, a concurrent /healthz queues behind it instead
    of answering immediately.

    Detection is held open by an Event rather than a sleep, so a passing run
    costs microseconds and never races: `entered` proves /detect really is
    inside the detector before /healthz is timed, which a fixed sleep could
    only guess at. If the offload is reverted, the detector runs ON the loop,
    /healthz cannot be answered until `release.wait` times out, and the
    elapsed assertion fails loudly instead of the test hanging.
    """
    import sys
    import threading
    import time
    import types

    import numpy as np

    from board_reader_core.wire import encode_jpeg

    quad = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    entered = threading.Event()  # the detector is running
    release = threading.Event()  # ...and the test decides when it returns

    def slow_detect(image):
        entered.set()
        release.wait(timeout=2)
        return quad

    fake = types.ModuleType("board_reader_core.boarddetect")
    fake.detect_board_quad = slow_detect
    monkeypatch.setitem(sys.modules, "board_reader_core.boarddetect", fake)

    body = encode_jpeg(render_board())
    app = create_app(StubEstimator(START_PLACEMENT), PositionHub())
    # A shared portal (single event loop) is required: without `with`, each
    # TestClient call gets its own throwaway portal/loop and would never
    # contend for the same loop regardless of whether the handler blocks it.
    with TestClient(app) as client:
        detect_done = threading.Event()
        detect_response: dict[str, object] = {}

        def run_detect():
            detect_response["r"] = client.post(
                "/detect", content=body, headers={"Content-Type": "image/jpeg"}
            )
            detect_done.set()

        t = threading.Thread(target=run_detect)
        t.start()
        try:
            assert entered.wait(timeout=5), "/detect never reached the detector"

            healthz_start = time.perf_counter()
            r = client.get("/healthz")
            healthz_elapsed = time.perf_counter() - healthz_start

            assert r.status_code == 200
            assert healthz_elapsed < 0.2, (
                f"/healthz took {healthz_elapsed:.3f}s -- it queued behind /detect"
            )
            assert not detect_done.is_set(), (
                "/detect already finished -- this run cannot tell blocking from fast"
            )
        finally:
            release.set()  # never leave the worker thread parked on a timeout
            t.join(timeout=5)

        assert detect_done.is_set()
        assert detect_response["r"].status_code == 200
        assert detect_response["r"].json()["corners"] == quad.tolist()


def test_viewer_replays_retained_latest_on_connect(gateway, drain_ingest):
    # Draining first is what keeps this test from HANGING the suite: if the frame
    # is never estimated, nothing is retained, and `receive_text()` below waits
    # forever for an event that will never come (see conftest.py).
    client = gateway(create_app(StubEstimator(START_PLACEMENT), PositionHub()))
    with client.websocket_connect("/ingest") as ingest:
        _send_frame(ingest, seq=9)
        ingest.receive_json()
    drain_ingest(client)
    # A viewer connecting AFTER the frame still gets the retained latest immediately.
    with client.websocket_connect("/viewer") as viewer:
        event = json.loads(viewer.receive_text())
    assert event["seq"] == 9


def _assert_valid_jpeg_b64(b64: str) -> None:
    import base64

    import cv2
    import numpy as np

    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None, "debug image did not decode as a valid JPEG"


def test_estimate_debug_off_by_default():
    from board_reader_core.wire import encode_jpeg

    r = _client().post(
        "/estimate",
        content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    assert "debug" not in r.json()


def test_estimate_debug_on_returns_valid_images_and_does_not_leak_into_snapshot():
    from board_reader_core.wire import encode_jpeg

    client = _client()
    r = client.post(
        "/estimate?debug=1",
        content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body["debug"].keys()) == {"boxes", "mosaic"}
    _assert_valid_jpeg_b64(body["debug"]["boxes"])
    _assert_valid_jpeg_b64(body["debug"]["mosaic"])
    # The debug key is HTTP-response-only; the hub's retained event has none.
    snap = client.get("/boards/snapshot").json()
    assert "debug" not in snap


def test_uploads_saved_when_retention_enabled(tmp_path, monkeypatch):
    from board_reader_core.wire import encode_jpeg

    monkeypatch.setenv("BR_SAVE_UPLOADS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(StubEstimator(START_PLACEMENT), PositionHub()))
    body = encode_jpeg(render_board())

    r = client.post("/estimate", content=body, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200
    jpgs = sorted(tmp_path.glob("*-estimate.jpg"))
    metas = sorted(tmp_path.glob("*-estimate.json"))
    assert len(jpgs) == 1 and len(metas) == 1
    assert jpgs[0].read_bytes() == body
    meta = json.loads(metas[0].read_text())
    assert meta["kind"] == "estimate"
    # The response's placement is annotated back onto the sidecar, so every
    # retained shot doubles as (predicted-)labeled eval data.
    assert meta["placement"] == r.json()["placement"]

    # undecodable bodies are rejected AND not retained
    client.post("/estimate", content=b"not a jpeg", headers={"Content-Type": "image/jpeg"})
    assert len(list(tmp_path.glob("*-estimate.jpg"))) == 1


def test_uploads_not_saved_by_default(tmp_path, monkeypatch):
    from board_reader_core.wire import encode_jpeg

    monkeypatch.delenv("BR_SAVE_UPLOADS", raising=False)
    monkeypatch.chdir(tmp_path)  # any accidental relative write would land here
    client = TestClient(create_app(StubEstimator(START_PLACEMENT), PositionHub()))
    r = client.post(
        "/estimate", content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_low_light_warning_on_dark_photos():
    import numpy as np

    from board_reader_core.wire import encode_jpeg

    client = _client()
    dark = (render_board() * 0.15).astype(np.uint8)
    r = client.post("/estimate", content=encode_jpeg(dark), headers={"Content-Type": "image/jpeg"})
    assert r.json().get("warnings") == ["low_light"]
    r = client.post(
        "/estimate", content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert "warnings" not in r.json()


def test_estimate_warns_on_very_low_camera_angle():
    """A corner quad compressed to <0.62 aspect (camera nearly board-level) gets
    a very_low_angle warning — recognition measurably collapses there."""

    from board_reader_core.wire import encode_jpeg

    # The warning comes from the quad's aspect alone, before any recognition runs.
    photo = render_board()
    h, w = photo.shape[:2]
    client = _client()
    flat = f"0,0,{w},0,{w},{int(h * 0.4)},0,{int(h * 0.4)}"  # aspect 0.4
    r = client.post(
        f"/estimate?corners={flat}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert "very_low_angle" in r.json().get("warnings", [])

    square = f"0,0,{w},0,{w},{h},0,{h}"
    r2 = client.post(
        f"/estimate?corners={square}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert "very_low_angle" not in r2.json().get("warnings", [])
