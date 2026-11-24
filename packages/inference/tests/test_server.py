"""Tests for the inference gateway (DESIGN.md §4.5) via Starlette TestClient."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.estimator import StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.recognizer import build_from_env
from board_reader_inference.server import create_app
from board_reader_inference.squares import BboxGeometry


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


def test_estimate_photo_path_used_for_photo_mode_centroids():
    """With photo-mode centroids the gateway calls estimate_photo (full-piece boxes)."""
    from board_reader_core.sim import simulate_angled_view
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.centroid import build_centroids
    from board_reader_inference.embed import LiteEmbedder
    from board_reader_inference.estimator import CentroidEstimator
    from board_reader_inference.squares import crop_squares_photo

    board = render_board(START_PLACEMENT)
    photo, quad = simulate_angled_view(board)
    emb = LiteEmbedder()
    clf = build_centroids(
        emb,
        crop_squares_photo(photo, quad, size=64),
        START_PLACEMENT,
        crop_mode="photo",
    )
    est = CentroidEstimator(emb, clf, square_size=64)
    calls = []
    orig = est.estimate_photo
    est.estimate_photo = lambda frame, corners: calls.append(1) or orig(frame, corners)
    client = TestClient(create_app(est, PositionHub()))
    corners = ",".join(f"{x:.1f},{y:.1f}" for x, y in quad)
    r = client.post(
        f"/estimate?corners={corners}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    # This asserts the ROUTING (photo-mode centroids -> estimate_photo with the
    # posted corners); classification quality on real photos is the DINOv2 eval's
    # job — the Lite embedder cannot read tall perspective boxes.
    assert calls, "gateway must route photo-mode centroids through estimate_photo"
    assert r.json()["board_id"] == "snapshot"


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


def test_calibrate_hot_swaps_estimator(tmp_path, monkeypatch):
    """POST /calibrate rebuilds a per-set kNN exemplar bank from one labeled photo
    and the running estimator immediately uses it (DESIGN §4.6 E1/E2)."""
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.centroid import ExemplarClassifier, build_exemplars
    from board_reader_inference.embed import LiteEmbedder
    from board_reader_inference.estimator import CentroidEstimator
    from board_reader_inference.squares import crop_squares_photo, full_frame_corners

    monkeypatch.chdir(tmp_path)  # exemplars_session.npz lands in a sandbox
    mid = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
    # Flat renders + full-frame corners: photo boxes with zero headroom are then
    # exact squares, which the Lite embedder can classify (it cannot read tall
    # perspective boxes — real-photo quality is the DINOv2 eval's job).
    photo = render_board(START_PLACEMENT)
    quad = full_frame_corners(photo)
    emb = LiteEmbedder()
    seed = build_exemplars(
        emb,
        crop_squares_photo(photo, quad, geometry=BboxGeometry(0.0, 0.0), size=64),
        START_PLACEMENT,
        box_headroom=0.0,
        box_side_pad=0.0,
    )
    est = CentroidEstimator(emb, seed, square_size=64, geometry=BboxGeometry(0.0, 0.0))
    client = TestClient(create_app(est, PositionHub()))

    cal_photo = render_board(mid)
    cal_quad = full_frame_corners(cal_photo)
    cal_corners = ",".join(f"{x:.1f},{y:.1f}" for x, y in cal_quad)
    r = client.post(
        f"/calibrate?corners={cal_corners}&placement={mid}",
        content=encode_jpeg(cal_photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["backbone"] == "lite"
    assert 0 < body["classes_calibrated"] <= 13
    assert (tmp_path / "models/exemplars_session.npz").exists() or body["saved"] is None
    swapped = _app_estimator(client)
    assert isinstance(swapped.classifier, ExemplarClassifier)
    exemplars_after_first = body["exemplars"]

    # The swapped-in estimator classifies the calibration position correctly.
    r2 = client.post(
        f"/estimate?corners={cal_corners}",
        content=encode_jpeg(cal_photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r2.status_code == 200
    assert r2.json()["placement"] == mid

    # A second calibration photo ADDS to the bank rather than replacing it.
    third = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1"
    third_photo = render_board(third)
    third_quad = full_frame_corners(third_photo)
    third_corners = ",".join(f"{x:.1f},{y:.1f}" for x, y in third_quad)
    r3 = client.post(
        f"/calibrate?corners={third_corners}&placement={third}",
        content=encode_jpeg(third_photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r3.status_code == 200
    assert r3.json()["exemplars"] > exemplars_after_first
    # The estimator still works after growing.
    r4 = client.post(
        f"/estimate?corners={third_corners}",
        content=encode_jpeg(third_photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r4.status_code == 200
    assert r4.json()["placement"] == third


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


def test_estimate_debug_on_photo_path():
    """debug=1 also works on the photo path (ExemplarClassifier + corners)."""
    from board_reader_core.sim import simulate_angled_view
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.centroid import build_exemplars
    from board_reader_inference.embed import LiteEmbedder
    from board_reader_inference.estimator import CentroidEstimator
    from board_reader_inference.squares import crop_squares_photo

    board = render_board(START_PLACEMENT)
    photo, quad = simulate_angled_view(board)
    emb = LiteEmbedder()
    clf = build_exemplars(
        emb,
        crop_squares_photo(photo, quad, size=64),
        START_PLACEMENT,
    )
    est = CentroidEstimator(emb, clf, square_size=64)
    client = TestClient(create_app(est, PositionHub()))
    corners = ",".join(f"{x:.1f},{y:.1f}" for x, y in quad)
    r = client.post(
        f"/estimate?corners={corners}&debug=true",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body["debug"].keys()) == {"boxes", "mosaic"}
    _assert_valid_jpeg_b64(body["debug"]["boxes"])
    _assert_valid_jpeg_b64(body["debug"]["mosaic"])


def test_calibrate_rejects_bad_input():
    from board_reader_core.wire import encode_jpeg

    client = _client()
    r = client.post(
        "/calibrate?corners=1,2,3",
        content=encode_jpeg(render_board()),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 400


def test_uploads_saved_when_retention_enabled(tmp_path, monkeypatch):
    from board_reader_core.wire import encode_jpeg

    monkeypatch.setenv("BR_SAVE_UPLOADS", str(tmp_path))
    # The /calibrate below persists a session bank relative to cwd — sandbox it,
    # or the test clobbers the real models/exemplars_session.npz (it did, once).
    # A lite photo-mode estimator keeps /calibrate off the DINOv2 weights too.
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(_occupancy_estimator(), PositionHub()))
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

    # calibrate keeps corners + placement so the shot is usable as labeled data
    h, w = render_board().shape[:2]
    corners = f"0,0,{w},0,{w},{h},0,{h}"
    client.post(
        f"/calibrate?corners={corners}",
        content=body,
        headers={"Content-Type": "image/jpeg"},
    )
    metas = sorted(tmp_path.glob("*-calibrate.json"))
    assert len(metas) == 1
    meta = json.loads(metas[0].read_text())
    assert meta["corners"] == corners and meta["placement"] == START_PLACEMENT

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


# --- Zero-setup path: occupancy-only + start-position auto-calibration ---------


def _occupancy_estimator():
    """Lite-backbone stand-in for the ChessReD occupancy bank, over flat sim
    renders (headroom 0 = exact squares, which the Lite embedder can read)."""
    from board_reader_inference.centroid import build_exemplars
    from board_reader_inference.embed import LiteEmbedder
    from board_reader_inference.estimator import CentroidEstimator
    from board_reader_inference.squares import crop_squares_photo, full_frame_corners

    photo = render_board(START_PLACEMENT)
    quad = full_frame_corners(photo)
    emb = LiteEmbedder()
    clf = build_exemplars(
        emb,
        crop_squares_photo(photo, quad, geometry=BboxGeometry(0.0, 0.0), size=64),
        START_PLACEMENT,
        box_headroom=0.0,
        box_side_pad=0.0,
    )
    return CentroidEstimator(emb, clf, square_size=64, geometry=BboxGeometry(0.0, 0.0))


def _corners_param(photo):
    from board_reader_inference.squares import full_frame_corners

    return ",".join(f"{x:.1f},{y:.1f}" for x, y in full_frame_corners(photo))


def test_uncalibrated_estimate_is_occupancy_only_and_unpublished(tmp_path, monkeypatch):
    from board_reader_core.protocol import parse_placement
    from board_reader_core.wire import encode_jpeg

    monkeypatch.chdir(tmp_path)
    client = TestClient(
        create_app(
            StubEstimator(START_PLACEMENT),
            PositionHub(),
            occupancy_estimator=_occupancy_estimator(),
        )
    )
    mid = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
    photo = render_board(mid)
    r = client.post(
        f"/estimate?corners={_corners_param(photo)}&board_id=Z1",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "occupancy"
    assert "placement" not in body
    assert "hint" in body
    truth = [
        cell != "empty" for row in parse_placement(mid) for cell in row
    ]
    got = [v for row in body["occupied"] for v in row]
    assert got == truth
    # Occupancy-only readouts are HTTP-response-only — never published to the hub.
    assert client.get("/boards/Z1").status_code == 404


def test_uncalibrated_estimate_autocalibrates_on_start_position(tmp_path, monkeypatch):
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.centroid import ExemplarClassifier

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BR_SESSION_BANK", str(tmp_path / "models/exemplars_session.npz"))
    client = TestClient(
        create_app(
            StubEstimator(START_PLACEMENT),
            PositionHub(),
            occupancy_estimator=_occupancy_estimator(),
        )
    )
    start_photo = render_board(START_PLACEMENT)
    r = client.post(
        f"/estimate?corners={_corners_param(start_photo)}&board_id=Z1",
        content=encode_jpeg(start_photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    body = r.json()
    auto = body["autocalibrated"]
    assert body["placement"] == START_PLACEMENT
    assert auto["matches"] == 64 and auto["rotation"] == 0
    assert auto["exemplars"] == 64 * 4  # photometric variants included
    assert auto["saved"] and (tmp_path / "models/exemplars_session.npz").exists()
    # The full event IS published (it's a real position reading).
    assert client.get("/boards/Z1").json()["placement"] == START_PLACEMENT
    # The estimator is hot-swapped: a midgame photo now gets full recognition.
    swapped = _app_estimator(client)
    assert isinstance(swapped.classifier, ExemplarClassifier)
    mid = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
    mid_photo = render_board(mid)
    r2 = client.post(
        f"/estimate?corners={_corners_param(mid_photo)}",
        content=encode_jpeg(mid_photo),
        headers={"Content-Type": "image/jpeg"},
    )
    body2 = r2.json()
    assert "mode" not in body2
    assert body2["placement"] == mid


def test_autocalibration_handles_rotated_corner_labeling(tmp_path, monkeypatch):
    """90-deg-off corner detection self-corrects: the bank's labels match the crops."""
    import numpy as np

    from board_reader_core.protocol import parse_placement
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.squares import full_frame_corners

    monkeypatch.chdir(tmp_path)
    client = TestClient(
        create_app(
            StubEstimator(START_PLACEMENT),
            PositionHub(),
            occupancy_estimator=_occupancy_estimator(),
        )
    )
    start_photo = render_board(START_PLACEMENT)
    rolled = np.roll(full_frame_corners(start_photo), 1, axis=0)
    corners = ",".join(f"{x:.1f},{y:.1f}" for x, y in rolled)
    r = client.post(
        f"/estimate?corners={corners}",
        content=encode_jpeg(start_photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    body = r.json()
    auto = body["autocalibrated"]
    assert auto["rotation"] != 0
    # The reported placement is the start position seen through the rotated
    # labeling — rotating it back by `rotation` recovers the canonical board.
    grid = np.array(parse_placement(body["placement"]), dtype="<U5")
    start = np.array(parse_placement(START_PLACEMENT), dtype="<U5")
    assert (np.rot90(grid, auto["rotation"]) == start).all()


def test_session_bank_restored_on_restart_when_autocalib_enabled(tmp_path, monkeypatch):
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.estimator import CentroidEstimator

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.setenv("BR_SESSION_BANK", str(tmp_path / "models/exemplars_session.npz"))
    occ = _occupancy_estimator()
    client = TestClient(
        create_app(StubEstimator(START_PLACEMENT), PositionHub(), occupancy_estimator=occ)
    )
    start_photo = render_board(START_PLACEMENT)
    r = client.post(
        f"/estimate?corners={_corners_param(start_photo)}",
        content=encode_jpeg(start_photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.json()["autocalibrated"]["saved"]

    # A fresh app (server restart) with autocalib enabled restores the session
    # bank as its style estimator — calibration survives restarts.
    app2 = create_app(hub=PositionHub(), occupancy_estimator=occ)
    restored = app2.state.estimator
    assert isinstance(restored, CentroidEstimator)
    assert restored.embedder is occ.embedder  # shared backbone, one ONNX session
    # Without autocalib enabled, the default stays the M0 stub (no implicit loads).
    app3 = create_app(hub=PositionHub())
    assert isinstance(app3.state.estimator, StubEstimator)


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


def test_calibrated_bank_refreshes_on_drifted_start_position(tmp_path, monkeypatch):
    """Seeing the start position at sub-threshold confidence appends
    label-confirmed exemplars — the bank tracks lighting drift on its own."""
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference import autocalib

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BR_SESSION_BANK", str(tmp_path / "models/exemplars_session.npz"))
    est = _occupancy_estimator()  # lite exemplar-bank estimator, 256 exemplars
    before = est.classifier.exemplars.shape[0]
    client = TestClient(create_app(est, PositionHub()))
    photo = render_board(START_PLACEMENT)

    # Confidence gate: with the threshold at 0 every read counts as "confident
    # enough already" -> no refresh. (The lite stand-in's absolute confidence is
    # not meaningful, so the gate is exercised via explicit thresholds.)
    monkeypatch.setattr(autocalib, "APPEND_MAX_CONF", 0.0)
    r = client.post(
        f"/estimate?corners={_corners_param(photo)}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert "bank_refreshed" not in r.json()

    # Simulate drifted lighting by disabling the confidence gate.
    monkeypatch.setattr(autocalib, "APPEND_MAX_CONF", 2.0)
    r = client.post(
        f"/estimate?corners={_corners_param(photo)}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    body = r.json()
    refresh = body["bank_refreshed"]
    assert refresh["agree"] >= autocalib.APPEND_MIN_AGREE
    assert refresh["total"] > before
    assert refresh["saved"] and (tmp_path / "models/exemplars_session.npz").exists()
    swapped = _app_estimator(client)
    assert swapped.classifier.exemplars.shape[0] == refresh["total"]

    # The cap stops unbounded growth.
    monkeypatch.setattr(autocalib, "APPEND_BANK_CAP", refresh["total"])
    r = client.post(
        f"/estimate?corners={_corners_param(photo)}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert "bank_refreshed" not in r.json()


def test_no_session_persistence_without_opt_in(tmp_path, monkeypatch):
    """Structural guard: without BR_SESSION_BANK, calibration flows never write
    (a cwd-relative default silently clobbered the real bank from test runs)."""
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference import autocalib

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BR_SESSION_BANK", raising=False)
    monkeypatch.setattr(autocalib, "APPEND_MAX_CONF", 2.0)  # force a refresh attempt
    client = TestClient(create_app(_occupancy_estimator(), PositionHub()))
    photo = render_board(START_PLACEMENT)
    r = client.post(
        f"/estimate?corners={_corners_param(photo)}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    refresh = r.json()["bank_refreshed"]
    assert refresh["saved"] is None  # refresh happened in-memory only
    assert list(tmp_path.rglob("*.npz")) == []  # and nothing landed on disk


def _cycling_oracle(placement: str):
    """Logits fn recognizing crops by position in the 64-crop sequence.

    Unlike test_trained's one-shot oracle it cycles, so any number of /estimate
    calls (and ensemble members re-cropping) keep getting consistent answers.
    """
    import numpy as np

    from board_reader_core.protocol import CLASSES, NUM_CLASSES, parse_placement

    labels = [c for row in parse_placement(placement) for c in row]
    state = {"i": 0}

    def fn(batch: np.ndarray) -> np.ndarray:
        n = batch.shape[0]
        out = np.full((n, NUM_CLASSES), -5.0, dtype=np.float32)
        for j in range(n):
            out[j, CLASSES.index(labels[(state["i"] + j) % 64])] = 5.0
        state["i"] = (state["i"] + n) % 64
        return out

    return fn


def _lite_start_estimator(size: int = 64):
    """A Lite-embedder CentroidEstimator calibrated on the start position, with
    the flat-render zero-headroom geometry the Lite embedder can actually read."""
    from board_reader_inference.centroid import build_exemplars
    from board_reader_inference.embed import LiteEmbedder
    from board_reader_inference.estimator import CentroidEstimator
    from board_reader_inference.squares import crop_squares_photo, full_frame_corners

    photo = render_board(START_PLACEMENT)
    quad = full_frame_corners(photo)
    emb = LiteEmbedder()
    bank = build_exemplars(
        emb,
        crop_squares_photo(photo, quad, geometry=BboxGeometry(0.0, 0.0), size=size),
        START_PLACEMENT,
        backbone="lite",
        box_headroom=0.0,
        box_side_pad=0.0,
    )
    return CentroidEstimator(
        emb, bank, square_size=size, geometry=BboxGeometry(0.0, 0.0)
    )


def _corners_query(photo):
    from board_reader_inference.squares import full_frame_corners

    return ",".join(f"{x:.1f},{y:.1f}" for x, y in full_frame_corners(photo))


def test_calibrate_boosts_trained_estimator(tmp_path, monkeypatch):
    """/calibrate over a zero-shot trained estimator wraps it into an ensemble —
    calibration boosts, never replaces (DESIGN §4.6)."""
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.estimator import EnsembleEstimator
    from board_reader_inference.trained import TrainedEstimator

    monkeypatch.chdir(tmp_path)
    trained = TrainedEstimator(
        _cycling_oracle(START_PLACEMENT), geometry=BboxGeometry(0.0, 0.0), square_size=64
    )
    occ = _lite_start_estimator()
    client = TestClient(create_app(trained, PositionHub(), occupancy_estimator=occ))
    photo = render_board(START_PLACEMENT)
    corners = _corners_query(photo)
    r = client.post(
        f"/calibrate?corners={corners}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    assert r.json()["backbone"] == "lite"
    est = _app_estimator(client)
    assert isinstance(est, EnsembleEstimator)
    assert est.primary is trained
    r2 = client.post(
        f"/estimate?corners={corners}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r2.status_code == 200
    assert r2.json()["placement"] == START_PLACEMENT


def test_trained_path_bootstraps_bank_on_start_position(tmp_path, monkeypatch):
    """A start-position sighting upgrades the bare trained model to the ensemble
    with zero user action, and persists the bank iff BR_SESSION_BANK opts in."""
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.estimator import EnsembleEstimator
    from board_reader_inference.trained import TrainedEstimator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BR_SESSION_BANK", str(tmp_path / "bank.npz"))
    trained = TrainedEstimator(
        _cycling_oracle(START_PLACEMENT), geometry=BboxGeometry(0.0, 0.0), square_size=64
    )
    occ = _lite_start_estimator()
    client = TestClient(create_app(trained, PositionHub(), occupancy_estimator=occ))
    photo = render_board(START_PLACEMENT)
    corners = _corners_query(photo)
    r = client.post(
        f"/estimate?corners={corners}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["placement"] == START_PLACEMENT
    assert body["autocalibrated"]["matches"] == 64
    assert body["autocalibrated"]["rotation"] == 0
    assert body["autocalibrated"]["saved"] == str(tmp_path / "bank.npz")
    assert (tmp_path / "bank.npz").exists()
    assert isinstance(_app_estimator(client), EnsembleEstimator)

    # Second sighting: the booster already exists — no re-bootstrap.
    r2 = client.post(
        f"/estimate?corners={corners}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert "autocalibrated" not in r2.json()


def test_estimate_autorotate_corrects_flipped_corners(tmp_path, monkeypatch):
    """autorotate=1 undoes mislabeled corner orientation via chess plausibility
    (the 20260718-0002 field failure: a8 tagged on White's home corner)."""
    import numpy as np

    from board_reader_core.protocol import encode_placement, parse_placement
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.trained import TrainedEstimator

    monkeypatch.chdir(tmp_path)
    mid = "3r2k1/ppp2pp1/7p/2Q5/5qbP/3B4/PPPRrP2/2K5"
    flipped = encode_placement(
        np.rot90(np.array(parse_placement(mid), dtype="<U5"), 2).tolist()
    )
    trained = TrainedEstimator(
        _cycling_oracle(flipped), geometry=BboxGeometry(0.0, 0.0), square_size=64
    )
    client = TestClient(create_app(trained, PositionHub()))
    photo = render_board(flipped)
    corners = _corners_query(photo)
    r = client.post(
        f"/estimate?corners={corners}&autorotate=1",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    body = r.json()
    assert body["autorotated"] == 180
    assert body["placement"] == mid

    r2 = client.post(
        f"/estimate?corners={corners}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    body2 = r2.json()
    assert "autorotated" not in body2
    assert body2["placement"] == flipped


@pytest.mark.parametrize(
    ("rendered_turns", "expected_deg"),
    [(1, 270), (2, 180), (3, 90)],
)
def test_estimate_debug_boxes_match_the_rotated_frame_when_autorotate_fires(
    rendered_turns, expected_deg, tmp_path, monkeypatch
):
    """?debug=1&autorotate=1: when autorotate turns the answer, the debug boxes
    must be turned with it, or mosaic tile (r,c) / box index r*8+c ends up
    showing a different square than square_dist[r][c] claims (the snapshot
    page's per-square thumbnail bug this test guards).

    All three quarter-turns are exercised deliberately: 180 alone cannot catch
    an inverted rotation direction, since np.rot90(x, 2) == np.rot90(x, -2).
    """
    import numpy as np

    from board_reader_core.protocol import encode_placement, parse_placement
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference import diagnostics
    from board_reader_inference.squares import full_frame_corners, square_boxes
    from board_reader_inference.trained import TrainedEstimator

    monkeypatch.chdir(tmp_path)
    # A board photographed with its corner labels a quarter-turn off: the
    # estimator reads it exactly as rendered, and resolve_rotation turns the
    # reading back by the complementary amount (pawns can't stand on ranks 1/8).
    mid = "3r2k1/ppp2pp1/7p/2Q5/5qbP/3B4/PPPRrP2/2K5"
    flipped = encode_placement(
        np.rot90(np.array(parse_placement(mid), dtype="<U5"), rendered_turns).tolist()
    )
    trained = TrainedEstimator(
        _cycling_oracle(flipped), geometry=BboxGeometry(0.3, 0.1), square_size=64
    )
    client = TestClient(create_app(trained, PositionHub()))
    photo = render_board(flipped)
    corners = _corners_query(photo)
    quad = full_frame_corners(photo)

    # Capture the exact boxes draw_boxes is called with, then let it run for
    # real so the response's JPEG payloads are still validated below.
    seen: dict = {}
    real_draw_boxes = diagnostics.draw_boxes

    def spy(image, boxes, square_dist):
        seen["boxes"] = boxes
        return real_draw_boxes(image, boxes, square_dist)

    monkeypatch.setattr(diagnostics, "draw_boxes", spy)

    r = client.post(
        f"/estimate?corners={corners}&autorotate=1&debug=1",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    body = r.json()
    assert body["autorotated"] == expected_deg  # the rotation actually fired
    assert body["placement"] == mid  # ...and it put the position back upright
    _assert_valid_jpeg_b64(body["debug"]["boxes"])
    _assert_valid_jpeg_b64(body["debug"]["mosaic"])

    # The correct debug boxes are the ORIGINAL (unrotated-corner) boxes —
    # recognition's actual sampling geometry — reindexed by the same rot90
    # permutation as square_dist, not re-derived from a shifted corner order
    # (see test_squares.py for why that alternative is wrong once headroom or
    # side_pad is nonzero: it rotates their fixed image direction too).
    k = body["autorotated"] // 90
    original = square_boxes(quad, BboxGeometry(0.3, 0.1), photo.shape)
    expected = np.rot90(original.reshape(8, 8, 4, 2), k, axes=(0, 1)).reshape(-1, 4, 2)
    assert np.allclose(seen["boxes"], expected)


def test_default_estimator_builds_ensemble_from_session_bank(tmp_path, monkeypatch):
    """BR_MODEL + a persisted session bank -> startup ensemble, not bare model."""
    import numpy as np

    from board_reader_inference import trained as trained_mod
    from board_reader_inference.estimator import EnsembleEstimator

    bank_path = tmp_path / "bank.npz"
    _lite_start_estimator().classifier.save(bank_path)
    monkeypatch.setenv("BR_MODEL", "whatever.onnx")
    monkeypatch.setenv("BR_SESSION_BANK", str(bank_path))
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.setattr(
        trained_mod,
        "_onnx_logits_fn",
        lambda path: (lambda b: np.zeros((b.shape[0], 13), dtype=np.float32)),
    )
    est = build_from_env(None)
    assert isinstance(est, EnsembleEstimator)

    # Without a persisted bank the model runs bare.
    monkeypatch.setenv("BR_SESSION_BANK", str(tmp_path / "missing.npz"))
    from board_reader_inference.trained import TrainedEstimator

    assert isinstance(build_from_env(None), TrainedEstimator)


def test_default_estimator_degrades_when_model_unloadable(tmp_path, monkeypatch):
    """BR_MODEL pointing at nothing (fresh clone without models/) must not kill
    the server — it degrades to the next tier so `docker compose up` still works."""
    monkeypatch.setenv("BR_MODEL", str(tmp_path / "nonexistent.onnx"))
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.delenv("BR_SESSION_BANK", raising=False)
    from board_reader_inference.estimator import StubEstimator

    assert isinstance(build_from_env(None), StubEstimator)


def _two_models(tmp_path, monkeypatch, *, sidecars: bool = True) -> tuple[str, str]:
    """Two loadable ONNX-shaped models with their crop-geometry sidecars.

    run_e's bbox geometry and run_f's pose geometry — the deployed pair — because
    "each member crops with its own geometry" is the thing the cascade's env
    surface has to carry from the sidecars all the way to the server.
    """
    from pathlib import Path

    import numpy as np

    from board_reader_inference import trained as trained_mod

    def fake_session(path):
        # Mirrors ORT: a path that is not there fails at session construction.
        if not Path(path).exists():
            raise OSError(f"no such model file: {path}")
        return lambda b: np.zeros((b.shape[0], 13), dtype=np.float32)

    monkeypatch.setattr(trained_mod, "_onnx_logits_fn", fake_session)
    geometries = [
        {"headroom": 1.6, "side_pad": 0.15, "square_size": 224},
        {
            "crop_geometry": {
                "kind": "pose",
                "piece_height": 1.9,
                "piece_radius": 0.5,
                "focal_ratio": 1.0,
            },
            "square_size": 224,
        },
    ]
    paths = []
    for name, meta in zip(("a.onnx", "b.onnx"), geometries, strict=True):
        path = tmp_path / name
        path.write_bytes(b"stub")
        if sidecars:
            (tmp_path / f"{name}.meta.json").write_text(json.dumps(meta))
        paths.append(str(path))
    return paths[0], paths[1]


def test_br_model_b_builds_the_cascade(tmp_path, monkeypatch):
    """BR_MODEL alone is unchanged; BR_MODEL_B turns the pair into the cascade,
    each member on its own sidecar geometry, with BR_CASCADE_TAU moving the gate."""
    from board_reader_inference.squares import BboxGeometry, PoseGeometry
    from board_reader_inference.trained import (
        DEFAULT_CASCADE_TAU,
        CascadeEstimator,
        TrainedEstimator,
    )

    model_a, model_b = _two_models(tmp_path, monkeypatch)
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.delenv("BR_SESSION_BANK", raising=False)
    monkeypatch.delenv("BR_CASCADE_TAU", raising=False)
    monkeypatch.setenv("BR_MODEL", model_a)

    assert isinstance(build_from_env(None), TrainedEstimator)

    monkeypatch.setenv("BR_MODEL_B", model_b)
    est = build_from_env(None)
    assert isinstance(est, CascadeEstimator)
    assert est.tau == DEFAULT_CASCADE_TAU
    assert est.primary.geometry == BboxGeometry(1.6, 0.15)
    assert est.secondary.geometry == PoseGeometry(1.9, 0.5, 1.0)
    # The cascade IS the trained component: geometry and photo-mode still read
    # off it exactly as off a single model.
    assert est.crop_mode == "photo" and est.geometry == est.primary.geometry

    monkeypatch.setenv("BR_CASCADE_TAU", "0.8")
    assert build_from_env(None).tau == 0.8


def test_cascade_still_gets_boosted_by_the_session_bank(tmp_path, monkeypatch):
    """The cascade replaces the trained *component*, not the bank arithmetic: a
    persisted bank still wraps it in the same EnsembleEstimator blend."""
    from board_reader_inference.estimator import EnsembleEstimator
    from board_reader_inference.trained import CascadeEstimator

    model_a, model_b = _two_models(tmp_path, monkeypatch)
    bank_path = tmp_path / "bank.npz"
    _lite_start_estimator().classifier.save(bank_path)
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.setenv("BR_MODEL", model_a)
    monkeypatch.setenv("BR_MODEL_B", model_b)
    monkeypatch.setenv("BR_SESSION_BANK", str(bank_path))

    est = build_from_env(None)
    assert isinstance(est, EnsembleEstimator)
    assert isinstance(est.primary, CascadeEstimator)
    # ...and the session still finds the trained member inside both wrappers, so
    # /calibrate's orientation probe and the bank bootstrap keep working.
    from board_reader_inference.session import EstimatorSession

    assert isinstance(EstimatorSession(est).base_trained(), CascadeEstimator)


def test_broken_second_model_is_fatal_not_silent(tmp_path, monkeypatch):
    """Unlike BR_MODEL, BR_MODEL_B has no fresh-clone excuse: setting it is
    deliberate, and quietly serving one model would surface months later as
    'the ensemble didn't help'."""

    model_a, model_b = _two_models(tmp_path, monkeypatch)
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.delenv("BR_SESSION_BANK", raising=False)
    monkeypatch.setenv("BR_MODEL", model_a)

    # A second model with no geometry sidecar: refused, never assumed.
    bare = tmp_path / "bare.onnx"
    bare.write_bytes(b"stub")
    monkeypatch.setenv("BR_MODEL_B", str(bare))
    with pytest.raises(ValueError, match="sidecar"):
        build_from_env(None)

    # A first model that cannot load is fatal too once a cascade was requested.
    monkeypatch.setenv("BR_MODEL_B", model_b)
    monkeypatch.setenv("BR_MODEL", str(tmp_path / "nonexistent.onnx"))
    with pytest.raises(RuntimeError, match="BR_MODEL_B"):
        build_from_env(None)


def test_cascade_with_both_models_absent_degrades_like_fresh_clone(tmp_path, monkeypatch):
    """compose.yaml ships the cascade env pointing into the git-ignored models/
    directory, so a fresh clone has NEITHER file — that is the same graceful
    fresh-clone case BR_MODEL alone always had, not a misconfiguration. Only a
    partial pair fails hard (previous test)."""
    from board_reader_inference import server as srv
    from board_reader_inference.trained import CascadeEstimator, TrainedEstimator

    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.delenv("BR_SESSION_BANK", raising=False)
    monkeypatch.setenv("BR_MODEL", str(tmp_path / "models/run_e/model.onnx"))
    monkeypatch.setenv("BR_MODEL_B", str(tmp_path / "models/run_f/model.onnx"))
    est = build_from_env(None)
    session = srv.EstimatorSession(est)
    assert not isinstance(session.base_trained(), (TrainedEstimator, CascadeEstimator))


def _base_and_tuned(tmp_path, monkeypatch) -> tuple[str, str]:
    """A base model and a calibration fine-tune of it: ONE shared geometry.

    That is the whole difference from :func:`_two_models` — a fine-tune inherits
    its base's crop geometry, and the served average asserts that it did.
    """
    from pathlib import Path

    import numpy as np

    from board_reader_inference import trained as trained_mod

    def fake_session(path):
        if not Path(path).exists():
            raise OSError(f"no such model file: {path}")
        return lambda b: np.zeros((b.shape[0], 13), dtype=np.float32)

    monkeypatch.setattr(trained_mod, "_onnx_logits_fn", fake_session)
    meta = json.dumps({"headroom": 1.6, "side_pad": 0.15, "square_size": 224})
    paths = []
    for name in ("base.onnx", "tuned.onnx"):
        path = tmp_path / name
        path.write_bytes(b"stub")
        (tmp_path / f"{name}.meta.json").write_text(meta)
        paths.append(str(path))
    return paths[0], paths[1]


def test_br_model_tuned_builds_the_calibration_average(tmp_path, monkeypatch):
    """BR_MODEL_TUNED averages the fine-tune with its base, 50/50, on one geometry."""
    from board_reader_inference import server as srv
    from board_reader_inference.squares import BboxGeometry
    from board_reader_inference.trained import AveragedEstimator, TrainedEstimator

    base, tuned = _base_and_tuned(tmp_path, monkeypatch)
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.delenv("BR_SESSION_BANK", raising=False)
    monkeypatch.delenv("BR_MODEL_B", raising=False)
    monkeypatch.setenv("BR_MODEL", base)

    assert isinstance(build_from_env(None), TrainedEstimator)

    monkeypatch.setenv("BR_MODEL_TUNED", tuned)
    est = build_from_env(None)
    assert isinstance(est, AveragedEstimator)
    assert est.model_path == base and est.tuned_path == tuned
    # It IS the trained component: geometry and photo-mode read off it exactly as
    # off a single model, and a session bank still finds it to boost.
    assert est.crop_mode == "photo" and est.geometry == BboxGeometry(1.6, 0.15)
    assert isinstance(srv.EstimatorSession(est).base_trained(), AveragedEstimator)


def test_tuned_and_model_b_together_are_refused_as_unmeasured(tmp_path, monkeypatch):
    """Two validated configurations, no validated stack of them. Guessing which
    one wins would ship a model nobody measured."""

    base, tuned = _base_and_tuned(tmp_path, monkeypatch)
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.delenv("BR_SESSION_BANK", raising=False)
    monkeypatch.setenv("BR_MODEL", base)
    monkeypatch.setenv("BR_MODEL_TUNED", tuned)
    monkeypatch.setenv("BR_MODEL_B", tuned)

    with pytest.raises(RuntimeError, match="UNMEASURED"):
        build_from_env(None)


def test_tuned_without_a_usable_base_is_fatal(tmp_path, monkeypatch):
    """No fresh-clone grace for BR_MODEL_TUNED: a tuned model only exists because
    somebody ran the tool, and alone it hallucinates pieces onto empty squares."""

    base, tuned = _base_and_tuned(tmp_path, monkeypatch)
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.delenv("BR_SESSION_BANK", raising=False)
    monkeypatch.delenv("BR_MODEL_B", raising=False)
    monkeypatch.setenv("BR_MODEL_TUNED", tuned)

    # Base named but absent — the graceful degrade BR_MODEL alone would get.
    monkeypatch.setenv("BR_MODEL", str(tmp_path / "models/run_e/model.onnx"))
    with pytest.raises(RuntimeError, match="BR_MODEL_TUNED"):
        build_from_env(None)

    # Base not named at all: the same error, not a silent fall-through to the stub
    # (which would serve a fixed placement under a "calibrated" banner).
    monkeypatch.delenv("BR_MODEL")
    with pytest.raises(RuntimeError, match="BR_MODEL_TUNED"):
        build_from_env(None)


def test_tuned_average_still_gets_boosted_by_the_session_bank(tmp_path, monkeypatch):
    """Calibration arithmetic is unchanged: the average replaces the trained
    *component*, and a persisted bank still wraps it in the same blend."""
    from board_reader_inference import server as srv
    from board_reader_inference.estimator import EnsembleEstimator
    from board_reader_inference.trained import AveragedEstimator

    base, tuned = _base_and_tuned(tmp_path, monkeypatch)
    bank_path = tmp_path / "bank.npz"
    _lite_start_estimator().classifier.save(bank_path)
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.delenv("BR_MODEL_B", raising=False)
    monkeypatch.setenv("BR_MODEL", base)
    monkeypatch.setenv("BR_MODEL_TUNED", tuned)
    monkeypatch.setenv("BR_SESSION_BANK", str(bank_path))

    est = build_from_env(None)
    assert isinstance(est, EnsembleEstimator)
    assert isinstance(est.primary, AveragedEstimator)
    assert isinstance(srv.EstimatorSession(est).base_trained(), AveragedEstimator)


def test_healthz_names_both_members_of_the_average(tmp_path, monkeypatch):
    """"Which model read my board?" must be answerable from outside the process —
    and for a tuned model, "which base is it averaged with?" is half the answer."""
    from starlette.testclient import TestClient

    from board_reader_inference import server as srv
    from board_reader_inference.hub import PositionHub

    base, tuned = _base_and_tuned(tmp_path, monkeypatch)
    monkeypatch.delenv("BR_CENTROIDS", raising=False)
    monkeypatch.delenv("BR_SESSION_BANK", raising=False)
    monkeypatch.delenv("BR_MODEL_B", raising=False)
    monkeypatch.setenv("BR_MODEL", base)
    monkeypatch.setenv("BR_MODEL_TUNED", tuned)
    monkeypatch.chdir(tmp_path)

    client = TestClient(srv.create_app(build_from_env(None), PositionHub()))
    health = client.get("/healthz").json()
    assert health["tuned"] == {"model": base, "model_tuned": tuned, "weight": 0.5}
    assert "cascade" not in health


def test_healthz_reports_cascade_invocations(tmp_path, monkeypatch):
    """The cascade's economic claim (second model on <30% of squares) has to be
    checkable against the operator's own boards, not just the experiment."""
    import numpy as np
    from starlette.testclient import TestClient

    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.hub import PositionHub
    from board_reader_inference.server import create_app
    from board_reader_inference.squares import (
        BboxGeometry,
        PoseGeometry,
        full_frame_corners,
    )
    from board_reader_inference.trained import CascadeEstimator, TrainedEstimator

    monkeypatch.chdir(tmp_path)
    photo = render_board(START_PLACEMENT)
    # Model A is torn on every square (uniform logits), so every square defers.
    flat = lambda b: np.zeros((b.shape[0], 13), dtype=np.float32)  # noqa: E731
    est = CascadeEstimator(
        TrainedEstimator(flat, geometry=BboxGeometry(1.6, 0.15)),
        TrainedEstimator(flat, geometry=PoseGeometry(1.9, 0.5, 1.0)),
    )
    client = TestClient(create_app(est, PositionHub()))

    assert client.get("/healthz").json()["cascade"]["squares"] == 0
    corners = ",".join(f"{x:.1f},{y:.1f}" for x, y in full_frame_corners(photo))
    client.post(
        f"/estimate?corners={corners}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    cascade = client.get("/healthz").json()["cascade"]
    assert cascade["frames"] == 1
    assert cascade["squares"] == 64 and cascade["deferred"] == 64
    assert cascade["deferred_fraction"] == 1.0
    assert cascade["tau"] == 0.95


def test_calibrate_corrects_mislabeled_orientation(tmp_path, monkeypatch):
    """/calibrate must not trust corner orientation blindly: labels that disagree
    with the trained model's own reading of the photo by a rotation are rotated
    into agreement (2026-07-18 field incident: a 180°-off calibrate pushed a
    color-swapped bank to disk)."""
    import numpy as np

    from board_reader_core.protocol import encode_placement, parse_placement
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.estimator import EnsembleEstimator
    from board_reader_inference.trained import TrainedEstimator

    monkeypatch.chdir(tmp_path)
    trained = TrainedEstimator(
        _cycling_oracle(START_PLACEMENT), geometry=BboxGeometry(0.0, 0.0), square_size=64
    )
    occ = _lite_start_estimator()
    client = TestClient(create_app(trained, PositionHub(), occupancy_estimator=occ))
    photo = render_board(START_PLACEMENT)
    corners = _corners_query(photo)
    flipped = encode_placement(
        np.rot90(np.array(parse_placement(START_PLACEMENT), dtype="<U5"), 2).tolist()
    )
    r = client.post(
        f"/calibrate?corners={corners}&placement={flipped}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    assert r.json()["label_rotation"] == 2
    est = _app_estimator(client)
    assert isinstance(est, EnsembleEstimator)
    # The bank was calibrated with the CORRECTED labels: reading the same photo
    # yields the true orientation, not the operator's flipped claim.
    r2 = client.post(
        f"/estimate?corners={corners}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r2.json()["placement"] == START_PLACEMENT


def test_calibrate_keeps_operator_labels_when_agreeing(tmp_path, monkeypatch):
    """Correctly-oriented calibrates pass through unrotated."""
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.trained import TrainedEstimator

    monkeypatch.chdir(tmp_path)
    trained = TrainedEstimator(
        _cycling_oracle(START_PLACEMENT), geometry=BboxGeometry(0.0, 0.0), square_size=64
    )
    occ = _lite_start_estimator()
    client = TestClient(create_app(trained, PositionHub(), occupancy_estimator=occ))
    photo = render_board(START_PLACEMENT)
    r = client.post(
        f"/calibrate?corners={_corners_query(photo)}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    assert r.json()["label_rotation"] == 0


def test_calibrate_rejects_photo_not_showing_claimed_position(tmp_path, monkeypatch):
    """A mid-game photo calibrated as 'start' (field incident #4) is refused —
    no rotation of the claimed labels agrees with the model's reading."""
    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.trained import TrainedEstimator

    monkeypatch.chdir(tmp_path)
    mid = "r2qr2k/ppp2Qpp/8/2b1N3/3npn2/2N5/PPP2PPP/R3R1K1"
    trained = TrainedEstimator(
        _cycling_oracle(mid), geometry=BboxGeometry(0.0, 0.0), square_size=64
    )
    occ = _lite_start_estimator()
    client = TestClient(create_app(trained, PositionHub(), occupancy_estimator=occ))
    photo = render_board(mid)
    r = client.post(
        f"/calibrate?corners={_corners_query(photo)}",  # default placement=START
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 422
    assert "does not match" in r.json()["error"]
    # Nothing was persisted or hot-swapped: the estimator is still the bare model.
    assert _app_estimator(client) is trained


def test_estimate_warns_on_very_low_camera_angle():
    """A corner quad compressed to <0.62 aspect (camera nearly board-level) gets
    a very_low_angle warning — recognition measurably collapses there."""

    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.trained import TrainedEstimator

    # Deliberately a mid-game position, not the start: a start-position reading
    # fires _maybe_bootstrap_bank, which loads DINOv2 and embeds all 64 crops on
    # every POST (~4.5s each). The warning under test comes from the quad's
    # aspect alone, before any recognition runs, so the extra work bought nothing.
    midgame = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1"
    photo = render_board(midgame)
    h, w = photo.shape[:2]
    est = TrainedEstimator(
        _cycling_oracle(midgame), geometry=BboxGeometry(0.0, 0.0), square_size=64
    )
    client = TestClient(create_app(est, PositionHub()))
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
