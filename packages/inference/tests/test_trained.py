"""Tests for the trained-ONNX crop-classifier estimator (trained.py)."""

from __future__ import annotations

import json

import numpy as np

from board_reader_core.protocol import (
    CLASSES,
    NUM_CLASSES,
    START_PLACEMENT,
    parse_placement,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.squares import (
    DEFAULT_GEOMETRY,
    BboxGeometry,
    PoseGeometry,
    full_frame_corners,
)
from board_reader_inference.trained import TrainedEstimator, preprocess


def _oracle_logits_for(placement: str):
    """A fake network that recognizes the sim renderer's crops by brightness.

    Not a real model — it returns high logits for the true class of each crop in
    the order they arrive, which exercises batching, softmax, reshape, and the
    event assembly without any ONNX file.
    """
    labels = [c for row in parse_placement(placement) for c in row]
    calls = {"i": 0, "batches": []}

    def fn(batch: np.ndarray) -> np.ndarray:
        assert batch.dtype == np.float32 and batch.shape[1:] == (3, 224, 224)
        n = batch.shape[0]
        calls["batches"].append(n)
        out = np.full((n, NUM_CLASSES), -5.0, dtype=np.float32)
        for j in range(n):
            out[j, CLASSES.index(labels[calls["i"] + j])] = 5.0
        calls["i"] += n
        return out

    fn.calls = calls
    return fn


def test_preprocess_contract():
    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    crop[..., 0] = 255  # pure blue in BGR
    batch = preprocess([crop])
    assert batch.shape == (1, 3, 224, 224) and batch.dtype == np.float32
    # BGR->RGB: channel 2 (B in RGB layout) must carry the 255 signal.
    r, g, b = batch[0, 0].mean(), batch[0, 1].mean(), batch[0, 2].mean()
    assert b > r and b > g
    # ImageNet normalization applied: zero-input channels go negative.
    assert r < 0 and g < 0


def test_estimate_photo_roundtrip_and_batching():
    mid = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
    photo = render_board(mid)
    est = TrainedEstimator(_oracle_logits_for(mid), batch_size=48)
    frame = Frame(board_id="T1", seq=1, capture_ts=0.0, image=photo)
    event = est.estimate_photo(frame, full_frame_corners(photo))
    assert event.placement == mid
    assert event.status == "ok"
    # Batching honored: 64 crops in ceil(64/48)=2 batches.
    assert est.logits_fn.calls["batches"] == [48, 16]
    dist = np.asarray(event.square_dist)
    assert dist.shape == (8, 8, NUM_CLASSES)
    np.testing.assert_allclose(dist.sum(axis=-1), 1.0, atol=1e-5)


def test_estimate_uses_full_frame():
    est = TrainedEstimator(_oracle_logits_for(START_PLACEMENT))
    frame = Frame(board_id="T1", seq=0, capture_ts=0.0, image=render_board())
    assert est.estimate(frame).placement == START_PLACEMENT


def test_wrong_output_shape_raises():
    def bad(batch):
        return np.zeros((batch.shape[0], 7), dtype=np.float32)

    est = TrainedEstimator(bad)
    frame = Frame(board_id="T1", seq=0, capture_ts=0.0, image=render_board())
    try:
        est.estimate(frame)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_server_routes_photo_path_and_default_env(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from board_reader_core.wire import encode_jpeg

    from board_reader_inference.hub import PositionHub
    from board_reader_inference.server import create_app

    monkeypatch.chdir(tmp_path)
    photo = render_board(START_PLACEMENT)
    est = TrainedEstimator(_oracle_logits_for(START_PLACEMENT))
    client = TestClient(create_app(est, PositionHub()))
    corners = ",".join(
        f"{x:.1f},{y:.1f}" for x, y in full_frame_corners(photo)
    )
    r = client.post(
        f"/estimate?corners={corners}",
        content=encode_jpeg(photo),
        headers={"Content-Type": "image/jpeg"},
    )
    body = r.json()
    assert body["placement"] == START_PLACEMENT
    assert "mode" not in body  # trained estimator IS the zero-shot path


def test_geometry_loaded_from_model_sidecar(tmp_path, monkeypatch):
    """Crop geometry ships with the model via <model>.meta.json — a geometry
    mismatch between training and serving silently decapitates crops."""
    import json

    from board_reader_inference import trained as trained_mod

    model = tmp_path / "m.onnx"
    model.write_bytes(b"stub")
    (tmp_path / "m.onnx.meta.json").write_text(
        json.dumps({"headroom": 1.6, "side_pad": 0.25, "square_size": 224})
    )
    monkeypatch.setattr(
        trained_mod, "_onnx_logits_fn", lambda p: (lambda b: None)
    )
    est = TrainedEstimator(model)
    assert est.geometry == BboxGeometry(1.6, 0.25) and est.square_size == 224
    # An explicit geometry overrides the sidecar; no sidecar -> historical default.
    override = BboxGeometry(0.0, 0.0)
    assert TrainedEstimator(model, geometry=override).geometry == override
    bare = tmp_path / "bare.onnx"
    bare.write_bytes(b"stub")
    assert TrainedEstimator(bare).geometry == BboxGeometry(0.15, 0.15)


def test_pose_geometry_round_trips_through_the_sidecar(tmp_path, monkeypatch):
    """A pose-geometry model must serve pose-geometry crops: the sidecar is the
    only thing carrying that from training to the server."""
    import json

    from board_reader_inference import trained as trained_mod

    model = tmp_path / "pose.onnx"
    model.write_bytes(b"stub")
    (tmp_path / "pose.onnx.meta.json").write_text(
        json.dumps(
            {
                "crop_geometry": {
                    "kind": "pose",
                    "piece_height": 1.9,
                    "piece_radius": 0.5,
                    "focal_ratio": 1.0,
                },
                "square_size": 224,
            }
        )
    )
    monkeypatch.setattr(trained_mod, "_onnx_logits_fn", lambda p: (lambda b: None))
    est = TrainedEstimator(model)
    assert est.geometry == PoseGeometry(1.9, 0.5, 1.0)
    # Pose geometry has no box-padding scalars; the bbox-only exemplar bank that
    # may sit beside this model gets the deployed default, not an invented number.
    assert (est.headroom, est.side_pad) == (DEFAULT_GEOMETRY.headroom, DEFAULT_GEOMETRY.side_pad)


def test_missing_sidecar_warns_loudly(tmp_path, monkeypatch, capsys):
    """The silent v1-geometry fallback shipped a decapitating server once; a
    sidecar-less model must say so on stderr."""
    from board_reader_inference import trained as trained_mod

    monkeypatch.setattr(trained_mod, "_onnx_logits_fn", lambda p: (lambda b: None))

    bare = tmp_path / "bare.onnx"
    bare.write_bytes(b"stub")
    TrainedEstimator(bare)
    err = capsys.readouterr().err
    assert "no crop-geometry sidecar" in err and "decapitated" in err

    # An explicit geometry is a deliberate choice, not an accident — no noise.
    TrainedEstimator(bare, geometry=BboxGeometry(1.6, 0.15))
    assert capsys.readouterr().err == ""

    # A model that ships its geometry is the normal case — also no noise.
    ok = tmp_path / "ok.onnx"
    ok.write_bytes(b"stub")
    (tmp_path / "ok.onnx.meta.json").write_text(json.dumps({"headroom": 1.6}))
    TrainedEstimator(ok)
    assert capsys.readouterr().err == ""
