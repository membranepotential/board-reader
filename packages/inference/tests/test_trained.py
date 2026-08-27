"""Tests for the trained-ONNX crop-classifier estimator (trained.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from board_reader_core.protocol import (
    CLASSES,
    NUM_CLASSES,
    START_PLACEMENT,
    parse_placement,
    placement_from_dist,
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
        # A tail batch is padded up to batch_size with repeats of its last real
        # crop (the arena-ratchet fix), so only the first `real` rows of the
        # final batch map onto labels; the padded ones are sliced off by the
        # caller and can answer anything.
        real = min(n, len(labels) - calls["i"])
        out = np.full((n, NUM_CLASSES), -5.0, dtype=np.float32)
        for j in range(n):
            out[j, CLASSES.index(labels[min(calls["i"] + j, len(labels) - 1)])] = 5.0
        calls["i"] += real
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
    # Batching honored: 64 crops in ceil(64/48)=2 batches — and BOTH runs see 48
    # rows, the 16-crop tail padded up (see test_partial_batches_* below).
    assert est.logits_fn.calls["batches"] == [48, 48]
    dist = np.asarray(event.square_dist)
    assert dist.shape == (8, 8, NUM_CLASSES)
    np.testing.assert_allclose(dist.sum(axis=-1), 1.0, atol=1e-5)


def _content_logits_fn(seen: list[int]):
    """Deterministic per-row logits from that row's own pixels.

    Content-keyed rather than call-order-keyed, so a padded row cannot shift
    another row's answer: any difference between a padded and an unpadded run is
    a real slicing bug, not fake bookkeeping. ``seen`` records every batch's row
    count.
    """

    def fn(batch: np.ndarray) -> np.ndarray:
        seen.append(batch.shape[0])
        flat = batch.reshape(batch.shape[0], -1)
        return np.ascontiguousarray(flat[:, :NUM_CLASSES] * 4.0, dtype=np.float32)

    return fn


@pytest.mark.parametrize("n", [1, 15, 16, 17, 33])
def test_partial_batches_are_padded_to_one_fixed_shape(n):
    """Every session.run sees exactly ``batch_size`` rows — the arena-ratchet fix.

    A CUDA BFC arena never frees and reserves a fresh chunk set per distinct
    input shape, so a varying tail batch (or a cascade secondary's varying
    "unsure" count) creeps the arena up to BR_GPU_MEM_LIMIT_MB, after which every
    large request fails. Padding to one shape must be invisible in the answer.
    """
    rng = np.random.default_rng(7)
    crops = [rng.integers(0, 256, (8, 8, 3), dtype=np.uint8) for _ in range(n)]

    seen: list[int] = []
    out = TrainedEstimator(_content_logits_fn(seen), batch_size=16).crop_dist(crops)
    assert out.shape == (n, NUM_CLASSES)
    assert seen == [16] * ((n + 15) // 16)

    # batch_size=1 never pads, so it is the unpadded reference. Bit-for-bit.
    ref_seen: list[int] = []
    ref = TrainedEstimator(_content_logits_fn(ref_seen), batch_size=1).crop_dist(crops)
    assert ref_seen == [1] * n
    np.testing.assert_array_equal(out, ref)


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


def test_photo_dist_is_raw_and_the_event_is_calibrated():
    """The two halves of the read seam: composites blend ``photo_dist`` (raw, so
    the black-bishop demotion cannot land on a member), and ``estimate_photo``
    is where that calibration is applied — exactly once, on the way out."""
    from board_reader_core.protocol import CLASS_INDEX

    from board_reader_inference.estimator import demote_attractor

    photo = render_board(START_PLACEMENT)
    probs = np.full((64, NUM_CLASSES), 0.001, dtype=np.float64)
    for i, cls in enumerate(c for row in parse_placement(START_PLACEMENT) for c in row):
        probs[i, CLASS_INDEX[cls]] = 0.9
    probs[2] = 0.001
    probs[2, CLASS_INDEX["b"]] = 0.40  # c8: a narrow bishop win over the rook
    probs[2, CLASS_INDEX["r"]] = 0.32
    probs /= probs.sum(axis=-1, keepdims=True)

    def logits(batch: np.ndarray) -> np.ndarray:
        # Board order across batches, clamped: a tail batch is padded up to
        # batch_size with repeats of its last real crop and sliced back off by
        # the caller, so the counter must advance by the REAL rows only. A fake
        # that ignored this would answer rank 8 to every batch and "read" a board
        # of four identical back ranks while the assertions still passed.
        n = batch.shape[0]
        rows = [probs[min(logits.i + j, 63)] for j in range(n)]
        logits.i += min(n, 64 - logits.i)
        return np.log(np.stack(rows)).astype(np.float32)

    logits.i = 0
    est = TrainedEstimator(logits)  # default batch_size=16: four batches
    corners = full_frame_corners(photo)
    event = est.estimate_photo(Frame(board_id="T1", seq=0, capture_ts=0.0, image=photo), corners)
    logits.i = 0
    raw = est.photo_dist(photo, corners)

    assert placement_from_dist(raw) == START_PLACEMENT  # the whole board, untouched
    assert event.placement == "rnrqkbnr/" + START_PLACEMENT.split("/", 1)[1]
    np.testing.assert_allclose(event.square_dist, demote_attractor(raw), atol=1e-7)
