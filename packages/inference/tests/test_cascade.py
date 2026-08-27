"""Tests for the confidence-gated two-model cascade (trained.CascadeEstimator).

The validated combiner (2026-08-11, run_e ⊕ run_f): read every square with model
A; re-read only the squares where A's max probability fell below tau with model
B — cropped with **B's own** geometry — and average the two distributions 50/50
there. These tests pin the three things that experiment measured: which squares
defer, what the deferred squares' distribution is, and that each member crops
with its own geometry.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from board_reader_core.protocol import CLASSES, NUM_CLASSES
from board_reader_core.wire import Frame

from board_reader_inference.squares import (
    BboxGeometry,
    PoseGeometry,
    crop_squares_photo,
    full_frame_corners,
)
from board_reader_inference.trained import (
    DEFAULT_CASCADE_TAU,
    CascadeEstimator,
    CascadeMetrics,
    TrainedEstimator,
    preprocess,
)

GEO_A = BboxGeometry(headroom=1.6, side_pad=0.15)  # run_e's
GEO_B = PoseGeometry(piece_height=1.9, piece_radius=0.5, focal_ratio=1.0)  # run_f's


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class ScriptedModel:
    """A fake network that answers per BOARD SQUARE, not per call position.

    It pre-cuts the 64 crops for its geometry, preprocesses them exactly as the
    estimator will, and keys its scripted logits on the resulting tensor bytes.
    So it returns square *i*'s answer whenever square *i*'s crop is handed to it —
    whichever subset arrives, in whatever order, and only for crops cut with *its*
    geometry. That is what lets it catch the cascade mapping its "unsure" indices
    onto the wrong squares of the second model.
    """

    def __init__(self, image, corners, geometry, logits: np.ndarray) -> None:
        crops = crop_squares_photo(image, corners, geometry=geometry, size=224)
        batch = preprocess(crops)
        self._by_crop = {batch[i].tobytes(): logits[i] for i in range(len(crops))}
        assert len(self._by_crop) == 64, "crops must be distinguishable for this fake"
        self.geometry = geometry
        self._asked: set[bytes] = set()
        self.runs: list[int] = []

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        rows = [row.tobytes() for row in batch]
        self._asked.update(rows)
        self.runs.append(len(rows))
        return np.stack([self._by_crop[row] for row in rows])

    @property
    def crops_seen(self) -> int:
        """Distinct board squares this model was asked about.

        Distinct, not rows fed: a partial batch is padded up to ``batch_size``
        with repeats of its last crop (the fixed-shape fix in
        ``TrainedEstimator.crop_dist`` that stops the CUDA arena ratcheting), and
        a repeat is not a square the cascade deferred. The 64 crops are distinct
        by construction, so de-duplicating recovers exactly the deferral count.
        """
        return len(self._asked)


def _confident(cls: str, n: int = 64) -> np.ndarray:
    """(n, 13) logits that make ``cls`` win at ~1.0 probability."""
    logits = np.full((n, NUM_CLASSES), -20.0, dtype=np.float32)
    logits[:, CLASSES.index(cls)] = 20.0
    return logits


def _photo() -> tuple[np.ndarray, np.ndarray]:
    """Noise, not a rendered board: :class:`ScriptedModel` keys on crop content,
    and a real board's identical empty squares are not distinguishable."""
    image = np.random.default_rng(7).integers(0, 256, (384, 384, 3), dtype=np.uint8)
    return image, full_frame_corners(image)


def _cascade(image, corners, logits_a, logits_b, *, tau=DEFAULT_CASCADE_TAU):
    a = ScriptedModel(image, corners, GEO_A, logits_a)
    b = ScriptedModel(image, corners, GEO_B, logits_b)
    est = CascadeEstimator(
        TrainedEstimator(a, geometry=GEO_A),
        TrainedEstimator(b, geometry=GEO_B),
        tau=tau,
    )
    return est, a, b


def test_confident_squares_never_reach_the_second_model():
    """The cascade's whole economic claim: a sure square costs one model, not two."""
    image, corners = _photo()
    est, _, b = _cascade(image, corners, _confident("empty"), _confident("K"))
    dist = est.square_dist(image, corners)

    assert b.crops_seen == 0
    assert est.metrics.deferred == 0 and est.metrics.squares == 64
    np.testing.assert_allclose(dist[:, CLASSES.index("empty")], 1.0, atol=1e-5)


def test_unsure_squares_average_5050_with_the_second_model():
    image, corners = _photo()
    logits_a = _confident("empty")
    # Squares 5 and 40: A is torn between empty and P, so its max lands at 0.5.
    unsure = [5, 40]
    logits_a[unsure] = -20.0
    logits_a[unsure, CLASSES.index("empty")] = 0.0
    logits_a[unsure, CLASSES.index("P")] = 0.0
    logits_b = _confident("P")

    est, _, b = _cascade(image, corners, logits_a, logits_b)
    dist = est.square_dist(image, corners)

    # Only the two unsure squares were re-read, and only they moved...
    assert b.crops_seen == 2
    # ...in ONE fixed-shape batch, never a ragged 2-row run: a CUDA BFC arena
    # reserves fresh chunks per distinct input shape and never frees them, so a
    # secondary whose batch varies with the unsure count ratchets it to the cap.
    assert b.runs == [16]
    assert est.metrics.deferred == 2 and est.metrics.squares == 64
    expected = 0.5 * _softmax(logits_a) + 0.5 * _softmax(logits_b)
    np.testing.assert_allclose(dist[unsure], expected[unsure], atol=1e-6)
    confident = [i for i in range(64) if i not in unsure]
    np.testing.assert_allclose(dist[confident], _softmax(logits_a)[confident], atol=1e-6)
    # ...and the average flipped their argmax to B's answer.
    assert dist[unsure].argmax(-1).tolist() == [CLASSES.index("P")] * 2


def test_second_model_is_asked_about_the_right_squares():
    """A misindexed defer would read square 5's crop as square 40's answer — the
    one bug a call-order-scripted fake cannot see, so the fake is crop-keyed."""
    image, corners = _photo()
    logits_a = np.full((64, NUM_CLASSES), 0.0, dtype=np.float32)  # max prob 1/13
    logits_b = np.random.default_rng(0).normal(size=(64, NUM_CLASSES)).astype(np.float32)

    est, _, b = _cascade(image, corners, logits_a, logits_b)
    dist = est.square_dist(image, corners)

    assert b.crops_seen == 64  # nothing is confident at 1/13
    np.testing.assert_allclose(
        dist, 0.5 * _softmax(logits_a) + 0.5 * _softmax(logits_b), atol=1e-6
    )


def test_per_square_confidence_comes_from_the_distribution_actually_used():
    """Confidence on the wire is the max of the blend where B was consulted, and
    A's own max where it was not — never a mix of the two."""
    image, corners = _photo()
    logits_a = _confident("empty")
    logits_a[9] = 0.0  # flat: max prob 1/13, well under tau
    logits_b = _confident("Q")

    est, _, _ = _cascade(image, corners, logits_a, logits_b)
    event = est.estimate_photo(
        Frame(board_id="C1", seq=3, capture_ts=0.0, image=image), corners
    )
    dist = np.asarray(event.square_dist).reshape(64, NUM_CLASSES)

    blended = dist[9]
    assert blended.argmax() == CLASSES.index("Q")
    # 0.5*(1/13) + 0.5*(~1.0) for Q; the untouched squares keep A's ~1.0.
    assert 0.5 < blended.max() < 0.56
    np.testing.assert_allclose(dist[0].max(), 1.0, atol=1e-5)
    np.testing.assert_allclose(dist.sum(axis=-1), 1.0, atol=1e-5)


def test_each_member_crops_with_its_own_geometry():
    """Two geometries is the point: feeding B the primary's crops throws away the
    complementary view the ensemble win came from."""
    image, corners = _photo()
    est, a, b = _cascade(
        image, corners, np.zeros((64, NUM_CLASSES), np.float32), _confident("k")
    )
    est.square_dist(image, corners)

    assert a.geometry == GEO_A and b.geometry == GEO_B
    # ScriptedModel only recognizes crops cut with the geometry it was built for,
    # so both members answering at all proves each got its own crops. Belt and
    # braces: the two geometries genuinely cut different pixels here.
    cuts_a = crop_squares_photo(image, corners, geometry=GEO_A, size=224)
    cuts_b = crop_squares_photo(image, corners, geometry=GEO_B, size=224)
    assert not np.array_equal(cuts_a[27], cuts_b[27])


def test_tau_moves_the_gate():
    image, corners = _photo()
    logits_a = _confident("empty")
    logits_a[3] = 0.0  # flat 13-way: max prob 1/13 ~ 0.077
    logits_a[4] = -20.0  # two-way tie: max prob 0.5
    logits_a[4, [CLASSES.index("empty"), CLASSES.index("P")]] = 0.0

    for tau, deferred in ((0.05, 0), (0.4, 1), (0.9, 2), (DEFAULT_CASCADE_TAU, 2)):
        est, _, b = _cascade(image, corners, logits_a, _confident("P"), tau=tau)
        est.square_dist(image, corners)
        assert b.crops_seen == deferred, f"tau={tau}"


def test_degenerate_tau_rejected():
    image, corners = _photo()
    a = TrainedEstimator(ScriptedModel(image, corners, GEO_A, _confident("empty")), geometry=GEO_A)
    for tau in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="tau"):
            CascadeEstimator(a, a, tau=tau)


def test_member_without_a_geometry_sidecar_is_refused(tmp_path, monkeypatch):
    """A lone model may fall back to the guessed v1 box with a warning; a cascade
    member may not — a member cropping on the wrong geometry decapitates its own
    crops and the ensemble just quietly stops helping."""
    from board_reader_inference import trained as trained_mod

    monkeypatch.setattr(trained_mod, "_onnx_logits_fn", lambda p: (lambda b: None))

    def _model(name: str, meta: dict | None) -> TrainedEstimator:
        path = tmp_path / name
        path.write_bytes(b"stub")
        if meta is not None:
            (tmp_path / f"{name}.meta.json").write_text(json.dumps(meta))
        return TrainedEstimator(path)

    with_sidecar = _model("a.onnx", {"headroom": 1.6, "side_pad": 0.15})
    bare = _model("b.onnx", None)
    assert with_sidecar.geometry_source == "sidecar"
    assert bare.geometry_source == "fallback"

    with pytest.raises(ValueError, match=r"secondary.*sidecar"):
        CascadeEstimator(with_sidecar, bare)
    with pytest.raises(ValueError, match=r"primary.*sidecar"):
        CascadeEstimator(bare, with_sidecar)
    # An explicit geometry is a deliberate choice and satisfies the requirement.
    CascadeEstimator(with_sidecar, TrainedEstimator(tmp_path / "b.onnx", geometry=GEO_B))


def test_geometry_and_mode_delegate_to_the_primary():
    """The cascade must sit where a single trained model sits: same photo-mode
    declaration, same geometry for ?debug=1 and for a bank built beside it."""
    image, corners = _photo()
    est, _, _ = _cascade(image, corners, _confident("empty"), _confident("P"))

    assert est.crop_mode == "photo"
    assert est.geometry == GEO_A
    assert (est.headroom, est.side_pad) == (1.6, 0.15)
    assert est.square_size == 224
    assert est.orientation == "a1_bottom_left"


def test_metrics_accumulate_across_frames():
    image, corners = _photo()
    logits_a = _confident("empty")
    logits_a[[1, 2]] = 0.0
    est, _, _ = _cascade(image, corners, logits_a, _confident("P"))

    for _ in range(3):
        est.square_dist(image, corners)

    assert est.metrics == CascadeMetrics(frames=3, squares=192, deferred=6)
    assert est.metrics.deferred_fraction == pytest.approx(6 / 192)


def test_estimate_falls_back_to_the_full_frame():
    """The rectified path: no corners means the whole frame is the board."""
    image, _ = _photo()
    corners = full_frame_corners(image)
    est, _, _ = _cascade(image, corners, _confident("empty"), _confident("P"))
    event = est.estimate(Frame(board_id="C2", seq=0, capture_ts=0.0, image=image))
    assert event.placement == "8/8/8/8/8/8/8/8"
