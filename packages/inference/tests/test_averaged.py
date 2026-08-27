"""Tests for the calibration average (trained.AveragedEstimator).

The validated composition (2026-08-11, ``scratch/experiments/calib_finetune_confirm``):
a model fine-tuned on the operator's own start-position photos, averaged 50/50 with
the base it was fine-tuned from, on **every** square — no gate. Both members share
one crop geometry, which is what lets the photo be warped once instead of twice.

These tests pin the three things that make that true in the serving code: the
arithmetic is a plain 50/50 probability average, the crops really are cut once and
handed to both models, and a pair that does *not* share a geometry is refused rather
than quietly averaged (it would be a different composition than the measured one).
"""

from __future__ import annotations

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
    AveragedEstimator,
    TrainedEstimator,
    preprocess,
)

GEO = BboxGeometry(headroom=1.6, side_pad=0.15)  # run_e's — the tuned model inherits it


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class CountingModel:
    """Scripted logits keyed on CROP CONTENT, counting the crops it was shown.

    Keyed on content, not call order, for the reason ``test_cascade``'s fake is:
    only that catches a composite handing a model the wrong squares. The count is
    what proves the single crop pass — two models, 64 crops each, one warp.
    """

    def __init__(self, image, corners, logits: np.ndarray) -> None:
        batch = preprocess(crop_squares_photo(image, corners, geometry=GEO, size=224))
        self._by_crop = {batch[i].tobytes(): logits[i] for i in range(len(batch))}
        assert len(self._by_crop) == 64, "crops must be distinguishable for this fake"
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
        a repeat is not a square anyone asked about. The 64 crops are distinct by
        construction, so de-duplicating recovers exactly the squares read.
        """
        return len(self._asked)


def _photo() -> tuple[np.ndarray, np.ndarray]:
    """Noise, not a rendered board: the fakes key on crop content, and a real
    board's identical empty squares are not distinguishable."""
    image = np.random.default_rng(11).integers(0, 256, (384, 384, 3), dtype=np.uint8)
    return image, full_frame_corners(image)


def _confident(cls: str, n: int = 64) -> np.ndarray:
    logits = np.full((n, NUM_CLASSES), -20.0, dtype=np.float32)
    logits[:, CLASSES.index(cls)] = 20.0
    return logits


def _averaged(image, corners, logits_base, logits_tuned):
    base_fn = CountingModel(image, corners, logits_base)
    tuned_fn = CountingModel(image, corners, logits_tuned)
    est = AveragedEstimator(
        TrainedEstimator(base_fn, geometry=GEO),
        TrainedEstimator(tuned_fn, geometry=GEO),
    )
    return est, base_fn, tuned_fn


def test_every_square_is_a_5050_average_of_both_models():
    """No gate: unlike the cascade, both models answer every square."""
    image, corners = _photo()
    rng = np.random.default_rng(3)
    logits_base = rng.normal(size=(64, NUM_CLASSES)).astype(np.float32)
    logits_tuned = rng.normal(size=(64, NUM_CLASSES)).astype(np.float32)

    est, base_fn, tuned_fn = _averaged(image, corners, logits_base, logits_tuned)
    dist = est.square_dist(image, corners)

    expected = 0.5 * _softmax(logits_base) + 0.5 * _softmax(logits_tuned)
    np.testing.assert_allclose(dist, expected, atol=1e-6)
    assert base_fn.crops_seen == 64 and tuned_fn.crops_seen == 64


def test_the_base_vetoes_the_tuned_models_hallucination():
    """The measured mechanism, in miniature: the tuned model is *fairly* sure a
    square holds a pawn, the base is *very* sure it is empty, and the average
    keeps it empty. That veto is why hallucinations stayed at the base's 1 on all
    ten composed runs, and why the tuned model is never served alone."""
    image, corners = _photo()
    logits_base = _confident("empty")
    logits_tuned = np.full((64, NUM_CLASSES), -20.0, dtype=np.float32)
    logits_tuned[:, CLASSES.index("P")] = 2.0
    logits_tuned[:, CLASSES.index("empty")] = 0.0

    est, _, _ = _averaged(image, corners, logits_base, logits_tuned)
    dist = est.square_dist(image, corners)

    assert set(np.asarray(CLASSES)[dist.argmax(-1)]) == {"empty"}
    # The tuned model alone would have said "P" on all 64.
    assert _softmax(logits_tuned).argmax(-1).tolist() == [CLASSES.index("P")] * 64


def test_the_photo_is_warped_once_for_both_models(monkeypatch):
    """The economic claim: sharing a geometry means one crop pass, not two."""
    import board_reader_inference.trained as trained_mod

    image, corners = _photo()
    est, base_fn, tuned_fn = _averaged(
        image, corners, _confident("empty"), _confident("K")
    )
    warps = 0

    def counted(*args, **kwargs):
        nonlocal warps
        warps += 1
        return crop_squares_photo(*args, **kwargs)

    monkeypatch.setattr(trained_mod, "crop_squares_photo", counted)
    est.square_dist(image, corners)

    assert warps == 1
    assert base_fn.crops_seen == 64 and tuned_fn.crops_seen == 64


def test_a_subset_of_squares_costs_that_subset_only():
    image, corners = _photo()
    est, base_fn, tuned_fn = _averaged(
        image, corners, _confident("empty"), _confident("K")
    )
    dist = est.square_dist(image, corners, squares=[0, 7, 63])
    assert dist.shape == (3, NUM_CLASSES)
    assert base_fn.crops_seen == 3 and tuned_fn.crops_seen == 3
    # Three squares, cut once each — but fed as ONE fixed-shape batch per model,
    # never a ragged 3-row run (the GPU arena reserves per input shape).
    assert base_fn.runs == [16] and tuned_fn.runs == [16]


def test_mismatched_geometry_is_refused():
    """A fine-tune inherits its base's geometry. Two different ones mean this is
    not a base/fine-tune pair at all, and averaging off one crop pass would feed
    one of them crops it never trained on."""
    with pytest.raises(ValueError, match="share one crop geometry"):
        AveragedEstimator(
            TrainedEstimator(lambda b: np.zeros((len(b), 13)), geometry=GEO),
            TrainedEstimator(
                lambda b: np.zeros((len(b), 13)),
                geometry=PoseGeometry(1.9, 0.5, 1.0),
            ),
        )


def test_mismatched_square_size_is_refused():
    with pytest.raises(ValueError, match="share one crop geometry"):
        AveragedEstimator(
            TrainedEstimator(lambda b: np.zeros((len(b), 13)), geometry=GEO),
            TrainedEstimator(
                lambda b: np.zeros((len(b), 13)), geometry=GEO, square_size=112
            ),
        )


def test_a_member_without_a_geometry_sidecar_is_refused(tmp_path, monkeypatch):
    """Same rule as the cascade: guessing a member's geometry decapitates its
    crops, and the symptom is "the tuned model didn't help", months later."""
    import board_reader_inference.trained as trained_mod

    monkeypatch.setattr(
        trained_mod,
        "_onnx_logits_fn",
        lambda path: (lambda b: np.zeros((len(b), 13), dtype=np.float32)),
    )
    bare = tmp_path / "bare.onnx"
    bare.write_bytes(b"stub")

    with pytest.raises(ValueError, match="sidecar"):
        AveragedEstimator(
            TrainedEstimator(bare),
            TrainedEstimator(lambda b: np.zeros((len(b), 13)), geometry=GEO),
        )


def test_it_is_a_drop_in_trained_component():
    """Everything downstream — the booster blend, /calibrate's probe, ?debug=1 —
    reads geometry, photo-mode and provenance off the trained component. The
    average answers all of them, delegating to the base."""
    image, corners = _photo()
    est, _, _ = _averaged(image, corners, _confident("empty"), _confident("K"))

    assert est.crop_mode == "photo"
    assert est.geometry == GEO
    assert est.headroom == GEO.headroom and est.side_pad == GEO.side_pad
    assert est.square_size == 224
    assert est.orientation == "a1_bottom_left"

    event = est.estimate_photo(
        Frame(board_id="A1", seq=3, capture_ts=1.5, image=image), corners
    )
    assert np.asarray(event.square_dist).shape == (8, 8, NUM_CLASSES)
    assert event.status == "ok" and event.seq == 3
