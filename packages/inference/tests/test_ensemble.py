"""Tests for the calibration-as-booster blend (EnsembleEstimator)."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import (
    NUM_CLASSES,
    PositionEvent,
    dist_from_placement,
    iso_utc,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.estimator import EnsembleEstimator


class FakePhotoEstimator:
    """Photo-mode estimator emitting a fixed square_dist (pixels ignored)."""

    crop_mode = "photo"

    def __init__(self, dist: np.ndarray, *, headroom: float = 0.15) -> None:
        self.dist = np.asarray(dist, dtype=np.float32)
        self.headroom = headroom
        self.side_pad = 0.15
        self.square_size = 224
        self.orientation = "a1_bottom_left"

    def estimate(self, frame: Frame) -> PositionEvent:
        return self.estimate_photo(frame, None)

    def estimate_photo(self, frame: Frame, corners) -> PositionEvent:
        return PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc=iso_utc(frame.capture_ts),
            square_dist=self.dist,
        )


def _frame() -> Frame:
    return Frame(board_id="E1", seq=0, capture_ts=0.0, image=render_board())


def test_blend_tracks_the_confident_member_per_square():
    """Where one member is sure and the other guesses, the blend follows the
    sure one — the measured mechanism behind 439 > max(415, 430) of 448."""
    a = np.asarray(dist_from_placement("8/8/8/8/8/8/8/8"), dtype=np.float32)
    b = a.copy()
    # Member A is confident 'P' on a1, member B mildly prefers 'empty' there.
    a[7, 0] = 0.0
    a[7, 0, 1] = 1.0  # CLASSES[1] == "P"
    b[7, 0] = 0.0
    b[7, 0, 0] = 0.55
    b[7, 0, 1] = 0.45
    ens = EnsembleEstimator(FakePhotoEstimator(a), FakePhotoEstimator(b))
    event = ens.estimate_photo(_frame(), None)
    assert event.placement.endswith("/P7")
    dist = np.asarray(event.square_dist)
    assert dist.shape == (8, 8, NUM_CLASSES)
    np.testing.assert_allclose(dist.sum(axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(dist[7, 0, 1], 0.5 * 1.0 + 0.5 * 0.45, atol=1e-6)


def test_weight_shifts_the_blend():
    a = np.asarray(dist_from_placement("8/8/8/8/8/8/8/8"), dtype=np.float32)
    b = a.copy()
    a[0, 0] = 0.0
    a[0, 0, 1] = 1.0
    b[0, 0] = 0.0
    b[0, 0, 0] = 1.0
    heavy_b = EnsembleEstimator(FakePhotoEstimator(a), FakePhotoEstimator(b), weight=0.2)
    assert heavy_b.estimate_photo(_frame(), None).placement.startswith("8/")
    heavy_a = EnsembleEstimator(FakePhotoEstimator(a), FakePhotoEstimator(b), weight=0.8)
    assert heavy_a.estimate_photo(_frame(), None).placement.startswith("P7/")


def test_degenerate_weights_rejected():
    a = FakePhotoEstimator(np.asarray(dist_from_placement("8/8/8/8/8/8/8/8")))
    for w in (0.0, 1.0, -0.1, 1.7):
        with pytest.raises(ValueError):
            EnsembleEstimator(a, a, weight=w)


def test_geometry_and_mode_delegate_to_primary():
    a = FakePhotoEstimator(np.asarray(dist_from_placement("8/8/8/8/8/8/8/8")), headroom=1.6)
    b = FakePhotoEstimator(np.asarray(dist_from_placement("8/8/8/8/8/8/8/8")), headroom=0.0)
    ens = EnsembleEstimator(a, b)
    assert ens.crop_mode == "photo"
    assert ens.headroom == 1.6
    assert ens.side_pad == 0.15
    assert ens.square_size == 224
    assert ens.orientation == "a1_bottom_left"
