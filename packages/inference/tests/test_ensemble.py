"""Tests for the calibration-as-booster blend (EnsembleEstimator)."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.protocol import (
    CLASS_INDEX,
    NUM_CLASSES,
    PositionEvent,
    dist_from_placement,
)
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame

from board_reader_inference.estimator import EnsembleEstimator, read_event


class FakePhotoEstimator:
    """Photo-mode member returning a fixed RAW square_dist (pixels ignored).

    A :class:`~board_reader_inference.estimator.PhotoDistReader`: the blend
    composes members' distributions, not their events, so the fake hands back the
    same thing a real member does — uncalibrated.
    """

    crop_mode = "photo"

    def __init__(self, dist: np.ndarray, *, headroom: float = 0.15) -> None:
        self.dist = np.asarray(dist, dtype=np.float32)
        self.headroom = headroom
        self.side_pad = 0.15
        self.square_size = 224
        self.orientation = "a1_bottom_left"
        self.reads = 0

    def photo_dist(self, image: np.ndarray, corners) -> np.ndarray:
        self.reads += 1
        return self.dist

    def estimate(self, frame: Frame) -> PositionEvent:
        return self.estimate_photo(frame, None)

    def estimate_photo(self, frame: Frame, corners) -> PositionEvent:
        return read_event(
            frame, self.photo_dist(frame.image, corners), orientation=self.orientation
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


def test_the_attractor_demotion_lands_on_the_blend_not_on_the_members():
    """The blend composes RAW member distributions, so the black-bishop
    calibration is applied once — to what leaves the ensemble.

    The discriminator is a member that would fire on its own: A reads c8 as 'b'
    over 'q' by 0.10 and B reads it 'b' by a mile. Blending first leaves c8 at
    0.55 'b' and nothing to demote; demoting A first would blend a swapped row
    and serve 0.50 instead. (A's own read is checked below, so this is not
    passing by the rule never firing.)"""
    a = np.asarray(dist_from_placement("8/8/8/8/8/8/8/8"), dtype=np.float32)
    b = a.copy()
    a[0, 2] = 0.0
    a[0, 2, CLASS_INDEX["b"]] = 0.50
    a[0, 2, CLASS_INDEX["q"]] = 0.40
    b[0, 2] = 0.0
    b[0, 2, CLASS_INDEX["b"]] = 0.60
    b[0, 2, CLASS_INDEX["q"]] = 0.05

    member_a = FakePhotoEstimator(a)
    frame = _frame()
    assert member_a.estimate_photo(frame, None).placement.startswith("2q5/")

    event = EnsembleEstimator(member_a, FakePhotoEstimator(b)).estimate_photo(frame, None)
    assert event.placement.startswith("2b5/")
    assert event.square_dist[0, 2, CLASS_INDEX["b"]] == pytest.approx(0.55)
    # Each member was asked for its distribution exactly once per read.
    assert member_a.reads == 2  # its own read above, then the blend's


def test_the_blend_demotes_what_the_members_could_not_see_alone():
    """The other direction: neither member's own read fires, the blend's does."""
    a = np.asarray(dist_from_placement("8/8/8/8/8/8/8/8"), dtype=np.float32)
    b = a.copy()
    a[0, 2] = 0.0
    a[0, 2, CLASS_INDEX["b"]] = 0.50
    a[0, 2, CLASS_INDEX["q"]] = 0.10
    b[0, 2] = 0.0
    b[0, 2, CLASS_INDEX["q"]] = 0.50
    b[0, 2, CLASS_INDEX["b"]] = 0.44

    # A wins c8 for 'b' by 0.40 (nothing to demote); B reads it 'q' already.
    assert FakePhotoEstimator(a).estimate_photo(_frame(), None).placement.startswith("2b5/")
    assert FakePhotoEstimator(b).estimate_photo(_frame(), None).placement.startswith("2q5/")

    event = EnsembleEstimator(FakePhotoEstimator(a), FakePhotoEstimator(b)).estimate_photo(
        _frame(), None
    )
    # Blended: b 0.47, q 0.30 — a 0.17 win, inside the band, so it goes to q.
    assert event.placement.startswith("2q5/")
    assert event.square_dist[0, 2, CLASS_INDEX["q"]] == pytest.approx(0.47)
